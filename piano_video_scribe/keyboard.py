"""Keyboard detection and note-to-pixel mapping utilities.

Functions for detecting piano keyboards in video frames, mapping MIDI notes
to pixel positions, and building detector regions for color sampling.
"""

import cv2
import numpy as np

from piano_video_scribe.constants import WHITE_SEMITONES


# ---------------------------------------------------------------------------
# Keyboard detection
# ---------------------------------------------------------------------------

def _frame_has_keyboard(frame):
    """Check if this frame shows a piano keyboard by scanning for regularly-spaced white keys.

    Looks for a horizontal row in the bottom half with 15+ white segments
    at consistent spacing (CV < 0.15). No dark pixel requirement — the
    regularity check alone rejects non-keyboard frames (blank screens,
    title cards, etc.) since they don't produce 15+ evenly-spaced segments.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for y in range(h - 5, h * 3 // 4, -5):
        row = hsv[y, :]
        white = int(np.sum((row[:, 1] < 60) & (row[:, 2] > 180)))
        if white < 400:
            continue
        keys = []
        in_k = False
        sx = 0
        for x in range(w):
            is_w = int(row[x, 1]) < 60 and int(row[x, 2]) > 180
            if is_w and not in_k:
                in_k = True
                sx = x
            elif not is_w and in_k:
                if x - sx > 8:
                    keys.append((sx + x) // 2)
                in_k = False
        if len(keys) >= 15:
            diffs = np.diff(keys)
            cv = float(np.std(diffs) / np.mean(diffs)) if np.mean(diffs) > 0 else 999
            if cv < 0.15:
                return True
    return False


def _auto_find_keyboard_frame(cap):
    """Scan forward from frame 0 to find the first frame showing a keyboard."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for candidate in range(0, min(total, 1800), 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ret, frm = cap.read()
        if ret and _frame_has_keyboard(frm):
            return candidate
    return 5  # fallback


def detect_keyboard(cap, frame_idx=None):
    """Detect white and black key x-centers from a clean keyboard frame (no notes).

    Uses HSV analysis to find the white-key row automatically (scans bottom-up
    for the first row with many white pixels and few dark pixels), then detects
    black keys just above that zone.

    If frame_idx is None, auto-scans forward from frame 0 to find the first
    frame showing a keyboard (handles videos with intros/title screens).

    Args:
        cap: OpenCV VideoCapture object.
        frame_idx: Frame index to use (auto-detected if None).

    Returns:
        (white_keys, black_keys, y_white, y_black).
    """
    if frame_idx is not None:
        # Try the specified frame; if it doesn't show a keyboard, auto-scan
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or not _frame_has_keyboard(frame):
            old = frame_idx
            frame_idx = _auto_find_keyboard_frame(cap)
            print(f"Frame {old} has no keyboard, auto-detected frame {frame_idx}")
    else:
        frame_idx = _auto_find_keyboard_frame(cap)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Found keyboard at frame {frame_idx} ({frame_idx / fps:.1f}s)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx}")

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Auto-find y for white keys: try multiple candidate rows and pick the one
    # with the most regular spacing (lowest coefficient of variation).
    # Some rows have artifacts (text labels like C2/C3/C4, key edges) that
    # create spurious detections, so we scan a range and pick the best.

    # Adaptive brightness thresholds for dark keyboards.
    # Standard: S<60, V>180. If that finds nothing, lower V threshold
    # based on actual frame brightness.
    bottom_half = hsv[h // 2:, :, 2]  # V channel of bottom half
    max_v = int(bottom_half.max())
    if max_v > 150:
        s_thresh, v_thresh = 60, 180
        min_white_px = 600
        min_white_scan = 400
    else:
        # Dark keyboard — scale thresholds to actual brightness
        v_thresh = max(3, max_v // 3)
        s_thresh = 30
        min_white_px = 20  # very few bright pixels on dim keyboards
        min_white_scan = 15
        print(f"Dark keyboard detected (max V={max_v}), using adaptive thresholds: S<{s_thresh}, V>{v_thresh}")

    min_seg_w = 3 if max_v < 150 else 8  # narrower segments on dark keyboards

    def _scan_white_row(hsv_img, y_scan):
        """Detect white key x-centers at a given y row."""
        row = hsv_img[y_scan, :]
        keys = []
        in_k = False
        sx = 0
        for x in range(hsv_img.shape[1]):
            is_w = int(row[x, 1]) < s_thresh and int(row[x, 2]) > v_thresh
            if is_w and not in_k:
                in_k = True
                sx = x
            elif not is_w and in_k:
                if x - sx > min_seg_w:
                    keys.append((sx + x) // 2)
                in_k = False
        if in_k and hsv_img.shape[1] - sx > min_seg_w:
            keys.append((sx + hsv_img.shape[1]) // 2)
        return keys

    # First find the white key zone (bottom-up scan for white-dominant row)
    # For dark keyboards, skip the dark_count check (everything looks dark).
    y_white_zone = h - 30
    for y_scan in range(h - 5, h // 2, -1):
        row_test = hsv[y_scan, :]
        white_count = int(np.sum((row_test[:, 1] < s_thresh) & (row_test[:, 2] > v_thresh)))
        if max_v > 150:
            dark_count = int(np.sum(row_test[:, 2] < 80))
            if white_count > min_white_px and dark_count < 100:
                y_white_zone = y_scan
                break
        else:
            # Dark keyboard: just find the row with the most white pixels
            if white_count > min_white_px:
                y_white_zone = y_scan
                break

    # Try multiple y-values around the detected zone, pick the most regular
    best_y = y_white_zone
    best_cv = float('inf')
    best_raw = _scan_white_row(hsv, y_white_zone)
    search_lo = max(y_white_zone - 80, h // 2)
    search_hi = min(y_white_zone + 5, h - 1)
    for y_try in range(search_lo, search_hi + 1):
        row_test = hsv[y_try, :]
        white_count = int(np.sum((row_test[:, 1] < s_thresh) & (row_test[:, 2] > v_thresh)))
        if white_count < min_white_scan:
            continue
        raw = _scan_white_row(hsv, y_try)
        if len(raw) < 15:
            continue
        diffs = np.diff(raw)
        cv = float(np.std(diffs) / np.mean(diffs)) if np.mean(diffs) > 0 else 999
        if cv < best_cv:
            best_cv = cv
            best_y = y_try
            best_raw = raw

    y_white = best_y
    raw_white = best_raw
    print(f"White key scan y={y_white} (regularity={best_cv:.3f})")

    # Regularize: apply linear regression to remove outliers (some keys may be
    # mis-detected when notes are partially visible on them)
    white_keys = regularize_positions(raw_white)
    print(f"Detected {len(raw_white)} raw white keys → {len(white_keys)} regularized")

    # Auto-find y for black keys: scan upward from white key zone and pick
    # the row with the most dark pixels that still has enough white pixels
    # (the actual black-key body, not just the narrow gap between keys).
    y_black = y_white - 50
    best_dark = 0
    max_search = min(y_white - 5, h - 1)
    min_search = max(y_white - int(h * 0.2), h // 2)
    for y_scan in range(max_search, min_search, -1):
        row_test = hsv[y_scan, :]
        dark_count = int(np.sum(row_test[:, 2] < 80))
        white_count = int(np.sum((row_test[:, 1] < s_thresh) & (row_test[:, 2] > v_thresh)))
        if dark_count > 100 and white_count > min_white_scan // 2 and dark_count > best_dark:
            best_dark = dark_count
            y_black = y_scan
    print(f"Black key scan y={y_black}")

    row_black = hsv[y_black, :]
    avg_white_width = np.diff(white_keys).mean() if len(white_keys) > 1 else 20
    black_keys = []
    in_key = False
    start_x = 0
    for x in range(w):
        is_dark = int(row_black[x, 2]) < 80
        if is_dark and not in_key:
            in_key = True
            start_x = x
        elif not is_dark and in_key:
            bw = x - start_x
            center = (start_x + x) // 2
            # Filter edge artifacts (x<20) and segments too wide or too narrow
            if 5 < bw < avg_white_width and center > 20:
                black_keys.append(center)
            in_key = False

    print(f"Detected {len(black_keys)} black keys")

    # Validate white keys against black keys.  If the black keys form valid
    # [2,3] octave groups, reconstruct white keys via least-squares fit —
    # this is far more reliable than scanning white pixel rows, which often
    # pick up text labels (C2, C3…) or edge artifacts.
    if len(black_keys) >= 5:
        bk_gaps = [black_keys[i + 1] - black_keys[i]
                   for i in range(len(black_keys) - 1)]
        gap_thresh = (min(bk_gaps) + max(bk_gaps)) / 2
        bk_groups = [[black_keys[0]]]
        for i in range(1, len(black_keys)):
            if black_keys[i] - black_keys[i - 1] > gap_thresh:
                bk_groups.append([])
            bk_groups[-1].append(black_keys[i])
        group_sizes = [len(g) for g in bk_groups]

        # Check if groups follow the piano [2,3] or [3,2] pattern
        valid_pattern = all(s in (2, 3) for s in group_sizes) and len(group_sizes) >= 2
        if valid_pattern:
            # Black key offsets in white-key units from C:
            #   C#=0.5, D#=1.5, F#=3.5, G#=4.5, A#=5.5
            bk_offsets_in_octave = {2: [0.5, 1.5], 3: [3.5, 4.5, 5.5]}

            all_offsets = []
            oct = 0
            # Determine starting group type
            first_is_2 = (group_sizes[0] == 2)
            for gi, g in enumerate(bk_groups):
                sz = len(g)
                if first_is_2:
                    offsets = bk_offsets_in_octave[sz]
                    if sz == 2:
                        base = oct * 7
                    else:
                        base = oct * 7
                    for off in offsets:
                        all_offsets.append(base + off)
                    if sz == 3:
                        oct += 1
                else:
                    offsets = bk_offsets_in_octave[sz]
                    base = oct * 7
                    for off in offsets:
                        all_offsets.append(base + off)
                    if sz == 2:
                        oct += 1

            if len(all_offsets) == len(black_keys):
                offsets_arr = np.array(all_offsets)
                bk_arr = np.array(black_keys, dtype=float)
                A = np.vstack([offsets_arr, np.ones(len(offsets_arr))]).T
                result = np.linalg.lstsq(A, bk_arr, rcond=None)
                w_fit, c_fit = result[0]
                residuals = bk_arr - (w_fit * offsets_arr + c_fit)
                max_res = float(np.max(np.abs(residuals)))

                if max_res < w_fit * 0.5:  # residuals within half a key width
                    n_octaves = oct if first_is_2 else oct + 1
                    # Partial leading keys (before first C)
                    if not first_is_2:
                        # Starts with group-of-3 (F#,G#,A#) → 3 white keys before C
                        n_before = 3  # F, G, A, B before next C
                    else:
                        n_before = 0
                    n_white = n_octaves * 7 + 1 + n_before  # +1 for top C
                    white_keys = [int(round(c_fit + i * w_fit))
                                  for i in range(-n_before, n_octaves * 7 + 1)]
                    print(f"Reconstructed {len(white_keys)} white keys from black keys "
                          f"(w={w_fit:.1f}px, max_residual={max_res:.1f}px)")

    print(f"Detected {len(black_keys)} black keys")

    # Find keyboard geometry by scanning pixel brightness.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Top of keyboard: scan a white key column upward from y_white.
    mid_wk = white_keys[len(white_keys) // 2]
    y_kb_top = 0
    for y in range(y_white - 5, 0, -1):
        if int(gray[y, mid_wk]) < 100:
            y_kb_top = y + 1
            break

    # Bottom of black keys: scan a black key column downward from y_black
    # until brightness rises (white key face appears). If it stays dark
    # all the way to y_white, the black key extends to y_white.
    y_bk_bottom = y_black
    if black_keys:
        mid_bk = black_keys[len(black_keys) // 2]
        for y in range(y_black, y_white):
            if int(gray[y, mid_bk]) > 100:
                y_bk_bottom = y
                break
        else:
            y_bk_bottom = y_white

    return white_keys, black_keys, y_white, y_black, y_kb_top, y_bk_bottom


def regularize_positions(positions):
    """Fit a line to detected key positions and reconstruct evenly-spaced ones.

    White keys in Synthesia are perfectly evenly spaced, so linear regression
    removes outliers from partial/occluded key detections.
    """
    if len(positions) < 3:
        return positions
    n = len(positions)
    indices = np.arange(n)
    x = np.array(positions, dtype=float)
    coeffs = np.polyfit(indices, x, 1)
    slope, intercept = coeffs[0], coeffs[1]
    return [int(round(slope * i + intercept)) for i in range(n)]


# ---------------------------------------------------------------------------
# Note → x-pixel mapping
# ---------------------------------------------------------------------------

def find_first_c(white_keys, black_keys):
    """Find which white key index is the first C, using black key grouping.

    Groups of 2 consecutive black keys (C#, D#) identify C as the white key
    immediately to their left.

    Returns:
        c_start_idx (int): index into white_keys of the first C.
    """
    if len(white_keys) < 7 or len(black_keys) < 2:
        print("Not enough keys, defaulting C to index 2")
        return 2

    # Group black keys by their x-pixel gaps.  Within a group (e.g. C#-D# or
    # F#-G#-A#) gaps are ~1 white-key width apart; between groups the gap is
    # ~1.5-2× wider.  Using the gap midpoint as threshold is robust regardless
    # of keyboard scale or letterboxing.
    bk_gaps = [black_keys[i + 1] - black_keys[i]
               for i in range(len(black_keys) - 1)]
    if not bk_gaps:
        return 2
    gap_thresh = (min(bk_gaps) + max(bk_gaps)) / 2
    bk_groups = [[black_keys[0]]]
    for i in range(1, len(black_keys)):
        if black_keys[i] - black_keys[i - 1] > gap_thresh:
            bk_groups.append([])
        bk_groups[-1].append(black_keys[i])

    print(f"Black key groups (sizes): {[len(g) for g in bk_groups]}")

    # First group of size 2 → C#, D# → C is the white key to their left.
    for bk_group in bk_groups:
        if len(bk_group) == 2:
            # Find the white key index just left of the first black key
            bx = bk_group[0]
            c_idx = 0
            for i, wx in enumerate(white_keys):
                if wx <= bx:
                    c_idx = i
                else:
                    break
            print(f"First C at white key index {c_idx} (x≈{white_keys[c_idx]})")
            return c_idx

    print("Could not find group-of-2; defaulting C to index 2")
    return 2


def build_note_x_map(white_keys, black_keys, min_midi_note):
    """Build a MIDI note number → x-pixel map for the visible keyboard.

    Args:
        white_keys: list of white key x-centers (from detect_keyboard).
        black_keys: list of black key x-centers (from detect_keyboard).
        min_midi_note: lowest MIDI pitch in the input file (used to determine
            which C octave is at the left of the visible keyboard).

    Returns:
        dict mapping MIDI note int → x pixel int.
    """
    c_start_idx = find_first_c(white_keys, black_keys)

    # Determine C octave.  Configurable via min_midi_note or auto-detected:
    n_white = len(white_keys)
    if n_white >= 40:
        # Large keyboard (near-full 88-key piano).
        # Assume starts at A0 (MIDI 21).
        BELOW_C_SEMITONES = [0, 1, 3, 5, 7, 8, 10]
        semitones_below = BELOW_C_SEMITONES[min(c_start_idx, 6)]
        c_midi = 21 + semitones_below
    elif min_midi_note <= 21:
        # Video-only mode (min_midi_note defaults to 21).
        # Partial Synthesia keyboards typically start around C2-C3.
        # Estimate from keyboard size: fewer keys → higher starting octave.
        if n_white >= 20:
            c_midi = 36   # C2 — typical 3+ octave keyboard
        elif n_white >= 14:
            c_midi = 48   # C3 — 2-3 octave keyboard
        else:
            c_midi = 60   # C4 — small keyboard
    else:
        # MIDI-based mode — estimate from MIDI note range.
        c_midi = ((min_midi_note + 3) // 12) * 12
        if c_midi < 12:
            c_midi = 12

    print(f"c_start_idx={c_start_idx}, c_midi={c_midi} (min_note={min_midi_note})")

    note_x_map = {}

    # Assign white keys going right from c_start_idx
    midi = c_midi
    for i in range(c_start_idx, len(white_keys)):
        note_x_map[midi] = white_keys[i]
        semi = midi % 12
        idx = WHITE_SEMITONES.index(semi)
        if idx < len(WHITE_SEMITONES) - 1:
            midi += WHITE_SEMITONES[idx + 1] - WHITE_SEMITONES[idx]
        else:
            midi += 1  # B → C
        if midi > 108:
            break

    # Assign white keys going left from c_start_idx-1
    midi = c_midi
    for i in range(c_start_idx - 1, -1, -1):
        semi = midi % 12
        idx = WHITE_SEMITONES.index(semi) if semi in WHITE_SEMITONES else -1
        if idx > 0:
            midi -= WHITE_SEMITONES[idx] - WHITE_SEMITONES[idx - 1]
        else:
            midi -= 1  # C → B
        if midi < 21:
            break
        note_x_map[midi] = white_keys[i]

    # Extrapolate one white key beyond each edge using average spacing.
    # Synthesia videos often have partially-visible keys at the edges that
    # the keyboard detector misses, but note bars still appear on them.
    extrapolated_whites = set()
    if len(white_keys) >= 2:
        avg_w = np.mean(np.diff(white_keys))
        # Left edge: add one key below the lowest mapped white key
        lowest_midi = min(m for m in note_x_map if m % 12 in [0, 2, 4, 5, 7, 9, 11])
        lowest_x = note_x_map[lowest_midi]
        left_x = int(lowest_x - avg_w)
        if left_x > 0 and lowest_midi - 1 >= 21:
            semi = lowest_midi % 12
            idx = WHITE_SEMITONES.index(semi) if semi in WHITE_SEMITONES else -1
            if idx > 0:
                prev_midi = lowest_midi - (WHITE_SEMITONES[idx] - WHITE_SEMITONES[idx - 1])
            else:
                prev_midi = lowest_midi - 1  # C → B
            if prev_midi >= 21:
                note_x_map[prev_midi] = left_x
                extrapolated_whites.add(prev_midi)
        # Right edge: add one key above the highest mapped white key
        highest_midi = max(m for m in note_x_map if m % 12 in [0, 2, 4, 5, 7, 9, 11])
        highest_x = note_x_map[highest_midi]
        right_x = int(highest_x + avg_w)
        if right_x < 1920 and highest_midi + 1 <= 108:
            semi = highest_midi % 12
            idx = WHITE_SEMITONES.index(semi) if semi in WHITE_SEMITONES else -1
            if idx < len(WHITE_SEMITONES) - 1:
                next_midi = highest_midi + (WHITE_SEMITONES[idx + 1] - WHITE_SEMITONES[idx])
            else:
                next_midi = highest_midi + 1  # B → C
            if next_midi <= 108:
                note_x_map[next_midi] = right_x
                extrapolated_whites.add(next_midi)

    # Assign black keys using the midpoint of flanking white keys.
    # The detected black-key x-positions can be off by up to half a white
    # key width — and a small offset is enough to drop the detector zone
    # onto an adjacent white key, where falling note bars (toward the white
    # key) trigger phantom black-key detections. White-key positions are
    # regularized via linear regression so their midpoints are far more
    # reliable than the raw black-key detections.
    for wm in list(note_x_map.keys()):
        semi = wm % 12
        if semi in [0, 2, 5, 7, 9]:  # C, D, F, G, A have a sharp to the right
            if wm in extrapolated_whites:
                continue
            black_midi = wm + 1
            next_white = wm + 2
            if next_white in extrapolated_whites:
                continue
            if next_white not in note_x_map:
                continue
            note_x_map[black_midi] = (note_x_map[wm] + note_x_map[next_white]) // 2

    print(f"Built note→x map for {len(note_x_map)} MIDI notes "
          f"(range {min(note_x_map)}–{max(note_x_map)})")
    return note_x_map


# ---------------------------------------------------------------------------
# Color sampling & hand classification
# ---------------------------------------------------------------------------

def build_detector_regions(note_x_map, white_keys, y_white, cfg=None, y_black=None, y_kb_top=None, y_bk_bottom=None):
    """Compute detector bounds for each key: x-range and y-range.

    Black keys are sampled in the upper zone (their body), white keys in the
    lower zone (their face).  This prevents bleed: a lit white key glows at its
    face level, which overlaps the black key gap — but NOT the black key body.

    Zone positions are derived from keyboard_height (y_white - y_black) as
    percentages, making them independent of resolution and keyboard zoom.

    Returns:
        dict mapping MIDI pitch → (x_left, x_right, y_top, y_bot).
    """
    BLACK_SEMITONES = {1, 3, 6, 8, 10}

    if len(white_keys) > 1:
        avg_white_w = np.mean(np.diff(white_keys))
    else:
        avg_white_w = 30

    # Keyboard height proxy for zone sizing: 5× white key width.
    # This controls detector zone sizes (which need to be proportional
    # to the visual key size, not the pixel distance between scan lines).
    kb_h = avg_white_w * 5

    # Detector zones as % of keyboard height — overridable via config.
    # White key face: bottom 12% of keyboard (just above y_white)
    # Black key body: 33-49% from bottom (above white glow, below halo)
    det = {
        'white_x_ratio': 0.25,
        'white_y_top_pct': 0.12,    # 12% above y_white
        'white_y_bot_pct': -0.02,   # 2% below y_white
        'black_x_hw': max(2, int(avg_white_w * 0.04)),
        'black_y_top_pct': 0.49,    # 49% above y_white
        'black_y_bot_pct': 0.33,    # 33% above y_white
    }
    if cfg and 'detector' in cfg:
        det.update(cfg['detector'])

    regions = {}
    for pitch, x_center in note_x_map.items():
        if pitch % 12 in BLACK_SEMITONES:
            hw = det['black_x_hw']
            if y_black is not None:
                # Place the detector at the BOTTOM of the black key body only.
                # Sampling the full body catches falling note bars (which travel
                # through the upper part of the body on the way down) as
                # phantom key presses. The actual key-press glow is most
                # concentrated near the playing edge of the black key, just
                # above the white-key face.
                bk_top = y_kb_top if y_kb_top is not None else y_black - int(avg_white_w * 2)
                bk_bot = y_bk_bottom if y_bk_bottom is not None else y_black + int(avg_white_w * 0.2)
                bk_h = bk_bot - bk_top
                # Sample only the bottom 25% of the black key — small enough
                # to dodge falling bars in mid-air, tall enough to register
                # the key-press glow reliably.
                bottom_frac = det.get('black_bottom_frac', 0.25)
                y_top = bk_bot - max(2, int(bk_h * bottom_frac))
                y_bot = bk_bot - max(1, int(bk_h * 0.05))
            else:
                y_top = y_white - int(kb_h * det['black_y_top_pct'])
                y_bot = y_white - int(kb_h * det['black_y_bot_pct'])
        else:
            hw = int(avg_white_w * det['white_x_ratio'])
            # White key detector sits on the exposed face below the black keys.
            if y_bk_bottom is not None and y_bk_bottom < y_white - 5:
                # There's a visible white key face area
                face_top = y_bk_bottom
                face_h = y_white - face_top
                margin = max(1, int(face_h * 0.1))
                y_top = face_top + margin
                y_bot = y_white - margin
            else:
                # No separate face area (e.g. midi2video) — use bottom of keyboard
                y_top = y_white - int(kb_h * det['white_y_top_pct'])
                y_bot = y_white - int(kb_h * det['white_y_bot_pct'])
                # Clamp: never go above the bottom of the black keys
                if y_bk_bottom is not None and y_top < y_bk_bottom:
                    y_top = y_bk_bottom
            # Shift detector for asymmetric keys: B/E have wider face on
            # the right (no adjacent black key), C/F on the left. The
            # color glow is offset toward the wider side, so centering
            # on the geometric midpoint dilutes the saturation reading.
            shift = int(avg_white_w * 0.15)
            if pitch % 12 in (4, 11):   # E, B — wider right
                x_center += shift
            elif pitch % 12 in (0, 5):  # C, F — wider left
                x_center -= shift
        regions[pitch] = (x_center - hw, x_center + hw, y_top, y_bot)

    return regions
