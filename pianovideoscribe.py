#!/usr/bin/env python3
"""
pianovideoscribe — Separate a Synthesia piano video into right/left hand MIDI tracks.

Uses the video's color coding (green = right hand, blue = left hand) to assign
each note from an input MIDI file to one of two output tracks.

Typical usage:
    python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120
"""

import argparse
import json
import os
import sys
from collections import deque

import cv2
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


# ---------------------------------------------------------------------------
# Keyboard detection
# ---------------------------------------------------------------------------

def detect_keyboard(cap, frame_idx=5):
    """Detect white and black key x-centers from a clean keyboard frame (no notes).

    Uses HSV analysis to find the white-key row automatically (scans bottom-up
    for the first row with many white pixels and few dark pixels), then detects
    black keys just above that zone.

    Args:
        cap: OpenCV VideoCapture object.
        frame_idx: Frame index to use — should be early and note-free (default 5).

    Returns:
        (white_keys, black_keys): lists of x-pixel centers.
    """
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

    def _scan_white_row(hsv_img, y_scan):
        """Detect white key x-centers at a given y row."""
        row = hsv_img[y_scan, :]
        keys = []
        in_k = False
        sx = 0
        for x in range(hsv_img.shape[1]):
            is_w = int(row[x, 1]) < 60 and int(row[x, 2]) > 180
            if is_w and not in_k:
                in_k = True
                sx = x
            elif not is_w and in_k:
                if x - sx > 8:
                    keys.append((sx + x) // 2)
                in_k = False
        if in_k and hsv_img.shape[1] - sx > 8:
            keys.append((sx + hsv_img.shape[1]) // 2)
        return keys

    # First find the white key zone (bottom-up scan for white-dominant row)
    y_white_zone = h - 30
    for y_scan in range(h - 5, h // 2, -1):
        row_test = hsv[y_scan, :]
        white_count = int(np.sum((row_test[:, 1] < 60) & (row_test[:, 2] > 180)))
        dark_count = int(np.sum(row_test[:, 2] < 80))
        if white_count > 600 and dark_count < 100:
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
        white_count = int(np.sum((row_test[:, 1] < 60) & (row_test[:, 2] > 180)))
        if white_count < 400:
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
        white_count = int(np.sum((row_test[:, 1] < 60) & (row_test[:, 2] > 180)))
        if dark_count > 100 and white_count > 200 and dark_count > best_dark:
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
    return white_keys, black_keys, y_white, y_black


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
    WHITE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B

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
        if n_white >= 28:
            c_midi = 36   # C2 — typical 4-5 octave keyboard
        elif n_white >= 20:
            c_midi = 48   # C3 — typical 3-4 octave keyboard
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

    # Assign black keys using actual detected positions where available,
    # falling back to midpoint of flanking white keys.
    for wm in list(note_x_map.keys()):
        semi = wm % 12
        if semi in [0, 2, 5, 7, 9]:  # C, D, F, G, A have a sharp to the right
            black_midi = wm + 1
            next_white = wm + 2
            if next_white not in note_x_map:
                continue
            midpoint = (note_x_map[wm] + note_x_map[next_white]) // 2
            # Find closest detected black key to this midpoint
            best_bk = None
            best_dist = 999
            for bk_x in black_keys:
                dist = abs(bk_x - midpoint)
                if dist < best_dist:
                    best_dist = dist
                    best_bk = bk_x
            # Use detected position if close enough, else midpoint
            if best_bk is not None and best_dist < 20:
                note_x_map[black_midi] = best_bk
            else:
                note_x_map[black_midi] = midpoint

    print(f"Built note→x map for {len(note_x_map)} MIDI notes "
          f"(range {min(note_x_map)}–{max(note_x_map)})")
    return note_x_map


# ---------------------------------------------------------------------------
# Color sampling & hand classification
# ---------------------------------------------------------------------------

def build_detector_regions(note_x_map, white_keys, y_white, cfg=None, y_black=None):
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
                # Anchor black key zones around the actual detected y_black,
                # not computed from percentages (which fail on letterboxed videos
                # where y_white-based offsets land in the falling-note area).
                # IMPORTANT: clip the bottom to stay ABOVE the white key face
                # zone to prevent white key glow from bleeding into black key
                # detectors when an adjacent white key is pressed.
                zone_h = max(10, int(kb_h * (det['black_y_top_pct'] - det['black_y_bot_pct'])))
                y_top = y_black - zone_h // 2
                y_bot = y_black + zone_h // 2
            else:
                y_top = y_white - int(kb_h * det['black_y_top_pct'])
                y_bot = y_white - int(kb_h * det['black_y_bot_pct'])
        else:
            hw = int(avg_white_w * det['white_x_ratio'])
            y_top = y_white - int(kb_h * det['white_y_top_pct'])
            y_bot = y_white - int(kb_h * det['white_y_bot_pct'])
        regions[pitch] = (x_center - hw, x_center + hw, y_top, y_bot)

    return regions


def sample_color(cap, frame_idx, x_center, y_top, y_bot, half_w=10):
    """Sample the dominant saturated color at a note's position in a video frame.

    Finds the brightest, most-saturated pixel in the region, filtering out dark
    pixels (V < 80) that produce false high-saturation readings at borders.

    Returns:
        (h, s, v) floats, or (None, None, None) if no suitable pixel found.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    fh, fw = frame.shape[:2]
    x1, x2 = max(0, x_center - half_w), min(fw, x_center + half_w)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return None, None, None

    region = frame[y1:y2, x1:x2]
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h_ch = hsv_region[:, :, 0].flatten().astype(float)
    s_ch = hsv_region[:, :, 1].flatten().astype(float)
    v_ch = hsv_region[:, :, 2].flatten().astype(float)

    # Filter out dark pixels — they produce noise-saturation artifacts
    bright = v_ch > 80
    if not np.any(bright):
        return None, None, None

    s_bright = s_ch.copy()
    s_bright[~bright] = 0
    best = int(np.argmax(s_bright))
    return float(h_ch[best]), float(s_ch[best]), float(v_ch[best])


def sample_color_avg(hsv_frame, x_left, x_right, y_top, y_bot):
    """Sample color by averaging bright saturated pixels in a detector region.

    More robust than single-pixel max: resistant to sparkles and edge bleed.

    Args:
        hsv_frame: full frame already converted to HSV.
        x_left, x_right: horizontal bounds of the detector region.
        y_top, y_bot: vertical bounds of the sampling zone.

    Returns:
        (h, s, v) average of bright saturated pixels, or (None, None, None).
    """
    fh, fw = hsv_frame.shape[:2]
    x1, x2 = max(0, x_left), min(fw, x_right)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return None, None, None

    region = hsv_frame[y1:y2, x1:x2]
    h_ch = region[:, :, 0].flatten().astype(float)
    s_ch = region[:, :, 1].flatten().astype(float)
    v_ch = region[:, :, 2].flatten().astype(float)

    # Keep only bright AND saturated pixels (the actual colored key face)
    mask = (v_ch > 80) & (s_ch > 50)
    if np.sum(mask) < 3:  # need at least a few pixels to trust the average
        return None, None, None

    return float(np.mean(h_ch[mask])), float(np.mean(s_ch[mask])), float(np.mean(v_ch[mask]))


def classify_hand(h, s, v, green_is_right=True, colors=None):
    """Classify a color sample as right hand (0), left hand (1), or unknown (None).

    Args:
        h, s, v: HSV values (OpenCV scale: H in [0,179]).
        green_is_right: if True, green = right (0), blue = left (1). Flip if needed.
            Ignored when colors dict contains 'right'/'left' keys.
        colors: dict with color keys, each containing h_min, h_max, s_min, v_min.
            Supports two formats:
            - Legacy: 'green' and 'blue' keys (uses green_is_right to map to hands)
            - Direct: 'right' and 'left' keys (maps directly, ignores green_is_right)

    Returns:
        0 (right), 1 (left), or None.
    """
    if h is None or s is None:
        return None

    if colors is None:
        colors = {
            'green': {'h_min': 40, 'h_max': 65, 's_min': 100, 'v_min': 80},
            'blue':  {'h_min': 100, 'h_max': 120, 's_min': 80, 'v_min': 80},
        }

    # Direct right/left config — no green_is_right logic needed
    if 'right' in colors and 'left' in colors:
        for hand_label, hand_id in [('right', 0), ('left', 1)]:
            c = colors[hand_label]
            if c['h_min'] <= h <= c['h_max'] and s > c['s_min'] and v > c['v_min']:
                return hand_id
        return None

    g = colors['green']
    b = colors['blue']
    is_green = g['h_min'] <= h <= g['h_max'] and s > g['s_min'] and v > g['v_min']
    is_blue = b['h_min'] <= h <= b['h_max'] and s > b['s_min'] and v > b['v_min']

    if is_green:
        return 0 if green_is_right else 1
    if is_blue:
        return 1 if green_is_right else 0
    return None


def fallback_hand(pitch, recent_right, recent_left):
    """Assign a hand by pitch proximity to recently seen notes of each hand.

    Falls back to: right if pitch >= 60 (middle C), left otherwise, when
    neither hand has any recent notes.
    """
    if not recent_right or not recent_left:
        return 0 if pitch >= 60 else 1
    d_right = abs(pitch - np.mean(list(recent_right)))
    d_left = abs(pitch - np.mean(list(recent_left)))
    return 0 if d_right <= d_left else 1


# ---------------------------------------------------------------------------
# Video-based note extraction
# ---------------------------------------------------------------------------

def _region_avg_saturation(hsv_frame, x_left, x_right, y_top, y_bot):
    """Return the average saturation in a detector region."""
    fh, fw = hsv_frame.shape[:2]
    x1, x2 = max(0, x_left), min(fw, x_right)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return 0.0
    return float(np.mean(hsv_frame[y1:y2, x1:x2, 1]))


def _region_avg_hue(hsv_frame, x_left, x_right, y_top, y_bot, s_min=50):
    """Return the average hue of saturated pixels in a detector region."""
    fh, fw = hsv_frame.shape[:2]
    x1, x2 = max(0, x_left), min(fw, x_right)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return None
    region = hsv_frame[y1:y2, x1:x2]
    h_ch = region[:, :, 0].flatten().astype(float)
    s_ch = region[:, :, 1].flatten().astype(float)
    mask = s_ch > s_min
    if np.sum(mask) < 3:
        return None
    return float(np.mean(h_ch[mask]))


def _classify_hand_from_hue(hue, green_is_right=True, colors=None):
    """Classify hand from average hue using config color ranges."""
    if hue is None:
        return None
    if colors is None:
        g_min, g_max = 40, 65
        b_min, b_max = 85, 125
    elif 'right' in colors and 'left' in colors:
        # Direct right/left config
        for hand_label, hand_id in [('right', 0), ('left', 1)]:
            c = colors[hand_label]
            if c['h_min'] <= hue <= c['h_max']:
                return hand_id
        return None
    else:
        g = colors['green']
        b = colors['blue']
        g_min, g_max = g['h_min'], g['h_max']
        b_min, b_max = b['h_min'], b['h_max']
    if g_min <= hue <= g_max:
        return 0 if green_is_right else 1
    if b_min <= hue <= b_max:
        return 1 if green_is_right else 0
    return None


def extract_notes_from_video(cap, note_x_map, y_sample_top, y_sample_bot,
                             half_w, fps, total_frames, green_is_right, colors,
                             start_frame=0, frame_step=1, white_keys=None, cfg=None,
                             y_black=None, start_time=None):
    """Extract notes from the video using frame-to-frame saturation delta.

    Instead of checking absolute color, detects the *moment* a key changes
    from neutral to colored (sudden saturation increase = note on, sudden
    decrease = note off).  This is robust against falling note bars (which
    create gradual color changes) vs actual key presses (sudden changes).

    Args:
        cap: OpenCV VideoCapture object.
        note_x_map: dict mapping MIDI pitch → x pixel center.
        y_sample_top, y_sample_bot: vertical bounds of the sampling zone.
        half_w: horizontal half-width for sampling (fallback).
        fps: video frame rate.
        total_frames: total number of frames.
        green_is_right: True if green = right hand (0).
        colors: color config dict (or None for defaults).
        start_frame: first frame to scan (skip intro).
        frame_step: scan every Nth frame (1 = every frame, 2 = every other, etc.)
        white_keys: list of white key x-centers (for computing detector bounds).

    Returns:
        List of (pitch, hand, onset_sec, offset_sec) tuples.
    """
    SAT_ON = 70       # saturation threshold for white keys: above = key pressed
    SAT_ON_BLACK = 90 # higher threshold for black keys to reject glow bleed
    SAT_OFF = 40      # saturation threshold: below = key released
    BLACK_SEMITONES_SET = {1, 3, 6, 8, 10}

    # Build detector regions
    if white_keys is not None:
        det_regions = build_detector_regions(note_x_map, white_keys,
                                            y_sample_bot, cfg=cfg, y_black=y_black)
    else:
        det_regions = {p: (x - half_w, x + half_w, y_sample_top, y_sample_bot)
                       for p, x in note_x_map.items()}

    pitches = sorted(note_x_map.keys())

    # Pre-compute detector slices for bulk numpy extraction
    # Group by y-zone to minimize per-pitch overhead
    pitch_slices = []
    for pitch in pitches:
        x_left, x_right, yt, yb = det_regions[pitch]
        pitch_slices.append((pitch, max(0, x_left), x_right, max(0, yt), yb))

    # Active notes: pitch → (hand, onset_frame)
    active = {}
    notes = []
    # Debounce: track consecutive frames above SAT_ON per pitch.
    # A note-on requires 2+ consecutive frames to filter single-frame
    # glows from adjacent key presses bleeding into black key zones.
    pending_on = {}  # pitch → (frame_idx, hand)

    # Auto-detect the first playing frame: scan forward from start_frame
    # looking for saturated pixels on the keyboard face (where keys light up
    # when pressed).  Uses a band 30px above y_white to 5px below — this
    # catches key presses but avoids falling note bars higher up.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    play_start = start_frame
    face_yt = max(0, y_sample_bot - 30)
    face_yb = min(y_sample_bot + 5, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1)
    x_lo = min(x for _, x, _, _, _ in pitch_slices) if pitch_slices else 0
    x_hi = max(x for _, _, x, _, _ in pitch_slices) if pitch_slices else 1920
    face_yt, face_yb = int(face_yt), int(face_yb)
    if start_time is not None:
        # Manual override: skip auto-detection
        play_start = int(start_time * fps)
        play_start = max(start_frame, min(play_start, total_frames - 1))
        print(f"  Using manual start time: frame {play_start} ({play_start/fps:.1f}s)")
    else:
        for f_scan in range(start_frame, min(start_frame + int(fps * 30), total_frames), 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_scan)
            ret, frame = cap.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            region = hsv[face_yt:face_yb, x_lo:x_hi, 1]
            n_saturated = int(np.sum(region > SAT_ON))
            if n_saturated > 200:  # enough saturated pixels = key actually pressed
                play_start = f_scan
                break
        if play_start > start_frame:
            print(f"  Skipping intro: first notes at frame {play_start} "
                  f"({play_start/fps:.1f}s)")

    # Seek once to play_start, then read sequentially (much faster than
    # seeking every frame — avoids H.264 keyframe backtracking)
    cap.set(cv2.CAP_PROP_POS_FRAMES, play_start)
    start_frame = play_start
    import time as _time
    t0 = _time.time()

    for f_idx in range(start_frame, total_frames, frame_step):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if frame_step > 1 (read but don't process)
        if frame_step > 1:
            for _ in range(frame_step - 1):
                cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]  # extract saturation once
        val_channel = hsv[:, :, 2]  # brightness — needed to gate dark pixels

        confirmed_this_frame = []  # track new onsets to detect bar artifacts

        for pitch, x1, x2, y1, y2 in pitch_slices:
            if y1 >= y2 or x1 >= x2:
                continue
            # Gate saturation by brightness: near-black pixels (V < 30)
            # produce noisy saturation from video compression artifacts.
            # Treat them as S=0 to prevent false note-on triggers.
            V_MIN = 30
            region_s = sat_channel[y1:y2, x1:x2]
            region_v = val_channel[y1:y2, x1:x2]
            bright_mask = region_v >= V_MIN
            if np.any(bright_mask):
                sat = float(np.mean(region_s[bright_mask]))
            else:
                sat = 0.0

            thresh = SAT_ON_BLACK if (pitch % 12) in BLACK_SEMITONES_SET else SAT_ON
            if pitch not in active and sat > thresh:
                # Debounce: require 2 consecutive frames above threshold
                # to filter single-frame glow bleed from adjacent keys.
                if pitch in pending_on and f_idx - pending_on[pitch][0] <= frame_step:
                    # Second consecutive frame — confirm note on
                    hand = pending_on.pop(pitch)[1]
                    active[pitch] = (hand, f_idx - frame_step)
                    confirmed_this_frame.append(pitch)
                else:
                    # First frame — classify hand and store as pending
                    region = hsv[y1:y2, x1:x2]
                    h_ch = region[:, :, 0].flatten().astype(float)
                    s_ch = region[:, :, 1].flatten().astype(float)
                    mask = s_ch > 50
                    hue = float(np.mean(h_ch[mask])) if np.sum(mask) >= 3 else None
                    hand = _classify_hand_from_hue(hue, green_is_right, colors)
                    if hand is not None:
                        pending_on[pitch] = (f_idx, hand)
                    # If hue doesn't match any configured color, skip —
                    # it's likely a UI element, border glow, or artifact.

            elif sat <= thresh and pitch in pending_on:
                # Saturation dropped before second frame — cancel pending
                del pending_on[pitch]

            note_off = False
            if pitch in active and sat < SAT_OFF:
                note_off = True
            elif pitch in active and sat > thresh:
                # Saturation is high — check if the hue still matches a
                # configured hand color.  If not, a non-musical overlay
                # (e.g. phone nav bar) has replaced the note bar.
                region_h = hsv[y1:y2, x1:x2, 0]
                region_s = hsv[y1:y2, x1:x2, 1]
                s_mask = region_s > 50
                hue = float(np.mean(region_h[s_mask])) if np.sum(s_mask) >= 3 else None
                if _classify_hand_from_hue(hue, green_is_right, colors) is None:
                    note_off = True
            if note_off and pitch in active:
                hand, onset_frame = active.pop(pitch)
                onset_sec = onset_frame / fps
                offset_sec = f_idx / fps
                if offset_sec - onset_sec > 0.02:
                    notes.append((pitch, hand, onset_sec, offset_sec))

        if f_idx % 500 == 0 and f_idx > start_frame:
            elapsed = _time.time() - t0
            rate = (f_idx - start_frame) / elapsed if elapsed > 0 else 0
            eta = (total_frames - f_idx) / rate if rate > 0 else 0
            print(f"  Frame {f_idx}/{total_frames} "
                  f"({100*f_idx/total_frames:.0f}%) — {len(notes)} notes "
                  f"— {rate:.0f} fps, ETA {eta:.0f}s")

    # Close remaining active notes
    for pitch, (hand, onset_frame) in active.items():
        onset_sec = onset_frame / fps
        offset_sec = (total_frames - 1) / fps
        if offset_sec - onset_sec > 0.02:
            notes.append((pitch, hand, onset_sec, offset_sec))

    notes.sort(key=lambda n: n[2])
    return notes


# ---------------------------------------------------------------------------
# Tick conversion & quantization
# ---------------------------------------------------------------------------

def make_tick_converters(orig_tpb, orig_bpm, out_tpb, out_bpm):
    """Return functions to convert original MIDI ticks to seconds and output ticks.

    NOTE: The correct formula is:   out_tick = t_sec * OUT_TPB * OUT_BPM / 60
    Do NOT use: t_sec * (US_PER_BEAT / 1e6) * TPB  — that is wrong by ~(BPM/60)².
    """
    orig_us_per_tick = (60_000_000 / orig_bpm) / orig_tpb

    def ticks_to_seconds(abs_tick):
        return abs_tick * orig_us_per_tick / 1_000_000

    def seconds_to_out_ticks(t_sec):
        return t_sec * out_tpb * out_bpm / 60

    return ticks_to_seconds, seconds_to_out_ticks


def quantize_tick(tick, grid):
    """Snap a tick to the nearest grid position."""
    return round(tick / grid) * grid


def quantize_tick_smart(tick, eighth, sixteenth):
    """Snap to nearest 16th (simple fallback, used by MIDI-based mode)."""
    return round(tick / sixteenth) * sixteenth


def quantize_onsets_pll(onset_secs, bpm, alpha=0.1, subdivisions=4):
    """Phase-locked loop quantizer: snap onsets to grid positions.

    Maintains a running phase offset (EMA) that self-corrects for BPM drift
    and per-note jitter.  Resets phase after large gaps (>= 3 grid units) to
    avoid carrying accumulated error from dense passages into sparse ones.

    Args:
        subdivisions: grid units per beat. 4 = 16th notes (default),
            3 = triplet 8th notes, 6 = triplet 16th notes.
    """
    s = 60.0 / bpm / subdivisions  # grid unit duration
    phase = 0.0
    initialized = False
    results = []

    for i, t in enumerate(onset_secs):
        # Reset phase after large gaps (accumulated error doesn't carry over).
        # Snap to nearest EIGHTH (2 sixteenths) since notes after gaps are
        # almost always on 8th positions, not 16ths.
        if i > 0 and (t - onset_secs[i - 1]) > 2.5 * s:
            grid_pos_8th = t / (2 * s)  # position in eighths
            idx_8th = round(grid_pos_8th)
            phase = t - idx_8th * 2 * s

        grid_pos = (t - phase) / s
        idx = round(grid_pos)
        ideal_time = idx * s + phase
        error = t - ideal_time

        if not initialized:
            phase = error
            initialized = True
        else:
            phase += alpha * error

        results.append(idx)

    # For combined grids (subdivisions=12), snap each index to the nearest
    # valid musical position. Valid positions per beat (in 12ths):
    # straight: 0,3,6,9  triplet: 0,4,8  → combined: 0,3,4,6,8,9
    # This eliminates quintuplets (5) and septuplets (7) from the output.
    if subdivisions == 12:
        valid_per_beat = [0, 3, 4, 6, 8, 9]
        snapped = []
        for idx in results:
            beat = idx // 12
            pos = idx % 12
            # Find nearest valid position (may cross beat boundary)
            best = min(valid_per_beat, key=lambda v: abs(v - pos))
            snapped.append(beat * 12 + best)
        results = snapped

    return results


def quantize_onsets_viterbi(onset_secs, bpm, abs_weight=0.1):
    """Viterbi DP quantizer: snap onsets to 16th-note grid positions.

    Finds the globally optimal assignment of grid positions that minimises
    a weighted combination of per-interval error and absolute-position error.
    Then refines ambiguous intervals by testing floor/ceil flips.

    100% accuracy on test data (vs 97% PLL, 91% global-fit).
    """
    import math

    s = 60.0 / bpm / 4  # sixteenth note duration
    n = len(onset_secs)
    if n == 0:
        return []

    # Phase 1: Viterbi DP
    dp = [dict() for _ in range(n)]
    dp[0][0] = (0.0, None)  # start at grid position 0

    for i in range(n - 1):
        interval = onset_secs[i + 1] - onset_secs[i]
        ratio = interval / s
        centre = round(ratio)
        candidates = set()
        for c in range(max(1, centre - 2), centre + 3):
            candidates.add(c)
        candidates.add(max(1, math.floor(ratio)))
        candidates.add(math.ceil(ratio))

        for pos, (cost, _) in dp[i].items():
            for k in candidates:
                new_pos = pos + k
                interval_err = (interval - k * s) ** 2
                abs_err = (onset_secs[i + 1] - new_pos * s) ** 2
                new_cost = cost + interval_err + abs_weight * abs_err
                if new_pos not in dp[i + 1] or dp[i + 1][new_pos][0] > new_cost:
                    dp[i + 1][new_pos] = (new_cost, pos)

    # Backtrace
    best_final = min(dp[n - 1], key=lambda p: dp[n - 1][p][0])
    positions = []
    pos = best_final
    for i in range(n - 1, -1, -1):
        positions.append(pos)
        pos = dp[i][pos][1]
    positions.reverse()

    # Phase 2: refine ambiguous intervals
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            interval = onset_secs[i] - onset_secs[i - 1]
            ratio = interval / s
            frac = ratio - math.floor(ratio)
            if not (0.25 < frac < 0.75):
                continue
            current_delta = positions[i] - positions[i - 1]
            lo = max(1, math.floor(ratio))
            hi = lo + 1
            other_delta = hi if current_delta == lo else lo
            shift = other_delta - current_delta
            curr_abs = sum((onset_secs[j] - positions[j] * s) ** 2 for j in range(i, n))
            flip_abs = sum((onset_secs[j] - (positions[j] + shift) * s) ** 2 for j in range(i, n))
            if flip_abs < curr_abs:
                for j in range(i, n):
                    positions[j] += shift
                changed = True

    return positions


# ---------------------------------------------------------------------------
# MIDI building
# ---------------------------------------------------------------------------

def remove_overlaps(events):
    """Cut held notes when a new onset arrives, but keep chords intact.

    Notes starting at the same tick are treated as a chord and kept together.
    Notes held from a previous tick are cut short when any new note_on arrives.
    This produces clean notation without destroying chordal voicing.

    Args:
        events: list of (abs_tick, type, pitch, velocity) tuples.

    Returns:
        New event list with no cross-onset overlaps.
    """
    sorted_events = sorted(events, key=lambda e: (e[0], e[1] != 'note_off'))
    result = []
    active = {}  # pitch -> note_on tick

    for ev in sorted_events:
        abs_tick, ev_type, pitch, vel = ev
        if ev_type == 'note_on':
            # End all notes from *previous* ticks (not same-tick chord mates)
            to_end = [p for p, t in active.items() if t < abs_tick]
            for p in to_end:
                result.append((abs_tick, 'note_off', p, 0))
                del active[p]
            active[pitch] = abs_tick
            result.append(ev)
        elif ev_type == 'note_off':
            if pitch in active:
                result.append(ev)
                del active[pitch]

    return result


def make_monophonic(events, keep='highest'):
    """Force a list of note events to be monophonic (single voice).

    At each tick where multiple note_ons occur (chords), keep only one note
    (highest or lowest pitch). Then cut each note when the next one begins,
    so no two notes overlap in time.

    Args:
        events: list of (abs_tick, type, pitch, velocity) tuples.
        keep: 'highest' to keep the top note (melody), 'lowest' for bass.

    Returns:
        New list with exactly one note sounding at any time.
    """
    from collections import defaultdict

    sorted_events = sorted(events, key=lambda e: (e[0], e[1] != 'note_off'))

    # Group note_ons by tick, pick one per tick
    tick_groups = defaultdict(list)
    for ev in sorted_events:
        if ev[1] == 'note_on':
            tick_groups[ev[0]].append(ev)

    kept_pitches = {}
    for tick, group in tick_groups.items():
        if len(group) == 1:
            kept_pitches[tick] = {group[0][2]}
        else:
            best = max(group, key=lambda e: e[2]) if keep == 'highest' else min(group, key=lambda e: e[2])
            kept_pitches[tick] = {best[2]}

    kept_notes = set()
    removed_notes = set()
    result = []
    active_pitch = None

    for ev in sorted_events:
        abs_tick, ev_type, pitch, vel = ev
        if ev_type == 'note_on':
            if pitch not in kept_pitches.get(abs_tick, set()):
                removed_notes.add(pitch)
                continue
            if active_pitch is not None:
                result.append((abs_tick, 'note_off', active_pitch, 0))
            active_pitch = pitch
            kept_notes.add(pitch)
            result.append(ev)
        elif ev_type == 'note_off':
            if pitch in removed_notes:
                removed_notes.discard(pitch)
                continue
            if pitch == active_pitch:
                result.append(ev)
                active_pitch = None

    return result


def build_track(events, name, tempo_us, channel=0):
    """Build a MidiTrack from a list of (abs_tick, type, pitch, velocity) events."""
    track = MidiTrack()
    track.name = name
    track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
    events_sorted = sorted(events, key=lambda e: (e[0], e[1] != 'note_off'))
    prev = 0
    for abs_tick, ev_type, pitch, vel in events_sorted:
        delta = max(0, abs_tick - prev)
        prev = abs_tick
        if ev_type == 'note_on':
            track.append(Message('note_on', channel=channel, note=pitch, velocity=vel, time=delta))
        else:
            track.append(Message('note_off', channel=channel, note=pitch, velocity=0, time=delta))
    track.append(MetaMessage('end_of_track', time=0))
    return track


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Separate a Synthesia video into right/left hand MIDI tracks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120

  # Dry run: detect keyboard only, print stats, exit
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --dry-run

  # If hands are swapped, flip green interpretation
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --green-hand left

  # Clean up left hand notation (recommended for most songs)
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --left-hand no-overlap

  # Single melody line in right hand, clean bass in left
  python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --right-hand monophonic --left-hand no-overlap
""")
    p.add_argument('video', help='Path to Synthesia MP4 video')
    p.add_argument('midi', nargs='?', default=None,
                   help='Path to input MIDI (optional — if omitted, extracts notes '
                        'directly from the video)')
    p.add_argument('output', help='Path to output MIDI')
    # All settings-overridable args default to None so --settings can fill them in.
    # Hardcoded defaults are applied after settings are merged.
    p.add_argument('--bpm', type=int, default=None,
                   help='BPM of the song. Auto-detected from video audio if omitted.')
    p.add_argument('--frame', type=int, default=None,
                   help='Frame index for keyboard detection (default: 5, should be note-free)')
    p.add_argument('--green-hand', choices=['right', 'left'], default=None,
                   help='Which hand is shown in green in the video (default: right)')
    hand_choices = ['normal', 'no-overlap', 'monophonic']
    p.add_argument('--right-hand', choices=hand_choices, default=None,
                   help='Right hand processing: normal, '
                        'no-overlap (default, cut held notes at next onset, keep chords), '
                        'monophonic (single voice, highest note only)')
    p.add_argument('--left-hand', choices=hand_choices, default=None,
                   help='Left hand processing: normal, '
                        'no-overlap (default, cut held notes at next onset, keep chords), '
                        'monophonic (single voice, lowest note only)')
    p.add_argument('--key', type=str, default=None,
                   help='Key signature (e.g. "E" for E major, "Em" for E minor). '
                        'Sets the MIDI key signature so MuseScore displays correct accidentals. '
                        'If omitted, MuseScore auto-detects (often wrong).')
    p.add_argument('--config', type=str, default=None,
                   help='Path to a JSON config file (colors, sampling zone, keyboard frame). '
                        'See configs/ directory for examples.')
    p.add_argument('--start-time', type=float, default=None,
                   help='Manual override: music start time in seconds (skips auto-detection)')
    p.add_argument('--end-time', type=float, default=None,
                   help='Manual override: music end time in seconds (skips auto-detection)')
    p.add_argument('--settings', type=str, default=None,
                   help='Path to a settings.json file with saved pipeline parameters. '
                        'CLI flags override values from the file.')
    p.add_argument('--triplet', action='store_true',
                   help='Use combined grid (12 per beat) that supports both straight 16ths '
                        'and triplet positions. Each note snaps to the nearest valid position.')
    p.add_argument('--dry-run', action='store_true',
                   help='Detect keyboard and print stats only — do not write output MIDI')

    args = p.parse_args()

    # --- Merge settings file (if provided) — fill None values from JSON ---
    if args.settings:
        with open(args.settings) as f:
            settings = json.load(f)
        SETTINGS_KEYS = ['bpm', 'key', 'green_hand', 'frame', 'right_hand',
                         'left_hand', 'start_time', 'end_time', 'config']
        for key in SETTINGS_KEYS:
            if getattr(args, key) is None and key in settings:
                setattr(args, key, settings[key])

    # --- Apply hardcoded defaults for anything still None ---
    HARDCODED_DEFAULTS = {
        'frame': 5,
        'green_hand': 'right',
        'right_hand': 'no-overlap',
        'left_hand': 'no-overlap',
    }
    for key, default in HARDCODED_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, default)

    # BPM: auto-detect from video audio if not provided
    if args.bpm is None:
        detected = detect_bpm_from_video(args.video)
        if detected is None:
            p.error('--bpm is required (auto-detection failed)')
        args.bpm = detected

    return args


def detect_bpm_from_video(video_path):
    """Extract audio from video and detect BPM using librosa.

    Returns the detected BPM as an integer, or None on failure.
    """
    import subprocess
    import tempfile

    # Extract audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
             '-ar', '44100', '-ac', '1', tmp_path, '-y'],
            capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print("WARNING: Could not extract audio from video for BPM detection.",
                  file=sys.stderr)
            return None

        import librosa
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)

        # Use beat_track for primary estimate
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        # tempo may be an array in some librosa versions
        primary_bpm = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])

        # Also get tempogram candidates for context
        oenv = librosa.onset.onset_strength(y=audio, sr=sr)
        tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
        ac = np.mean(tg, axis=1)
        bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr)
        mask = (bpms >= 40) & (bpms <= 200)
        top_idx = np.argsort(ac[mask])[::-1][:3]
        candidates = [(bpms[mask][i], ac[mask][i]) for i in top_idx]

        print(f"Auto-detected BPM: {primary_bpm:.0f}")
        print(f"  Candidates: {', '.join(f'{b:.0f}' for b, _ in candidates)}")

        # Librosa often picks double-time; prefer half if it's in 80-140 range
        bpm = round(primary_bpm)
        if bpm > 140:
            half = bpm / 2
            if 60 <= half <= 140:
                print(f"  Halving {bpm} -> {half:.0f} (likely double-time)")
                bpm = round(half)

        print(f"  Using: {bpm} BPM")
        return bpm

    except Exception as e:
        print(f"WARNING: BPM auto-detection failed: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_config(config_path):
    """Load a JSON config file and return its contents as a dict.

    Config files can override: colors (HSV thresholds for green/blue),
    sampling zone (y offsets, half_w), and keyboard detection frame.
    Missing keys fall back to defaults.
    """
    defaults = {
        'colors': {
            'green': {'h_min': 40, 'h_max': 65, 's_min': 100, 'v_min': 80},
            'blue':  {'h_min': 100, 'h_max': 120, 's_min': 80, 'v_min': 80},
        },
        'sampling': {
            'y_offset_top': 90,
            'y_offset_bot': 0,
            'half_w': 10,
        },
        'keyboard': {
            'frame': 5,
        },
    }
    if config_path is None:
        return defaults

    with open(config_path) as f:
        cfg = json.load(f)

    # Deep merge: config values override defaults (2 levels deep)
    for section in defaults:
        if section in cfg:
            if isinstance(defaults[section], dict):
                for key in cfg[section]:
                    if isinstance(defaults[section].get(key), dict) and isinstance(cfg[section][key], dict):
                        defaults[section][key].update(cfg[section][key])
                    else:
                        defaults[section][key] = cfg[section][key]
            else:
                defaults[section] = cfg[section]
    return defaults


def main():
    args = parse_args()
    cfg = load_config(args.config)

    OUT_TPB = 960
    OUT_BPM = args.bpm
    OUT_US_PER_BEAT = int(60_000_000 / OUT_BPM)
    EIGHTH = OUT_TPB // 2     # ticks per 8th note
    SIXTEENTH = OUT_TPB // 4  # ticks per 16th note
    THIRTYSECOND = OUT_TPB // 8  # ticks per 32nd note
    # Grid subdivisions per beat: 4 = 16th notes (default),
    # 12 = combined grid (supports both straight and triplet positions:
    #   valid positions per beat in 12ths: 0,3,4,6,8,9
    #   each hand's PLL naturally phase-locks to its own grid type)
    GRID_SUBDIVISIONS = 12 if args.triplet else 4
    GRID_TICKS = OUT_TPB // GRID_SUBDIVISIONS  # ticks per grid unit (80 or 240)
    green_is_right = (args.green_hand == 'right')

    # Config overrides for --frame (CLI takes precedence)
    frame_idx = args.frame if args.frame != 5 else cfg['keyboard'].get('frame', 5)

    print("=== pianovideoscribe ===\n")
    print(f"Video:       {args.video}")
    print(f"Source:      {'video (direct extraction)' if args.midi is None else args.midi}")
    print(f"Output MIDI: {args.output}")
    print(f"BPM:         {OUT_BPM}")
    if args.triplet:
        print(f"Grid:        triplet (3 per beat, {GRID_TICKS} ticks)")
    print(f"Green hand:  {args.green_hand}")
    if args.config:
        print(f"Config:      {args.config}")
    print()

    # --- Load video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fps:.2f} fps, {total_frames} frames, h={vid_h}\n")

    # --- Step 1: Detect keyboard ---
    print("--- Step 1: Detect keyboard ---")
    white_keys, black_keys, y_white, y_black = detect_keyboard(cap, frame_idx=frame_idx)

    y_sample_top = y_white - cfg['sampling']['y_offset_top']
    y_sample_bot = y_white - cfg['sampling']['y_offset_bot']
    half_w = cfg['sampling']['half_w']

    # --- Step 2: Build note → x map ---
    print("\n--- Step 2: Build note→x map ---")

    if args.midi is not None:
        # MIDI-based mode (fallback for non-keyboard videos)
        mid = MidiFile(args.midi)
        orig_tpb = mid.ticks_per_beat
        orig_bpm = 120
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    orig_bpm = round(60_000_000 / msg.tempo)
                    break

        all_pitches = [msg.note for track in mid.tracks for msg in track
                       if msg.type == 'note_on' and msg.velocity > 0]
        if not all_pitches:
            print("ERROR: No note_on events found in input MIDI.", file=sys.stderr)
            sys.exit(1)

        min_note = min(all_pitches)
        print(f"MIDI pitch range: {min_note}–{max(all_pitches)}  ({len(all_pitches)} notes)")
        print(f"Original MIDI: tpb={orig_tpb}, bpm={orig_bpm}")
    else:
        # Video-only mode — use full keyboard range
        min_note = 21  # A0

    note_x_map = build_note_x_map(white_keys, black_keys, min_note)

    if args.dry_run:
        print("\n--- Dry run complete. Use these stats to calibrate. ---")
        print(f"White keys detected: {len(white_keys)}")
        print(f"Black keys detected: {len(black_keys)}")
        print(f"Notes in x-map: {len(note_x_map)}")
        cap.release()
        return

    # --- Step 3: Extract notes ---
    if args.midi is None:
        # VIDEO-ONLY MODE: scan frames for key state changes
        print("\n--- Step 3: Extract notes from video ---")
        video_notes = extract_notes_from_video(
            cap, note_x_map, y_sample_top, y_sample_bot,
            half_w, fps, total_frames, green_is_right, cfg['colors'],
            start_frame=frame_idx, frame_step=1, white_keys=white_keys, cfg=cfg,
            y_black=y_black, start_time=args.start_time)

        # --- End-of-music detection ---
        # Scan backward from the last frame to find the last frame with lit keys.
        # Uses the same saturated-pixel approach as start detection but in reverse.
        SAT_ON_THRESH = 70  # same as SAT_ON in extract_notes_from_video
        END_BUFFER_SEC = 2.0  # buffer to avoid cutting sustained notes
        face_yt = max(0, y_sample_bot - 30)
        face_yb = min(y_sample_bot + 5, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 1)
        # Keyboard x-bounds from note_x_map
        all_x = list(note_x_map.values())
        x_lo = min(all_x) if all_x else 0
        x_hi = max(all_x) if all_x else 1920

        if args.end_time is not None:
            # Manual override
            music_end_sec = args.end_time
            print(f"  Using manual end time: {music_end_sec:.1f}s")
        else:
            # Auto-detect: scan backward from last frame
            music_end_frame = total_frames - 1
            for f_scan in range(total_frames - 1, max(0, total_frames - int(fps * 60)), -5):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_scan)
                ret, frame = cap.read()
                if not ret:
                    continue
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                region = hsv[face_yt:face_yb, x_lo:x_hi, 1]
                n_saturated = int(np.sum(region > SAT_ON_THRESH))
                if n_saturated > 200:
                    music_end_frame = f_scan
                    break
            music_end_sec = music_end_frame / fps + END_BUFFER_SEC
            print(f"  End-of-music detected: frame {music_end_frame} "
                  f"({music_end_frame/fps:.1f}s, +{END_BUFFER_SEC}s buffer)")

        # Trim notes that start after the detected end time
        pre_trim_count = len(video_notes)
        video_notes = [n for n in video_notes if n[2] <= music_end_sec]

        # Trim end-screen popup artifacts: YouTube end screens overlay colored
        # elements (mini keyboards, subscribe buttons) that trigger a burst of
        # simultaneous false notes.  If the last timestamp has 3+ notes starting
        # together, remove them — real music rarely has 3+ simultaneous onsets
        # at the very end.
        if video_notes:
            last_t = video_notes[-1][2]
            tail = [n for n in video_notes if abs(n[2] - last_t) < 0.1]
            if len(tail) >= 3:
                video_notes = [n for n in video_notes if abs(n[2] - last_t) >= 0.1]
                print(f"  Trimmed {len(tail)} end-screen popup notes at {last_t:.1f}s")

        trimmed = pre_trim_count - len(video_notes)
        if trimmed > 0:
            print(f"  Trimmed {trimmed} total trailing/popup notes")

        # Determine effective start time for logging
        if args.start_time is not None:
            music_start_sec = args.start_time
        elif video_notes:
            music_start_sec = video_notes[0][2]
        else:
            music_start_sec = 0.0

        duration = music_end_sec - music_start_sec
        print(f"\nMusic range: {music_start_sec:.1f}s \u2013 {music_end_sec:.1f}s "
              f"({duration:.1f}s duration)")

        cap.release()

        right_count = sum(1 for _, h, _, _ in video_notes if h == 0)
        left_count = sum(1 for _, h, _, _ in video_notes if h == 1)
        print(f"Extracted {len(video_notes)} notes: right={right_count}, left={left_count}")

        if not video_notes:
            print("ERROR: No notes detected in video.", file=sys.stderr)
            sys.exit(1)

        # Convert to quantized events using phase-locked loop quantizer.
        # PLL self-corrects for BPM drift and per-note jitter (97% accuracy).
        print("\n--- Step 4: Build quantized 2-track MIDI ---")

        # Quantize each hand separately: when --triplet is set, left hand
        # uses triplet grid (sub=6) while right hand stays straight (sub=4).
        # This avoids swing artifacts on a straight melody.
        first_onset = video_notes[0][2]

        RIGHT_SUB = GRID_SUBDIVISIONS  # combined (12) or straight (4)
        LEFT_SUB = GRID_SUBDIVISIONS  # combined (12) or straight (4)
        RIGHT_GRID_TICKS = OUT_TPB // RIGHT_SUB
        LEFT_GRID_TICKS = OUT_TPB // LEFT_SUB

        # Split notes by hand, keeping original indices
        right_indices = [i for i, (_, h, _, _) in enumerate(video_notes) if h == 0]
        left_indices = [i for i, (_, h, _, _) in enumerate(video_notes) if h == 1]

        # Quantize each hand with its own PLL and grid
        def quantize_hand(indices, subdivisions):
            onsets = [video_notes[i][2] - first_onset for i in indices]
            offsets = [video_notes[i][3] - first_onset for i in indices]
            on_g = quantize_onsets_pll(onsets, OUT_BPM, alpha=0.1, subdivisions=subdivisions)
            off_g = quantize_onsets_pll(offsets, OUT_BPM, alpha=0.1, subdivisions=subdivisions)
            return on_g, off_g

        r_on_grid, r_off_grid = quantize_hand(right_indices, RIGHT_SUB)
        l_on_grid, l_off_grid = quantize_hand(left_indices, LEFT_SUB)

        right_events = []
        left_events = []

        for j, i in enumerate(right_indices):
            pitch = video_notes[i][0]
            on_tick = r_on_grid[j] * RIGHT_GRID_TICKS
            off_tick = r_off_grid[j] * RIGHT_GRID_TICKS
            on_tick = max(0, on_tick)
            off_tick = max(on_tick + 1, off_tick)
            right_events.append((on_tick, 'note_on', pitch, 80))
            right_events.append((off_tick, 'note_off', pitch, 0))

        for j, i in enumerate(left_indices):
            pitch = video_notes[i][0]
            on_tick = l_on_grid[j] * LEFT_GRID_TICKS
            off_tick = l_off_grid[j] * LEFT_GRID_TICKS
            on_tick = max(0, on_tick)
            off_tick = max(on_tick + 1, off_tick)
            left_events.append((on_tick, 'note_on', pitch, 80))
            left_events.append((off_tick, 'note_off', pitch, 0))

    else:
        # MIDI-BASED MODE: use MIDI notes, video for hand assignment
        print("\n--- Step 3: Assign hands via video color ---")
        ticks_to_seconds, seconds_to_out_ticks = make_tick_converters(
            orig_tpb, orig_bpm, OUT_TPB, OUT_BPM)

        note_ons = []
        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_ons.append((abs_tick, msg.note, msg.velocity))
        note_ons.sort()
        print(f"Note-on events: {len(note_ons)}")

        recent_right = deque(maxlen=5)
        recent_left = deque(maxlen=5)
        hand_assignments = {}
        color_count = {0: 0, 1: 0, 'fallback': 0}
        unmatched_pitches = set()

        for abs_tick, pitch, vel in note_ons:
            t_sec = ticks_to_seconds(abs_tick)
            frame_idx = int(t_sec * fps)
            frame_idx = max(0, min(frame_idx, total_frames - 1))

            hand = None
            if pitch in note_x_map:
                x_center = note_x_map[pitch]
                hv, sv, vv = sample_color(cap, frame_idx, x_center,
                                           y_sample_top, y_sample_bot, half_w=half_w)
                hand = classify_hand(hv, sv, vv, green_is_right, cfg['colors'])
                if hand is None:
                    unmatched_pitches.add(pitch)
            else:
                unmatched_pitches.add(pitch)

            if hand is None:
                hand = fallback_hand(pitch, recent_right, recent_left)
                color_count['fallback'] += 1
            else:
                color_count[hand] += 1

            hand_assignments[(abs_tick, pitch)] = hand
            (recent_right if hand == 0 else recent_left).append(pitch)

        total = len(note_ons)
        r, l, fb = color_count[0], color_count[1], color_count['fallback']
        detected_pct = round(100 * (r + l) / total) if total else 0
        fallback_pct = round(100 * fb / total) if total else 0
        print(f"Color detection: right={r}, left={l}, fallback={fb}")
        print(f"Detection rate: {detected_pct}% color, {fallback_pct}% fallback")
        if unmatched_pitches:
            print(f"Pitches using fallback: {sorted(unmatched_pitches)}")

        cap.release()

        # --- Step 4: Build quantized 2-track MIDI ---
        print("\n--- Step 4: Build quantized 2-track MIDI ---")

        first_note_sec = ticks_to_seconds(note_ons[0][0])
        tick_offset = quantize_tick(int(seconds_to_out_ticks(first_note_sec)), EIGHTH)
        print(f"First note at {first_note_sec:.3f}s → removing {tick_offset} tick offset")

        right_events = []
        left_events = []

        for track in mid.tracks:
            abs_tick = 0
            active = {}
            for msg in track:
                abs_tick += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    hand = hand_assignments.get((abs_tick, msg.note),
                                                fallback_hand(msg.note, recent_right, recent_left))
                    active[msg.note] = hand
                    t_sec = ticks_to_seconds(abs_tick)
                    out_tick = quantize_tick_smart(int(seconds_to_out_ticks(t_sec)), EIGHTH, SIXTEENTH) - tick_offset
                    ev = (max(0, out_tick), 'note_on', msg.note, msg.velocity)
                    (right_events if hand == 0 else left_events).append(ev)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active:
                        hand = active.pop(msg.note)
                        t_sec = ticks_to_seconds(abs_tick)
                        out_tick = quantize_tick_smart(int(seconds_to_out_ticks(t_sec)), EIGHTH, SIXTEENTH) - tick_offset
                        ev = (max(0, out_tick), 'note_off', msg.note, 0)
                        (right_events if hand == 0 else left_events).append(ev)

    for hand_name, hand_mode, events_ref in [
        ('right', args.right_hand, 'right_events'),
        ('left', args.left_hand, 'left_events'),
    ]:
        evts = locals()[events_ref]
        if hand_mode == 'no-overlap':
            evts = remove_overlaps(evts)
            print(f"No-overlap {hand_name} hand: held notes cut at next onset, chords kept")
        elif hand_mode == 'monophonic':
            keep = 'highest' if hand_name == 'right' else 'lowest'
            evts = make_monophonic(evts, keep=keep)
            print(f"Monophonic {hand_name} hand: single voice ({keep} note)")
        if hand_name == 'right':
            right_events = evts
        else:
            left_events = evts

    out_mid = MidiFile(type=1, ticks_per_beat=OUT_TPB)
    out_mid.tracks.append(build_track(right_events, 'Right Hand', OUT_US_PER_BEAT))
    out_mid.tracks.append(build_track(left_events, 'Left Hand', OUT_US_PER_BEAT))

    # Add key signature: use --key if provided, otherwise auto-detect
    key_sig = args.key
    if not key_sig:
        try:
            from music21 import converter as m21_converter
            score = m21_converter.parse(args.output)  # parse what we just built
        except Exception:
            score = None
        if score is None:
            # music21 not available or parse failed — save first, then try
            out_mid.save(args.output)
            try:
                from music21 import converter as m21_converter
                score = m21_converter.parse(args.output)
            except Exception:
                score = None
        if score is not None:
            detected = score.analyze('key')
            # Convert music21 key to MIDI key_signature format
            tonic = detected.tonic.name.replace('-', 'b')
            if detected.mode == 'minor':
                key_sig = tonic + 'm'
            else:
                key_sig = tonic
            print(f"Key auto-detected: {detected} (confidence={detected.correlationCoefficient:.2f})")

    if key_sig:
        out_mid.tracks[0].insert(0, MetaMessage('key_signature', key=key_sig, time=0))

    out_mid.save(args.output)

    rn = sum(1 for e in right_events if e[1] == 'note_on')
    ln = sum(1 for e in left_events if e[1] == 'note_on')
    print(f"\nSaved: {args.output}")
    print(f"Right hand: {rn} notes")
    print(f"Left hand:  {ln} notes")
    print(f"Total:      {rn + ln} notes")
    print(f"\nDone! Open {args.output} in MuseScore.")
    print("Tip: Preferences → Import → MIDI → Shortest note: 16th")

    # --- Auto-save settings.json next to the output MIDI ---
    settings_to_save = {}
    SAVE_KEYS = ['bpm', 'key', 'green_hand', 'frame', 'right_hand',
                 'left_hand', 'start_time', 'end_time', 'config']
    for key in SAVE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            settings_to_save[key] = val
    settings_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), 'settings.json')
    with open(settings_path, 'w') as f:
        json.dump(settings_to_save, f, indent=2)
        f.write('\n')
    print(f"Settings saved: {settings_path}")


if __name__ == '__main__':
    main()
