#!/usr/bin/env python3
"""Detect piano notes from falling colored blocks above the keyboard.

Works with videos where hands are visible on the keyboard (Rousseau, Kassia
style). Detects colored rectangles in the "waterfall" area above the keyboard,
maps them to pitches, and extracts timing from vertical position.

Usage:
    from detect_falling_blocks import detect_falling_notes_pipeline
    notes, bpm = detect_falling_notes_pipeline('video.mp4', 'output.mid')
"""

import cv2
import numpy as np


def find_ui_bar(hsv_frame, waterfall_top, waterfall_bottom):
    """Detect horizontal UI bars (full-width saturated bands) in the waterfall.

    Returns (bar_top, bar_bottom) or None if no bar found.
    """
    h, w = hsv_frame.shape[:2]
    bar_top = None
    for y in range(waterfall_top, waterfall_bottom):
        row_s = hsv_frame[y, :, 1]
        wide_sat = int(np.sum(row_s > 20))
        if wide_sat > w * 0.4 and bar_top is None:
            bar_top = y
        elif wide_sat < w * 0.2 and bar_top is not None:
            return (bar_top, y)
    if bar_top is not None:
        return (bar_top, waterfall_bottom)
    return None


def build_note_x_map_from_keyboard(cap, waterfall_bottom=None):
    """Build MIDI→x-pixel map by analyzing the physical keyboard in the video.

    Averages many frames to wash out the moving blocks and reveal the static
    keyboard.  Detects white key spacing from the lower half of the frame,
    then identifies C positions from the black-key gap pattern (E-F and B-C
    have no black key between them).

    Args:
        cap: cv2.VideoCapture (position will be changed).
        waterfall_bottom: y where waterfall ends / keyboard starts.  If None,
            auto-detected.

    Returns:
        (note_x_map, waterfall_top, waterfall_bottom) where note_x_map maps
        MIDI pitch int → x pixel float.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Average many frames to reveal static keyboard structure -----------
    acc = None
    count = 0
    for i in range(0, min(total, 2600), 50):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        if acc is None:
            acc = frame.astype(np.float64)
        else:
            acc += frame.astype(np.float64)
        count += 1
    avg = (acc / count).astype(np.uint8)
    gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
    hsv_avg = cv2.cvtColor(avg, cv2.COLOR_BGR2HSV)

    waterfall_top = 10  # small margin below frame top

    # --- Detect white keys by scanning ALL rows in the lower half -----------
    # Search the entire lower portion of the frame for rows with many evenly
    # spaced bright segments (white keys).  This avoids needing to know
    # waterfall_bottom ahead of time.
    best_y_wk = None
    best_n_keys = 0
    best_centers = []
    best_cv = 1.0

    for y_scan in range(h_frame - 20, h_frame // 3, -1):
        row = gray[y_scan, :]
        row_max = int(row.max())
        if row_max < 20:
            continue
        # Adaptive threshold at 30% of row max
        thresh = max(row_max * 0.3, 15)

        # Run-length detection of bright segments (white keys)
        in_key = False
        sx = 0
        centers = []
        for x in range(w_frame):
            if row[x] > thresh and not in_key:
                in_key = True
                sx = x
            elif (row[x] <= thresh or x == w_frame - 1) and in_key:
                seg_w = x - sx
                if seg_w > 8:
                    centers.append((sx + x) // 2)
                in_key = False

        if len(centers) >= 25:
            spacings = np.diff(centers)
            cv_val = np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else 999
            avg_w = np.mean(spacings)
            # White keys should be 20-60px wide depending on resolution
            if cv_val < 0.15 and 15 < avg_w < 70:
                if len(centers) > best_n_keys or (len(centers) == best_n_keys and cv_val < best_cv):
                    best_n_keys = len(centers)
                    best_y_wk = y_scan
                    best_centers = centers
                    best_cv = cv_val

    if best_n_keys < 10:
        raise RuntimeError(f"Could not detect enough white keys (found {best_n_keys})")

    avg_spacing = float(np.median(np.diff(best_centers)))
    print(f"  Keyboard scan: {best_n_keys} white keys at y={best_y_wk}, "
          f"spacing={avg_spacing:.1f}px")

    # --- Regularize white key positions ------------------------------------
    # Hand occlusion in the averaged frame can distort some key centers.
    # Use the median spacing and a robust reference to regenerate uniform
    # positions for any keys that deviate too much.
    spacings = np.diff(best_centers)
    # Find a run of consecutive well-spaced keys to anchor the grid
    # (at least 5 keys with spacing close to median)
    best_run_start = 0
    best_run_len = 0
    run_start = 0
    for i, sp in enumerate(spacings):
        if abs(sp - avg_spacing) <= avg_spacing * 0.15:
            run_len = i - run_start + 1
            if run_len > best_run_len:
                best_run_len = run_len
                best_run_start = run_start
        else:
            run_start = i + 1

    # Use the midpoint of the best run as reference
    ref_idx = best_run_start + best_run_len // 2
    ref_x = float(best_centers[ref_idx])

    # Regenerate all positions from the reference
    regularized = []
    for i in range(best_n_keys):
        regularized.append(ref_x + (i - ref_idx) * avg_spacing)
    best_centers = regularized
    print(f"  Regularized {best_n_keys} white keys from reference WK{ref_idx} "
          f"(x={ref_x:.0f})")

    # --- Auto-detect waterfall_bottom if not provided -----------------------
    # The keyboard top is where regular white-key segments first appear when
    # scanning upward from the detected key row.  The transition from
    # waterfall (few/irregular segments) to keyboard (many regular segments)
    # gives us the boundary.
    if waterfall_bottom is None:
        # Scan upward from the detected keyboard row until the pattern breaks
        for y_scan in range(best_y_wk, h_frame // 4, -1):
            row = gray[y_scan, :]
            row_max = int(row.max())
            if row_max < 10:
                waterfall_bottom = y_scan + 1
                break
            thresh = max(row_max * 0.3, 15)
            in_key = False
            sx = 0
            centers = []
            for x in range(w_frame):
                if row[x] > thresh and not in_key:
                    in_key = True
                    sx = x
                elif (row[x] <= thresh or x == w_frame - 1) and in_key:
                    if x - sx > 8:
                        centers.append((sx + x) // 2)
                    in_key = False
            if len(centers) < 15:
                waterfall_bottom = y_scan + 1
                break
        if waterfall_bottom is None:
            waterfall_bottom = best_y_wk - 50

    print(f"  Waterfall region: y={waterfall_top} to y={waterfall_bottom}")

    # --- Identify C positions using the black-key gap pattern ---------------
    # At a y-level where black keys exist (above the main white key surface),
    # the midpoint between two white keys is dark if there IS a black key,
    # and brighter if there is NO black key (E-F or B-C gap).
    # The black key zone is between waterfall_bottom and the detected white
    # key row.

    bk_zone_lo = waterfall_bottom + 10
    bk_zone_hi = min(best_y_wk - 10, waterfall_bottom + 90)

    has_black_raw = []
    for i in range(len(best_centers) - 1):
        mid_x = int((best_centers[i] + best_centers[i + 1]) / 2)
        wk_x = int(best_centers[i])
        if mid_x < 0 or mid_x >= w_frame or wk_x < 0 or wk_x >= w_frame:
            has_black_raw.append(None)
            continue
        if bk_zone_hi >= h_frame:
            bk_zone_hi = h_frame - 1

        bk_strip = gray[bk_zone_lo:bk_zone_hi,
                         max(0, mid_x - 3):min(w_frame, mid_x + 4)]
        wk_strip = gray[bk_zone_lo:bk_zone_hi,
                         max(0, wk_x - 3):min(w_frame, wk_x + 4)]
        bk_val = float(np.mean(bk_strip)) if bk_strip.size > 0 else 0
        wk_val = float(np.mean(wk_strip)) if wk_strip.size > 0 else 0

        if wk_val < 5:
            has_black_raw.append(None)  # can't determine (hand occlusion / too dark)
        elif bk_val < wk_val * 0.6:
            has_black_raw.append(True)
        else:
            has_black_raw.append(False)

    # Find the E-F / B-C gap pattern from the reliably detected gaps.
    # Only use pairs where the detection was conclusive (not None).
    reliable_no_bk = [i for i, v in enumerate(has_black_raw) if v is False]

    if len(reliable_no_bk) < 2:
        print("  WARNING: Could not detect black-key pattern, assuming WK0=C2")
        c_index = 0
    else:
        diffs = np.diff(reliable_no_bk)
        # Pattern should be alternating 3 and 4
        # If first diff is 4: first no-BK is E-F, second is B-C
        # If first diff is 3: first no-BK is B-C, second is E-F
        if len(diffs) > 0 and diffs[0] == 4:
            ef_first = True
        elif len(diffs) > 0 and diffs[0] == 3:
            ef_first = False
        else:
            ef_first = True

        if ef_first:
            e_index = reliable_no_bk[0]
        else:
            e_index = reliable_no_bk[1] if len(reliable_no_bk) > 1 else reliable_no_bk[0]

        c_index = e_index - 2

    # Now fill in the full has_black pattern using the known piano layout.
    # The piano has black keys between white keys at positions (relative to C):
    # C-D (0), D-E (1), F-G (3), G-A (4), A-B (5) — yes
    # E-F (2), B-C (6) — no
    BLACK_KEY_POSITIONS = {0, 1, 3, 4, 5}  # relative to C within an octave
    has_black = []
    for i in range(len(best_centers) - 1):
        pos = (i - c_index) % 7
        has_black.append(pos in BLACK_KEY_POSITIONS)

    # Determine the MIDI octave of the reference C.
    # c_index is the white key index of the reference C.  Keys below c_index
    # are in lower octaves.  base_c_midi is the MIDI number of THIS C.
    n_white = len(best_centers)

    # Estimate the leftmost visible C based on total keyboard span.
    # c_index white keys sit below the reference C (= c_index // 7 full octaves).
    octaves_below = c_index // 7  # full octaves below reference C

    if n_white >= 35:
        first_c_midi = 36  # C2 — large keyboard
    elif n_white >= 28:
        first_c_midi = 36  # C2
    elif n_white >= 20:
        first_c_midi = 48  # C3
    else:
        first_c_midi = 60  # C4

    base_c_midi = first_c_midi + octaves_below * 12

    # Simple approach: map each white key index to MIDI using offset from
    # the reference C.  White keys follow the pattern C D E F G A B with
    # semitone offsets [0, 2, 4, 5, 7, 9, 11].
    WHITE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]

    note_x_map = {}

    for i in range(n_white):
        # How many white keys is this from the reference C?
        rel = i - c_index  # positive = above C, negative = below
        # Convert white-key offset to semitone offset
        # rel // 7 gives full octaves, rel % 7 gives position within octave
        # Python's % handles negatives correctly: (-1) % 7 = 6
        octave = rel // 7
        pos = rel % 7  # 0=C, 1=D, 2=E, 3=F, 4=G, 5=A, 6=B
        midi = base_c_midi + octave * 12 + WHITE_SEMITONES[pos]
        note_x_map[midi] = float(best_centers[i])

    # Assign black keys at midpoints between white key pairs that have them
    for i in range(n_white - 1):
        if has_black[i] is not True:
            continue

        # The white key at index i has a known MIDI pitch
        rel = i - c_index
        pos = rel % 7  # position in octave: 0=C, 1=D, 2=E, 3=F, 4=G, 5=A, 6=B
        octave = rel // 7

        # Black keys exist between: C-D(C#), D-E(D#), F-G(F#), G-A(G#), A-B(A#)
        # Not between: E-F, B-C
        BK_SEMITONE = {0: 1, 1: 3, 3: 6, 4: 8, 5: 10}  # pos → black key semitone
        if pos in BK_SEMITONE:
            midi_bk = base_c_midi + octave * 12 + BK_SEMITONE[pos]
            bk_x = (best_centers[i] + best_centers[i + 1]) / 2.0
            note_x_map[midi_bk] = bk_x

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    sorted_notes = sorted(note_x_map.keys())
    lo_name = NOTE_NAMES[sorted_notes[0] % 12] + str(sorted_notes[0] // 12 - 1)
    hi_name = NOTE_NAMES[sorted_notes[-1] % 12] + str(sorted_notes[-1] // 12 - 1)
    print(f"  Note map: {len(note_x_map)} notes, {lo_name} to {hi_name}")

    return note_x_map, waterfall_top, waterfall_bottom


def detect_blocks_in_frame(hsv_frame, waterfall_top, waterfall_bottom,
                           note_x_map, color_ranges, min_area=50):
    """Detect colored note blocks in a single frame's waterfall region.

    Args:
        hsv_frame: Full frame in HSV color space.
        waterfall_top: y-coordinate of the top of the waterfall region.
        waterfall_bottom: y-coordinate of the bottom (keyboard top).
        note_x_map: dict mapping MIDI pitch → x pixel center.
        color_ranges: dict with 'right' and 'left' keys, each containing
            h_min, h_max, s_min, v_min for HSV thresholding.
        min_area: minimum pixel area for a valid block.

    Returns:
        list of dicts: [{pitch, hand, bottom_y, top_y, width, height, area}]
    """
    waterfall = hsv_frame[waterfall_top:waterfall_bottom, :]
    wf_h = waterfall_bottom - waterfall_top

    if wf_h <= 0:
        return []

    # Pre-compute sorted pitch→x for fast nearest-neighbor lookup
    sorted_pitches = sorted(note_x_map.keys(), key=lambda p: note_x_map[p])
    sorted_x = np.array([note_x_map[p] for p in sorted_pitches])

    blocks = []
    for hand_name, color in color_ranges.items():
        h_min = color.get('h_min', 0)
        h_max = color.get('h_max', 180)
        s_min = color.get('s_min', 80)
        v_min = color.get('v_min', 80)

        # HSV threshold (handle red hue wrap-around: H near 0 and 180)
        if h_min > h_max:
            # Wrap-around: e.g., h_min=170, h_max=10 means red
            mask1 = cv2.inRange(waterfall, np.array([h_min, s_min, v_min]),
                                np.array([180, 255, 255]))
            mask2 = cv2.inRange(waterfall, np.array([0, s_min, v_min]),
                                np.array([h_max, 255, 255]))
            mask = mask1 | mask2
        else:
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, 255, 255])
            mask = cv2.inRange(waterfall, lower, upper)

        # Morphological cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        for i in range(1, n_labels):  # skip background (label 0)
            x, y, w, h, area = stats[i]
            if area < min_area or h < 3:
                continue

            x_center = centroids[i][0]

            # Fast nearest-pitch lookup using sorted arrays
            idx = np.searchsorted(sorted_x, x_center)
            candidates = []
            if idx > 0:
                candidates.append(idx - 1)
            if idx < len(sorted_x):
                candidates.append(idx)
            if not candidates:
                continue

            best_ci = min(candidates, key=lambda c: abs(sorted_x[c] - x_center))
            best_pitch = sorted_pitches[best_ci]
            dist = abs(sorted_x[best_ci] - x_center)

            # Max distance: smaller than half the gap between adjacent semitones.
            # With 36px white key spacing, the smallest gap (white↔black) is ~18px,
            # so half that is ~9px.  Use 10px to keep matching tight.
            max_dist = 10
            if dist > max_dist:
                continue

            blocks.append({
                'pitch': best_pitch,
                'hand': 0 if hand_name == 'right' else 1,
                'bottom_y': waterfall_top + y + h,  # in full-frame coords
                'top_y': waterfall_top + y,
                'width': w,
                'height': h,
                'area': area,
            })

    return blocks


def estimate_note_speed(cap, frame_idx_a, frame_idx_b, waterfall_top,
                        waterfall_bottom, note_x_map, color_ranges):
    """Estimate note speed (pixels/second) by tracking blocks between two frames.

    Args:
        cap: cv2.VideoCapture
        frame_idx_a, frame_idx_b: two frame indices to compare
        Others: same as detect_blocks_in_frame

    Returns:
        float: estimated speed in pixels per second, or None on failure
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = (frame_idx_b - frame_idx_a) / fps

    frames = []
    for fi in [frame_idx_a, frame_idx_b]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            return None
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    blocks_a = detect_blocks_in_frame(frames[0], waterfall_top, waterfall_bottom,
                                       note_x_map, color_ranges)
    blocks_b = detect_blocks_in_frame(frames[1], waterfall_top, waterfall_bottom,
                                       note_x_map, color_ranges)

    # Match blocks by pitch and approximate y-position
    displacements = []
    for ba in blocks_a:
        for bb in blocks_b:
            if ba['pitch'] != bb['pitch'] or ba['hand'] != bb['hand']:
                continue
            dy = bb['bottom_y'] - ba['bottom_y']
            # Block should have moved DOWN (increasing y = toward keyboard)
            if dy <= 0:
                continue
            # Sanity: displacement should be reasonable
            expected_dy = dt * 200  # rough guess: ~200 px/s
            if 0.1 * expected_dy < dy < 5 * expected_dy:
                displacements.append(dy)

    if not displacements:
        return None

    median_dy = float(np.median(displacements))
    speed = median_dy / dt
    return speed


def detect_notes_from_blocks(cap, note_x_map, waterfall_top, waterfall_bottom,
                              fps, total_frames, color_ranges,
                              note_speed_pps=None, frame_step=3,
                              start_frame=0):
    """Extract notes from falling blocks across all video frames.

    Args:
        cap: cv2.VideoCapture
        note_x_map: dict mapping MIDI pitch → x pixel center
        waterfall_top: y of waterfall region top
        waterfall_bottom: y of waterfall region bottom (keyboard top)
        fps: video frame rate
        total_frames: total number of frames
        color_ranges: dict with 'right'/'left' HSV ranges
        note_speed_pps: pixels per second (auto-detected if None)
        frame_step: process every Nth frame (3 = every 3rd frame)
        start_frame: first frame to process

    Returns:
        list of (pitch, hand, onset_sec, offset_sec) tuples
    """
    # Auto-detect note speed if not provided
    if note_speed_pps is None:
        speeds = []
        # Try multiple pairs of frames spread across the video
        for frac in [0.3, 0.5, 0.7]:
            mid = int(total_frames * frac)
            for gap_sec in [0.3, 0.5, 1.0]:
                gap = int(fps * gap_sec)
                if mid + gap >= total_frames:
                    continue
                speed = estimate_note_speed(
                    cap, mid, mid + gap, waterfall_top, waterfall_bottom,
                    note_x_map, color_ranges)
                if speed is not None and 50 < speed < 1000:
                    speeds.append(speed)

        if speeds:
            note_speed_pps = float(np.median(speeds))
            print(f"  Note speed: {note_speed_pps:.0f} px/s (from {len(speeds)} estimates)")
        else:
            print("WARNING: Could not auto-detect note speed, using default 200 px/s")
            note_speed_pps = 200

    # Collect detections from all frames
    all_detections = []  # (pitch, hand, onset_sec, offset_sec, area)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = 0

    for f_idx in range(start_frame, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_time = f_idx / fps

        blocks = detect_blocks_in_frame(hsv, waterfall_top, waterfall_bottom,
                                         note_x_map, color_ranges)

        for b in blocks:
            # Convert y-position to time
            # bottom_y is where the note ends (at keyboard)
            # top_y is where the note begins (further from keyboard)
            # The keyboard line is at waterfall_bottom
            # Distance from bottom of block to keyboard = how far in the future
            # the note onset is
            onset_offset_px = waterfall_bottom - b['bottom_y']
            onset_sec = frame_time + onset_offset_px / note_speed_pps
            duration_sec = b['height'] / note_speed_pps
            offset_sec = onset_sec + duration_sec

            all_detections.append((
                b['pitch'], b['hand'], onset_sec, offset_sec, b['area']
            ))

        processed += 1
        if processed % 100 == 0:
            progress = f_idx / total_frames * 100
            print(f"  Frame {f_idx}/{total_frames} ({progress:.0f}%) — "
                  f"{len(all_detections)} raw detections")

    print(f"  {len(all_detections)} raw detections from {processed} frames")

    # Merge overlapping detections (same pitch, overlapping time)
    notes = merge_detections(all_detections)

    # Post-processing: remove spurious notes
    notes = _postprocess_notes(notes)
    notes = _merge_jittery_pitches(notes)
    return notes


def _postprocess_notes(notes, min_duration=0.05):
    """Remove spurious notes and clean up the detection.

    - Remove very short notes (< min_duration) that are likely noise.
    - When two notes on adjacent semitones overlap in time, keep the one
      with longer duration (resolves F/F# boundary confusion from wide blocks).

    Args:
        notes: list of (pitch, hand, onset_sec, offset_sec)
        min_duration: minimum note duration in seconds.

    Returns:
        cleaned list of notes.
    """
    # Step 1: Remove very short notes
    notes = [(p, h, on, off) for p, h, on, off in notes if off - on >= min_duration]

    # Step 2: Remove adjacent-semitone overlaps (near-simultaneous ghost notes)
    # When a wide block straddles two keys, both get detected at nearly the
    # same time. Remove the shorter one only when they are nearly simultaneous.
    notes.sort(key=lambda n: n[2])
    to_remove = set()

    for i in range(len(notes)):
        if i in to_remove:
            continue
        pi, hi, oni, offi = notes[i]
        duri = offi - oni

        for j in range(i + 1, len(notes)):
            if j in to_remove:
                continue
            pj, hj, onj, offj = notes[j]
            # Only check nearly simultaneous notes (within 0.08s)
            if onj - oni > 0.08:
                break

            # Check if adjacent semitones from the same hand
            if abs(pi - pj) == 1 and hi == hj:
                durj = offj - onj
                # Remove the shorter one (likely the ghost detection)
                if duri >= durj:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break

    notes = [n for i, n in enumerate(notes) if i not in to_remove]
    return notes


def _merge_jittery_pitches(notes, max_gap=0.15):
    """Absorb very short ghost notes into adjacent-semitone neighbors.

    When a block's x-center jitters between two adjacent key positions across
    frames, the merge step can produce a short spurious note alongside the
    real one.  This step removes the short note if a longer note on an
    adjacent semitone exists nearby.

    Args:
        notes: list of (pitch, hand, onset_sec, offset_sec)
        max_gap: maximum time gap between notes to consider them related.

    Returns:
        cleaned list of notes.
    """
    notes.sort(key=lambda n: n[2])
    to_remove = set()

    for i in range(len(notes)):
        if i in to_remove:
            continue
        pi, hi, oni, offi = notes[i]
        duri = offi - oni

        # Only consider short notes as potential ghosts
        if duri > 0.15:
            continue

        for j in range(len(notes)):
            if i == j or j in to_remove:
                continue
            pj, hj, onj, offj = notes[j]
            if hi != hj or abs(pi - pj) != 1:
                continue

            durj = offj - onj
            # If the neighbor is substantially longer and overlaps in time
            if durj > duri * 2 and onj < offi + max_gap and offj > oni - max_gap:
                to_remove.add(i)
                break

    notes = [n for i, n in enumerate(notes) if i not in to_remove]
    return notes


def merge_detections(detections, time_tolerance=0.3):
    """Merge overlapping detections of the same pitch into single notes.

    Groups detections by (pitch, hand), sorts by onset, and merges
    overlapping or near-overlapping entries. Similar to NMS in object
    detection but in (pitch, time) space.

    Uses the median onset of the first cluster of detections as the note
    onset (more robust than the earliest detection which may be noisy).

    Args:
        detections: list of (pitch, hand, onset_sec, offset_sec, area)
        time_tolerance: max gap in seconds to still merge (0.3s handles
            the fact that the same block appears in many consecutive frames
            with slightly different computed onset times).

    Returns:
        list of (pitch, hand, onset_sec, offset_sec) tuples
    """
    from collections import defaultdict

    # Group by (pitch, hand)
    groups = defaultdict(list)
    for pitch, hand, onset, offset, area in detections:
        groups[(pitch, hand)].append((onset, offset, area))

    notes = []
    for (pitch, hand), dets in groups.items():
        dets.sort()  # sort by onset

        # Merge overlapping detections
        merged = []
        cluster_onsets = [dets[0][0]]
        current_onset, current_offset, total_area = dets[0]
        count = 1

        for onset, offset, area in dets[1:]:
            if onset <= current_offset + time_tolerance:
                # Overlapping or close — extend
                current_offset = max(current_offset, offset)
                total_area += area
                cluster_onsets.append(onset)
                count += 1
            else:
                # Gap — emit current and start new
                # Use median onset for robustness
                median_onset = float(np.median(cluster_onsets))
                merged.append((median_onset, current_offset))
                current_onset, current_offset = onset, offset
                cluster_onsets = [onset]
                total_area = area
                count = 1

        median_onset = float(np.median(cluster_onsets))
        merged.append((median_onset, current_offset))

        for onset, offset in merged:
            notes.append((pitch, hand, onset, offset))

    # Sort by onset time
    notes.sort(key=lambda n: n[2])
    return notes


def detect_bpm(notes, min_bpm=50, max_bpm=140):
    """Detect BPM from note onset times using inter-onset interval analysis.

    Uses an eighth-note grid (subdivisions=2) to match against common musical
    subdivisions.  Tests each BPM candidate and scores by how well inter-onset
    intervals align to multiples of the eighth-note duration.

    Args:
        notes: list of (pitch, hand, onset_sec, offset_sec)
        min_bpm, max_bpm: BPM search range.

    Returns:
        float: detected BPM
    """
    onsets = sorted(set(round(n[2], 3) for n in notes))
    if len(onsets) < 4:
        return 80.0  # fallback

    # Compute consecutive inter-onset intervals (skip near-simultaneous)
    intervals = []
    for i in range(len(onsets) - 1):
        dt = onsets[i + 1] - onsets[i]
        if dt > 0.08:  # skip simultaneous notes
            intervals.append(dt)

    if len(intervals) < 3:
        return 80.0

    intervals = np.array(intervals)

    # Search for BPM using eighth-note as the basic unit
    best_bpm = 80.0
    best_score = -1e9

    for bpm_try in np.arange(min_bpm, max_bpm + 0.5, 0.5):
        eighth = 60.0 / bpm_try / 2  # eighth note duration
        # For each interval, compute how close it is to an integer multiple
        # of the eighth note
        ratios = intervals / eighth
        rounded = np.round(ratios).astype(int)
        rounded = np.maximum(rounded, 1)
        errors = np.abs(intervals - rounded * eighth)
        # Weighted score: penalize errors, reward accuracy
        # Normalize errors by eighth duration
        norm_errors = errors / eighth
        # Score: sum of (1 - error) for errors < 0.15, penalty for larger errors
        score = 0
        for e in norm_errors:
            if e < 0.15:
                score += 1.0 - e
            else:
                score -= 0.5
        if score > best_score:
            best_score = score
            best_bpm = bpm_try

    return best_bpm


def build_midi(notes, bpm, output_path):
    """Build a MIDI file from detected notes.

    Args:
        notes: list of (pitch, hand, onset_sec, offset_sec)
        bpm: beats per minute
        output_path: path to write the MIDI file

    Returns:
        mido.MidiFile
    """
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage

    mid = MidiFile(ticks_per_beat=480)

    # Separate hands into tracks
    hands = {0: [], 1: []}
    for pitch, hand, onset, offset in notes:
        hands[hand].append((pitch, onset, offset))

    tempo = mido.bpm2tempo(bpm)

    for hand_idx in [0, 1]:
        track = MidiTrack()
        mid.tracks.append(track)

        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        hand_name = 'Right' if hand_idx == 0 else 'Left'
        track.append(MetaMessage('track_name', name=hand_name, time=0))

        hand_notes = hands[hand_idx]
        if not hand_notes:
            continue

        # Build events sorted by time
        events = []
        for pitch, onset, offset in hand_notes:
            events.append((onset, 'note_on', pitch, 80))
            events.append((offset, 'note_off', pitch, 0))
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

        prev_tick = 0
        for time_sec, msg_type, pitch, vel in events:
            tick = int(mido.second2tick(time_sec, mid.ticks_per_beat, tempo))
            delta = max(0, tick - prev_tick)
            track.append(Message(msg_type, note=pitch, velocity=vel, time=delta))
            prev_tick = tick

    mid.save(output_path)
    print(f"  MIDI saved to {output_path}")
    return mid


def detect_falling_notes_pipeline(video_path, output_midi_path,
                                   color_ranges=None, waterfall_bottom=None,
                                   frame_step=3, note_speed_pps=None):
    """Complete pipeline: video → MIDI for falling-notes style videos.

    Args:
        video_path: path to the input video
        output_midi_path: path for the output MIDI file
        color_ranges: HSV color ranges for right/left hand blocks.
            If None, uses blue (right) / red (left) defaults.
        waterfall_bottom: y-coordinate where waterfall ends. Auto-detected if None.
        frame_step: process every Nth frame.
        note_speed_pps: override note speed in pixels/second.

    Returns:
        (notes, bpm) where notes is a list of (pitch, hand, onset_sec, offset_sec)
    """
    if color_ranges is None:
        color_ranges = {
            'right': {'h_min': 95, 'h_max': 125, 's_min': 60, 'v_min': 60},
            'left':  {'h_min': 170, 'h_max': 10, 's_min': 60, 'v_min': 60},
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"{fps:.0f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    # Step 1: Build pitch mapping from keyboard analysis
    print("Step 1: Analyzing keyboard...")
    note_x_map, waterfall_top, wf_bottom = build_note_x_map_from_keyboard(
        cap, waterfall_bottom=waterfall_bottom)

    # Step 2: Detect all notes
    print("Step 2: Detecting notes...")
    notes = detect_notes_from_blocks(
        cap, note_x_map, waterfall_top, wf_bottom,
        fps, total_frames, color_ranges,
        note_speed_pps=note_speed_pps, frame_step=frame_step)

    cap.release()

    print(f"  {len(notes)} notes after merging")

    # Step 3: Detect BPM
    print("Step 3: Detecting BPM...")
    bpm = detect_bpm(notes)
    print(f"  Detected BPM: {bpm:.1f}")

    # Step 4: Build MIDI
    print("Step 4: Building MIDI...")
    build_midi(notes, bpm, output_midi_path)

    # Print summary
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    print(f"\nSummary: {len(notes)} notes, BPM={bpm:.1f}")
    pitches = [n[0] for n in notes]
    lo, hi = min(pitches), max(pitches)
    lo_name = NOTE_NAMES[lo % 12] + str(lo // 12 - 1)
    hi_name = NOTE_NAMES[hi % 12] + str(hi // 12 - 1)
    print(f"  Pitch range: {lo_name} (MIDI {lo}) to {hi_name} (MIDI {hi})")

    # Print first 20 notes
    print("\nFirst 20 notes:")
    for i, (pitch, hand, onset, offset) in enumerate(notes[:20]):
        name = NOTE_NAMES[pitch % 12] + str(pitch // 12 - 1)
        hand_str = 'R' if hand == 0 else 'L'
        dur = offset - onset
        print(f"  {i+1:3d}. {name:4s} ({hand_str}) t={onset:6.2f}s dur={dur:.2f}s")

    return notes, bpm
