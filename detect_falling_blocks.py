#!/usr/bin/env python3
"""
detect_falling_blocks.py — Detect notes from falling-block piano tutorial videos.

Builds a pitch map directly from falling colored blocks (no keyboard detection):
  1. Scan many frames to collect (x_center, width) of every colored block
  2. Cluster x-positions into discrete keys using histogram peaks
  3. Fit a regular white-key grid (linear regression on wide blocks)
  4. Use detected narrow (black-key) blocks to identify which grid position is C
  5. Assign MIDI numbers and extract note onsets/offsets via scan-line tracking

Colors: red = left hand (H wrapping around 0), blue = right hand (H=100-140).

Usage:
    python detect_falling_blocks.py tests/test-falling-1/video.mp4 output.mid
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from piano_video_scribe.constants import NOTE_NAMES, WHITE_SEMITONES, midi_to_name


# ---------------------------------------------------------------------------
# Color detection
# ---------------------------------------------------------------------------

def auto_detect_colors(cap, y_min, y_max, n_frames=20):
    """Auto-detect the two hand colors from block pixels.

    Samples multiple frames, collects hues of saturated pixels in the
    waterfall region, and finds the two dominant hue clusters.

    Returns list of dicts: [{'name': str, 'h_center': int, 'h_min': int, 'h_max': int}]
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_hues = []

    for fi in range(0, total, max(1, total // n_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        region = hsv[y_min:y_max, :, :]
        sat_mask = (region[:, :, 1] > 80) & (region[:, :, 2] > 80)
        hues = region[:, :, 0][sat_mask]
        all_hues.extend(int(h) for h in hues)

    if not all_hues:
        # Fallback: standard green/blue
        return [
            {'name': 'color1', 'h_center': 55, 'h_min': 35, 'h_max': 75},
            {'name': 'color2', 'h_center': 110, 'h_min': 90, 'h_max': 130},
        ]

    # Build hue histogram (0-180)
    hist = np.zeros(180, dtype=int)
    for h in all_hues:
        hist[h % 180] += 1

    # Smooth and find peaks
    from scipy.ndimage import uniform_filter1d
    try:
        smooth = uniform_filter1d(hist.astype(float), size=10, mode='wrap')
    except ImportError:
        # No scipy — manual smoothing
        smooth = np.convolve(hist.astype(float), np.ones(10) / 10, mode='same')

    # Find peaks: local maxima above 10% of max
    threshold = smooth.max() * 0.1
    peaks = []
    for i in range(180):
        if smooth[i] > threshold:
            prev = smooth[(i - 1) % 180]
            next_ = smooth[(i + 1) % 180]
            if smooth[i] >= prev and smooth[i] >= next_:
                peaks.append((i, int(smooth[i])))

    # Merge nearby peaks (within 15 hue units)
    merged_peaks = []
    for h, c in sorted(peaks, key=lambda x: -x[1]):
        if not any(abs(h - mh) < 15 or abs(h - mh) > 165 for mh, _ in merged_peaks):
            merged_peaks.append((h, c))
        if len(merged_peaks) >= 2:
            break

    colors = []
    for h_center, _ in merged_peaks:
        colors.append({
            'name': f'h{h_center}',
            'h_center': h_center,
            'h_min': (h_center - 15) % 180,
            'h_max': (h_center + 15) % 180,
        })

    return colors


def make_color_mask(hsv_region, color):
    """Create a boolean mask for a detected color range."""
    h_min = color['h_min']
    h_max = color['h_max']
    s_min = 80
    v_min = 80

    if h_min > h_max:  # wraps around 0/180 (red)
        return (((hsv_region[:, :, 0] >= h_min) | (hsv_region[:, :, 0] <= h_max)) &
                (hsv_region[:, :, 1] > s_min) & (hsv_region[:, :, 2] > v_min))
    else:
        return ((hsv_region[:, :, 0] >= h_min) & (hsv_region[:, :, 0] <= h_max) &
                (hsv_region[:, :, 1] > s_min) & (hsv_region[:, :, 2] > v_min))


# ---------------------------------------------------------------------------
# Block detection via contours
# ---------------------------------------------------------------------------

def detect_blocks_in_frame(hsv, y_min, y_max, colors=None):
    """Detect colored blocks in the falling region of a single frame.

    Args:
        colors: list of color dicts from auto_detect_colors(). If None,
                uses hardcoded blue/green fallback.

    Returns list of (x_center, width, height, y_bottom, color_idx) tuples.
    color_idx is the index into the colors list (0 = first/higher pitch = RH).
    """
    if colors is None:
        colors = [
            {'name': 'green', 'h_min': 35, 'h_max': 75},
            {'name': 'blue', 'h_min': 90, 'h_max': 130},
        ]

    region = hsv[y_min:y_max, :, :]
    blocks = []

    for ci, color in enumerate(colors):
        mask = make_color_mask(region, color).astype(np.uint8) * 255
        # Erode horizontally to separate adjacent blocks that touch.
        # A 5x1 kernel removes 2px from each side, breaking bridges
        # between blocks on neighboring keys (e.g. C#5/D5 stacked
        # vertically in midi2video) while preserving block height.
        erode_kernel = np.ones((1, 5), dtype=np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            bbox_area = bw * bh
            if bbox_area == 0:
                continue
            fill_ratio = area / bbox_area

            # Basic size filter
            if bw < 8 or bw > 55 or bh < 5 or area < 50:
                continue

            # Shape filter: real falling blocks are tall narrow rectangles.
            # Reject wide/short shapes (UI elements, progress bars).
            # For blocks near keyboard (short), allow wider aspect ratios.
            if bh >= 10 and bw > bh * 3:
                continue  # Too wide relative to height (UI element)
            if fill_ratio < 0.5:
                continue  # Not a solid rectangle

            cx = x + bw / 2.0
            y_bot = y + bh + y_min
            blocks.append((cx, bw, bh, y_bot, color['name']))

    return blocks


# ---------------------------------------------------------------------------
# Step 1-2: Collect and cluster block x-positions
# ---------------------------------------------------------------------------

def collect_and_cluster_blocks(cap, y_min, y_max, sample_step=2, colors=None):
    """Scan frames, detect blocks, cluster x-positions into discrete keys.

    Returns list of (x_center, count, median_width) sorted by x.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_xs = []
    all_ws = []

    for fi in range(50, total, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for (cx, bw, bh, y_bot, color) in detect_blocks_in_frame(hsv, y_min, y_max, colors=colors):
            all_xs.append(cx)
            all_ws.append(bw)

    print(f"  Collected {len(all_xs)} block measurements")

    xs = np.array(all_xs)
    ws = np.array(all_ws)

    # Histogram-based peak finding
    x_lo, x_hi = xs.min() - 5, xs.max() + 5
    bins = np.arange(x_lo, x_hi + 1, 1)
    hist, edges = np.histogram(xs, bins=bins)
    smoothed = np.convolve(hist.astype(float), np.ones(5) / 5, mode='same')

    peaks = []
    for i in range(3, len(smoothed) - 3):
        if (smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]
                and smoothed[i] > smoothed[i - 3] and smoothed[i] > smoothed[i + 3]
                and smoothed[i] > 3):
            peaks.append(((edges[i] + edges[i + 1]) / 2, smoothed[i]))

    # Merge close peaks, but only if they have similar widths.
    # A narrow peak (black key ~22px) next to a wide peak (white key ~30px)
    # should NOT be merged — they're different keys.
    merged = []
    for px, ps in sorted(peaks):
        if merged and px - merged[-1][0] < 12:
            # Check widths: get median width for each peak's blocks
            mask_new = np.abs(xs - px) < 8
            mask_old = np.abs(xs - merged[-1][0]) < 8
            w_new = float(np.median(ws[mask_new])) if np.sum(mask_new) >= 3 else 30
            w_old = float(np.median(ws[mask_old])) if np.sum(mask_old) >= 3 else 30
            # Only merge if both are similar width (within 30%)
            if min(w_new, w_old) / max(w_new, w_old) > 0.7:
                if ps > merged[-1][1]:
                    merged[-1] = (px, ps)
            else:
                # Different widths = different keys, keep both
                merged.append((px, ps))
        else:
            merged.append((px, ps))

    # Refine and filter
    result = []
    for px, _ in merged:
        mask = np.abs(xs - px) < 8
        if np.sum(mask) >= 3:
            center = float(np.median(xs[mask]))
            count = int(np.sum(mask))
            med_w = float(np.median(ws[mask]))
            # Skip edge artifacts (x < 30)
            if center > 30:
                result.append((center, count, med_w))

    return result


# ---------------------------------------------------------------------------
# Step 3: Fit white key grid
# ---------------------------------------------------------------------------

def fit_white_key_grid(key_positions, width_threshold=None):
    """Fit a regular linear grid to white-key positions (wider blocks).

    Returns (spacing, offset, n_white_keys, black_positions).
    black_positions is list of (x_center,) for black keys.
    """
    if width_threshold is None:
        # Adaptive threshold: find the gap between black-key and white-key
        # widths. Constraints:
        # - Black keys are narrower (width ratio ~0.55-0.80 of white keys)
        # - Count ratio: black/white should be roughly 5/7 ≈ 0.71
        #   (but can be lower if not all keys are played)
        # - There should be more white keys than black keys
        all_widths = sorted(w for _, _, w in key_positions)
        if len(all_widths) >= 4:
            best_gap_score = -1
            best_threshold = 25
            for i in range(len(all_widths) - 1):
                gap = all_widths[i+1] - all_widths[i]
                if gap < 1.0:
                    continue
                threshold = (all_widths[i] + all_widths[i+1]) / 2
                below = [w for w in all_widths if w < threshold]
                above = [w for w in all_widths if w >= threshold]
                if len(below) < 1 or len(above) < 3:
                    continue
                # White keys should outnumber black keys
                if len(above) < len(below):
                    continue
                mean_below = np.mean(below)
                mean_above = np.mean(above)
                width_ratio = mean_below / mean_above
                count_ratio = len(below) / len(above)
                # Width ratio: 0.50-0.82 (black keys ~55-75% of white width)
                if not (0.45 <= width_ratio <= 0.85):
                    continue
                # Count ratio: 0.1-1.0 (some pieces don't use all black keys)
                if not (0.1 <= count_ratio <= 1.0):
                    continue
                # Score: prefer gap size, ratio near 0.65, count ratio near 0.71
                score = (gap
                         + 2.0 * max(0, 1.0 - abs(width_ratio - 0.65) / 0.2)
                         + 1.0 * max(0, 1.0 - abs(count_ratio - 0.5) / 0.4))
                if score > best_gap_score:
                    best_gap_score = score
                    best_threshold = threshold

            width_threshold = best_threshold
        else:
            width_threshold = 25

    print(f"  White/black width threshold: {width_threshold:.1f}")

    whites = [(x, n, w) for x, n, w in key_positions if w >= width_threshold]
    blacks = [(x,) for x, n, w in key_positions if w < width_threshold]

    if len(whites) < 5:
        raise ValueError(f"Only {len(whites)} white keys detected (need >= 5)")

    white_xs = np.array([x for x, _, _ in whites])
    diffs = np.diff(white_xs)
    typical = np.median(diffs)

    # Assign grid indices (handle gaps where keys are missing)
    indices = [0]
    for d in diffs:
        skip = max(1, round(d / typical))
        indices.append(indices[-1] + skip)
    indices = np.array(indices, dtype=float)

    # Linear regression: x = spacing * idx + offset
    A = np.vstack([indices, np.ones(len(indices))]).T
    result = np.linalg.lstsq(A, white_xs, rcond=None)
    spacing, offset = result[0]
    residuals = white_xs - (spacing * indices + offset)

    n_keys = int(indices[-1]) + 1
    print(f"  Grid: spacing={spacing:.2f}px, offset={offset:.2f}, "
          f"n_white={n_keys}, max_res={np.max(np.abs(residuals)):.2f}px")

    return spacing, offset, n_keys, blacks


# ---------------------------------------------------------------------------
# Step 4: Identify C position
# ---------------------------------------------------------------------------

def identify_c_position(spacing, offset, n_white, black_positions, base_octave=4):
    """Determine which grid index is C by matching black keys to piano layout.

    Because the black key pattern repeats every 7 white keys (one octave),
    multiple C positions may score equally. To disambiguate:
    1. First, find all candidates with the best black-key match score.
    2. Among tied candidates, use the grouping pattern of consecutive black
       keys: real pianos have groups of 2 (C#D#) then 3 (F#G#A#), separated
       by gaps (E-F and B-C have no sharps).
    3. If still tied, prefer the C position that places the keyboard center
       near C4 (typical for Synthesia and tutorial videos).
    """
    bk_offsets = [0.5, 1.5, 3.5, 4.5, 5.5]  # C# D# F# G# A# in WKW from C

    candidates = []
    for c_idx in range(n_white):
        c_x = c_idx * spacing + offset
        expected = []
        for oct in range(-2, 4):
            for off in bk_offsets:
                bk_x = c_x + (oct * 7 + off) * spacing
                if offset - spacing < bk_x < offset + (n_white + 1) * spacing:
                    expected.append(bk_x)

        total_dist, matches = 0.0, 0
        for (bk_x,) in black_positions:
            if expected:
                d = min(abs(bk_x - ex) for ex in expected)
                if d < spacing * 0.25:
                    matches += 1
                    total_dist += d

        avg = total_dist / matches if matches else 999
        candidates.append((matches, avg, c_idx))

    # Find the best match count
    best_score = max(c[0] for c in candidates)
    best_avg = min(c[1] for c in candidates if c[0] == best_score)
    # Use tolerance for floating-point comparison
    tied = [c for c in candidates if c[0] == best_score and c[1] - best_avg < 0.5]

    if len(tied) <= 1:
        best_c = tied[0][2]
        print(f"  C at grid index {best_c} ({best_score}/{len(black_positions)} "
              f"black keys matched)")
        return best_c

    # Disambiguate tied candidates using black key grouping pattern.
    # The piano's black keys come in groups of 2 and 3. Score each C position
    # by how well the detected black keys cluster into (2, 3) groups.
    best_group_score = -1
    group_scores = {}
    for _, _, c_idx in tied:
        c_x = c_idx * spacing + offset
        # For each detected black key, find its position within the octave
        positions_in_oct = []
        for (bk_x,) in black_positions:
            rel = ((bk_x - c_x) / spacing) % 7
            positions_in_oct.append(rel)

        # Score: each black key gets a point for being close to an expected
        # position, plus a bonus for having a neighbor at the right distance
        score = 0
        for pos in positions_in_oct:
            min_d = min(abs(pos - e) for e in bk_offsets)
            if min_d < 0.35:
                score += 1
                # Check for adjacent black key in same group
                for pos2 in positions_in_oct:
                    if pos2 != pos:
                        gap = abs(pos2 - pos)
                        if gap > 3.5:
                            gap = 7 - gap  # wrap
                        if 0.8 < gap < 1.3:
                            score += 0.5

        group_scores[c_idx] = score
        if score > best_group_score:
            best_group_score = score

    # Filter to best group score
    group_tied = [c for c in tied if group_scores[c[2]] >= best_group_score - 0.1]

    if len(group_tied) == 1:
        best_c = group_tied[0][2]
    else:
        # Final tiebreak: prefer C position that places the keyboard center
        # near C4 (MIDI 60). For base_octave=4, C4 is at c_idx.
        # The center of the keyboard is at grid index n_white/2.
        # We want c_idx such that the center note is near C4-E4.
        # Center white key index relative to C: n_white/2 - c_idx
        # This should be a small number (3-5 = E to G above C).
        mid_idx = n_white / 2.0
        best_c = min(group_tied, key=lambda t: abs(mid_idx - t[2] - 3.5))[2]

    print(f"  C at grid index {best_c} ({best_score}/{len(black_positions)} "
          f"black keys matched, {len(tied)} tied, disambiguated)")
    return best_c


# ---------------------------------------------------------------------------
# Step 5: Build pitch map
# ---------------------------------------------------------------------------

def build_pitch_map(spacing, offset, n_white, c_idx, black_positions, base_octave=4):
    """Build MIDI pitch -> x_center mapping.

    base_octave: octave of the first C (c_idx).
    """
    c_midi = (base_octave + 1) * 12  # C4 = 60

    note_map = {}

    # White keys: go right from C
    grid_i = c_idx
    oct = 0
    while grid_i < n_white:
        note_in_oct = grid_i - c_idx - oct * 7
        if 0 <= note_in_oct < 7:
            midi = c_midi + oct * 12 + WHITE_SEMITONES[note_in_oct]
            note_map[midi] = spacing * grid_i + offset
        grid_i += 1
        if (grid_i - c_idx) % 7 == 0:
            oct += 1

    # White keys: go left from C
    grid_i = c_idx - 1
    note_in_oct = 6  # B
    oct = -1
    while grid_i >= 0:
        midi = c_midi + oct * 12 + WHITE_SEMITONES[note_in_oct]
        if midi >= 21:
            note_map[midi] = spacing * grid_i + offset
        grid_i -= 1
        note_in_oct -= 1
        if note_in_oct < 0:
            note_in_oct = 6
            oct -= 1

    # Black keys
    for wm in sorted(note_map.keys()):
        semi = wm % 12
        if semi in (0, 2, 5, 7, 9):  # C D F G A have sharp
            bk_midi = wm + 1
            nxt = wm + 2
            if nxt not in note_map:
                continue
            mid_x = (note_map[wm] + note_map[nxt]) / 2.0
            # Use detected position if close
            best_x, best_d = mid_x, 999
            for (bk_x,) in black_positions:
                d = abs(bk_x - mid_x)
                if d < best_d:
                    best_d = d
                    best_x = bk_x
            note_map[bk_midi] = best_x if best_d < spacing * 0.4 else mid_x

    return note_map


# ---------------------------------------------------------------------------
# Step 6: Scan-line note extraction
# ---------------------------------------------------------------------------

def find_keyboard_y(cap):
    """Find the y-coordinate where the keyboard top edge starts.

    Uses multiple strategies with voting across frames:
    1. Look for the first row (scanning downward) where many white-key pixels
       appear — bright (V>180), low-saturation (S<40) pixels.
    2. Fall back to a fully dark horizontal line (divider above keyboard).

    Skips early frames which may be title cards. Returns the y-coordinate
    of the keyboard top edge.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Skip early frames (title cards), sample from ~25% onward
    start_frame = max(100, total // 4)
    sample_frames = [fi for fi in range(start_frame, total, max(1, total // 8))
                     if fi < total][:6]
    if not sample_frames:
        sample_frames = [total // 2]

    white_key_candidates = []
    dark_line_candidates = []

    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Strategy 1: Find where white-key pixels suddenly appear.
        # Scan from 40% height down to 95% height (some videos have
        # keyboards very low, e.g., midi2video at ~88% of frame).
        # Look for a sharp transition: row N has <5% white-key pixels,
        # row N+K has >20%. The keyboard top is around that transition.
        wk_counts = []
        y_start = int(h * 0.4)
        y_end = int(h * 0.95)
        for y in range(y_start, y_end):
            row = hsv[y, :]
            wk = int(np.sum((row[:, 1] < 40) & (row[:, 2] > 180)))
            wk_counts.append((y, wk))

        # Find the first row where wk > 20% of width, with a lookback
        # confirming that a few rows earlier had < 5%
        for idx, (y, wk) in enumerate(wk_counts):
            if wk > w * 0.20:
                # Check that ~5 rows before had < 5%
                lookback = max(0, idx - 5)
                prev_min = min(c for _, c in wk_counts[lookback:idx+1]) if idx > 0 else 0
                if prev_min < w * 0.05:
                    white_key_candidates.append(y)
                    break

        # Strategy 2: Fully dark row (>=90% dark pixels)
        for y in range(h // 2, int(h * 0.95)):
            row = hsv[y, :]
            dark = int(np.sum(row[:, 2] < 30))
            if dark > w * 0.9:
                dark_line_candidates.append(y)
                break

    # Prefer white-key detection (finds the actual keyboard top)
    if white_key_candidates:
        kb_y = int(np.median(white_key_candidates))
        print(f"  Keyboard top (white-key onset) at y={kb_y}")
        return kb_y

    # Fall back to dark line (this finds the divider, which may be
    # above OR below the keyboard depending on the video layout)
    if dark_line_candidates:
        kb_y = int(np.median(dark_line_candidates))
        print(f"  Keyboard boundary (dark line) at y={kb_y}")
        return kb_y

    # Fallback
    y_default = int(h * 0.62)
    print(f"  Keyboard boundary (default): y={y_default}")
    return y_default


def measure_fall_speed(cap, y_min, y_max):
    """Measure fall speed in pixels/frame by tracking blocks across frame pairs."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    speeds = []

    # Sample throughout the video, not just early frames
    sample_frames = list(range(200, min(total - 20, 900), 100))
    # Add samples from later in the video
    for frac in [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]:
        fi = int(total * frac)
        if fi not in sample_frames and fi < total - 20:
            sample_frames.append(fi)
    sample_frames.sort()

    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, f1 = cap.read()
        if not ret:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi + 10)
        ret, f2 = cap.read()
        if not ret:
            continue

        hsv1 = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV)
        b1 = detect_blocks_in_frame(hsv1, y_min, y_max)
        b2 = detect_blocks_in_frame(hsv2, y_min, y_max)

        for cx1, w1, h1, yb1, c1 in b1:
            for cx2, w2, h2, yb2, c2 in b2:
                if c1 == c2 and abs(cx1 - cx2) < 5 and abs(w1 - w2) < 5:
                    dy = yb2 - yb1
                    if 10 < dy < 150:
                        speeds.append(dy / 10.0)

    speed = float(np.median(speeds)) if speeds else 2.7
    print(f"  Fall speed: {speed:.2f} px/frame ({len(speeds)} measurements)")
    return speed


def _build_x_to_pitch(note_map):
    """Build a function mapping x-coordinate to MIDI pitch."""
    pitches_sorted = sorted(note_map.items(), key=lambda kv: kv[1])
    boundaries = []
    for i, (midi, x) in enumerate(pitches_sorted):
        x_lo = (pitches_sorted[i - 1][1] + x) / 2.0 if i > 0 else x - 20
        x_hi = (x + pitches_sorted[i + 1][1]) / 2.0 if i < len(pitches_sorted) - 1 else x + 20
        boundaries.append((midi, x_lo, x_hi, x))

    def x_to_pitch(cx):
        for midi, x_lo, x_hi, _ in boundaries:
            if x_lo <= cx <= x_hi:
                return midi
        return min(boundaries, key=lambda b: abs(b[3] - cx))[0]

    return x_to_pitch


def extract_notes_scanband(cap, note_map, keyboard_y, fall_speed,
                           merge_tolerance=0.15, min_duration=0.08):
    """Extract notes using a tall scan band just above the keyboard.

    Uses a band from (keyboard_y - band_height) to (keyboard_y - 2).
    Blocks are tracked as they pass through this band.
    Onset time = first appearance + offset to reach keyboard.
    Offset time = last appearance + offset for top of block to reach keyboard.

    This gives accurate timing and reliable detection since blocks spend
    many frames passing through the tall band.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Band: 50px tall, just above the keyboard dark line
    band_top = keyboard_y - 52
    band_bot = keyboard_y - 2
    band_h = band_bot - band_top

    # Time for a block to fall from band_top to keyboard
    frames_band_to_kb = (keyboard_y - band_top) / fall_speed
    time_offset = frames_band_to_kb / fps

    # Time for a block to traverse the full band
    frames_across_band = band_h / fall_speed

    print(f"  Scan band: y={band_top}..{band_bot} ({band_h}px), "
          f"onset offset: {time_offset:.2f}s")

    x_to_pitch = _build_x_to_pitch(note_map)
    hand_map = {'blue': 0, 'red': 1}

    # Track active notes: (pitch, color) -> {'first_frame', 'last_frame'}
    active = {}
    raw_notes = []

    for fi in range(0, total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        band = hsv[band_top:band_bot, :, :]

        # Find colored runs for each color
        current = set()
        for color_name, mask_fn in [('blue', blue_mask), ('red', red_mask)]:
            mask = mask_fn(band)
            col_any = np.any(mask, axis=0)

            in_run = False
            sx = 0
            w = col_any.shape[0]
            min_run_width = 12  # filter stray pixels
            x_margin = 30  # ignore edges of frame
            for x in range(w):
                if col_any[x] and not in_run:
                    in_run = True
                    sx = x
                elif not col_any[x] and in_run:
                    bw = x - sx
                    cx = (sx + x) / 2.0
                    if bw >= min_run_width and x_margin < cx < w - x_margin:
                        pitch = x_to_pitch(cx)
                        current.add((pitch, color_name))
                    in_run = False
            if in_run and w - sx >= min_run_width:
                cx = (sx + w) / 2.0
                if x_margin < cx < w - x_margin:
                    pitch = x_to_pitch(cx)
                    current.add((pitch, color_name))

        # New onsets
        for key in current:
            if key not in active:
                active[key] = {'first_frame': fi, 'last_frame': fi}
            else:
                active[key]['last_frame'] = fi

        # Check for ended notes: not seen for > gap_frames
        gap_frames = 3  # allow small gaps (video compression artifacts)
        ended = []
        for key, info in active.items():
            if key not in current and fi - info['last_frame'] > gap_frames:
                pitch, color = key
                hand = hand_map.get(color)
                # Onset: when bottom of block reaches keyboard
                # = first_frame / fps + time_offset
                onset = info['first_frame'] / fps + time_offset
                # Offset: when top of block passes keyboard
                # The block was visible from first_frame to last_frame in the band
                # Total visible duration in band = (last - first) frames
                # Real note duration = visible duration + band traversal time
                visible_dur = (info['last_frame'] - info['first_frame']) / fps
                offset = onset + visible_dur
                raw_notes.append((pitch, hand, onset, offset))
                ended.append(key)
        for key in ended:
            del active[key]

        if fi % 500 == 0 and fi > 0:
            print(f"    Frame {fi}/{total} ({100 * fi / total:.0f}%) "
                  f"— {len(raw_notes)} notes")

    # Close remaining
    for key, info in active.items():
        pitch, color = key
        hand = hand_map.get(color)
        onset = info['first_frame'] / fps + time_offset
        visible_dur = (info['last_frame'] - info['first_frame']) / fps
        raw_notes.append((pitch, hand, onset, onset + visible_dur))

    raw_notes.sort(key=lambda n: n[2])

    # Merge close notes of same pitch/hand
    merged = []
    for pitch, hand, onset, offset in raw_notes:
        found = False
        for i in range(len(merged) - 1, max(-1, len(merged) - 30), -1):
            mp, mh, mon, moff = merged[i]
            if mp == pitch and mh == hand and onset - moff < merge_tolerance:
                merged[i] = (mp, mh, mon, max(moff, offset))
                found = True
                break
            if onset - mon > 3.0:
                break
        if not found:
            merged.append((pitch, hand, onset, offset))

    # Filter short notes
    filtered = [(p, h, on, off) for p, h, on, off in merged
                if off - on >= min_duration]

    print(f"  Raw: {len(raw_notes)}, merged: {len(merged)}, "
          f"filtered (>={min_duration}s): {len(filtered)}")
    return filtered


def extract_notes_contour(cap, note_map, y_min, y_max, fall_speed,
                          colors=None, merge_tolerance=0.06, min_duration=0.12,
                          sample_step=None, keyboard_y_for_onset=None):
    """Extract notes by detecting block contours and projecting onset/duration.

    Each block in each frame independently provides:
      - pitch (from x-position)
      - onset_time = frame_time + (keyboard_y - block_bottom) / speed
      - duration = block_height / speed
      - hand (from color — auto-assigned based on pitch range)

    Multiple frames observing the same block produce multiple observations
    with nearly identical onset times. These are merged (median onset,
    median duration) to produce the final note list.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    px_per_sec = fall_speed * fps

    # Adaptive sample step: each block needs ~10 observations to merge
    # reliably. A block is visible for fall_time frames. Sample every
    # fall_time/10 frames to get 10 obs per block. Cap at step=10 to
    # keep merge_tolerance reasonable.
    if sample_step is None:
        visible_height = y_max - y_min
        fall_time_frames = visible_height / max(0.1, fall_speed)
        sample_step = max(1, min(10, int(fall_time_frames / 10)))
        merge_tolerance = max(0.06, 0.02 * sample_step)
        n_frames = total // sample_step
        print(f"  Sample step: {sample_step} frames ({n_frames} frames, "
              f"merge_tol={merge_tolerance:.2f}s)")

    # keyboard_y_for_onset: the y-coordinate where blocks reach the keyboard.
    # Used to project block y-position to onset time. Defaults to y_max.
    onset_ref_y = keyboard_y_for_onset if keyboard_y_for_onset is not None else y_max

    # Build x -> pitch mapping with boundaries.
    # First, deduplicate notes with nearly identical x-positions (< 5px apart).
    # When the keyboard detector places a black key at almost the same x as a
    # white key, the boundary between them is unreliable. Keep the white key.
    white_semitones_set = {0, 2, 4, 5, 7, 9, 11}
    pitches_sorted = sorted(note_map.items(), key=lambda kv: kv[1])
    deduped = []
    for midi, x in pitches_sorted:
        if deduped and abs(x - deduped[-1][1]) < 5:
            # Keep whichever is a white key, or if both same type, keep first
            prev_midi = deduped[-1][0]
            prev_is_white = prev_midi % 12 in white_semitones_set
            curr_is_white = midi % 12 in white_semitones_set
            if curr_is_white and not prev_is_white:
                deduped[-1] = (midi, x)
            # else keep previous (either both white or prev is white)
        else:
            deduped.append((midi, x))

    boundaries = []
    for i, (midi, x) in enumerate(deduped):
        x_lo = (deduped[i - 1][1] + x) / 2.0 if i > 0 else x - 20
        x_hi = (x + deduped[i + 1][1]) / 2.0 if i < len(deduped) - 1 else x + 20
        boundaries.append((midi, x_lo, x_hi, x))

    def x_to_pitch(cx):
        for midi, x_lo, x_hi, _ in boundaries:
            if x_lo <= cx <= x_hi:
                return midi
        return min(boundaries, key=lambda b: abs(b[3] - cx))[0]

    # Phase 1: Collect observations with color names (hand assigned later)
    # Scan only the top portion (y_min to y_max) — clean, no keyboard glow.
    observations = []  # (pitch, color_name, onset_sec, dur_sec)

    for fi in range(0, total, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blocks = detect_blocks_in_frame(hsv, y_min, y_max, colors=colors)

        t_now = fi / fps
        for cx, bw, bh, y_bot, color_name in blocks:
            pitch = x_to_pitch(cx)
            # Project onset: how long until this block's bottom reaches keyboard
            onset_sec = t_now + max(0, onset_ref_y - y_bot) / px_per_sec
            dur_sec = bh / px_per_sec
            observations.append((pitch, color_name, onset_sec, dur_sec))

        if fi % 500 == 0 and fi > 0:
            print(f"    Frame {fi}/{total} ({100 * fi / total:.0f}%) "
                  f"— {len(observations)} observations")

    print(f"  {len(observations)} observations from {total // sample_step} frames")

    # Phase 2: Auto-assign hands based on average pitch per color
    color_pitches = defaultdict(list)
    for pitch, color_name, onset, dur in observations:
        color_pitches[color_name].append(pitch)

    if len(color_pitches) >= 2:
        # Color with higher average pitch = RH (hand 0)
        color_avg = {c: np.mean(ps) for c, ps in color_pitches.items()}
        sorted_colors = sorted(color_avg, key=lambda c: -color_avg[c])
        hand_map = {sorted_colors[0]: 0}  # highest pitch = RH
        for c in sorted_colors[1:]:
            hand_map[c] = 1  # lower pitch = LH
        assignments = ', '.join(f'{c}={"RH" if h==0 else "LH"}' for c, h in hand_map.items())
        print(f"  Hand assignment: {assignments}")
    elif len(color_pitches) == 1:
        # Single color — assign hand by pitch (>=60 = RH)
        hand_map = {list(color_pitches.keys())[0]: None}
        print(f"  Single color detected — hand from pitch")
    else:
        hand_map = {}

    # Replace color names with hand indices
    obs_with_hands = [(pitch, hand_map.get(cn), onset, dur)
                      for pitch, cn, onset, dur in observations]

    # Sort by (pitch, hand, onset) then cluster overlapping onsets
    obs_with_hands.sort(key=lambda o: (o[0], o[1] or 0, o[2]))

    notes = []
    i = 0
    while i < len(obs_with_hands):
        pitch, hand, onset, dur = obs_with_hands[i]
        group_onsets = [onset]
        group_durs = [dur]

        # Grow cluster: merge observations of the SAME block seen across
        # frames. Chain-merge with max span: each observation must be
        # within tolerance of the previous AND the total cluster span
        # can't exceed the note's duration + a small margin.
        # This prevents multiple distinct notes from being merged when
        # several blocks of the same pitch are visible simultaneously.
        j = i + 1
        last_onset = onset
        # Use tighter margin for short notes to separate rapid repeated notes,
        # but keep looser margin for long notes to avoid splitting sustained notes.
        margin = 0.05 if dur < 0.4 else 0.1
        max_span = dur + margin
        while j < len(obs_with_hands):
            p2, h2, on2, dur2 = obs_with_hands[j]
            if p2 != pitch or h2 != hand:
                break
            if abs(on2 - last_onset) <= merge_tolerance and (on2 - onset) <= max_span:
                group_onsets.append(on2)
                group_durs.append(dur2)
                last_onset = on2
                j += 1
            else:
                break

        # Emit note: use 25th percentile of onsets (reduces bias from
        # merged contours that inflate onset times), median duration
        med_onset = float(np.percentile(group_onsets, 25))
        med_dur = float(np.median(group_durs))
        if med_dur >= min_duration:
            notes.append((pitch, hand, med_onset, med_onset + med_dur))

        i = j

    notes.sort(key=lambda n: n[2])

    # Deduplicate: drop same-pitch same-hand notes that overlap in time.
    # This removes false duplicates from contour detection seeing the same
    # block at slightly different positions across frames.
    pre_dedup = len(notes)
    deduped = []
    for note in notes:
        pitch, hand, onset, offset = note
        is_dup = False
        for k in range(len(deduped) - 1, max(-1, len(deduped) - 20), -1):
            dp, dh, don, doff = deduped[k]
            if dp == pitch and dh == hand and onset < doff:
                is_dup = True
                break
        if not is_dup:
            deduped.append(note)
    notes = deduped
    if len(notes) < pre_dedup:
        print(f"  Deduplication: {pre_dedup} → {len(notes)} notes "
              f"(removed {pre_dedup - len(notes)} overlapping duplicates)")

    rh = sum(1 for _, h, _, _ in notes if h == 0)
    lh = sum(1 for _, h, _, _ in notes if h == 1)
    print(f"  {len(observations)} obs → {len(notes)} notes ({rh} RH, {lh} LH)")
    return notes


# ---------------------------------------------------------------------------
# BPM detection
# ---------------------------------------------------------------------------

def detect_bpm(notes):
    """Estimate BPM from note onset intervals."""
    if len(notes) < 4:
        return 78

    onsets = sorted(set(round(n[2], 3) for n in notes))
    intervals = [onsets[i] - onsets[i - 1] for i in range(1, len(onsets))
                 if 0.15 < onsets[i] - onsets[i - 1] < 4.0]

    if not intervals:
        return 78

    intervals = np.array(intervals)

    # Histogram with fine bins
    hist, edges = np.histogram(intervals, bins=200, range=(0.15, 4.0))
    kernel = np.exp(-np.arange(-5, 6) ** 2 / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    smoothed = np.convolve(hist.astype(float), kernel, mode='same')

    # Find dominant interval
    peak_idx = np.argmax(smoothed)
    dominant = (edges[peak_idx] + edges[peak_idx + 1]) / 2

    # Try interpreting as quarter note, eighth note, half note, dotted quarter
    candidates = []
    for divisor, name in [(1, 'quarter'), (0.5, 'eighth'), (2, 'half'),
                          (1.5, 'dotted quarter'), (0.75, 'dotted eighth')]:
        bpm = 60.0 / (dominant / divisor) if dominant / divisor > 0 else 999
        if 50 <= bpm <= 200:
            candidates.append((bpm, name, dominant))

    if candidates:
        # Prefer quarter note if reasonable, otherwise pick closest to 78
        for bpm, name, _ in candidates:
            if name == 'quarter' and 50 <= bpm <= 120:
                return round(bpm)
        # Fallback: closest to 78
        best = min(candidates, key=lambda c: abs(c[0] - 78))
        return round(best[0])

    return 78


# ---------------------------------------------------------------------------
# MIDI output
# ---------------------------------------------------------------------------

def notes_to_midi(notes, output_path, bpm=78, velocity=80):
    """Write notes to MIDI with separate RH/LH tracks."""
    mid = MidiFile(ticks_per_beat=480)
    tpb = mid.ticks_per_beat
    us_per_beat = int(60_000_000 / bpm)

    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(MetaMessage('set_tempo', tempo=us_per_beat, time=0))
    tempo_track.append(MetaMessage('time_signature', numerator=3, denominator=4, time=0))
    tempo_track.append(MetaMessage('end_of_track', time=0))

    rh = [(p, on, off) for p, h, on, off in notes if h == 0]
    lh = [(p, on, off) for p, h, on, off in notes if h == 1]
    unk = [(p, on, off) for p, h, on, off in notes if h is None]
    for p, on, off in unk:
        (rh if p >= 60 else lh).append((p, on, off))

    def sec_to_ticks(sec):
        return int(sec * tpb * bpm / 60.0)

    for track_notes, name, ch in [(rh, 'Right Hand', 0), (lh, 'Left Hand', 1)]:
        if not track_notes:
            continue
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('track_name', name=name, time=0))

        events = []
        for pitch, onset, offset in track_notes:
            t_on = sec_to_ticks(onset)
            t_off = sec_to_ticks(offset)
            if t_off <= t_on:
                t_off = t_on + sec_to_ticks(0.1)
            events.append((t_on, 'note_on', pitch, velocity))
            events.append((t_off, 'note_off', pitch, 0))

        events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

        prev = 0
        for tick, mtype, pitch, vel in events:
            delta = max(0, tick - prev)
            track.append(Message(mtype, note=pitch, velocity=vel,
                                 channel=ch, time=delta))
            prev = tick
        track.append(MetaMessage('end_of_track', time=0))

    mid.save(output_path)
    print(f"  Saved MIDI: {output_path}")
    return mid


# ---------------------------------------------------------------------------
# Calibration: align keyboard note_map to block positions
# ---------------------------------------------------------------------------

def _calibrate_note_map_with_blocks(note_map, key_positions):
    """Adjust note_map x-coordinates to match detected block positions.

    The keyboard detector and the falling blocks may use slightly different
    x-coordinate scales (e.g., keyboard keys at 26px spacing, blocks at 25px).
    This finds the best affine transform by matching block positions to
    note positions (both white AND black keys).

    Args:
        note_map: dict {midi_note: x_position} from keyboard detector
        key_positions: list of (x_center, count, width) from block clustering

    Returns:
        adjusted note_map with x-coordinates in block space
    """
    block_xs = sorted([x for x, _, _ in key_positions])

    # Use ALL note positions (white + black) for matching
    all_notes = sorted(note_map.items(), key=lambda kv: kv[1])
    all_xs = np.array([x for _, x in all_notes])

    # Also get white-key-only positions for the anchor fitting
    white_semitones = {0, 2, 4, 5, 7, 9, 11}
    white_notes = sorted([(midi, x) for midi, x in note_map.items()
                          if midi % 12 in white_semitones],
                         key=lambda kv: kv[1])
    white_xs = np.array([x for _, x in white_notes])

    n_blocks = len(block_xs)
    n_all = len(all_xs)

    # Estimate keyboard spacing from white keys
    white_diffs = np.diff(white_xs)
    white_spacing = float(np.median(white_diffs))

    print(f"  Calibration: {n_blocks} blocks; "
          f"{len(white_notes)} white + {n_all - len(white_notes)} black keys, "
          f"spacing≈{white_spacing:.1f}px")

    # Strategy: try different affine transforms and match each block to
    # its nearest note (white or black). Use pairs of blocks as anchors.
    best_score = float('inf')
    best_a, best_b = 1.0, 0.0
    best_pairs = None

    # For efficiency, only use a subset of block pairs as anchors
    # Use first block + each other block to define the affine transform
    for anchor_i in range(min(n_blocks, 5)):
        for anchor_j in range(max(anchor_i + 1, n_blocks - 5), n_blocks):
            bx0, bx1 = block_xs[anchor_i], block_xs[anchor_j]
            if abs(bx1 - bx0) < 50:
                continue
            # Try matching these two blocks to different notes
            # Find candidate notes near each block position
            for ni in range(n_all):
                kx0 = all_xs[ni]
                if abs(kx0 - bx0) > white_spacing * 2:
                    continue
                for nj in range(ni + 1, n_all):
                    kx1 = all_xs[nj]
                    if abs(kx1 - bx1) > white_spacing * 2:
                        continue
                    if abs(kx1 - kx0) < 1:
                        continue
                    a = (bx1 - bx0) / (kx1 - kx0)
                    b_val = bx0 - a * kx0
                    if not (0.85 <= a <= 1.15):
                        continue

                    # Match each block to nearest note after transform
                    transformed = a * all_xs + b_val
                    pairs = []
                    total_err = 0.0
                    matched_indices = set()
                    for bx in block_xs:
                        dists = np.abs(transformed - bx)
                        best_j = int(np.argmin(dists))
                        d = float(dists[best_j])
                        total_err += d * d
                        pairs.append((all_xs[best_j], bx, all_notes[best_j][0], d))
                        matched_indices.add(best_j)

                    # All blocks should match different notes
                    if len(matched_indices) < n_blocks:
                        continue

                    # Penalize transforms far from identity
                    identity_penalty = 1000 * (a - 1.0) ** 2 + 0.1 * b_val ** 2
                    score = total_err + identity_penalty

                    if score < best_score:
                        best_score = score
                        best_a, best_b = a, b_val
                        best_pairs = pairs

    if best_pairs is None:
        print(f"  Calibration: no good match found, keeping original")
        return note_map

    max_res = max(p[3] for p in best_pairs)
    print(f"  Calibration: scale={best_a:.4f}, offset={best_b:.1f}, "
          f"max_residual={max_res:.1f}px ({len(best_pairs)} anchors)")
    for kb_x, blk_x, midi, d in best_pairs:
        pred = best_a * kb_x + best_b
        print(f"    {midi_to_name(midi):5s} kb_x={kb_x:.0f} → block_x={blk_x:.0f} "
              f"(predicted={pred:.0f}, err={blk_x - pred:.1f})")

    # Apply transform to note_map
    calibrated = {midi: best_a * x + best_b for midi, x in note_map.items()}
    return calibrated


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def detect_falling_notes_pipeline(video_path, output_path, base_octave=None):
    """Full pipeline: video -> MIDI via falling block detection."""
    print(f"\n{'=' * 60}")
    print(f"Falling Block Note Detection")
    print(f"{'=' * 60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}, {fps:.0f}fps, {total} frames, {total / fps:.1f}s")

    # Step 1: Find keyboard boundary
    print(f"\n--- Step 1: Finding keyboard boundary ---")
    kb_y = find_keyboard_y(cap)
    y_min = 5
    # Scan the full waterfall area but exclude a small band near the keyboard
    # where glow contaminates block detection. 50px margin is enough.
    y_max = max(y_min + 10, kb_y - 50)
    print(f"  Keyboard at y={kb_y}")
    print(f"  Block region: y={y_min}..{y_max}")

    # Step 1b: Auto-detect block colors
    print(f"\n--- Step 1b: Detecting block colors ---")
    colors = auto_detect_colors(cap, y_min, y_max)
    for c in colors:
        print(f"  {c['name']}: H={c['h_min']}-{c['h_max']} (center={c['h_center']})")

    # Step 2: Build pitch map.
    # Try block-based first (uses actual block positions, most accurate).
    # Fall back to keyboard detector if not enough blocks for a grid fit.
    print(f"\n--- Step 2: Building pitch map ---")
    key_positions = collect_and_cluster_blocks(cap, y_min, y_max, colors=colors)
    print(f"  {len(key_positions)} discrete keys")

    note_map = None
    use_block_based = False
    if len(key_positions) >= 15:
        try:
            spacing, offset, n_white, blacks = fit_white_key_grid(key_positions)
            # Only use block-based mapping if we have enough black keys
            # for reliable C identification (need at least 5 black keys
            # per octave span to avoid ambiguous identification)
            n_blacks = len(blacks)
            n_octaves = max(1, n_white / 7.0)
            blacks_per_octave = n_blacks / n_octaves
            if blacks_per_octave >= 2.0 and n_white >= 20:
                c_idx = identify_c_position(spacing, offset, n_white, blacks)
                # Auto-detect octave from keyboard size if not explicitly set
                if base_octave is None:
                    if n_white >= 28:
                        base_octave = 2  # C2 start — 4+ octave keyboard
                    else:
                        base_octave = 2  # C2 start — 3+ octave keyboard
                note_map = build_pitch_map(spacing, offset, n_white, c_idx, blacks,
                                           base_octave=base_octave)
                use_block_based = True
                print(f"  Block-based pitch map: {len(note_map)} notes")
            else:
                print(f"  Too few black keys ({n_blacks} across {n_octaves:.1f} octaves, "
                      f"{blacks_per_octave:.1f}/oct) for reliable C identification")
        except Exception as e:
            print(f"  Block-based mapping failed: {e}")

    if note_map is None:
        # Fall back to keyboard detector + offset correction
        print(f"  Using keyboard detector for pitch map")
        from pianovideoscribe import detect_keyboard as _detect_kb
        from pianovideoscribe import build_note_x_map as _build_nxm
        try:
            kb_result = _detect_kb(cap, frame_idx=None)
            wk, bk = kb_result[0], kb_result[1]
            note_map = _build_nxm(wk, bk, 21)
            print(f"  Keyboard detector: {len(wk)} white, {len(bk)} black keys")

            print(f"  Pitch map: {len(note_map)} notes "
                  f"(MIDI {min(note_map)}-{max(note_map)})")

            # Calibrate note_map using block positions
            if len(key_positions) >= 3:
                note_map = _calibrate_note_map_with_blocks(
                    note_map, key_positions)

        except Exception as e:
            print(f"  Keyboard detector failed: {e}")
            note_map = {}

    if not note_map:
        print("  ERROR: No pitch map available", file=sys.stderr)
        cap.release()
        return []

    for midi in sorted(note_map.keys()):
        print(f"    {midi_to_name(midi):5s} (MIDI {midi:3d}) -> x={note_map[midi]:7.1f}")

    # Step 6: Measure fall speed
    print(f"\n--- Step 6: Measuring fall speed ---")
    fall_speed = measure_fall_speed(cap, y_min, kb_y)

    # Step 7: Extract notes via contour detection + cross-frame merge
    print(f"\n--- Step 7: Extracting notes (contour) ---")
    notes = extract_notes_contour(cap, note_map, y_min, y_max, fall_speed,
                                  colors=colors, keyboard_y_for_onset=kb_y)

    # Step 8: BPM
    print(f"\n--- Step 8: BPM detection ---")
    bpm = detect_bpm(notes)
    print(f"  BPM: {bpm}")

    # Print results
    print(f"\n--- Results: {len(notes)} notes ---")
    hand_names = {0: 'RH', 1: 'LH', None: '??'}
    for i, (pitch, hand, onset, offset) in enumerate(notes[:40]):
        dur = offset - onset
        print(f"  {i + 1:3d}  {onset:7.2f}s  {midi_to_name(pitch):<5s}  "
              f"{hand_names.get(hand, '??'):>3s}  {dur:.02f}s")
    if len(notes) > 40:
        print(f"  ... ({len(notes) - 40} more)")

    rh_count = sum(1 for _, h, _, _ in notes if h == 0)
    lh_count = sum(1 for _, h, _, _ in notes if h == 1)
    print(f"\n  RH={rh_count}, LH={lh_count}, total={len(notes)}")

    # Write MIDI (skip when called as a library with output_path=None)
    if output_path is not None:
        print(f"\n--- Writing MIDI ---")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        notes_to_midi(notes, output_path, bpm=bpm)

    cap.release()

    # Verification: print first RH notes
    rh_notes = sorted([(p, on) for p, h, on, _ in notes if h == 0], key=lambda x: x[1])
    print(f"\n--- First 10 RH note names ---")
    print("  " + " ".join(midi_to_name(p) for p, _ in rh_notes[:10]))

    return notes


def main():
    parser = argparse.ArgumentParser(
        description='Detect notes from falling-block piano tutorial videos')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('output', help='Path for output MIDI file')
    parser.add_argument('--base-octave', type=int, default=4,
                        help='Octave of the first C (default: 4)')
    args = parser.parse_args()
    detect_falling_notes_pipeline(args.video, args.output, args.base_octave)


if __name__ == '__main__':
    main()
