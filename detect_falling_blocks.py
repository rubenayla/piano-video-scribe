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

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
WHITE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B


def midi_to_name(midi_num):
    octave = (midi_num // 12) - 1
    return f"{NOTE_NAMES[midi_num % 12]}{octave}"


# ---------------------------------------------------------------------------
# Color masks
# ---------------------------------------------------------------------------

def blue_mask(hsv_region):
    """Boolean mask for blue (RH) pixels."""
    return ((hsv_region[:, :, 0] >= 100) & (hsv_region[:, :, 0] <= 140) &
            (hsv_region[:, :, 1] > 80) & (hsv_region[:, :, 2] > 80))


def red_mask(hsv_region):
    """Boolean mask for red (LH) pixels. Red hue wraps around 0/180."""
    return (((hsv_region[:, :, 0] <= 10) | (hsv_region[:, :, 0] >= 170)) &
            (hsv_region[:, :, 1] > 80) & (hsv_region[:, :, 2] > 80))


# ---------------------------------------------------------------------------
# Block detection via contours
# ---------------------------------------------------------------------------

def detect_blocks_in_frame(hsv, y_min, y_max):
    """Detect colored blocks in the falling region of a single frame.

    Returns list of (x_center, width, height, y_bottom, color) tuples.
    """
    region = hsv[y_min:y_max, :, :]
    blocks = []

    for color_name, mask_fn in [('blue', blue_mask), ('red', red_mask)]:
        mask = mask_fn(region).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if 8 < bw < 55 and bh > 3 and area > 40:
                cx = x + bw / 2.0
                y_bot = y + bh + y_min
                blocks.append((cx, bw, bh, y_bot, color_name))

    return blocks


# ---------------------------------------------------------------------------
# Step 1-2: Collect and cluster block x-positions
# ---------------------------------------------------------------------------

def collect_and_cluster_blocks(cap, y_min, y_max, sample_step=2):
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
        for (cx, bw, bh, y_bot, color) in detect_blocks_in_frame(hsv, y_min, y_max):
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

    # Merge close peaks
    merged = []
    for px, ps in sorted(peaks):
        if merged and px - merged[-1][0] < 12:
            if ps > merged[-1][1]:
                merged[-1] = (px, ps)
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

def fit_white_key_grid(key_positions, width_threshold=25):
    """Fit a regular linear grid to white-key positions (wider blocks).

    Returns (spacing, offset, n_white_keys, black_positions).
    black_positions is list of (x_center,) for black keys.
    """
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

def identify_c_position(spacing, offset, n_white, black_positions):
    """Determine which grid index is C by matching black keys to piano layout."""
    bk_offsets = [0.5, 1.5, 3.5, 4.5, 5.5]  # C# D# F# G# A# in WKW from C

    best_score, best_avg, best_c = -1, 999, 1
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
        if matches > best_score or (matches == best_score and avg < best_avg):
            best_score, best_avg, best_c = matches, avg, c_idx

    print(f"  C at grid index {best_c} ({best_score}/{len(black_positions)} black keys matched)")
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
    """Find the y-coordinate of the keyboard boundary (dark divider line)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample a few frames and find the consistent dark horizontal line
    for fi in [200, 400, 600]:
        if fi >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Scan from middle down, looking for a fully dark row
        for y in range(h // 2, h - 10):
            row = hsv[y, :]
            dark = int(np.sum(row[:, 2] < 30))
            if dark > hsv.shape[1] * 0.9:
                print(f"  Keyboard boundary at y={y}")
                return y

    # Fallback
    y_default = int(h * 0.62)
    print(f"  Keyboard boundary (default): y={y_default}")
    return y_default


def measure_fall_speed(cap, y_min, y_max):
    """Measure fall speed in pixels/frame by tracking blocks across frame pairs."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    speeds = []

    for fi in [200, 300, 400, 500, 600, 700, 800]:
        if fi >= total - 20:
            continue
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
                    if 15 < dy < 50:
                        speeds.append(dy / 10.0)

    speed = float(np.median(speeds)) if speeds else 2.7
    print(f"  Fall speed: {speed:.2f} px/frame")
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


def extract_notes_contour(cap, note_map, y_min, keyboard_y, fall_speed,
                          merge_tolerance=0.2, min_duration=0.05,
                          sample_step=3):
    """Extract notes by detecting block contours and projecting onset/duration.

    Each block in each frame independently provides:
      - pitch (from x-position)
      - onset_time = frame_time + (keyboard_y - block_bottom) / speed
      - duration = block_height / speed
      - hand (from color)

    Multiple frames observing the same block produce multiple observations
    with nearly identical onset times. These are merged (median onset,
    median duration) to produce the final note list.

    This is like NMS in object detection: each frame is an independent
    detector, and we merge overlapping detections in (pitch, time) space.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    px_per_sec = fall_speed * fps

    # Build x -> pitch mapping with boundaries
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

    hand_map = {'blue': 0, 'red': 1}

    # Phase 1: Collect observations from every Nth frame
    observations = []  # (pitch, hand, onset_sec, dur_sec)

    for fi in range(0, total, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blocks = detect_blocks_in_frame(hsv, y_min, keyboard_y)

        t_now = fi / fps
        for cx, bw, bh, y_bot, color in blocks:
            pitch = x_to_pitch(cx)
            hand = hand_map.get(color)
            onset_sec = t_now + max(0, keyboard_y - y_bot) / px_per_sec
            dur_sec = bh / px_per_sec
            observations.append((pitch, hand, onset_sec, dur_sec))

        if fi % 500 == 0 and fi > 0:
            print(f"    Frame {fi}/{total} ({100 * fi / total:.0f}%) "
                  f"— {len(observations)} observations")

    print(f"  {len(observations)} observations from {total // sample_step} frames")

    # Phase 2: Merge observations into notes
    # Sort by (pitch, hand, onset) then cluster overlapping onsets
    observations.sort(key=lambda o: (o[0], o[1] or 0, o[2]))

    notes = []
    i = 0
    while i < len(observations):
        pitch, hand, onset, dur = observations[i]
        group_onsets = [onset]
        group_durs = [dur]

        # Grow cluster: add next observations if they overlap in time
        j = i + 1
        cluster_end = onset + dur + merge_tolerance
        while j < len(observations):
            p2, h2, on2, dur2 = observations[j]
            if p2 != pitch or h2 != hand:
                break
            if on2 <= cluster_end:
                group_onsets.append(on2)
                group_durs.append(dur2)
                # Extend cluster end if this observation reaches further
                cluster_end = max(cluster_end, on2 + dur2 + merge_tolerance)
                j += 1
            else:
                break

        # Emit note: median onset, median duration (robust to outliers)
        med_onset = float(np.median(group_onsets))
        med_dur = float(np.median(group_durs))
        if med_dur >= min_duration:
            notes.append((pitch, hand, med_onset, med_onset + med_dur))

        i = j

    notes.sort(key=lambda n: n[2])

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
# Main pipeline
# ---------------------------------------------------------------------------

def detect_falling_notes_pipeline(video_path, output_path, base_octave=4):
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
    y_max = kb_y - 2  # blocks fall to just above the dark line
    print(f"  Block region: y={y_min}..{y_max}")

    # Step 2: Collect and cluster block positions
    print(f"\n--- Step 2: Collecting & clustering blocks ---")
    key_positions = collect_and_cluster_blocks(cap, y_min, y_max)
    print(f"  {len(key_positions)} discrete keys:")
    for x, n, mw in key_positions:
        print(f"    x={x:7.1f}  n={n:4d}  w={mw:5.1f}  "
              f"{'BLACK' if mw < 25 else 'white'}")

    # Step 3: Fit white key grid
    print(f"\n--- Step 3: Fitting white key grid ---")
    spacing, offset, n_white, blacks = fit_white_key_grid(key_positions)

    # Step 4: Identify C
    print(f"\n--- Step 4: Identifying C position ---")
    c_idx = identify_c_position(spacing, offset, n_white, blacks)

    # Step 5: Build pitch map
    print(f"\n--- Step 5: Building pitch map ---")
    note_map = build_pitch_map(spacing, offset, n_white, c_idx, blacks,
                               base_octave=base_octave)
    for midi in sorted(note_map.keys()):
        print(f"    {midi_to_name(midi):5s} (MIDI {midi:3d}) -> x={note_map[midi]:7.1f}")

    # Step 6: Measure fall speed
    print(f"\n--- Step 6: Measuring fall speed ---")
    fall_speed = measure_fall_speed(cap, y_min, kb_y)

    # Step 7: Extract notes via contour detection + cross-frame merge
    print(f"\n--- Step 7: Extracting notes (contour) ---")
    notes = extract_notes_contour(cap, note_map, y_min, kb_y, fall_speed)

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

    # Write MIDI
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
