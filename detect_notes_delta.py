#!/usr/bin/env python3
"""
detect_notes_delta.py — Prototype: detect notes via frame-to-frame saturation delta.

Instead of checking absolute color ("is this pixel green/blue?"), this checks
whether the average saturation in each key's detector region CHANGED suddenly
from the previous frame.  A sudden saturation increase = note on, a sudden
decrease = note off.  This is more robust against falling note bars (which
move gradually) vs actual key presses (which cause a sudden color change at
the key face).

Usage:
    python detect_notes_delta.py /path/to/video.mp4
"""

import argparse
import sys

import cv2
import numpy as np

# Reuse keyboard detection and note mapping from the main module
from pianovideoscribe import detect_keyboard, build_note_x_map

# MIDI note number -> note name
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_name(midi_num):
    octave = (midi_num // 12) - 1
    name = NOTE_NAMES[midi_num % 12]
    return f"{name}{octave}"


def build_detector_regions(note_x_map, white_keys, y_white):
    """Compute detector bounds for each key: (x_left, x_right, y_top, y_bot).

    White key detector: y_white-30 to y_white+5, x +/- (avg_white_width * 0.25)
    Black key detector: y_white-80 to y_white-40, x +/- 3
    """
    BLACK_SEMITONES = {1, 3, 6, 8, 10}
    avg_white_w = np.mean(np.diff(white_keys)) if len(white_keys) > 1 else 30

    regions = {}
    for pitch, x_center in note_x_map.items():
        if pitch % 12 in BLACK_SEMITONES:
            hw = 3
            y_top = y_white - 80
            y_bot = y_white - 40
        else:
            hw = int(avg_white_w * 0.25)
            y_top = y_white - 30
            y_bot = y_white + 5
        regions[pitch] = (x_center - hw, x_center + hw, y_top, y_bot)
    return regions


def region_avg_saturation(hsv_frame, x_left, x_right, y_top, y_bot):
    """Return the average saturation in a detector region."""
    fh, fw = hsv_frame.shape[:2]
    x1, x2 = max(0, x_left), min(fw, x_right)
    y1, y2 = max(0, y_top), min(fh, y_bot)
    if y1 >= y2 or x1 >= x2:
        return 0.0
    region = hsv_frame[y1:y2, x1:x2, 1]  # saturation channel
    return float(np.mean(region))


def region_avg_hue(hsv_frame, x_left, x_right, y_top, y_bot, s_min=50):
    """Return the average hue of saturated pixels in a detector region.

    Only considers pixels with S > s_min to get the actual colored hue.
    Returns None if not enough saturated pixels.
    """
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


def classify_hand_from_hue(hue):
    """Classify hand from average hue: green (40-65) = right(0), blue (85-125) = left(1)."""
    if hue is None:
        return None
    if 40 <= hue <= 65:
        return 0  # green = right
    if 85 <= hue <= 125:
        return 1  # blue = left
    return None


def main():
    parser = argparse.ArgumentParser(description='Detect notes via saturation delta')
    parser.add_argument('video', help='Path to Synthesia MP4 video')
    parser.add_argument('--frame', type=int, default=250,
                        help='Frame index for keyboard detection (default: 250)')
    parser.add_argument('--threshold-on', type=float, default=30,
                        help='Saturation increase threshold for note-on (default: 30)')
    parser.add_argument('--threshold-off', type=float, default=30,
                        help='Saturation decrease threshold for note-off (default: 30)')
    parser.add_argument('--min-sat', type=float, default=80,
                        help='Minimum absolute saturation to confirm note-on (default: 80)')
    parser.add_argument('--max-notes', type=int, default=20,
                        help='Number of note onsets to print (default: 20)')
    parser.add_argument('--debug-pitch', type=int, default=None,
                        help='Print per-frame saturation trace for this MIDI pitch')
    parser.add_argument('--debug-time', type=str, default=None,
                        help='Time range for debug output, e.g. "8-16" (seconds)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.2f} fps, {total_frames} frames, "
          f"duration {total_frames/fps:.1f}s\n")

    # Step 1: Detect keyboard
    print("--- Detect keyboard ---")
    white_keys, black_keys, y_white = detect_keyboard(cap, frame_idx=args.frame)

    # Step 2: Build note -> x map (assume full keyboard, min_midi=21)
    print("\n--- Build note map ---")
    note_x_map = build_note_x_map(white_keys, black_keys, min_midi_note=21)

    # Step 3: Build detector regions
    regions = build_detector_regions(note_x_map, white_keys, y_white)
    pitches = sorted(note_x_map.keys())
    print(f"\nDetector regions built for {len(pitches)} pitches")
    print(f"y_white={y_white}")
    print(f"Thresholds: on_delta={args.threshold_on}, off_delta={args.threshold_off}, "
          f"min_sat={args.min_sat}")

    # Debug setup
    debug_pitch = args.debug_pitch
    debug_t0, debug_t1 = 0, 9999
    if args.debug_time:
        parts = args.debug_time.split('-')
        debug_t0, debug_t1 = float(parts[0]), float(parts[1])

    # Step 4: Scan frames, compute saturation delta
    print(f"\n--- Scanning frames ---")

    # Previous frame's average saturation per key
    prev_sat = {p: 0.0 for p in pitches}
    # Active notes: pitch -> (hand, onset_frame)
    active = {}
    notes = []  # (pitch, hand, onset_sec, offset_sec)

    for f_idx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = f_idx / fps
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for pitch in pitches:
            x_left, x_right, y_top, y_bot = regions[pitch]
            sat = region_avg_saturation(hsv, x_left, x_right, y_top, y_bot)
            delta = sat - prev_sat[pitch]
            prev_sat[pitch] = sat

            # Debug trace
            if debug_pitch == pitch and debug_t0 <= t_sec <= debug_t1:
                state = "ON " if pitch in active else "off"
                if abs(delta) > 5:
                    hue = region_avg_hue(hsv, x_left, x_right, y_top, y_bot)
                    print(f"  [{state}] t={t_sec:6.2f}s  sat={sat:5.1f}  "
                          f"delta={delta:+6.1f}  hue={hue}  "
                          f"pitch={midi_to_name(pitch)}")

            if pitch not in active and delta > args.threshold_on and sat > args.min_sat:
                # Note on — require both a big delta AND high absolute saturation.
                # This filters out small blips where delta is large but sat is low.
                hue = region_avg_hue(hsv, x_left, x_right, y_top, y_bot)
                hand = classify_hand_from_hue(hue)
                active[pitch] = (hand, f_idx)

            elif pitch in active and delta < -args.threshold_off:
                # Note off
                hand, onset_frame = active.pop(pitch)
                onset_sec = onset_frame / fps
                offset_sec = t_sec
                if offset_sec - onset_sec > 0.02:
                    notes.append((pitch, hand, onset_sec, offset_sec))

        if f_idx % 500 == 0 and f_idx > 0:
            print(f"  Frame {f_idx}/{total_frames} "
                  f"({100*f_idx/total_frames:.0f}%) — "
                  f"{len(notes)} notes, {len(active)} active")

    # Close remaining active notes
    for pitch, (hand, onset_frame) in active.items():
        onset_sec = onset_frame / fps
        offset_sec = (total_frames - 1) / fps
        if offset_sec - onset_sec > 0.02:
            notes.append((pitch, hand, onset_sec, offset_sec))

    notes.sort(key=lambda n: n[2])

    # Print results
    print(f"\n=== Results: {len(notes)} total notes detected ===\n")
    hand_names = {0: 'R', 1: 'L', None: '?'}
    print(f"{'#':>3}  {'Time':>7}  {'Note':<5}  {'Hand':>4}  {'Dur':>6}")
    print(f"{'---':>3}  {'-------':>7}  {'-----':<5}  {'----':>4}  {'------':>6}")
    for i, (pitch, hand, onset, offset) in enumerate(notes[:args.max_notes]):
        dur = offset - onset
        print(f"{i+1:>3}  {onset:>7.2f}s  {midi_to_name(pitch):<5}  "
              f"{hand_names.get(hand, '?'):>4}  {dur:>5.02f}s")

    if len(notes) > args.max_notes:
        print(f"  ... ({len(notes) - args.max_notes} more notes)")

    # Summary
    right_count = sum(1 for _, h, _, _ in notes if h == 0)
    left_count = sum(1 for _, h, _, _ in notes if h == 1)
    unknown_count = sum(1 for _, h, _, _ in notes if h is None)
    print(f"\nSummary: R={right_count}, L={left_count}, ?={unknown_count}")

    cap.release()


if __name__ == '__main__':
    main()
