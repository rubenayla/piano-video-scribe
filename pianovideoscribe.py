#!/usr/bin/env python3
"""
pianovideoscribe — Separate a Synthesia piano video into right/left hand MIDI tracks.

Uses the video's color coding (green = right hand, blue = left hand) to assign
each note from an input MIDI file to one of two output tracks.

Typical usage:
    python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120
"""

import argparse
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

    # Auto-find y for white keys: scan bottom-up for row with many white, few dark pixels
    y_white = h - 30
    for y_scan in range(h - 5, h // 2, -1):
        row_test = hsv[y_scan, :]
        white_count = int(np.sum((row_test[:, 1] < 60) & (row_test[:, 2] > 180)))
        dark_count = int(np.sum(row_test[:, 2] < 80))
        if white_count > 600 and dark_count < 100:
            y_white = y_scan
            break
    print(f"White key scan y={y_white}")

    # Detect white key x-centers from that row
    row_white = hsv[y_white, :]
    raw_white = []
    in_key = False
    start_x = 0
    for x in range(w):
        is_white = int(row_white[x, 1]) < 60 and int(row_white[x, 2]) > 180
        if is_white and not in_key:
            in_key = True
            start_x = x
        elif not is_white and in_key:
            if x - start_x > 8:
                raw_white.append((start_x + x) // 2)
            in_key = False
    if in_key and w - start_x > 8:
        raw_white.append((start_x + w) // 2)

    # Regularize: apply linear regression to remove outliers (some keys may be
    # mis-detected when notes are partially visible on them)
    white_keys = regularize_positions(raw_white)
    print(f"Detected {len(raw_white)} raw white keys → {len(white_keys)} regularized")

    # Auto-find y for black keys: just above white zone, has many dark pixels
    y_black = y_white - 50
    for y_scan in range(y_white - 5, y_white - 80, -1):
        row_test = hsv[y_scan, :]
        dark_count = int(np.sum(row_test[:, 2] < 80))
        white_count = int(np.sum((row_test[:, 1] < 60) & (row_test[:, 2] > 180)))
        if 100 < dark_count < 700 and white_count > 200:
            y_black = y_scan
            break
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
    return white_keys, black_keys


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

    # Map each black key to the white key interval (left neighbor index)
    bk_intervals = []
    for bx in black_keys:
        left_idx = None
        for i, wx in enumerate(white_keys):
            if wx <= bx:
                left_idx = i
            else:
                break
        if left_idx is not None:
            bk_intervals.append(left_idx)

    if not bk_intervals:
        return 2

    # Group consecutive intervals
    groups = []
    group = [bk_intervals[0]]
    for prev, curr in zip(bk_intervals, bk_intervals[1:]):
        if curr - prev == 1:
            group.append(curr)
        else:
            groups.append(group)
            group = [curr]
    groups.append(group)

    print(f"Black key groups (sizes): {[len(g) for g in groups]}")

    # First group of size 2 → C#, D# → C is at group[0][0]
    for group in groups:
        if len(group) == 2:
            c_idx = group[0]
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

    # Determine C octave from MIDI range.
    # A typical partial keyboard starts at A (3 semitones below C).
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

    # Assign black keys as midpoint of flanking white keys
    for wm in list(note_x_map.keys()):
        semi = wm % 12
        if semi in [0, 2, 5, 7, 9]:  # C, D, F, G, A have a sharp to the right
            black_midi = wm + 1
            next_white = wm + 2
            if next_white in note_x_map:
                note_x_map[black_midi] = (note_x_map[wm] + note_x_map[next_white]) // 2

    print(f"Built note→x map for {len(note_x_map)} MIDI notes "
          f"(range {min(note_x_map)}–{max(note_x_map)})")
    return note_x_map


# ---------------------------------------------------------------------------
# Color sampling & hand classification
# ---------------------------------------------------------------------------

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


def classify_hand(h, s, v, green_is_right=True):
    """Classify a color sample as right hand (0), left hand (1), or unknown (None).

    Synthesia default colors:
        Green  H 40–65,  S > 100, V > 80  → one hand
        Blue   H 100–120, S > 80,  V > 80  → other hand

    Args:
        h, s, v: HSV values (OpenCV scale: H in [0,179]).
        green_is_right: if True, green = right (0), blue = left (1). Flip if needed.

    Returns:
        0 (right), 1 (left), or None.
    """
    if h is None or s is None:
        return None
    is_green = 40 <= h <= 65 and s > 100 and v > 80
    is_blue = 100 <= h <= 120 and s > 80 and v > 80
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
    if not recent_right and not recent_left:
        return 0 if pitch >= 60 else 1
    if not recent_right:
        return 1
    if not recent_left:
        return 0
    d_right = abs(pitch - np.mean(list(recent_right)))
    d_left = abs(pitch - np.mean(list(recent_left)))
    return 0 if d_right <= d_left else 1


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
    return round(tick / grid) * grid


# ---------------------------------------------------------------------------
# MIDI building
# ---------------------------------------------------------------------------

def build_track(events, name, tempo_us):
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
            track.append(Message('note_on', channel=0, note=pitch, velocity=vel, time=delta))
        else:
            track.append(Message('note_off', channel=0, note=pitch, velocity=0, time=delta))
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
""")
    p.add_argument('video', help='Path to Synthesia MP4 video')
    p.add_argument('midi', help='Path to input MIDI (e.g. from piano_transcription_inference)')
    p.add_argument('output', help='Path to output MIDI')
    p.add_argument('--bpm', type=int, required=True,
                   help='Actual BPM of the song (required)')
    p.add_argument('--frame', type=int, default=5,
                   help='Frame index for keyboard detection (default: 5, should be note-free)')
    p.add_argument('--green-hand', choices=['right', 'left'], default='right',
                   help='Which hand is shown in green in the video (default: right)')
    p.add_argument('--dry-run', action='store_true',
                   help='Detect keyboard and print stats only — do not write output MIDI')
    return p.parse_args()


def main():
    args = parse_args()

    OUT_TPB = 960
    OUT_BPM = args.bpm
    OUT_US_PER_BEAT = int(60_000_000 / OUT_BPM)
    SIXTEENTH = OUT_TPB // 4  # ticks per 16th note
    green_is_right = (args.green_hand == 'right')

    print("=== pianovideoscribe ===\n")
    print(f"Video:       {args.video}")
    print(f"Input MIDI:  {args.midi}")
    print(f"Output MIDI: {args.output}")
    print(f"BPM:         {OUT_BPM}")
    print(f"Green hand:  {args.green_hand}")
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
    white_keys, black_keys = detect_keyboard(cap, frame_idx=args.frame)

    # --- Step 2: Build note → x map ---
    print("\n--- Step 2: Build note→x map ---")
    mid = MidiFile(args.midi)

    # Read original MIDI timing (piano_transcription default: 120 BPM, 384 TPB)
    orig_tpb = mid.ticks_per_beat
    orig_bpm = 120  # piano_transcription always outputs at 120 BPM
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

    min_note, max_note = min(all_pitches), max(all_pitches)
    print(f"MIDI pitch range: {min_note}–{max_note}  ({len(all_pitches)} notes)")
    print(f"Original MIDI: tpb={orig_tpb}, bpm={orig_bpm}")

    note_x_map = build_note_x_map(white_keys, black_keys, min_note)

    if args.dry_run:
        print("\n--- Dry run complete. Use these stats to calibrate. ---")
        print(f"White keys detected: {len(white_keys)}")
        print(f"Black keys detected: {len(black_keys)}")
        print(f"Notes in x-map: {len(note_x_map)}")
        unmapped = [p for p in set(all_pitches) if p not in note_x_map]
        if unmapped:
            print(f"Pitches NOT in x-map (will use fallback): {sorted(unmapped)}")
        cap.release()
        return

    # --- Step 3: Assign hands via video color ---
    print("\n--- Step 3: Assign hands via video color ---")
    ticks_to_seconds, seconds_to_out_ticks = make_tick_converters(
        orig_tpb, orig_bpm, OUT_TPB, OUT_BPM)

    # Collect all note-on events with absolute ticks
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

    # Color sample zone: covers falling note bar and lit key area
    # (y ≈ 480–600 for typical 1280×720 Synthesia video)
    y_sample_top = 480
    y_sample_bot = 600

    for abs_tick, pitch, vel in note_ons:
        t_sec = ticks_to_seconds(abs_tick)
        frame_idx = int(t_sec * fps)
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        hand = None
        if pitch in note_x_map:
            x_center = note_x_map[pitch]
            hv, sv, vv = sample_color(cap, frame_idx, x_center,
                                       y_sample_top, y_sample_bot, half_w=10)
            hand = classify_hand(hv, sv, vv, green_is_right)
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
    tick_offset = quantize_tick(int(seconds_to_out_ticks(first_note_sec)), SIXTEENTH)
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
                out_tick = quantize_tick(int(seconds_to_out_ticks(t_sec)), SIXTEENTH) - tick_offset
                ev = (max(0, out_tick), 'note_on', msg.note, msg.velocity)
                (right_events if hand == 0 else left_events).append(ev)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    hand = active.pop(msg.note)
                    t_sec = ticks_to_seconds(abs_tick)
                    out_tick = quantize_tick(int(seconds_to_out_ticks(t_sec)), SIXTEENTH) - tick_offset
                    ev = (max(0, out_tick), 'note_off', msg.note, 0)
                    (right_events if hand == 0 else left_events).append(ev)

    out_mid = MidiFile(type=1, ticks_per_beat=OUT_TPB)
    t0 = MidiTrack()
    t0.append(MetaMessage('set_tempo', tempo=OUT_US_PER_BEAT, time=0))
    t0.append(MetaMessage('end_of_track', time=0))
    out_mid.tracks.append(t0)
    out_mid.tracks.append(build_track(right_events, 'Right Hand', OUT_US_PER_BEAT))
    out_mid.tracks.append(build_track(left_events, 'Left Hand', OUT_US_PER_BEAT))

    out_mid.save(args.output)

    rn = sum(1 for e in right_events if e[1] == 'note_on')
    ln = sum(1 for e in left_events if e[1] == 'note_on')
    print(f"\nSaved: {args.output}")
    print(f"Right hand: {rn} notes")
    print(f"Left hand:  {ln} notes")
    print(f"Total:      {rn + ln} notes")
    print(f"\nDone! Open {args.output} in MuseScore.")
    print("Tip: Preferences → Import → MIDI → Shortest note: 16th")


if __name__ == '__main__':
    main()
