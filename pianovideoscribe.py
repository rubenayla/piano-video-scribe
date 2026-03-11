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
    return white_keys, black_keys, y_white


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


def classify_hand(h, s, v, green_is_right=True, colors=None):
    """Classify a color sample as right hand (0), left hand (1), or unknown (None).

    Args:
        h, s, v: HSV values (OpenCV scale: H in [0,179]).
        green_is_right: if True, green = right (0), blue = left (1). Flip if needed.
        colors: dict with 'green' and 'blue' keys, each containing
                h_min, h_max, s_min, v_min thresholds. Uses defaults if None.

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
    p.add_argument('midi', help='Path to input MIDI (e.g. from piano_transcription_inference)')
    p.add_argument('output', help='Path to output MIDI')
    p.add_argument('--bpm', type=int, required=True,
                   help='Actual BPM of the song (required)')
    p.add_argument('--frame', type=int, default=5,
                   help='Frame index for keyboard detection (default: 5, should be note-free)')
    p.add_argument('--green-hand', choices=['right', 'left'], default='right',
                   help='Which hand is shown in green in the video (default: right)')
    hand_choices = ['normal', 'no-overlap', 'monophonic']
    p.add_argument('--right-hand', choices=hand_choices, default='no-overlap',
                   help='Right hand processing: normal, '
                        'no-overlap (default, cut held notes at next onset, keep chords), '
                        'monophonic (single voice, highest note only)')
    p.add_argument('--left-hand', choices=hand_choices, default='no-overlap',
                   help='Left hand processing: normal, '
                        'no-overlap (default, cut held notes at next onset, keep chords), '
                        'monophonic (single voice, lowest note only)')
    p.add_argument('--config', type=str, default=None,
                   help='Path to a JSON config file (colors, sampling zone, keyboard frame). '
                        'See configs/ directory for examples.')
    p.add_argument('--dry-run', action='store_true',
                   help='Detect keyboard and print stats only — do not write output MIDI')
    return p.parse_args()


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
    SIXTEENTH = OUT_TPB // 4  # ticks per 16th note
    THIRTYSECOND = OUT_TPB // 8  # ticks per 32nd note
    green_is_right = (args.green_hand == 'right')

    # Config overrides for --frame (CLI takes precedence)
    frame_idx = args.frame if args.frame != 5 else cfg['keyboard'].get('frame', 5)

    print("=== pianovideoscribe ===\n")
    print(f"Video:       {args.video}")
    print(f"Input MIDI:  {args.midi}")
    print(f"Output MIDI: {args.output}")
    print(f"BPM:         {OUT_BPM}")
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
    white_keys, black_keys, y_white = detect_keyboard(cap, frame_idx=frame_idx)

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

    # Color sample zone: sample the lit key faces directly.
    # Keys light up green/blue when pressed — more reliable than falling bars.
    # Position relative to white key scan row (y_white), slightly above it.
    y_sample_top = y_white - cfg['sampling']['y_offset_top']
    y_sample_bot = y_white - cfg['sampling']['y_offset_bot']
    half_w = cfg['sampling']['half_w']

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
    tick_offset = quantize_tick(int(seconds_to_out_ticks(first_note_sec)), THIRTYSECOND)
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
                out_tick = quantize_tick(int(seconds_to_out_ticks(t_sec)), THIRTYSECOND) - tick_offset
                ev = (max(0, out_tick), 'note_on', msg.note, msg.velocity)
                (right_events if hand == 0 else left_events).append(ev)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    hand = active.pop(msg.note)
                    t_sec = ticks_to_seconds(abs_tick)
                    out_tick = quantize_tick(int(seconds_to_out_ticks(t_sec)), THIRTYSECOND) - tick_offset
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
