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

from piano_video_scribe.keyboard import (detect_keyboard, build_note_x_map,
    build_detector_regions, regularize_positions, find_first_c)
from piano_video_scribe.color import (sample_color, sample_color_avg, classify_hand,
    fallback_hand, _region_avg_saturation, _region_avg_hue, _classify_hand_from_hue)
from piano_video_scribe.quantization import (make_tick_converters, quantize_tick,
    quantize_tick_smart, quantize_onsets_pll, quantize_onsets_viterbi,
    estimate_local_bpm, warp_onsets, quantize_onsets_adaptive)
from piano_video_scribe.midi_output import remove_overlaps, make_monophonic, build_track
from piano_video_scribe.detectors.keys import extract_notes_from_video
from piano_video_scribe.visualization import generate_summary_image


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
                   help='Key signature for MIDI metadata (e.g. "E", "Em"). '
                        'Auto-detected if omitted.')
    p.add_argument('--config', type=str, default=None,
                   help='Path to a JSON config file (colors, sampling zone, keyboard frame). '
                        'See configs/ directory for examples.')
    p.add_argument('--start-time', type=float, default=None,
                   help='Manual override: music start time in seconds (skips auto-detection)')
    p.add_argument('--end-time', type=float, default=None,
                   help='Manual override: music end time in seconds (skips auto-detection)')
    p.add_argument('--start-beat', type=float, default=None,
                   help='Beat position of the first note (default: auto-detect). '
                        'Use 1.5 for pickup on the "and" of beat 1, '
                        '4.5 for pickup on the "and" of beat 4, etc.')
    p.add_argument('--settings', type=str, default=None,
                   help='Path to a settings.json file with saved pipeline parameters. '
                        'CLI flags override values from the file.')
    p.add_argument('--time-sig', type=str, default=None, dest='time_sig',
                   help='Time signature (e.g. "6/8", "3/4"). Default: 4/4')
    p.add_argument('--transpose', type=int, default=0,
                   help='Transpose all notes by N semitones (e.g. -12 for one octave down)')
    p.add_argument('--triplet', action='store_true',
                   help='Use combined grid (12 per beat) that supports both straight 16ths '
                        'and triplet positions. Each note snaps to the nearest valid position.')
    p.add_argument('--detector', choices=['keys', 'falling-blocks'], default=None,
                   help='Note detection method: falling-blocks (default, detects notes from '
                        'falling colored blocks) or keys (key-lighting saturation fallback)')
    p.add_argument('--dry-run', action='store_true',
                   help='Detect keyboard and print stats only — do not write output MIDI')

    args = p.parse_args()

    # --- Merge settings file (if provided) — fill None values from JSON ---
    if args.settings:
        with open(args.settings) as f:
            settings = json.load(f)
        SETTINGS_KEYS = ['bpm', 'key', 'green_hand', 'frame', 'right_hand',
                         'left_hand', 'start_time', 'end_time', 'start_beat', 'config',
                         'time_sig', 'transpose', 'detector']
        for key in SETTINGS_KEYS:
            if key in settings:
                # Only override if the CLI didn't explicitly set it
                cli_val = getattr(args, key, None)
                if cli_val is None:
                    setattr(args, key, settings[key])

    # --- Apply hardcoded defaults for anything still None ---
    HARDCODED_DEFAULTS = {
        'frame': 5,
        'green_hand': 'right',
        'right_hand': 'no-overlap',
        'left_hand': 'no-overlap',
        'detector': 'falling-blocks',
    }
    for key, default in HARDCODED_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, default)

    # BPM: auto-detect from video audio if not provided
    args._had_explicit_bpm = args.bpm is not None
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

    # Config overrides for --frame (CLI takes precedence, None = auto-detect)
    frame_idx = args.frame if args.frame is not None else cfg['keyboard'].get('frame', None)

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
    white_keys, black_keys, y_white, y_black, y_kb_top, y_bk_bottom = detect_keyboard(cap, frame_idx=frame_idx)

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
    if args.midi is None and args.detector == 'falling-blocks':
        # FALLING-BLOCKS MODE: detect notes from colored blocks above keyboard
        print("\n--- Step 3: Extract notes from falling blocks ---")
        cap.release()
        from piano_video_scribe.detectors.falling_blocks import detect_falling_notes_pipeline
        video_notes = detect_falling_notes_pipeline(args.video, None)
        video_notes.sort(key=lambda n: n[2])  # sort by onset time

        if args.end_time is not None:
            video_notes = [n for n in video_notes if n[2] <= args.end_time]
        if args.start_time is not None:
            video_notes = [n for n in video_notes if n[2] >= args.start_time]

        right_count = sum(1 for _, h, _, _ in video_notes if h == 0)
        left_count = sum(1 for _, h, _, _ in video_notes if h == 1)
        print(f"Extracted {len(video_notes)} notes: right={right_count}, left={left_count}")

        if not video_notes:
            print("ERROR: No notes detected in video.", file=sys.stderr)
            sys.exit(1)

    elif args.midi is None:
        # VIDEO-ONLY MODE (key-press): scan frames for key state changes
        print("\n--- Step 3: Extract notes from video ---")
        video_notes = extract_notes_from_video(
            cap, note_x_map, y_sample_top, y_sample_bot,
            half_w, fps, total_frames, green_is_right, cfg['colors'],
            start_frame=frame_idx, frame_step=1, white_keys=white_keys, cfg=cfg,
            y_black=y_black, start_time=args.start_time, y_kb_top=y_kb_top,
            y_bk_bottom=y_bk_bottom)

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

        right_count = sum(1 for _, h, _, _ in video_notes if h == 0)
        left_count = sum(1 for _, h, _, _ in video_notes if h == 1)
        print(f"Extracted {len(video_notes)} notes: right={right_count}, left={left_count}")

        # Generate summary image alongside the output MIDI
        det_regions = build_detector_regions(note_x_map, white_keys, y_white,
                                             cfg=cfg, y_black=y_black, y_kb_top=y_kb_top,
                                             y_bk_bottom=y_bk_bottom)
        summary_path = os.path.splitext(args.output)[0] + '-summary.png'
        generate_summary_image(cap, frame_idx, white_keys, black_keys,
                               y_white, y_black, y_kb_top, note_x_map,
                               det_regions, OUT_BPM, args.key, green_is_right,
                               right_count, left_count, summary_path)

        cap.release()

        if not video_notes:
            print("ERROR: No notes detected in video.", file=sys.stderr)
            sys.exit(1)

    # --- Shared quantization for all video-based detectors ---
    if args.midi is None:
        # Refine BPM using detected note onsets: test candidates near the
        # initial estimate and pick the one with least quantization error.
        if not args._had_explicit_bpm:
            onsets = sorted(set(n[2] for n in video_notes))
            if len(onsets) >= 10:
                intervals = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)
                              if 0.05 < onsets[i+1] - onsets[i] < 2.0]
                if intervals:
                    # Test BPM candidates: original ± 15%, in steps of 1
                    best_bpm, best_err = OUT_BPM, float('inf')
                    for candidate in range(max(40, OUT_BPM - 15), min(220, OUT_BPM + 16)):
                        s16 = 60.0 / candidate / 4
                        # Quantization error: how well do intervals snap to 16th-note multiples?
                        err = sum(abs(iv / s16 - round(iv / s16)) for iv in intervals[:50])
                        if err < best_err:
                            best_err = err
                            best_bpm = candidate
                    if best_bpm != OUT_BPM:
                        print(f"  BPM refined: {OUT_BPM} → {best_bpm} "
                              f"(onset-based, {len(intervals)} intervals)")
                        OUT_BPM = best_bpm

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

        # Quantize each hand with its own quantizer and grid
        # start_beat shifts onsets so the first note lands on the right beat.
        # Each hand uses its own first onset as reference, and start_beat
        # only applies to the RH (melody hand) — LH follows from there.
        # Compute beat_pre_offset so the RH's first note lands on start_beat.
        # The RH first onset relative to first_onset may be > 0 if LH starts earlier.
        rh_first = video_notes[right_indices[0]][2] if right_indices else first_onset
        rh_rel = rh_first - first_onset  # RH's offset from the global first note
        effective_start_beat = args.start_beat if args.start_beat is not None else 1.0
        beat_target_sec = (effective_start_beat - 1.0) * (60.0 / OUT_BPM)
        beat_pre_offset = beat_target_sec - rh_rel
        if abs(beat_pre_offset) > 0.001:
            print(f"  start_beat={effective_start_beat}, pre-offset={beat_pre_offset:.4f}s")

        def quantize_hand(indices, subdivisions, target_start_grid=0):
            """Quantize note onsets for one hand.

            target_start_grid: the grid position where the first note should
            land.  The Viterbi quantizer always starts at 0, so we offset all
            positions afterward.
            """
            onsets = [video_notes[i][2] - first_onset for i in indices]
            offsets = [video_notes[i][3] - first_onset for i in indices]
            s = 60.0 / OUT_BPM / subdivisions  # grid unit duration

            # Group chord notes: notes within 50ms share the same onset
            # for quantization purposes. The Viterbi quantizer requires
            # min step=1 between consecutive entries, so chord notes sent
            # individually would be spread to consecutive 16th positions.
            CHORD_THRESH = 0.050  # seconds
            groups = []  # [(representative_onset, [member_indices])]
            for idx, onset in enumerate(onsets):
                if groups and onset - onsets[groups[-1][1][0]] <= CHORD_THRESH:
                    groups[-1][1].append(idx)
                else:
                    groups.append((onset, [idx]))

            # Quantize representative onsets only.
            # Shift to start at t=0 so Viterbi's absolute-position cost
            # doesn't distort intervals when a hand enters late.
            rep_onsets = [g[0] for g in groups]
            hand_t0 = rep_onsets[0]
            rep_onsets_shifted = [t - hand_t0 for t in rep_onsets]
            if subdivisions == 4:
                rep_grid = quantize_onsets_adaptive(rep_onsets_shifted, OUT_BPM)
            else:
                rep_grid = quantize_onsets_pll(rep_onsets_shifted, OUT_BPM, alpha=0.1, subdivisions=subdivisions)

            # Offset so the first note lands at target_start_grid
            grid_offset = target_start_grid - rep_grid[0]
            if grid_offset != 0:
                rep_grid = [p + grid_offset for p in rep_grid]

            # Expand back: all members of a chord group share the same grid position
            on_g = [0] * len(onsets)
            for gi, (_, members) in enumerate(groups):
                for mi in members:
                    on_g[mi] = rep_grid[gi]

            # Derive offset grid from onset grid + quantized duration
            off_g = []
            for i_idx, on_pos in enumerate(on_g):
                raw_dur = offsets[i_idx] - onsets[i_idx]
                dur_grid = max(1, round(raw_dur / s))
                off_g.append(on_pos + dur_grid)
            return on_g, off_g

        # Compute target grid position for each hand's first note.
        # Grid units = subdivisions per beat.
        if args.start_beat is not None:
            # Manual override: RH lands on the specified beat, LH at beat 1
            rh_target = int(round((args.start_beat - 1.0) * RIGHT_SUB))
            lh_target = 0
        else:
            # Auto-detect: compute offset from actual time gap between hands
            rh_first_sec = video_notes[right_indices[0]][2] if right_indices else first_onset
            lh_first_sec = video_notes[left_indices[0]][2] if left_indices else first_onset
            gap_sec = rh_first_sec - lh_first_sec  # positive = RH enters later
            beat_dur = 60.0 / OUT_BPM
            rh_gap_grid = gap_sec / (beat_dur / RIGHT_SUB)
            lh_gap_grid = gap_sec / (beat_dur / LEFT_SUB)
            rh_target = max(0, int(round(rh_gap_grid)))
            lh_target = max(0, int(round(-lh_gap_grid)))
            if rh_target > 0 or lh_target > 0:
                print(f"  Auto-detected hand offset: RH grid={rh_target}, LH grid={lh_target} "
                      f"(gap={gap_sec:.3f}s)")

        r_on_grid, r_off_grid = quantize_hand(right_indices, RIGHT_SUB, rh_target)
        l_on_grid, l_off_grid = quantize_hand(left_indices, LEFT_SUB, lh_target)

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

    # --- Transpose ---
    if args.transpose != 0:
        right_events = [(t, ty, p + args.transpose, v) for t, ty, p, v in right_events]
        left_events = [(t, ty, p + args.transpose, v) for t, ty, p, v in left_events]
        print(f"Transposed all notes by {args.transpose:+d} semitones")

    # --- Key signature: use explicit --key or auto-detect for metadata ---
    key_sig = args.key
    if not key_sig:
        # Auto-detect key for MIDI metadata (informational only)
        tmp_mid = MidiFile(type=1, ticks_per_beat=OUT_TPB)
        tmp_mid.tracks.append(build_track(right_events, 'Right Hand', OUT_US_PER_BEAT))
        tmp_mid.tracks.append(build_track(left_events, 'Left Hand', OUT_US_PER_BEAT))
        tmp_mid.save(args.output)
        try:
            from music21 import converter as m21_converter
            score = m21_converter.parse(args.output)
        except Exception:
            score = None
        if score is not None:
            detected = score.analyze('key')
            tonic = detected.tonic.name.replace('-', 'b')
            if detected.mode == 'minor':
                key_sig = tonic + 'm'
            else:
                key_sig = tonic
            print(f"Key auto-detected: {detected} (confidence={detected.correlationCoefficient:.2f})")

    out_mid = MidiFile(type=1, ticks_per_beat=OUT_TPB)
    out_mid.tracks.append(build_track(right_events, 'Right Hand', OUT_US_PER_BEAT))
    out_mid.tracks.append(build_track(left_events, 'Left Hand', OUT_US_PER_BEAT))

    if key_sig:
        out_mid.tracks[0].insert(0, MetaMessage('key_signature', key=key_sig, time=0))

    if args.time_sig:
        num, den = map(int, args.time_sig.split('/'))
        out_mid.tracks[0].insert(0, MetaMessage('time_signature', numerator=num, denominator=den, time=0))

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
                 'left_hand', 'start_time', 'end_time', 'start_beat', 'config',
                 'time_sig', 'transpose', 'detector']
    for key in SAVE_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            # Skip defaults that don't need saving
            if key == 'start_beat' and val is None:
                continue
            if key == 'detector' and val == 'falling-blocks':
                continue
            settings_to_save[key] = val
    settings_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), 'settings.json')
    with open(settings_path, 'w') as f:
        json.dump(settings_to_save, f, indent=2)
        f.write('\n')
    print(f"Settings saved: {settings_path}")


if __name__ == '__main__':
    main()
