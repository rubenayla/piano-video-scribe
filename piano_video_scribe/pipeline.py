"""Main pipeline orchestration — ties all modules together."""

import os
import sys
from collections import deque

import cv2
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage

from piano_video_scribe.config import parse_args, detect_bpm_from_video, load_config
from piano_video_scribe.keyboard import (
    detect_keyboard, build_note_x_map, build_detector_regions,
)
from piano_video_scribe.color import (
    sample_color, classify_hand, fallback_hand,
)
from piano_video_scribe.quantization import (
    make_tick_converters, quantize_tick, quantize_tick_smart,
    quantize_onsets_pll, quantize_onsets_viterbi, quantize_onsets_adaptive,
    quantize_onsets_simple, build_shared_local_bpm_lookup,
    build_shared_warp_lookup, _refine_bpm_for_grid,
)
from piano_video_scribe.midi_output import (
    dedup_same_tick_pitch, remove_overlaps, make_monophonic, build_track,
)
from piano_video_scribe.visualization import generate_summary_image
from piano_video_scribe.detectors.keys import extract_notes_from_video
from piano_video_scribe.detectors.keys_refine import refine_notes_via_narrow_trace
from piano_video_scribe.detectors.keys_continuity import extract_notes_continuity


def main():
    args = parse_args()
    cfg = load_config(args.config, inline_config=getattr(args, '_inline_config', None))

    OUT_TPB = 960
    # BPM always means quarter note = N, regardless of time signature.
    # In 6/8, the user sets BPM to the quarter-note rate (e.g. 90),
    # and each eighth note is half a quarter = 0.333s at BPM=90.
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
    raw_cap = cv2.VideoCapture(args.video)
    if not raw_cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)
    src_h = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_h = getattr(args, 'max_height', 1080) or 0
    if max_h > 0 and src_h > max_h:
        from piano_video_scribe.video_io import ResizingCapture
        cap = ResizingCapture(raw_cap, max_h)
        print(f"Resizing frames: {src_w}x{src_h} -> "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"(scale={cap.scale:.3f})")
    else:
        cap = raw_cap
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
        fb_kwargs = {}
        for fb_key in ('glow_margin', 's_min', 's_min_per_color', 'sample_step'):
            val = getattr(args, fb_key)
            if val is not None:
                fb_kwargs[fb_key] = val
        # Pass --frame through so the falling-blocks pipeline's internal
        # keyboard detection uses the same clean frame as the keys path.
        if getattr(args, 'frame', None) is not None:
            fb_kwargs['frame_idx'] = args.frame
        fb_kwargs['max_height'] = getattr(args, 'max_height', 1080)
        video_notes = detect_falling_notes_pipeline(args.video, None, **fb_kwargs)
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

    elif args.midi is None and args.detector == 'keys-continuity':
        # CONTINUITY MODE: per-pitch saturation time-series + attack/release
        print("\n--- Step 3: Extract notes via continuity detector ---")
        cont_yt = y_kb_top + 5
        cont_yb = y_kb_top + max(10, int((y_bk_bottom - y_kb_top) * 0.30))
        cont_start = frame_idx
        if args.start_time is not None:
            cont_start = max(cont_start, int(args.start_time * fps))
        video_notes = extract_notes_continuity(
            cap, note_x_map, cont_yt, cont_yb,
            fps, total_frames, green_is_right, cfg['colors'],
            start_frame=cont_start, white_keys=white_keys, cfg=cfg,
            y_black=y_black, y_kb_top=y_kb_top, y_bk_bottom=y_bk_bottom,
        )
        if args.end_time is not None:
            video_notes = [n for n in video_notes if n[2] <= args.end_time]
        if args.start_time is not None:
            video_notes = [n for n in video_notes if n[2] >= args.start_time]
        right_count = sum(1 for _, h, _, _ in video_notes if h == 0)
        left_count = sum(1 for _, h, _, _ in video_notes if h == 1)
        print(f"Extracted {len(video_notes)} notes: right={right_count}, left={left_count}")
        cap.release()
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

        # --- Narrow-column refinement: split merged re-strikes / trim noise tails ---
        # Operates on raw (unfiltered) saturation in a 9-pixel column under
        # each long note. Counters the v_min=30 + bright-mean filter inside
        # KeySampler that flattens within-note variation.
        if not getattr(args, 'no_refine_keys', False) and video_notes:
            refine_yt = y_kb_top + 5
            refine_yb = y_kb_top + max(10, int((y_bk_bottom - y_kb_top) * 0.30))
            video_notes = refine_notes_via_narrow_trace(
                video_notes, args.video, note_x_map,
                y_top=refine_yt, y_bot=refine_yb, fps=fps,
                max_height=getattr(args, 'max_height', 0) or 0,
            )

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
        # When BPM was auto-detected (not user-specified), refine it using
        # the detected note onsets for better precision.
        if not args._had_explicit_bpm:
            # For falling-blocks, use note-based BPM as starting point
            # (more reliable than audio-based for Synthesia videos).
            if args.detector == 'falling-blocks' and len(video_notes) >= 10:
                from piano_video_scribe.detectors.falling_blocks import detect_bpm as _fb_detect_bpm
                fb_bpm = _fb_detect_bpm(video_notes)
                if 40 <= fb_bpm <= 220:
                    print(f"  Note-based BPM: {fb_bpm:.1f} (replaces audio estimate {OUT_BPM})")
                    OUT_BPM = fb_bpm
            # Refine BPM: search ±15 in 0.1 steps, pick the value where
            # the most notes land on 8th-note grid positions (even grid16).
            all_onsets = sorted(on for _, _, on, _ in video_notes)
            first = all_onsets[0]
            rel_onsets = [on - first for on in all_onsets]
            if len(rel_onsets) >= 10:
                best_bpm, best_on_grid = OUT_BPM, 0
                low = max(40.0, OUT_BPM - 15)
                high = min(220.0, OUT_BPM + 15)
                candidate = low
                while candidate <= high:
                    s16 = 60.0 / candidate / 4
                    on_grid = sum(1 for t in rel_onsets if round(t / s16) % 2 == 0)
                    if on_grid > best_on_grid:
                        best_on_grid = on_grid
                        best_bpm = candidate
                    candidate += 0.1
                best_bpm = round(best_bpm, 1)
                if best_bpm != OUT_BPM:
                    print(f"  BPM refined: {OUT_BPM} → {best_bpm} "
                          f"({best_on_grid}/{len(rel_onsets)} notes on 8th grid)")
                    OUT_BPM = best_bpm

        # Recompute derived values after any BPM changes
        OUT_US_PER_BEAT = int(60_000_000 / OUT_BPM)

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

        # Pre-compute a SHARED effective BPM and local-tempo lookup using
        # both hands together. The default 'simple' quantizer otherwise
        # runs drift-correction independently per hand, which lets each
        # hand land on a different effective BPM and slowly desync —
        # exactly the symptom seen on this video starting at the second
        # half of measure 1.
        all_onsets_rel = sorted([video_notes[i][2] - first_onset
                                 for i in right_indices + left_indices])
        shared_effective_bpm = _refine_bpm_for_grid(all_onsets_rel, OUT_BPM)
        shared_local_bpm = build_shared_local_bpm_lookup(
            all_onsets_rel, shared_effective_bpm)
        # Attach a continuous warp function built over BOTH hands so the
        # piecewise-integrated warp doesn't diverge per hand. Each hand's
        # quantizer call uses the same t→warped_t map, keeping them
        # locked to a single tempo curve over the whole song.
        shared_local_bpm.warp_lookup = build_shared_warp_lookup(
            all_onsets_rel, shared_effective_bpm)

        # Quantize ALL onsets on a SINGLE shared timeline in one pass,
        # then split the resulting grid positions back per hand. There is
        # no scenario where the two hands of one performance should be
        # quantized to independent grids — separate calls would let drift
        # correction or local-tempo estimation diverge per hand and pull
        # them out of sync. Single combined call by construction.
        if RIGHT_SUB != LEFT_SUB:
            raise ValueError(
                f"Hand subdivisions must match for shared quantization "
                f"(got RIGHT={RIGHT_SUB}, LEFT={LEFT_SUB}). Triplet/straight "
                f"split is not supported in single-pass mode."
            )
        SUB = RIGHT_SUB
        s = 60.0 / OUT_BPM / SUB  # grid unit duration in seconds

        # All onsets relative to first_onset, in original video_notes order.
        # We need to remember each onset's index so we can split back per
        # hand after quantization.
        onsets_all = [video_notes[i][2] - first_onset
                      for i in range(len(video_notes))]
        offsets_all = [video_notes[i][3] - first_onset
                       for i in range(len(video_notes))]
        # Sort by onset time, keeping original indices.
        order = sorted(range(len(video_notes)), key=lambda i: onsets_all[i])
        sorted_onsets = [onsets_all[i] for i in order]

        # Debug hook: PVS_DUMP_ONSETS=<path> dumps the exact pre-quantization
        # onsets (seconds, sorted) so detector timing can be inspected
        # separately from quantizer behaviour.
        dump_path = os.environ.get('PVS_DUMP_ONSETS')
        if dump_path:
            import json
            with open(dump_path, 'w') as f:
                json.dump({
                    'bpm': OUT_BPM,
                    'onsets': sorted_onsets,
                    'hands': [video_notes[i][1] for i in order],
                    'notes': [video_notes[i][0] for i in order],
                }, f)

        # Default is viterbi fed the shared *effective* BPM: 'simple' snaps
        # each onset independently, so frame jitter can flip a borderline
        # onset one 16th off and break visually-identical repeating bars
        # (Billie Jean intro: simple 4/64 bad intervals, viterbi@effective
        # 0/64; Beat It syncopation preserved note-for-note — see
        # .agents/investigations/rhythm-quantization-pattern-drift.md).
        quantizer = getattr(args, 'quantizer', None) or 'viterbi'
        if quantizer == 'simple':
            on_g_sorted = list(quantize_onsets_simple(
                sorted_onsets, OUT_BPM,
                effective_bpm=shared_effective_bpm,
                local_bpm_lookup=shared_local_bpm,
            ))
        elif quantizer == 'viterbi':
            on_g_sorted = list(quantize_onsets_viterbi(
                sorted_onsets, shared_effective_bpm, subdivisions=SUB))
        elif quantizer == 'adaptive':
            on_g_sorted = list(quantize_onsets_adaptive(sorted_onsets, OUT_BPM))
        elif quantizer == 'pll':
            on_g_sorted = list(quantize_onsets_pll(sorted_onsets, OUT_BPM,
                                                    subdivisions=SUB))
        else:
            on_g_sorted = list(quantize_onsets_viterbi(
                sorted_onsets, shared_effective_bpm, subdivisions=SUB))

        # Un-sort back to original index order.
        on_g_all = [0] * len(video_notes)
        for sort_pos, orig_i in enumerate(order):
            on_g_all[orig_i] = on_g_sorted[sort_pos]

        # Derive each note's grid offset from quantized onset + raw duration.
        off_g_all = []
        for i in range(len(video_notes)):
            raw_dur = offsets_all[i] - onsets_all[i]
            dur_grid = max(1, round(raw_dur / s))
            off_g_all.append(on_g_all[i] + dur_grid)

        # Split back per hand.
        r_on_grid = [on_g_all[i] for i in right_indices]
        r_off_grid = [off_g_all[i] for i in right_indices]
        l_on_grid = [on_g_all[i] for i in left_indices]
        l_off_grid = [off_g_all[i] for i in left_indices]

        # Decollide: two same-hand notes of *different* pitches that
        # quantize to the same on-grid are real consecutive strikes whose
        # raw timings happened to round together. Shift the earlier-in-
        # raw-time ones back by 1/16th-note grid steps so they read as
        # sequential, not as a simultaneous chord.
        # Only shift apart notes whose RAW onsets were genuinely
        # sequential — same-tick same-raw-time notes are real chords and
        # must stay together. Threshold = half a 16th note (well above
        # frame-jitter for chord onsets, well below any real 16th gap).
        sixteenth_grid = max(1, SUB // 4)
        chord_window_sec = 7.5 / OUT_BPM  # half a 16th note

        def _decollide(on_grid, off_grid, indices_local):
            by_pos = {}
            for j in range(len(indices_local)):
                by_pos.setdefault(on_grid[j], []).append(j)
            for pos, group in by_pos.items():
                if len(group) <= 1:
                    continue
                pitches = {video_notes[indices_local[j]][0] for j in group}
                if len(pitches) <= 1:
                    continue
                group.sort(key=lambda j: video_notes[indices_local[j]][2])
                latest_raw = video_notes[indices_local[group[-1]]][2]
                # Skip notes whose raw onset is within the chord window —
                # those are intentional chord mates of the latest one.
                shift_targets = [
                    j for j in group[:-1]
                    if (latest_raw - video_notes[indices_local[j]][2])
                       > chord_window_sec
                ]
                for k, j in enumerate(reversed(shift_targets)):
                    new_on = pos - (k + 1) * sixteenth_grid
                    if new_on < 0:
                        continue
                    on_grid[j] = new_on
                    off_grid[j] = pos

        _decollide(r_on_grid, r_off_grid, right_indices)
        _decollide(l_on_grid, l_off_grid, left_indices)

        # start_beat: grid units to add after quantization
        # (subdivisions per beat * beats of silence before first note)
        effective_start_beat = args.start_beat if args.start_beat is not None else 1.0
        start_grid_offset = int(round((effective_start_beat - 1.0) * SUB))
        if start_grid_offset > 0:
            print(f"  start_beat={effective_start_beat}, grid_offset={start_grid_offset}")

        # Apply start_beat offset: shift all grid positions so the first note
        # of the piece lands on the right beat in the measure. Same
        # subdivision for both hands (enforced above), same offset.
        if start_grid_offset > 0:
            r_on_grid = [p + start_grid_offset for p in r_on_grid]
            r_off_grid = [p + start_grid_offset for p in r_off_grid]
            l_on_grid = [p + start_grid_offset for p in l_on_grid]
            l_off_grid = [p + start_grid_offset for p in l_off_grid]

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
        evts = dedup_same_tick_pitch(evts)
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
            # Prefer simpler enharmonic spelling (Db over C#, Ab over G#)
            enharmonic_map = {'C#': 'Db', 'G#': 'Ab', 'D#': 'Eb', 'A#': 'Bb',
                              'E#': 'F', 'B#': 'C', 'Fb': 'E', 'Cb': 'B'}
            tonic = enharmonic_map.get(tonic, tonic)
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

    if getattr(args, 'lead_sheet', False):
        from piano_video_scribe.lead_sheet import save_lead_sheet
        ts = tuple(map(int, args.time_sig.split('/'))) if args.time_sig else (4, 4)
        save_lead_sheet(args.output, right_events, left_events, OUT_US_PER_BEAT, OUT_TPB, ts)

    rn = sum(1 for e in right_events if e[1] == 'note_on')
    ln = sum(1 for e in left_events if e[1] == 'note_on')
    print(f"\nSaved: {args.output}")
    print(f"Right hand: {rn} notes")
    print(f"Left hand:  {ln} notes")
    print(f"Total:      {rn + ln} notes")
    print(f"\nDone! Open {args.output} in MuseScore.")
    print("Tip: Preferences → Import → MIDI → Shortest note: 16th")

