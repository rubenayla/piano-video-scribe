"""Post-extraction refinement for the keys detector.

Re-traces long-duration notes through a narrow column of *unfiltered*
saturation pixels to detect:
  - merged re-strikes (rising-edge splits within a held note)
  - noise tails (saturation drops well below the within-note peak even
    though the per-key averaged signal stays above the off-threshold)

The hot per-frame loop in `extract_notes_from_video` filters out exactly
the variation we need here (it gates on V>=30 and averages only bright
pixels, so a continuously-lit key reads as a flat plateau even across
re-strikes). This refinement runs once at the end on raw means, so the
fast path stays fast.
"""

import cv2
import numpy as np

BLACK_SEMITONES = {1, 3, 6, 8, 10}


def _filter_adjacent_phantoms(notes, fps, max_dur=0.20, dur_ratio=2.0):
    """Drop short white-key notes that overlap a much longer adjacent
    white-key note in the same hand.

    Glow from a struck white key bleeds into its neighbor's detector
    column. The bleed registers as a brief note overlapping (or
    sandwiched between) the real strike(s) on the neighbor key. Black
    adjacency is already suppressed inside extract_notes_from_video.
    """
    if not notes:
        return notes
    tol = 1.0 / fps + 1e-6
    by_hand = {}
    for idx, (pitch, hand, on, off) in enumerate(notes):
        by_hand.setdefault(hand, []).append((idx, pitch, on, off))

    drop = set()
    for hand, group in by_hand.items():
        group.sort(key=lambda x: x[2])
        for i, (idx_n, p_n, on_n, off_n) in enumerate(group):
            dur_n = off_n - on_n
            if dur_n >= max_dur:
                continue
            if (p_n % 12) in BLACK_SEMITONES:
                continue
            # Search a window of nearby notes for an overlapping adjacent
            # white-key neighbor with substantially longer duration.
            for j in range(max(0, i - 8), min(len(group), i + 9)):
                if j == i:
                    continue
                idx_m, p_m, on_m, off_m = group[j]
                d = abs(p_m - p_n)
                # White keys are adjacent on the keyboard when their
                # semitone distance is 1 (E-F, B-C) or 2 (D-E, C-D, F-G,
                # G-A, A-B — black key between).
                if d not in (1, 2):
                    continue
                if (p_m % 12) in BLACK_SEMITONES:
                    continue
                # Overlap (with frame tolerance) in time.
                if not (on_n < off_m + tol and on_m < off_n + tol):
                    continue
                if (off_m - on_m) >= dur_n * dur_ratio:
                    drop.add(idx_n)
                    break

    if not drop:
        return notes
    print(f"  Adjacent-phantom filter: dropped {len(drop)} short white-key bleed notes")
    return [n for i, n in enumerate(notes) if i not in drop]


def refine_notes_via_narrow_trace(
    notes,
    video_path,
    note_x_map,
    y_top,
    y_bot,
    fps,
    min_dur_sec=0.40,
    column_half_width=4,
    valley_depth=15.0,
    rebound_depth=15.0,
    restrike_min_sat=80.0,
    restrike_min_gap_frames=3,
    peak_floor_frac=0.45,
    peak_floor_min=40.0,
    max_height=0,
):
    """Refine long notes using raw narrow-column saturation traces.

    Args:
        notes: list of (pitch, hand, onset_sec, offset_sec).
        video_path: path to source video (a fresh VideoCapture is opened).
        note_x_map: MIDI pitch -> x pixel center.
        y_top, y_bot: vertical bounds of the strip to sample.
        fps: video frame rate.
        min_dur_sec: only notes longer than this are refined.
        column_half_width: x ± hw → 2*hw+1 pixel column.
        restrike_rise: minimum sat jump (current - prev) to consider a re-strike.
        restrike_min_sat: minimum absolute sat to count as a re-strike.
        restrike_min_gap_frames: minimum frames since current segment start.
        peak_floor_frac: fraction of segment peak below which we trim the tail.
        peak_floor_min: absolute floor below which we trim regardless of peak.

    Returns:
        New list of (pitch, hand, onset_sec, offset_sec). Short notes pass
        through unchanged. Long notes may be split into multiple segments
        and/or have their offset trimmed.
    """
    candidates = []  # (orig_idx, pitch, hand, onset_f, offset_f)
    for i, (pitch, hand, onset_sec, offset_sec) in enumerate(notes):
        if offset_sec - onset_sec <= min_dur_sec:
            continue
        if pitch not in note_x_map:
            continue
        onset_f = int(round(onset_sec * fps))
        offset_f = int(round(offset_sec * fps))
        if offset_f <= onset_f:
            continue
        candidates.append((i, pitch, hand, onset_f, offset_f))

    if not candidates:
        return list(notes)

    raw_cap = cv2.VideoCapture(video_path)
    if not raw_cap.isOpened():
        return list(notes)
    src_h = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_height and max_height > 0 and src_h > max_height:
        from piano_video_scribe.video_io import ResizingCapture
        cap = ResizingCapture(raw_cap, max_height)
    else:
        cap = raw_cap
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    f_start = max(0, min(c[3] for c in candidates))
    f_end = min(total_frames - 1, max(c[4] for c in candidates))

    # Frame -> list of (cand_idx, pitch, x)
    cand_by_frame = {}
    for ci, (_, pitch, _, of, ef) in enumerate(candidates):
        x = note_x_map[pitch]
        of_c = max(0, of)
        ef_c = min(total_frames - 1, ef)
        for f in range(of_c, ef_c + 1):
            cand_by_frame.setdefault(f, []).append((ci, pitch, x))

    traces = [[] for _ in candidates]  # per cand: list of (frame_idx, sat)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    yt = max(0, int(y_top))
    yb = max(yt + 1, int(y_bot))
    f_idx = f_start
    print(f"  Refining {len(candidates)} long notes "
          f"(frames {f_start}–{f_end}, {f_end - f_start + 1} frames)...")
    while f_idx <= f_end:
        ret, frame = cap.read()
        if not ret:
            break
        needs = cand_by_frame.get(f_idx)
        if needs:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sat_strip = hsv[yt:yb, :, 1]
            for ci, _pitch, x in needs:
                x_lo = max(0, x - column_half_width)
                x_hi = min(width, x + column_half_width + 1)
                if x_hi > x_lo:
                    traces[ci].append((f_idx, float(sat_strip[:, x_lo:x_hi].mean())))
        f_idx += 1
    cap.release()

    refined_by_orig = {}
    n_split = 0
    n_trimmed = 0
    for ci, (orig_idx, pitch, hand, _of, _ef) in enumerate(candidates):
        trace = traces[ci]
        if len(trace) < 2:
            continue

        segments = []  # list of (start_frame, end_frame)
        seg_start_f = trace[0][0]
        seg_start_k = 0
        seg_peak = trace[0][1]
        seg_min_after_peak = trace[0][1]
        early_end_f = None

        for k in range(1, len(trace)):
            f, sat = trace[k]
            frames_in_seg = k - seg_start_k
            # Re-strike: signal must have dipped (valley) below peak, then
            # rebounded sharply above that valley, AND reach an absolute
            # minimum saturation. Thresholds scale with peak so that a 6%
            # noise dip on a sat-250 black key doesn't fire (393 false
            # splits on Beat It with the previous fixed 15.0 thresholds).
            valley_thresh   = max(valley_depth,  seg_peak * 0.30)
            rebound_thresh  = max(rebound_depth, seg_peak * 0.30)
            valley_seen = (seg_peak - seg_min_after_peak) >= valley_thresh
            is_restrike = (
                valley_seen
                and (sat - seg_min_after_peak) >= rebound_thresh
                and sat >= restrike_min_sat
                and frames_in_seg >= restrike_min_gap_frames
            )
            if is_restrike:
                end_f = trace[k - 1][0]
                if early_end_f is not None and early_end_f < end_f:
                    end_f = early_end_f
                segments.append((seg_start_f, end_f))
                seg_start_f = f
                seg_start_k = k
                seg_peak = sat
                seg_min_after_peak = sat
                early_end_f = None
            else:
                if sat > seg_peak:
                    seg_peak = sat
                    seg_min_after_peak = sat
                    early_end_f = None  # peak still climbing — reset tail mark
                elif sat < seg_min_after_peak:
                    seg_min_after_peak = sat
                floor = max(peak_floor_min, seg_peak * peak_floor_frac)
                if early_end_f is None and sat < floor and frames_in_seg >= 2:
                    early_end_f = f

        end_f = trace[-1][0]
        if early_end_f is not None and early_end_f < end_f:
            end_f = early_end_f
        segments.append((seg_start_f, end_f))

        if len(segments) > 1:
            n_split += 1
        elif segments and (segments[0][1] - segments[0][0]) < (trace[-1][0] - trace[0][0]):
            n_trimmed += 1

        refined = []
        for s_f, e_f in segments:
            onset_sec = s_f / fps
            # Ensure at least one-frame duration
            offset_sec = max(onset_sec + 1.0 / fps, e_f / fps)
            if offset_sec - onset_sec >= 0.02:
                refined.append((pitch, hand, onset_sec, offset_sec))

        if refined:
            refined_by_orig[orig_idx] = refined

    result = []
    for i, n in enumerate(notes):
        if i in refined_by_orig:
            result.extend(refined_by_orig[i])
        else:
            result.append(n)
    result.sort(key=lambda x: x[2])

    print(f"  Refinement: split={n_split}, trimmed={n_trimmed}, "
          f"total notes {len(notes)} -> {len(result)}")
    return result
