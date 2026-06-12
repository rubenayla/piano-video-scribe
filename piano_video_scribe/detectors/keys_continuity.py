"""Continuity-based keys detector.

Sibling to the per-frame threshold detector in ``keys.py``. Instead of
making an independent on/off decision every frame from the column
mean, this detector:

  Phase A — scans the video once into a per-pitch raw saturation
            time-series (and a hue time-series for hand classification).
  Phase B — finds onsets as prominence peaks above a moving baseline.
  Phase C — finds offsets via relative drop from the per-strike peak.
  Phase D — classifies each onset's hand from its hue.

The point: the column mean is allowed to drift (background glow,
sparkles, animations) without spurious on/offs, because every
decision is made against a *baseline that drifts with it*. One
decision per onset, not 60 per second.

Designed for cartoon/animation-heavy videos (Magic Bus etc.) where
the v-min + bright-pixel-mean filter inside ``keys.py`` flattens
within-note variation. Reuses the same column geometry that
``keys_refine`` already validated empirically.
"""

import time as _time

import cv2
import numpy as np

from piano_video_scribe.color import _classify_hand_from_hue


BLACK_SEMITONES = {1, 3, 6, 8, 10}


def extract_notes_continuity(
    cap, note_x_map, y_top, y_bot,
    fps, total_frames, green_is_right, colors,
    start_frame=0, white_keys=None, cfg=None,
    y_black=None, y_kb_top=None, y_bk_bottom=None,
    column_half_width=4,
    # Onset detector
    prominence_min=35.0,
    absolute_min_white=70.0,
    absolute_min_black=90.0,
    refractory_frames=6,
    baseline_window_sec=0.5,
    # Offset detector
    release_frac=0.45,
    release_abs=55.0,
    # Black-key bleed gate
    black_neighbor_ratio=0.95,
):
    """Extract notes via per-pitch saturation time-series.

    Returns: list of (pitch, hand, onset_sec, offset_sec).
    """
    pitches = sorted(note_x_map.keys())
    n_pitches = len(pitches)
    if total_frames <= 0 or n_pitches == 0:
        return []

    yt = max(0, int(y_top))
    yb = max(yt + 1, int(y_bot))

    # Pre-compute per-pitch column slices, clipped to frame width once
    # we read the first frame.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    col_slices = []
    for p in pitches:
        x = note_x_map[p]
        x_lo = max(0, x - column_half_width)
        x_hi = min(width, x + column_half_width + 1)
        col_slices.append((p, x_lo, x_hi))

    # Phase A — single sequential scan into per-pitch arrays
    sat_trace = np.zeros((n_pitches, total_frames), dtype=np.float32)
    hue_trace = np.full((n_pitches, total_frames), np.nan, dtype=np.float32)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    t0 = _time.time()
    f_idx = start_frame
    print(f"  Continuity scan: frames {start_frame}–{total_frames - 1} "
          f"({n_pitches} pitches)...")
    while f_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[yt:yb, :, 1]
        hue = hsv[yt:yb, :, 0]
        for k, (_p, x_lo, x_hi) in enumerate(col_slices):
            if x_hi <= x_lo:
                continue
            col_s = sat[:, x_lo:x_hi]
            sat_trace[k, f_idx] = float(col_s.mean())
            mask = col_s > 50
            if np.any(mask):
                hue_trace[k, f_idx] = float(hue[:, x_lo:x_hi][mask].mean())
        if f_idx % 500 == 0 and f_idx > start_frame:
            elapsed = _time.time() - t0
            rate = (f_idx - start_frame) / elapsed if elapsed > 0 else 0
            print(f"    Frame {f_idx}/{total_frames} "
                  f"({100*f_idx/total_frames:.0f}%) — {rate:.0f} fps")
        f_idx += 1

    # Phase B/C — per-pitch onset & offset detection
    pitch_to_idx = {p: k for k, (p, *_rest) in enumerate(col_slices)}
    notes = []
    baseline_w = max(3, int(round(baseline_window_sec * fps)))

    for k, p in enumerate(pitches):
        is_black = (p % 12) in BLACK_SEMITONES
        sat = sat_trace[k]
        if not np.any(sat > 30):
            continue
        baseline = _rolling_min(sat, baseline_w)
        # Smooth the rolling-min so single-frame valleys don't
        # artificially inflate prominence.
        baseline = _moving_average(baseline, max(3, baseline_w // 4))
        # Smooth saturation lightly before computing prominence — kills
        # single-frame jitter that creates spurious local maxima during
        # the flat top of a strike.
        sat_smooth = _moving_average(sat, 3)
        prom = sat_smooth - baseline

        abs_min = absolute_min_black if is_black else absolute_min_white
        onsets = _find_onsets(
            sat, prom, prominence_min, abs_min, refractory_frames,
        )

        if is_black and onsets:
            # Drop onsets that don't out-saturate the adjacent white-key
            # columns at the same frame (= bleed from white neighbor).
            left_idx = pitch_to_idx.get(p - 1)
            right_idx = pitch_to_idx.get(p + 1)
            kept = []
            for t_on in onsets:
                neigh_max = 0.0
                if left_idx is not None:
                    neigh_max = max(neigh_max, float(sat_trace[left_idx, t_on]))
                if right_idx is not None:
                    neigh_max = max(neigh_max, float(sat_trace[right_idx, t_on]))
                if neigh_max <= 0 or sat[t_on] >= black_neighbor_ratio * neigh_max:
                    kept.append(t_on)
            onsets = kept

        if not onsets:
            continue

        # Phase C — offsets per onset
        for i, t_on in enumerate(onsets):
            t_next = onsets[i + 1] if i + 1 < len(onsets) else total_frames
            t_off = _find_offset(sat, t_on, t_next, release_frac, release_abs)

            # Phase D — hand from hue at attack window
            attack_end = min(t_on + 3, total_frames)
            window_hues = hue_trace[k, t_on:attack_end]
            hue_vals = window_hues[~np.isnan(window_hues)]
            hue = float(np.mean(hue_vals)) if hue_vals.size else None
            hand = _classify_hand_from_hue(hue, green_is_right, colors)
            if hand is None:
                continue  # UI/overlay artifact — skip

            on_sec = t_on / fps
            off_sec = max(on_sec + 1.0 / fps, t_off / fps)
            if off_sec - on_sec >= 0.02:
                notes.append((p, hand, on_sec, off_sec))

    notes.sort(key=lambda n: n[2])
    print(f"  Continuity detector: {len(notes)} notes")
    return notes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rolling_min(a, w):
    """Rolling min with window ``w``, same length as ``a`` (edge-padded)."""
    n = a.shape[0]
    if n == 0:
        return a.copy()
    half = w // 2
    out = np.empty_like(a)
    # O(n*w) is fine here — w ~= 30 frames, n ~= 4000.
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = a[lo:hi].min()
    return out


def _moving_average(a, w):
    if w <= 1 or a.shape[0] == 0:
        return a.copy()
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(a, kernel, mode='same')


def _find_onsets(sat, prom, prominence_min, abs_min, refractory):
    """Find local maxima of ``prom`` that pass thresholds, with refractory."""
    n = sat.shape[0]
    onsets = []
    last = -10**9
    for t in range(1, n - 1):
        if prom[t] < prominence_min:
            continue
        if sat[t] < abs_min:
            continue
        # Local maximum on the prominence trace
        if not (prom[t] >= prom[t - 1] and prom[t] >= prom[t + 1]):
            continue
        if t - last < refractory:
            # Within refractory — keep only the higher prominence one.
            if onsets and prom[t] > prom[onsets[-1]]:
                onsets[-1] = t
                last = t
            continue
        onsets.append(t)
        last = t
    return onsets


def _find_offset(sat, t_on, t_limit, release_frac, release_abs):
    n = sat.shape[0]
    end = min(n, t_limit)
    peak = sat[t_on]
    for t in range(t_on + 1, end):
        if sat[t] > peak:
            peak = sat[t]
        floor = peak * release_frac
        if sat[t] < floor and sat[t] < release_abs:
            return t
    return end - 1
