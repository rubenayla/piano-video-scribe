"""Tick conversion, onset quantization, and adaptive tempo tracking.

Pure math/algorithm functions with no OpenCV dependency.
"""

import math
import statistics


# ---------------------------------------------------------------------------
# Simple round-to-nearest quantizer (default)
# ---------------------------------------------------------------------------

def quantize_onsets_simple(onset_secs: list[float], bpm: float) -> list[int]:
    """Snap each onset independently to the nearest 16th-note grid position.

    Two-stage correction:
    1. Drift correction: refines the effective BPM so that raw onsets don't
       accumulate a progressive phase offset (common with falling-blocks
       detector where fall speed ≠ exact musical tempo).
    2. Per-interval BPM correction: estimates local BPM from each interval
       and warps to match the nominal grid.
    """
    s = 60.0 / bpm / 4  # 16th note duration
    if not onset_secs:
        return []

    # Stage 1: Drift correction — find the effective BPM that best aligns
    # onsets to even (8th-note) grid positions.  Search is narrow (±1%)
    # to avoid accidentally snapping to a wrong tempo.
    effective_bpm = _refine_bpm_for_grid(onset_secs, bpm)
    s_eff = 60.0 / effective_bpm / 4

    if len(onset_secs) >= 4:
        grid0 = [round(t / s_eff) for t in onset_secs]
        local_bpms = estimate_local_bpm(onset_secs, grid0, effective_bpm)
        warped = warp_onsets(onset_secs, local_bpms, effective_bpm)
        return [round(t / s_eff) for t in warped]

    return [round(t / s_eff) for t in onset_secs]


def _refine_bpm_for_grid(onset_secs: list[float], nominal_bpm: float,
                          max_pct: float = 1.5, step: float = 0.01) -> float:
    """Find the effective BPM (within ±max_pct% of nominal) that maximizes
    the number of onsets landing on 8th-note grid positions (even grid16).

    This corrects for systematic drift caused by fall-speed calibration
    errors in the detector.  The search range is kept tight to avoid
    accidentally locking onto a harmonic (double/half time).

    Returns the refined BPM, or nominal_bpm if no improvement is found.
    """
    if len(onset_secs) < 10:
        return nominal_bpm

    low = nominal_bpm * (1 - max_pct / 100)
    high = nominal_bpm * (1 + max_pct / 100)

    best_bpm = nominal_bpm
    s_nom = 60.0 / nominal_bpm / 4
    best_on_grid = sum(1 for t in onset_secs if round(t / s_nom) % 2 == 0)

    candidate = low
    while candidate <= high:
        s16 = 60.0 / candidate / 4
        on_grid = sum(1 for t in onset_secs if round(t / s16) % 2 == 0)
        if on_grid > best_on_grid:
            best_on_grid = on_grid
            best_bpm = candidate
        candidate += step

    return round(best_bpm, 2)


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
    """Snap a tick to the nearest grid position."""
    return round(tick / grid) * grid


def quantize_tick_smart(tick, eighth, sixteenth):
    """Snap to nearest 16th (simple fallback, used by MIDI-based mode)."""
    return round(tick / sixteenth) * sixteenth


def quantize_onsets_pll(onset_secs, bpm, alpha=0.1, subdivisions=4):
    """Phase-locked loop quantizer: snap onsets to grid positions.

    Maintains a running phase offset (EMA) that self-corrects for BPM drift
    and per-note jitter.  Resets phase after large gaps (>= 3 grid units) to
    avoid carrying accumulated error from dense passages into sparse ones.

    Args:
        subdivisions: grid units per beat. 4 = 16th notes (default),
            3 = triplet 8th notes, 6 = triplet 16th notes.
    """
    s = 60.0 / bpm / subdivisions  # grid unit duration
    phase = 0.0
    initialized = False
    results = []

    for i, t in enumerate(onset_secs):
        # Reset phase after large gaps (accumulated error doesn't carry over).
        # Snap to nearest EIGHTH (2 sixteenths) since notes after gaps are
        # almost always on 8th positions, not 16ths.
        if i > 0 and (t - onset_secs[i - 1]) > 2.5 * s:
            grid_pos_8th = t / (2 * s)  # position in eighths
            idx_8th = round(grid_pos_8th)
            phase = t - idx_8th * 2 * s

        grid_pos = (t - phase) / s
        idx = round(grid_pos)
        ideal_time = idx * s + phase
        error = t - ideal_time

        if not initialized:
            phase = error
            initialized = True
        else:
            phase += alpha * error

        results.append(idx)

    # For combined grids (subdivisions=12), snap each index to the nearest
    # valid musical position. Valid positions per beat (in 12ths):
    # straight: 0,3,6,9  triplet: 0,4,8  → combined: 0,3,4,6,8,9
    # This eliminates quintuplets (5) and septuplets (7) from the output.
    if subdivisions == 12:
        valid_per_beat = [0, 3, 4, 6, 8, 9]
        snapped = []
        for idx in results:
            beat = idx // 12
            pos = idx % 12
            # Find nearest valid position (may cross beat boundary)
            best = min(valid_per_beat, key=lambda v: abs(v - pos))
            snapped.append(beat * 12 + best)
        results = snapped

    return results


def quantize_onsets_viterbi(onset_secs, bpm, abs_weight=0.1):
    """Viterbi DP quantizer: snap onsets to 16th-note grid positions.

    Finds the globally optimal assignment of grid positions that minimises
    a weighted combination of per-interval error and absolute-position error.
    Then refines ambiguous intervals by testing floor/ceil flips.

    100% accuracy on test data (vs 97% PLL, 91% global-fit).
    """
    s = 60.0 / bpm / 4  # sixteenth note duration
    n = len(onset_secs)
    if n == 0:
        return []

    # Phase 1: Viterbi DP
    dp = [dict() for _ in range(n)]
    dp[0][0] = (0.0, None)  # start at grid position 0

    for i in range(n - 1):
        interval = onset_secs[i + 1] - onset_secs[i]
        ratio = interval / s
        centre = round(ratio)
        # Allow step=0 for chord notes (interval < half a grid unit)
        min_step = 0 if ratio < 0.5 else 1
        candidates = set()
        for c in range(max(min_step, centre - 2), centre + 3):
            candidates.add(c)
        candidates.add(max(min_step, math.floor(ratio)))
        candidates.add(max(min_step, math.ceil(ratio)))

        for pos, (cost, _) in dp[i].items():
            for k in candidates:
                new_pos = pos + k
                interval_err = (interval - k * s) ** 2
                abs_err = (onset_secs[i + 1] - new_pos * s) ** 2
                new_cost = cost + interval_err + abs_weight * abs_err
                if new_pos not in dp[i + 1] or dp[i + 1][new_pos][0] > new_cost:
                    dp[i + 1][new_pos] = (new_cost, pos)

    # Backtrace
    best_final = min(dp[n - 1], key=lambda p: dp[n - 1][p][0])
    positions = []
    pos = best_final
    for i in range(n - 1, -1, -1):
        positions.append(pos)
        pos = dp[i][pos][1]
    positions.reverse()

    # Phase 2: refine ambiguous intervals
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            interval = onset_secs[i] - onset_secs[i - 1]
            ratio = interval / s
            frac = ratio - math.floor(ratio)
            if not (0.25 < frac < 0.75):
                continue
            current_delta = positions[i] - positions[i - 1]
            lo = max(1, math.floor(ratio))
            hi = lo + 1
            other_delta = hi if current_delta == lo else lo
            shift = other_delta - current_delta
            curr_abs = sum((onset_secs[j] - positions[j] * s) ** 2 for j in range(i, n))
            flip_abs = sum((onset_secs[j] - (positions[j] + shift) * s) ** 2 for j in range(i, n))
            if flip_abs < curr_abs:
                for j in range(i, n):
                    positions[j] += shift
                changed = True

    return positions


def estimate_local_bpm(onset_secs: list[float], grid_positions: list[int], nominal_bpm: float) -> list[float]:
    """Estimate a smooth per-note BPM curve from grid positions.

    Uses median filtering + rate limiting to reject bad estimates from
    misquantized grid positions while following gradual tempo drift.
    """
    n = len(onset_secs)
    if n < 2:
        return [nominal_bpm] * n

    s = 60.0 / nominal_bpm / 4  # nominal 16th duration

    # Compute implied BPM from each interval
    raw_bpms: list[float] = [nominal_bpm]  # first note has no prior interval
    for i in range(1, n):
        interval = onset_secs[i] - onset_secs[i - 1]
        grid_step = grid_positions[i] - grid_positions[i - 1]
        # Skip chords (zero interval) and large gaps (rests)
        if interval < 1e-6 or grid_step <= 0 or interval > 2.5 * grid_step * s:
            raw_bpms.append(nominal_bpm)
        else:
            raw_bpms.append(grid_step * 15.0 / interval)

    # Median filter (window=7) to remove outlier BPM values
    WINDOW = 7
    half = WINDOW // 2
    filtered: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        filtered.append(statistics.median(raw_bpms[lo:hi]))

    # Rate limiter: clamp BPM change per note
    median_step = statistics.median(
        max(1, grid_positions[i] - grid_positions[i - 1])
        for i in range(1, n)
    ) if n > 1 else 4
    max_delta = 0.005 * nominal_bpm * median_step / 4
    limited: list[float] = [filtered[0]]
    for i in range(1, n):
        prev = limited[-1]
        clamped = max(prev - max_delta, min(prev + max_delta, filtered[i]))
        limited.append(clamped)

    return limited


def warp_onsets(onset_secs: list[float], local_bpms: list[float], nominal_bpm: float) -> list[float]:
    """Warp onset times to compensate for tempo drift.

    Transforms onsets so that under constant nominal_bpm, the Viterbi
    quantizer sees the same relative spacing as the actual notes under
    the variable tempo.  Identity when local_bpms == nominal_bpm everywhere.
    """
    n = len(onset_secs)
    if n == 0:
        return []
    warped = [onset_secs[0]]
    for i in range(1, n):
        dt = onset_secs[i] - onset_secs[i - 1]
        avg_bpm = (local_bpms[i - 1] + local_bpms[i]) / 2
        warped.append(warped[-1] + dt * avg_bpm / nominal_bpm)
    return warped


def quantize_onsets_adaptive(onset_secs, bpm, abs_weight=0.1, drift_threshold=0.3):
    """Two-pass adaptive quantizer: handles gradual BPM drift.

    Pass 1: standard Viterbi with nominal BPM to get grid positions.
    Pass 2: estimate local BPM curve, warp onsets, re-run Viterbi.

    Rate-limited BPM tracking ensures rhythmic patterns (triplets,
    syncopation) don't cause false tempo adjustments.
    """
    n = len(onset_secs)
    if n < 4:
        return quantize_onsets_viterbi(onset_secs, bpm, abs_weight)

    # Pass 1: initial quantization
    grid_positions = quantize_onsets_viterbi(onset_secs, bpm, abs_weight)

    # Estimate smooth local BPM curve
    local_bpms = estimate_local_bpm(onset_secs, grid_positions, bpm)

    # Short-circuit if no significant drift detected
    if max(local_bpms) - min(local_bpms) < drift_threshold:
        return grid_positions

    # Pass 2: warp onsets and re-quantize
    warped = warp_onsets(onset_secs, local_bpms, bpm)
    return quantize_onsets_viterbi(warped, bpm, abs_weight)
