"""
Test quantization of raw note onsets against expected grid positions.

Test data from a Synthesia video at ~90 BPM in 6/8 time.
Grid positions are in 16th notes (1 sixteenth = 60/(90*4) ~= 0.1667s).

Usage:
    from test_quantization import run_quantization_test
    score = run_quantization_test(my_quantize_fn, bpm=90)

The quantize function signature: fn(onset_secs: float, bpm: float) -> int
    - onset_secs: note onset in seconds, relative to the first note
    - bpm: tempo in quarter-note beats per minute
    - returns: grid position in 16th notes
"""

from typing import Callable, NamedTuple

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# First 12 notes: should all be on 8th note grid (every 0.3333s)
RAW_ONSETS_8THS = [0.0000, 0.3337, 0.7007, 1.0344, 1.3680, 1.7017, 2.0354, 2.3690, 2.7027, 3.0364, 3.3700, 3.7371]
EXPECTED_GRID_8THS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]  # in 16th note positions

# Measure 5 notes (t_rel ~8-10s): mix of 8ths and 16ths
RAW_ONSETS_M5 = [8.0374, 8.5712, 8.7381, 8.9718, 9.3055, 9.6725, 9.8395]
EXPECTED_GRID_M5 = [48, 51, 52, 54, 56, 58, 59]  # 16th positions -- has real 16ths

# Measure 6 notes (t_rel ~10-12s): should all be 8ths
RAW_ONSETS_M6 = [10.0767, 10.4771, 10.7441, 11.0777, 11.4448, 11.7784]
EXPECTED_GRID_M6 = [60, 62, 64, 66, 68, 70]  # all even = 8th positions

# Measure 9 notes (t_rel ~16-18s): mix of 8ths and 16ths
RAW_ONSETS_M9 = [16.0374, 16.5712, 16.7381, 16.9718, 17.1722, 17.5725, 17.7395, 17.9731]
EXPECTED_GRID_M9 = [96, 99, 100, 102, 103, 105, 106, 108]  # has real 16ths

TEST_CASES = [
    ("First 12 notes (straight 8ths)", RAW_ONSETS_8THS, EXPECTED_GRID_8THS),
    ("Measure 5 (8ths + 16ths)", RAW_ONSETS_M5, EXPECTED_GRID_M5),
    ("Measure 6 (straight 8ths)", RAW_ONSETS_M6, EXPECTED_GRID_M6),
    ("Measure 9 (8ths + 16ths)", RAW_ONSETS_M9, EXPECTED_GRID_M9),
]


class QuantizationResult(NamedTuple):
    score: int  # 0-100
    correct: int
    total: int
    mismatches: list  # list of (case_name, onset, expected, got, error_secs)


def run_quantization_test(
    quantize_fn: Callable[[float, float], int],
    bpm: float = 90,
    *,
    verbose: bool = True,
) -> int:
    """Run the quantization test suite and return a score (0-100).

    Args:
        quantize_fn: function(onset_secs, bpm) -> grid_position_in_16ths
        bpm: tempo in quarter-note BPM (default 90)
        verbose: if True, print results and mismatches

    Returns:
        Integer score 0-100 based on percentage of correctly placed notes.
    """
    sixteenth_dur = 60.0 / (bpm * 4)
    correct = 0
    total = 0
    mismatches: list[tuple[str, float, int, int, float]] = []

    for case_name, raw_onsets, expected_grid in TEST_CASES:
        for onset, expected_pos in zip(raw_onsets, expected_grid):
            got_pos = quantize_fn(onset, bpm)
            total += 1
            if got_pos == expected_pos:
                correct += 1
            else:
                error_secs = (got_pos - expected_pos) * sixteenth_dur
                mismatches.append((case_name, onset, expected_pos, got_pos, error_secs))

    score = round(100 * correct / total) if total > 0 else 0

    if verbose:
        print(f"\nQuantization test results: {correct}/{total} correct  ({score}%)")
        print(f"  BPM={bpm}  16th={sixteenth_dur:.4f}s  8th={sixteenth_dur*2:.4f}s")
        if mismatches:
            print(f"\n  {len(mismatches)} mismatches:")
            for case_name, onset, expected, got, err in mismatches:
                print(
                    f"    [{case_name}] onset={onset:.4f}s  "
                    f"expected={expected}  got={got}  "
                    f"error={err:+.4f}s ({err/sixteenth_dur:+.1f} 16ths)"
                )
        else:
            print("  All notes snapped correctly.")

    return score


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------

def _naive_quantize(onset_secs: float, bpm: float) -> int:
    """Round to the nearest 16th note -- baseline quantizer for smoke test."""
    sixteenth_dur = 60.0 / (bpm * 4)
    return round(onset_secs / sixteenth_dur)


def test_run_quantization_test_returns_score():
    """Smoke test: run_quantization_test runs without error and returns 0-100."""
    score = run_quantization_test(_naive_quantize, bpm=90, verbose=False)
    assert isinstance(score, int)
    assert 0 <= score <= 100


def test_perfect_quantizer_scores_100():
    """A quantizer that returns the expected values must score 100."""
    # Build a lookup from onset -> expected grid position
    lookup: dict[float, int] = {}
    for _, onsets, grid in TEST_CASES:
        for onset, pos in zip(onsets, grid):
            lookup[onset] = pos

    def perfect_fn(onset: float, bpm: float) -> int:
        return lookup[onset]

    score = run_quantization_test(perfect_fn, bpm=90, verbose=False)
    assert score == 100


def test_naive_quantize_baseline():
    """Naive rounding should get most straight 8ths right but not all 16ths."""
    score = run_quantization_test(_naive_quantize, bpm=90, verbose=False)
    # Straight 8ths are roughly on grid, so baseline should beat random chance
    assert score > 50


# ---------------------------------------------------------------------------
# Adaptive quantizer tests
# ---------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pianovideoscribe import (
    quantize_onsets_viterbi,
    quantize_onsets_adaptive,
)
from piano_video_scribe.quantization import (
    quantize_onsets_pll,
    quantize_onsets_simple,
    _refine_bpm_for_grid,
)


def apply_tempo_drift(onsets, grid_positions, nominal_bpm, drift_rate):
    """Warp constant-BPM onsets as if tempo drifts linearly.

    drift_rate: BPM change per second (e.g., 0.25 = +1 BPM over 4 seconds)
    """
    result = [0.0]
    for i in range(1, len(onsets)):
        grid_step = grid_positions[i] - grid_positions[i - 1]
        actual_bpm = nominal_bpm + drift_rate * result[-1]
        s_actual = 60.0 / actual_bpm / 4
        result.append(result[-1] + grid_step * s_actual)
    return result


def test_adaptive_handles_drift():
    """Adaptive quantizer should recover correct grid under gradual BPM drift."""
    bpm = 90
    drifted = apply_tempo_drift(RAW_ONSETS_8THS, EXPECTED_GRID_8THS, bpm, 0.25)
    result = quantize_onsets_adaptive(drifted, bpm)
    assert result == EXPECTED_GRID_8THS, (
        f"Adaptive failed on drifted 8ths: {result} != {EXPECTED_GRID_8THS}"
    )


def test_adaptive_ignores_triplets():
    """Adaptive should not mistake rhythmic variation for tempo drift."""
    bpm = 90
    s = 60.0 / bpm / 4  # 16th note duration

    # Build onsets: 4 8ths, then a triplet (3 notes in 2 8ths), then 4 more 8ths
    onsets = [0.0]
    expected = [0]
    for i in range(1, 5):
        onsets.append(i * 2 * s)
        expected.append(i * 2)
    # 3 notes in the space of 2 8ths (triplet-like)
    base = onsets[-1]
    triplet_dur = 2 * s / 3
    onsets.append(base + triplet_dur)
    expected.append(11)
    onsets.append(base + 2 * triplet_dur)
    expected.append(12)
    # 4 more 8ths after the triplet section
    resume = base + 2 * 2 * s
    for i in range(4):
        onsets.append(resume + i * 2 * s)
        expected.append(14 + i * 2)

    result_adaptive = quantize_onsets_adaptive(onsets, bpm)
    result_viterbi = quantize_onsets_viterbi(onsets, bpm)
    assert result_adaptive == result_viterbi, (
        f"Adaptive diverged from Viterbi on triplet pattern: "
        f"{result_adaptive} != {result_viterbi}"
    )


def test_adaptive_regression_constant_tempo():
    """With constant tempo, adaptive must match Viterbi exactly."""
    bpm = 90
    for case_name, onsets, expected in TEST_CASES:
        t0 = onsets[0]
        shifted = [t - t0 for t in onsets]
        result_adaptive = quantize_onsets_adaptive(shifted, bpm)
        result_viterbi = quantize_onsets_viterbi(shifted, bpm)
        assert result_adaptive == result_viterbi, (
            f"[{case_name}] Adaptive != Viterbi at constant tempo: "
            f"{result_adaptive} != {result_viterbi}"
        )


# ---------------------------------------------------------------------------
# PLL quantizer tests
# ---------------------------------------------------------------------------

def _make_constant_wrong_bpm_onsets(nominal_bpm, true_bpm, grid_positions):
    """Build onsets sampled at true_bpm given expected grid_positions in 16ths.

    The caller will pass nominal_bpm to the quantizer; the PLL should adapt
    its internal grid spacing toward true_bpm and recover grid_positions.
    """
    s_true = 60.0 / true_bpm / 4
    return [pos * s_true for pos in grid_positions]


def test_pll_matches_viterbi_at_constant_correct_tempo():
    """With no drift and correct BPM, PLL should land on the expected grid."""
    bpm = 90
    for case_name, onsets, expected in TEST_CASES:
        t0 = onsets[0]
        shifted = [t - t0 for t in onsets]
        result = quantize_onsets_pll(shifted, bpm)
        # Shift result so it starts at the expected first position (PLL
        # returns raw grid indices; the pipeline adds the hand_grid_offset).
        offset = expected[0] - result[0]
        normalized = [r + offset for r in result]
        assert normalized == expected, (
            f"[{case_name}] PLL at correct tempo: {normalized} != {expected}"
        )


def test_pll_adapts_to_constant_bpm_offset():
    """If nominal BPM is off from the true tempo by 2%, the old PLL tracked
    only phase and let quantization drift into the next measure.  The fixed
    PLL also tracks grid spacing and should recover correct positions
    within a few onsets.
    """
    # 32 onsets, two 16ths apart on a 117.6 BPM grid
    expected = list(range(0, 64, 2))
    onsets = _make_constant_wrong_bpm_onsets(120, 117.6, expected)

    result = quantize_onsets_pll(onsets, 120)
    # Allow initial transient to settle — first 8 onsets may lag, but the
    # tail must land perfectly.  (Prior PLL drifted monotonically; this one
    # should converge.)
    assert result[-8:] == expected[-8:], (
        f"PLL did not adapt to 2% slow tempo.\n"
        f"expected tail: {expected[-8:]}\n"
        f"     got tail: {result[-8:]}"
    )


def test_pll_adapts_to_gradual_drift():
    """PLL should track a slow linear BPM drift."""
    bpm = 90
    drifted = apply_tempo_drift(RAW_ONSETS_8THS, EXPECTED_GRID_8THS, bpm, 0.25)
    result = quantize_onsets_pll(drifted, bpm)
    offset = EXPECTED_GRID_8THS[0] - result[0]
    normalized = [r + offset for r in result]
    assert normalized == EXPECTED_GRID_8THS, (
        f"PLL failed on drifted 8ths: {normalized} != {EXPECTED_GRID_8THS}"
    )


def test_pll_period_clamp_prevents_runaway():
    """A single bad onset shouldn't push the grid outside ±10% of nominal."""
    bpm = 120
    s = 60.0 / bpm / 4
    # Normal onsets then one 50%-late outlier then back to normal
    onsets = [i * 2 * s for i in range(6)]
    onsets += [onsets[-1] + 3 * s]  # outlier at +3/16 instead of +2/16
    onsets += [onsets[-1] + 2 * s for _ in range(5)]
    # Recompute last 5 based on last previous
    onsets = [i * 2 * s for i in range(6)]
    onsets.append(onsets[-1] + 3 * s)
    for _ in range(5):
        onsets.append(onsets[-1] + 2 * s)

    result = quantize_onsets_pll(onsets, bpm)
    # After the outlier, spacing between consecutive grid positions should
    # stay close to 2 (the true cadence), not balloon.
    tail_diffs = [result[i+1] - result[i] for i in range(len(result) - 5, len(result) - 1)]
    assert all(1 <= d <= 3 for d in tail_diffs), (
        f"PLL spacing after outlier: {tail_diffs}"
    )


# ---------------------------------------------------------------------------
# _refine_bpm_for_grid tests
# ---------------------------------------------------------------------------

def test_refine_bpm_catches_2pct_slow_tempo():
    """Refinement must find the true BPM within the ±5% window."""
    true_bpm = 117.6
    expected = list(range(0, 40, 2))
    onsets = _make_constant_wrong_bpm_onsets(120, true_bpm, expected)
    effective = _refine_bpm_for_grid(onsets, 120.0)
    assert abs(effective - true_bpm) < 0.5, (
        f"Refinement missed true BPM: got {effective}, expected ~{true_bpm}"
    )


def test_refine_bpm_keeps_nominal_when_correct():
    """If onsets are already on the nominal grid, refinement should not move."""
    bpm = 120.0
    s = 60.0 / bpm / 4
    onsets = [i * 2 * s for i in range(40)]
    effective = _refine_bpm_for_grid(onsets, bpm)
    assert abs(effective - bpm) < 0.1


def test_refine_bpm_short_input_returns_nominal():
    """Too few onsets → nominal (no reliable refinement possible)."""
    effective = _refine_bpm_for_grid([0.0, 0.5, 1.0], 120.0)
    assert effective == 120.0


# ---------------------------------------------------------------------------
# Synthetic regression for the "F drifts onto next bar" bug
# ---------------------------------------------------------------------------
#
# Made-up two-bar phrase sampled 2% slow against a 120 BPM nominal.
# The test checks that the last onset (the last 16th of bar 2) does not
# get pushed onto the bar-3 downbeat.  The positions are defined here,
# not taken from any real song.

def test_simple_quantizer_stable_end_of_bar_onset():
    """Under 2% slow tempo, a last-16th onset must stay inside its bar.

    Made-up two-bar phrase with onsets at 16th positions
    [0, 3, 4, 15, 16, 19, 20, 31].  With the old ±1.5% refinement window
    the final onset quantized to 32 (bar-3 downbeat); with the widened
    window + RMS objective it must stay at 31.
    """
    nominal = 120.0
    true_bpm = 117.6
    expected = [0, 3, 4, 15, 16, 19, 20, 31]
    onsets = _make_constant_wrong_bpm_onsets(nominal, true_bpm, expected)
    grid = quantize_onsets_simple(onsets, nominal)
    assert grid[-1] == 31, (
        f"Last onset drifted onto next bar's downbeat: "
        f"grid={grid}, expected {expected}"
    )
    assert grid == expected, f"grid={grid}, expected={expected}"


# ---------------------------------------------------------------------------
# Billie Jean intro regression — identical repeating patterns must quantize
# to identical intervals.
#
# Real pre-quantization detector onsets (seconds, sorted, combined hands)
# from ~/piano/songs/billie-jean-michael-jackson/video.mp4 at nominal
# 117 BPM. The left hand is a metronomically steady stream of eighth
# notes, so after quantization every LH inter-onset interval must be
# exactly 2 sixteenths — zero exceptions. The old default 'simple'
# quantizer snapped onsets independently and produced 4/64 wrong
# intervals here; this is the bug that made repeating bars engrave
# differently. Full brief:
# .agents/investigations/rhythm-quantization-pattern-drift.md
# ---------------------------------------------------------------------------

BILLIE_JEAN_INTRO_ONSETS = [
    0.0, 0.2667, 0.5333, 0.7667, 1.0333, 1.3, 1.5667, 1.8, 2.0667, 2.3333,
    2.6, 2.8333, 3.1, 3.3667, 3.6333, 3.8667, 4.1333, 4.4, 4.6667, 4.9,
    5.1667, 5.4333, 5.7, 5.9667, 6.2, 6.4667, 6.7333, 7.0, 7.2333, 7.5,
    7.7667, 8.0333, 8.2667, 8.5333, 8.8, 9.0667, 9.3, 9.5667, 9.8333, 10.1,
    10.3333, 10.6, 10.8667, 11.1333, 11.4, 11.6333, 11.9, 12.1667, 12.4333,
    12.6667, 12.9333, 13.2, 13.4667, 13.7, 13.9667, 14.2333, 14.5, 14.7333,
    15.0, 15.2667, 15.5333, 15.7667, 16.0333, 16.3, 16.5667, 16.5667,
    16.5667, 16.5667, 16.8333,
]
# 1 = left hand, 0 = right hand (the three RH notes are the first chord).
BILLIE_JEAN_INTRO_HANDS = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
]


def test_billie_jean_intro_equal_intervals_viterbi():
    """Steady LH eighths must come out as 64 identical 2-sixteenth intervals.

    Mirrors the pipeline's default path: refine the nominal BPM on the
    combined onsets, quantize in one shared pass with viterbi, then split
    the left hand back out and check its inter-onset intervals.
    """
    from collections import Counter

    eff = _refine_bpm_for_grid(BILLIE_JEAN_INTRO_ONSETS, 117.0)
    grid = quantize_onsets_viterbi(BILLIE_JEAN_INTRO_ONSETS, eff)
    lh = sorted({p for p, h in zip(grid, BILLIE_JEAN_INTRO_HANDS) if h == 1})
    intervals = [lh[i] - lh[i - 1] for i in range(1, len(lh))][:64]
    assert intervals == [2] * 64, (
        f"Identical input pattern quantized non-identically: "
        f"interval histogram {dict(Counter(intervals))}, want {{2: 64}}"
    )


if __name__ == "__main__":
    print("=== Naive round-to-nearest-16th baseline ===")
    score = run_quantization_test(_naive_quantize, bpm=90)
    print(f"\nFinal score: {score}")
