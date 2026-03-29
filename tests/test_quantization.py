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


if __name__ == "__main__":
    print("=== Naive round-to-nearest-16th baseline ===")
    score = run_quantization_test(_naive_quantize, bpm=90)
    print(f"\nFinal score: {score}")
