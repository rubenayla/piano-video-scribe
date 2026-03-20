"""
Global least-squares fit for snapping raw note onsets to a 16th-note grid.

Problem: raw onset times from a Synthesia video at ~90 BPM with +-33ms jitter
(~30fps). Mostly 8th notes (0.333s) with occasional 16ths (0.167s).

Approach:
  1. Naive snap: round each onset to nearest 16th-note position using initial BPM.
  2. Fit linear model: t = pos * duration + offset  (least-squares).
  3. Re-snap on the fitted grid using the refined duration and offset.
  4. Iterate steps 2-3 for N iterations.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

raw_8ths = [0.0000, 0.3337, 0.7007, 1.0344, 1.3680, 1.7017, 2.0354, 2.3690, 2.7027, 3.0364, 3.3700, 3.7371]
expected_8ths = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

raw_m5 = [8.0374, 8.5712, 8.7381, 8.9718, 9.3055, 9.6725, 9.8395]
expected_m5 = [48, 51, 52, 54, 56, 58, 59]

raw_m6 = [10.0767, 10.4771, 10.7441, 11.0777, 11.4448, 11.7784]
expected_m6 = [60, 62, 64, 66, 68, 70]

raw_m9 = [16.0374, 16.5712, 16.7381, 16.9718, 17.1722, 17.5725, 17.7395, 17.9731]
expected_m9 = [96, 99, 100, 102, 103, 105, 106, 108]

SECTIONS = [
    ("8ths (m1-3)", raw_8ths, expected_8ths),
    ("m5",         raw_m5,    expected_m5),
    ("m6",         raw_m6,    expected_m6),
    ("m9",         raw_m9,    expected_m9),
]

# Concatenate all sections into one sequence.
all_raw = np.array(raw_8ths + raw_m5 + raw_m6 + raw_m9)
all_expected = np.array(expected_8ths + expected_m5 + expected_m6 + expected_m9)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

def snap_to_grid(times: np.ndarray, duration: float, offset: float) -> np.ndarray:
    """Round each onset time to the nearest 16th-note grid position."""
    return np.round((times - offset) / duration).astype(int)


def fit_grid(times: np.ndarray, positions: np.ndarray) -> tuple[float, float]:
    """Fit t = pos * duration + offset via least-squares. Returns (duration, offset)."""
    # Design matrix: [positions, 1]
    A = np.column_stack([positions, np.ones(len(positions))])
    result, *_ = np.linalg.lstsq(A, times, rcond=None)
    duration, offset = result
    return float(duration), float(offset)


def global_fit(times: np.ndarray, initial_bpm: float, n_iter: int = 3) -> tuple[np.ndarray, float, float]:
    """
    Iterative snap-then-fit quantization.

    Returns (positions, fitted_duration, fitted_offset).
    """
    sixteenth_dur = 60.0 / initial_bpm / 4.0  # initial 16th-note duration
    offset = 0.0

    for i in range(n_iter):
        positions = snap_to_grid(times, sixteenth_dur, offset)
        sixteenth_dur, offset = fit_grid(times, positions)

    # Final snap with converged parameters.
    positions = snap_to_grid(times, sixteenth_dur, offset)
    return positions, sixteenth_dur, offset


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(predicted: np.ndarray, expected: np.ndarray, label: str) -> int:
    """Print per-note comparison and return number correct."""
    correct = int(np.sum(predicted == expected))
    total = len(expected)
    wrong_indices = np.where(predicted != expected)[0]
    if len(wrong_indices) > 0:
        details = ", ".join(
            f"idx {i}: got {predicted[i]} want {expected[i]}" for i in wrong_indices
        )
        print(f"  {label}: {correct}/{total}  WRONG: {details}")
    else:
        print(f"  {label}: {correct}/{total}  all correct")
    return correct


def run_experiment(initial_bpm: float, n_iter: int) -> None:
    """Run global fit and report accuracy."""
    positions, dur, off = global_fit(all_raw, initial_bpm, n_iter=n_iter)
    fitted_bpm = 60.0 / (dur * 4)

    print(f"  BPM_init={initial_bpm:5.1f}  iters={n_iter}  ->  "
          f"fitted_dur={dur*1000:.3f}ms  offset={off*1000:.2f}ms  "
          f"fitted_BPM={fitted_bpm:.3f}")

    # Evaluate per section.
    idx = 0
    total_correct = 0
    total_notes = 0
    for name, raw_sec, exp_sec in SECTIONS:
        n = len(exp_sec)
        pred_sec = positions[idx:idx + n]
        exp_arr = np.array(exp_sec)
        total_correct += evaluate(pred_sec, exp_arr, name)
        total_notes += n
        idx += n

    print(f"  OVERALL: {total_correct}/{total_notes} "
          f"({100.0 * total_correct / total_notes:.1f}%)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Global least-squares grid fit — onset quantization experiment")
    print("=" * 70)

    for bpm in [89.0, 89.5, 90.0]:
        for n_iter in [1, 2, 3]:
            run_experiment(bpm, n_iter)
        print("-" * 70)


if __name__ == "__main__":
    main()
