#!/usr/bin/env python3
"""Interval-aware quantizer for Synthesia video note onsets.

Snaps raw onset times to a 16th-note grid by classifying each inter-onset
interval as N sixteenths, then building grid positions incrementally.

Two-phase approach:
  1. **Viterbi DP** — finds the globally optimal path that minimises a
     weighted combination of per-interval quantisation error and absolute-
     position error.  The absolute term anchors the path against long-range
     drift but is kept lightweight so it never overrides clear interval
     evidence.
  2. **Refinement pass** — any interval whose ratio falls in the ambiguous
     zone (fractional part 0.25–0.75) is tested: would flipping floor<->ceil
     reduce the total absolute error for the remaining sequence?  This fixes
     the rare cases where cumulative DP cost masks a locally wrong choice.

Usage:  python quantize_interval_aware.py
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Core quantiser
# ---------------------------------------------------------------------------

def quantize_interval_aware(
    onsets: list[float],
    sixteenth_dur: float,
    *,
    start_pos: int = 0,
    abs_weight: float = 0.1,
) -> list[int]:
    """Return 16th-note grid positions for *onsets*.

    Parameters
    ----------
    onsets : list[float]
        Absolute onset times in seconds (must be sorted).
    sixteenth_dur : float
        Duration of one sixteenth note in seconds.
    start_pos : int
        Grid position assigned to the first onset.
    abs_weight : float
        Weight of the absolute-position cost relative to the interval cost
        in the Viterbi DP.  Values around 0.05–0.3 work well for ~30 fps
        video jitter at typical tempos.

    Returns
    -------
    list[int]
        One grid position per onset.
    """
    if not onsets:
        return []

    n = len(onsets)
    s = sixteenth_dur

    # ------------------------------------------------------------------
    # Phase 1: Viterbi DP
    #
    # State = grid position.
    # Transition cost = (actual_interval - k * s)^2
    #                 + abs_weight * (onset - pos * s)^2
    # ------------------------------------------------------------------
    dp: list[dict[int, tuple[float, int | None]]] = [dict() for _ in range(n)]
    dp[0][start_pos] = (0.0, None)

    for i in range(n - 1):
        interval = onsets[i + 1] - onsets[i]
        ratio = interval / s
        centre = round(ratio)

        # Candidate steps: centre ± 2, plus floor/ceil
        candidates: set[int] = set()
        for c in range(max(1, centre - 2), centre + 3):
            candidates.add(c)
        candidates.add(max(1, math.floor(ratio)))
        candidates.add(math.ceil(ratio))

        for pos, (cost, _) in dp[i].items():
            for k in candidates:
                new_pos = pos + k
                interval_err = (interval - k * s) ** 2
                abs_err = (onsets[i + 1] - new_pos * s) ** 2
                new_cost = cost + interval_err + abs_weight * abs_err

                if new_pos not in dp[i + 1] or dp[i + 1][new_pos][0] > new_cost:
                    dp[i + 1][new_pos] = (new_cost, pos)

    # Back-track
    best_final = min(dp[n - 1], key=lambda p: dp[n - 1][p][0])
    positions: list[int] = []
    pos: int | None = best_final
    for i in range(n - 1, -1, -1):
        assert pos is not None
        positions.append(pos)
        pos = dp[i][pos][1]
    positions.reverse()

    # ------------------------------------------------------------------
    # Phase 2: Refinement of ambiguous intervals
    #
    # For each interval whose ratio falls in the ambiguous zone, check
    # whether flipping floor<->ceil would reduce the total absolute error
    # for the rest of the sequence.  Iterate until stable.
    # ------------------------------------------------------------------
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            interval = onsets[i] - onsets[i - 1]
            ratio = interval / s
            frac = ratio - math.floor(ratio)
            if not (0.25 < frac < 0.75):
                continue

            current_delta = positions[i] - positions[i - 1]
            lo = max(1, math.floor(ratio))
            hi = lo + 1
            other_delta = hi if current_delta == lo else lo
            shift = other_delta - current_delta

            # Would shifting positions[i:] by *shift* reduce abs error?
            curr_abs = sum(
                (onsets[j] - positions[j] * s) ** 2 for j in range(i, n)
            )
            flip_abs = sum(
                (onsets[j] - (positions[j] + shift) * s) ** 2 for j in range(i, n)
            )
            if flip_abs < curr_abs:
                for j in range(i, n):
                    positions[j] += shift
                changed = True

    return positions


# ---------------------------------------------------------------------------
# Test data — raw onsets extracted from a ~90 BPM Synthesia video
# ---------------------------------------------------------------------------

# Eighths (measures 1–2): all 8th notes = 2 sixteenths apart
raw_8ths = [
    0.0000, 0.3337, 0.7007, 1.0344, 1.3680,
    1.7017, 2.0354, 2.3690, 2.7027, 3.0364,
    3.3700, 3.7371,
]
expected_8ths = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

# Measure 5: mixed dotted-8th, 16th, 8th patterns
raw_m5 = [8.0374, 8.5712, 8.7381, 8.9718, 9.3055, 9.6725, 9.8395]
expected_m5 = [48, 51, 52, 54, 56, 58, 59]

# Measure 6: all 8th notes
raw_m6 = [10.0767, 10.4771, 10.7441, 11.0777, 11.4448, 11.7784]
expected_m6 = [60, 62, 64, 66, 68, 70]

# Measure 9: mixed dotted-8th, 16th, 8th patterns with some 16ths
raw_m9 = [
    16.0374, 16.5712, 16.7381, 16.9718,
    17.1722, 17.5725, 17.7395, 17.9731,
]
expected_m9 = [96, 99, 100, 102, 103, 105, 106, 108]

SECTIONS = [
    ("8ths (m1-2)", raw_8ths, expected_8ths),
    ("m5",         raw_m5,    expected_m5),
    ("m6",         raw_m6,    expected_m6),
    ("m9",         raw_m9,    expected_m9),
]


def main() -> None:
    # Derive sixteenth duration from the first two 8th-note onsets
    # (known to be exactly 2 sixteenths apart).
    sixteenth_dur = (raw_8ths[1] - raw_8ths[0]) / 2
    print(f"Sixteenth-note duration: {sixteenth_dur:.4f} s  "
          f"(BPM ~ {60 / (sixteenth_dur * 4):.1f})\n")

    # Build one concatenated sequence
    all_raw = raw_8ths + raw_m5 + raw_m6 + raw_m9
    all_expected = expected_8ths + expected_m5 + expected_m6 + expected_m9

    # Quantise
    all_predicted = quantize_interval_aware(all_raw, sixteenth_dur)

    # Report per section
    offset = 0
    total_correct = 0
    total_notes = 0
    for name, raw, expected in SECTIONS:
        n = len(raw)
        predicted = all_predicted[offset: offset + n]
        correct = sum(p == e for p, e in zip(predicted, expected))
        total_correct += correct
        total_notes += n
        status = "PASS" if correct == n else "FAIL"
        print(f"[{status}] {name:12s}  {correct}/{n} correct")
        if correct < n:
            for i, (p, e) in enumerate(zip(predicted, expected)):
                marker = " <-- WRONG" if p != e else ""
                print(f"       note {i}: predicted={p:3d}  expected={e:3d}{marker}")
        offset += n

    pct = 100 * total_correct / total_notes
    print(f"\nOverall: {total_correct}/{total_notes} correct ({pct:.1f}%)")

    if total_correct == total_notes:
        print("\nAll notes quantised correctly.")
    else:
        print(f"\n{total_notes - total_correct} note(s) quantised incorrectly.")


if __name__ == "__main__":
    main()
