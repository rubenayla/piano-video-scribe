"""Phase-locked loop quantizer for snapping raw note onsets to a 16th-note grid.

Problem: raw onset times from a Synthesia video at ~90 BPM have +-33ms jitter
from the ~30fps frame rate. We maintain a running phase estimate and snap each
onset to the nearest 16th-note position on the phase-adjusted grid, then update
the phase via exponential moving average.

At 90 BPM one beat = 60/90 = 0.6667s, one 16th = 0.1667s.
"""

from __future__ import annotations

import dataclasses


# ── Config ──────────────────────────────────────────────────────────────────

BEATS_PER_MINUTE = 90.0
BEAT_DURATION = 60.0 / BEATS_PER_MINUTE          # 0.6667s
SIXTEENTH_DURATION = BEAT_DURATION / 4.0          # 0.1667s


# ── Quantizer ───────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PLLQuantizer:
    """Phase-locked loop quantizer.

    Maintains a phase offset that is updated with each new onset via an
    exponential moving average.  The phase offset represents the estimated
    difference between the ideal grid origin and the actual performance.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor.  Larger = faster adaptation, more noise.
    sixteenth : float
        Duration of one 16th note in seconds.
    """

    alpha: float = 0.1
    sixteenth: float = SIXTEENTH_DURATION
    phase_offset: float = 0.0  # running EMA of phase error
    _initialized: bool = False

    def quantize(self, onsets: list[float]) -> list[int]:
        """Snap a sequence of raw onset times to 16th-note grid indices."""
        self.phase_offset = 0.0
        self._initialized = False
        result: list[int] = []

        for t in onsets:
            # Position on the grid (fractional 16th-note index) adjusted by phase
            grid_pos = (t - self.phase_offset) / self.sixteenth
            idx = round(grid_pos)

            # Phase error: how far the onset was from the snapped grid point
            ideal_time = idx * self.sixteenth + self.phase_offset
            error = t - ideal_time

            # Update phase offset via EMA
            if not self._initialized:
                self.phase_offset = error
                self._initialized = True
            else:
                self.phase_offset += self.alpha * error

            result.append(idx)

        return result


# ── Test data ───────────────────────────────────────────────────────────────

# fmt: off
RAW_8THS     = [0.0000, 0.3337, 0.7007, 1.0344, 1.3680, 1.7017, 2.0354, 2.3690, 2.7027, 3.0364, 3.3700, 3.7371]
EXPECTED_8THS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

RAW_M5       = [8.0374, 8.5712, 8.7381, 8.9718, 9.3055, 9.6725, 9.8395]
EXPECTED_M5  = [48, 51, 52, 54, 56, 58, 59]

RAW_M6       = [10.0767, 10.4771, 10.7441, 11.0777, 11.4448, 11.7784]
EXPECTED_M6  = [60, 62, 64, 66, 68, 70]

RAW_M9       = [16.0374, 16.5712, 16.7381, 16.9718, 17.1722, 17.5725, 17.7395, 17.9731]
EXPECTED_M9  = [96, 99, 100, 102, 103, 105, 106, 108]
# fmt: on

SECTIONS = [
    ("8ths",  RAW_8THS,  EXPECTED_8THS),
    ("m5",    RAW_M5,    EXPECTED_M5),
    ("m6",    RAW_M6,    EXPECTED_M6),
    ("m9",    RAW_M9,    EXPECTED_M9),
]


# ── Runner ──────────────────────────────────────────────────────────────────

def run_test(alpha: float) -> None:
    """Run the quantizer on the full concatenated sequence, then score per section."""
    # Build the single concatenated sequence and its expected output
    all_raw: list[float] = []
    all_expected: list[int] = []
    section_slices: list[tuple[str, int, int]] = []  # (name, start, end)

    for name, raw, expected in SECTIONS:
        start = len(all_raw)
        all_raw.extend(raw)
        all_expected.extend(expected)
        end = len(all_raw)
        section_slices.append((name, start, end))

    # Quantize the whole thing in one pass
    q = PLLQuantizer(alpha=alpha)
    all_result = q.quantize(all_raw)

    # Score per section
    print(f"\n{'=' * 60}")
    print(f"  alpha = {alpha}")
    print(f"{'=' * 60}")

    total_correct = 0
    total_count = 0

    for name, start, end in section_slices:
        result = all_result[start:end]
        expected = all_expected[start:end]
        correct = sum(r == e for r, e in zip(result, expected))
        count = len(expected)
        total_correct += correct
        total_count += count
        ok = "PASS" if correct == count else "FAIL"
        print(f"  {name:6s}  {correct}/{count}  {ok}")
        if correct != count:
            for i, (r, e, t) in enumerate(zip(result, expected, all_raw[start:end])):
                marker = " <-- MISS" if r != e else ""
                print(f"          onset={t:.4f}  got={r:3d}  exp={e:3d}{marker}")

    pct = 100.0 * total_correct / total_count if total_count else 0.0
    overall_ok = "PASS" if total_correct == total_count else "FAIL"
    print(f"  {'TOTAL':6s}  {total_correct}/{total_count}  ({pct:.1f}%)  {overall_ok}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for alpha in [0.05, 0.1, 0.2, 0.3]:
        run_test(alpha)
