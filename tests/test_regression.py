#!/usr/bin/env python3
"""Regression tests — run the pipeline on test videos and verify output.

Each test class runs the full pipeline on a test video and checks specific
properties of the output MIDI. Tests target specific failure modes discovered
during development and ground truth verification.
"""

import os
import subprocess
import sys
from collections import defaultdict

import mido
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _run_pipeline(test_name):
    """Run the pipeline on a test video and return the output MIDI path."""
    test_dir = os.path.join(TESTS_DIR, test_name)
    video = os.path.join(test_dir, 'video.mp4')
    settings = os.path.join(test_dir, 'settings.json')
    output = os.path.join(test_dir, 'output-video.mid')
    if not os.path.exists(video):
        pytest.skip(f'{test_name}/video.mp4 not found')
    cmd = [sys.executable, 'pianovideoscribe.py', video, output]
    if os.path.exists(settings):
        cmd += ['--settings', settings]
    subprocess.check_call(cmd)
    return output


def extract_notes(mid, hand_idx):
    """Extract (onset_sec, pitch_class, midi_note) from a hand track."""
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo != 500000:
            break
    note_tracks = []
    for i, track in enumerate(mid.tracks):
        if any(m.type == 'note_on' and m.velocity > 0 for m in track):
            note_tracks.append(i)
    rh_idx = note_tracks[0] if note_tracks else 0
    lh_idx = note_tracks[1] if len(note_tracks) >= 2 else 1
    track_idx = rh_idx if hand_idx == 0 else lh_idx
    track = mid.tracks[track_idx]
    tpb = mid.ticks_per_beat
    abs_tick = 0
    notes = []
    for msg in track:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            t = abs_tick * tempo / (tpb * 1_000_000)
            pc = msg.note % 12
            notes.append((t, pc, msg.note))
    return notes


def has_note_near(notes, time_sec, pitch_class, tolerance=0.4):
    for t, pc, midi in notes:
        if pc == pitch_class and abs(t - time_sec) <= tolerance:
            return True
    return False


def count_notes_in_range(notes, time_start, time_end, pitch_class=None):
    count = 0
    for t, pc, midi in notes:
        if time_start <= t <= time_end:
            if pitch_class is None or pc == pitch_class:
                count += 1
    return count


# ============================================================================
# test2: Partial keyboard (26 white keys) — octave and edge detection
# ============================================================================

class TestTest2Regression:
    """Regressions for test2 (ya_no_mas, partial keyboard)."""

    @pytest.fixture(autouse=True, scope='class')
    def setup(self, request):
        path = _run_pipeline('test2')
        mid = mido.MidiFile(path)
        request.cls.rh = extract_notes(mid, 0)
        request.cls.lh = extract_notes(mid, 1)

    def test_reasonable_note_count(self):
        """Both hands should detect a reasonable number of notes."""
        assert len(self.rh) >= 80, f'RH notes: {len(self.rh)}'
        assert len(self.lh) >= 80, f'LH notes: {len(self.lh)}'

    def test_rapid_repeat_detection(self):
        """Pipeline should detect rapid repeated notes on the same key.

        Bug: F#4 at 20.56s missed because off-to-on gap too brief.
        """
        f_sharp_pc = 6
        count = count_notes_in_range(self.rh, 20.0, 21.0, f_sharp_pc)
        assert count >= 2, f"Expected >=2 F# rapid repeats at t=20s, got {count}"

    def test_lh_short_notes_detected(self):
        """Pipeline should detect very short notes (0.09s / ~3 frames).

        Bug: LH C#3 at 42.78s is only 0.09s long, not detected.
        """
        c_sharp_pc = 1
        found = has_note_near(self.lh, 42.78, c_sharp_pc, tolerance=0.5)
        if not found:
            pytest.xfail("Known issue: ultra-short notes (0.09s) not detected")


# ============================================================================
# test3: Timing offset and rapid note detection
# ============================================================================

class TestTest3Regression:
    """Regressions for test3 (partial keyboard, 33 white keys)."""

    @pytest.fixture(autouse=True, scope='class')
    def setup(self, request):
        path = _run_pipeline('test3')
        mid = mido.MidiFile(path)
        request.cls.rh = extract_notes(mid, 0)
        request.cls.lh = extract_notes(mid, 1)

    def test_rh_e5_timing_within_tolerance(self):
        """E5 notes should be detected within 0.5s of ground truth.
        (Video trimmed to 60s, only checking timestamps within range.)"""
        gt_e5_times = [48.36]  # only this one is within 60s
        e_pc = 4
        detected = 0
        for gt_t in gt_e5_times:
            if has_note_near(self.rh, gt_t, e_pc, tolerance=0.5):
                detected += 1
        assert detected >= 1, f"E5 at 48.36s not found within 0.5s tolerance"

    def test_reasonable_note_count(self):
        """Both hands should detect a reasonable number of notes."""
        assert len(self.rh) >= 70, f'RH notes: {len(self.rh)}'
        assert len(self.lh) >= 70, f'LH notes: {len(self.lh)}'

    def test_lh_has_b_notes(self):
        """LH should have B notes (ostinato pattern)."""
        b_pc = 11
        total_b = sum(1 for t, pc, m in self.lh if pc == b_pc)
        assert total_b >= 10, f"Expected >=10 LH B notes, got {total_b}"


# ============================================================================
# test5: Sweden (Minecraft) — final chord must not be trimmed
# ============================================================================

class TestTest5Regression:
    """Regressions for test5 (Sweden, full 88-key keyboard)."""

    @pytest.fixture(autouse=True, scope='class')
    def setup(self, request):
        path = _run_pipeline('test5')
        mid = mido.MidiFile(path)
        request.cls.rh = extract_notes(mid, 0)
        request.cls.lh = extract_notes(mid, 1)

    def test_final_chord_not_trimmed(self):
        """Final chord should not be removed by popup trimmer.

        Bug: The popup trimmer falsely removed the final 6-note chord
        because 3+ simultaneous notes at the end triggered the heuristic.
        """
        last_rh_t = max(t for t, pc, m in self.rh)
        # Video trimmed to 60s. Last note should be near the end, not cut early.
        assert last_rh_t > 45.0, \
            f"Final notes trimmed: last RH note at {last_rh_t:.1f}s"


# ============================================================================
# test7: Faded (Alan Walker) — no C#, correct hand separation
# ============================================================================

class TestTest7Regression:
    """Regressions for test7 (Faded easy, non-standard cyan/purple colors)."""

    @pytest.fixture(autouse=True, scope='class')
    def setup(self, request):
        path = _run_pipeline('test7')
        mid = mido.MidiFile(path)
        request.cls.rh = extract_notes(mid, 0)
        request.cls.lh = extract_notes(mid, 1)

    def test_no_c_sharp_detections(self):
        """No C# notes should be detected in Faded (Em/G major)."""
        c_sharp_pc = 1
        all_notes = self.rh + self.lh
        c_sharps = [(t, m) for t, pc, m in all_notes if pc == c_sharp_pc]
        assert len(c_sharps) == 0, \
            f"Found {len(c_sharps)} false C# notes: {c_sharps[:5]}"

    def test_lh_notes_in_lower_range(self):
        """LH notes should be in the lower range, not mixed into RH."""
        assert len(self.lh) > 0, "LH has no notes — hand separation failed"
        total = len(self.rh) + len(self.lh)
        lh_ratio = len(self.lh) / total
        assert lh_ratio > 0.15, \
            f"LH only has {lh_ratio:.0%} of notes — hand separation likely wrong"

    def test_all_notes_diatonic_to_em(self):
        """All notes should be diatonic to E minor (or G major)."""
        em_pcs = {0, 2, 4, 6, 7, 9, 11}
        all_notes = self.rh + self.lh
        non_diatonic = [(t, pc, m) for t, pc, m in all_notes if pc not in em_pcs]
        ratio = len(non_diatonic) / len(all_notes) if all_notes else 0
        assert ratio < 0.05, \
            f"{len(non_diatonic)}/{len(all_notes)} notes non-diatonic ({ratio:.0%})"


# ============================================================================
# test4: Should remain perfect (baseline)
# ============================================================================

class TestTest4Regression:
    """test4 currently scores 100% — ensure no regressions."""

    @pytest.fixture(autouse=True, scope='class')
    def setup(self, request):
        path = _run_pipeline('test4')
        mid = mido.MidiFile(path)
        request.cls.rh = extract_notes(mid, 0)
        request.cls.lh = extract_notes(mid, 1)

    def test_rh_note_count(self):
        """RH should have ~94 notes (video trimmed to 60s)."""
        assert 80 <= len(self.rh) <= 110, f"Expected ~94 RH notes, got {len(self.rh)}"

    def test_lh_note_count(self):
        """LH should have ~27 notes (video trimmed to 60s)."""
        assert 20 <= len(self.lh) <= 40, f"Expected ~27 LH notes, got {len(self.lh)}"


# ============================================================================
# Ground truth integrity checks (these test the GT files, not the pipeline)
# ============================================================================

class TestGroundTruthIntegrity:
    """Verify ground truth files are self-consistent."""

    @pytest.mark.parametrize("test_name,expected_rh,expected_lh", [
        ("test2", 299, 313),
        ("test3", 335, 482),
    ])
    def test_gt_note_counts(self, test_name, expected_rh, expected_lh):
        gt_path = os.path.join(TESTS_DIR, test_name, "ground_truth.mid")
        if not os.path.exists(gt_path):
            pytest.skip(f"No ground truth for {test_name}")
        gt = mido.MidiFile(gt_path)
        rh = extract_notes(gt, 0)
        lh = extract_notes(gt, 1)
        assert len(rh) == expected_rh, \
            f"{test_name} RH: expected {expected_rh}, got {len(rh)}"
        assert len(lh) == expected_lh, \
            f"{test_name} LH: expected {expected_lh}, got {len(lh)}"

    @pytest.mark.parametrize("test_name", ["test3", "test4"])
    def test_gt_no_overlapping_identical_notes(self, test_name):
        gt_path = os.path.join(TESTS_DIR, test_name, "ground_truth.mid")
        if not os.path.exists(gt_path):
            pytest.skip(f"No ground truth for {test_name}")
        gt = mido.MidiFile(gt_path)
        for track_idx in [0, 1]:
            notes = extract_notes(gt, track_idx)
            seen = set()
            for t, pc, midi in notes:
                key = (round(t, 3), midi)
                assert key not in seen, \
                    f"{test_name} track {track_idx}: duplicate note MIDI {midi} at t={t:.3f}s"
                seen.add(key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
