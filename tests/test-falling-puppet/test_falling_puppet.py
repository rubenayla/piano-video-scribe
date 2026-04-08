"""
Test: Ib — Mary's Theme (Puppet arrangement)
Video: Synthesia-style falling blocks, ~55s, 6/8 time, ~90 BPM, key of E minor.

This is a HARD test for the falling-blocks detector because:
- The video tempo drifts slightly (~89 vs 90 BPM), requiring adaptive tempo tracking.
- LH has repeating chords on every 8th note (bass-chord-chord pattern in 6/8) —
  the key-press detector misses these because the keys stay lit between re-attacks,
  but the falling blocks clearly show distinct blocks with gaps.
- RH has both 8th-note passages and 16th-note runs (measures 5, 9, etc.).
- Some blocks are short (~30px, dur≈0.15s) for the 16th-note runs.
- D# (leading tone in Em) must not be snapped to D.
- A few notes (~3) land on 16th positions due to video timing anomalies
  (single block detected ~80ms late, beyond quantizer threshold).

Known remaining issues (accepted):
- ~3 LH notes may land on 16th positions instead of 8th (video timing drift at edges)
- ~8 RH notes land on 16th positions (real 16th-note runs + a few anomalies)
- F#4 at measure 12 beat 34.25 is a persistent 16th-note anomaly from video timing
"""
import os
import subprocess
import sys
from collections import defaultdict

import mido
import pytest

TEST_DIR = os.path.dirname(__file__)
VIDEO = os.path.join(TEST_DIR, 'video.mp4')
SETTINGS = os.path.join(TEST_DIR, 'settings.json')
OUTPUT = os.path.join(TEST_DIR, 'output-video.mid')
GROUND_TRUTH = os.path.join(TEST_DIR, 'ground_truth.mid')


def _run_pipeline():
    """Run the falling-blocks pipeline and return the output MIDI path."""
    subprocess.check_call([
        sys.executable, 'pianovideoscribe.py',
        VIDEO, OUTPUT,
        '--settings', SETTINGS,
        '--detector', 'falling-blocks',
        '--bpm', '90',
    ])
    return OUTPUT


@pytest.fixture(scope='module')
def midi():
    from mido import MidiFile
    if not os.path.exists(VIDEO):
        pytest.skip('video.mp4 not found (symlink broken?)')
    _run_pipeline()
    return MidiFile(OUTPUT)


def _count_notes(track):
    abs_tick = 0
    notes = []
    for msg in track:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append((abs_tick, msg.note))
    return notes


class TestFallingPuppet:
    """Falling-blocks detector on Ib — Mary's Theme (Puppet)."""

    def test_rh_note_count(self, midi):
        """RH should have ~96 notes (94 in reference + a few from 16th runs)."""
        rh_notes = _count_notes(midi.tracks[0])
        assert 85 <= len(rh_notes) <= 110, f'RH notes: {len(rh_notes)}'

    def test_lh_note_count(self, midi):
        """LH should have ~247 notes (includes repeated chords the key-press
        detector misses). The old reference had only 143 because the key-press
        detector couldn't see re-attacks on held keys."""
        lh_notes = _count_notes(midi.tracks[1])
        assert 230 <= len(lh_notes) <= 265, f'LH notes: {len(lh_notes)}'

    def test_lh_mostly_eighths(self, midi):
        """LH notes should be almost entirely on the 8th-note grid.
        Allow up to 10 off-grid notes (video timing anomalies)."""
        lh_notes = _count_notes(midi.tracks[1])
        off_grid = sum(1 for tick, _ in lh_notes if tick % 480 != 0)
        assert off_grid <= 10, f'LH off 8th grid: {off_grid}'

    def test_lh_first_measure_pattern(self, midi):
        """First measure LH must be: E EGB EGB B EGB EGB (6 eighth notes).
        This is the key pattern the key-press detector gets wrong (misses
        the repeated chords at 8ths 3 and 6)."""
        lh_notes = _count_notes(midi.tracks[1])
        first_measure = [(tick, note) for tick, note in lh_notes if tick < 960 * 3]

        # Group by tick position
        from collections import defaultdict
        by_tick = defaultdict(set)
        nn = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for tick, note in first_measure:
            by_tick[tick].add(nn[note % 12])

        ticks_sorted = sorted(by_tick.keys())
        assert len(ticks_sorted) == 6, f'Expected 6 8th-note positions, got {len(ticks_sorted)}'

        # All should be on 8th grid
        for tick in ticks_sorted:
            assert tick % 480 == 0, f'tick {tick} not on 8th grid'

        # Check the pattern: E, EGB, EGB, B, EGB, EGB
        patterns = [by_tick[t] for t in ticks_sorted]
        assert 'E' in patterns[0], f'8th 1 should have E, got {patterns[0]}'
        assert patterns[1] >= {'E', 'G', 'B'}, f'8th 2 should have EGB, got {patterns[1]}'
        assert patterns[2] >= {'E', 'G', 'B'}, f'8th 3 should have EGB, got {patterns[2]}'
        assert 'B' in patterns[3], f'8th 4 should have B, got {patterns[3]}'
        assert patterns[4] >= {'E', 'G', 'B'}, f'8th 5 should have EGB, got {patterns[4]}'
        assert patterns[5] >= {'E', 'G', 'B'}, f'8th 6 should have EGB, got {patterns[5]}'

    def test_lh_has_d_sharp(self, midi):
        """D# (leading tone in Em) must be present, not snapped to D."""
        lh_notes = _count_notes(midi.tracks[1])
        d_sharp_count = sum(1 for _, note in lh_notes if note % 12 == 3)
        assert d_sharp_count >= 10, f'D# count: {d_sharp_count} (expected ~20+)'

    def test_rh_has_sixteenth_runs(self, midi):
        """RH should contain some 16th-note runs (240-tick gaps)."""
        rh_notes = _count_notes(midi.tracks[0])
        sixteenth_gaps = 0
        for i in range(1, len(rh_notes)):
            gap = rh_notes[i][0] - rh_notes[i - 1][0]
            if gap == 240:
                sixteenth_gaps += 1
        assert sixteenth_gaps >= 4, f'16th gaps: {sixteenth_gaps} (expected some runs)'

    def test_rh_starts_at_beat_7(self, midi):
        """RH first note should be at beat 7 (start_beat=7.0 in settings)."""
        rh_notes = _count_notes(midi.tracks[0])
        first_tick = rh_notes[0][0]
        first_beat = first_tick / 960
        assert 5.5 <= first_beat <= 7.5, f'RH starts at beat {first_beat}'


# ============================================================================
# Ground truth note-by-note comparison
# GT confirmed by user (listened to full piece) on 2026-03-31.
# Every note must match exactly — pitch AND tick position.
# ============================================================================

def _extract_note_events(track):
    """Extract (tick, midi_note) for all note-on events, sorted by (tick, pitch)."""
    abs_tick = 0
    events = []
    for msg in track:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            events.append((abs_tick, msg.note))
    events.sort()  # sort by (tick, pitch) — chord note order is arbitrary
    return events


class TestPuppetGroundTruth:
    """Note-by-note ground truth comparison.

    The GT MIDI was confirmed correct by the user on 2026-03-31.
    Any deviation means a regression — no tolerance, every note must match.
    """

    @pytest.fixture(autouse=True, scope='class')
    def setup(self, request, midi):
        if not os.path.exists(GROUND_TRUTH):
            pytest.skip('ground_truth.mid not found')
        gt = mido.MidiFile(GROUND_TRUTH)
        # Find note tracks in GT
        gt_note_tracks = [i for i, t in enumerate(gt.tracks)
                          if any(m.type == 'note_on' and m.velocity > 0 for m in t)]
        request.cls.gt_rh = _extract_note_events(gt.tracks[gt_note_tracks[0]])
        request.cls.gt_lh = _extract_note_events(gt.tracks[gt_note_tracks[1]])
        request.cls.out_rh = _extract_note_events(midi.tracks[0])
        request.cls.out_lh = _extract_note_events(midi.tracks[1])

    def test_rh_exact_note_count(self):
        """RH note count must match GT exactly."""
        assert len(self.out_rh) == len(self.gt_rh), \
            f'RH: output has {len(self.out_rh)} notes, GT has {len(self.gt_rh)}'

    def test_lh_exact_note_count(self):
        """LH note count must match GT exactly."""
        assert len(self.out_lh) == len(self.gt_lh), \
            f'LH: output has {len(self.out_lh)} notes, GT has {len(self.gt_lh)}'

    def test_rh_notes_match_gt(self):
        """Every RH note must match GT in both pitch and tick position."""
        mismatches = []
        for i, (gt_ev, out_ev) in enumerate(zip(self.gt_rh, self.out_rh)):
            gt_tick, gt_note = gt_ev
            out_tick, out_note = out_ev
            if gt_tick != out_tick or gt_note != out_note:
                nn = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                mismatches.append(
                    f'  note {i}: GT=({gt_tick}, {nn[gt_note%12]}{gt_note//12-1}) '
                    f'vs OUT=({out_tick}, {nn[out_note%12]}{out_note//12-1})')
        assert not mismatches, \
            f'RH has {len(mismatches)} mismatches:\n' + '\n'.join(mismatches[:20])

    def test_lh_notes_match_gt(self):
        """Every LH note must match GT in both pitch and tick position."""
        mismatches = []
        for i, (gt_ev, out_ev) in enumerate(zip(self.gt_lh, self.out_lh)):
            gt_tick, gt_note = gt_ev
            out_tick, out_note = out_ev
            if gt_tick != out_tick or gt_note != out_note:
                nn = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                mismatches.append(
                    f'  note {i}: GT=({gt_tick}, {nn[gt_note%12]}{gt_note//12-1}) '
                    f'vs OUT=({out_tick}, {nn[out_note%12]}{out_note//12-1})')
        assert not mismatches, \
            f'LH has {len(mismatches)} mismatches:\n' + '\n'.join(mismatches[:20])
