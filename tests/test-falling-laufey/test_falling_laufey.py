"""
Test: From the Start — Laufey
Video: Synthesia-style falling blocks, ~185s, 4/4 time, 149.9 BPM, key of Db major.

Ground truth confirmed by user on 2026-03-31 (listened + hand-corrected in MuseScore).

IMPORTANT LIMITATIONS OF THIS GROUND TRUTH:
- Measures 1–104: fully verified, tests check exact note match.
- Measures 105–108: ritardando section. The video slows down but the GT was
  written at a constant (slower) tempo — the ritardando is NOT encoded as
  tempo changes. Arpeggios in the ending are approximate (the pipeline doesn't
  support arpeggio detection yet). Tests only check up to m104.
- The .mscz file (ground_truth.mscz) is the MuseScore source used to produce
  the GT MIDI. It can be re-exported if needed.
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

# Only check notes before m105 — m105+ has ritardando and approximate arpeggios
GT_TPB = 960
M105_TICK = 104 * 4 * GT_TPB  # 399360


def _run_pipeline():
    """Run the falling-blocks pipeline and return the output MIDI path."""
    subprocess.check_call([
        sys.executable, 'pianovideoscribe.py',
        VIDEO, OUTPUT,
        '--settings', SETTINGS,
    ])
    return OUTPUT


@pytest.fixture(scope='module')
def midi():
    if not os.path.exists(VIDEO):
        pytest.skip('video.mp4 not found (symlink broken?)')
    _run_pipeline()
    return mido.MidiFile(OUTPUT)


def _extract_note_events(track, max_tick=None):
    """Extract (tick, midi_note) for all note-on events, optionally up to max_tick."""
    abs_tick = 0
    events = []
    for msg in track:
        abs_tick += msg.time
        if max_tick is not None and abs_tick >= max_tick:
            break
        if msg.type == 'note_on' and msg.velocity > 0:
            events.append((abs_tick, msg.note))
    return events


class TestLaufeyGroundTruth:
    """Note-by-note ground truth comparison for measures 1–104.

    The GT MIDI was hand-corrected by the user on 2026-03-31.
    Measures 105–108 are excluded because:
    - The video has a ritardando that is not encoded as tempo changes in the GT.
    - The ending has arpeggios that the pipeline doesn't support yet.
    - The user approximated these measures but they are not exact.
    """

    @pytest.fixture(autouse=True, scope='class')
    def setup(self, request, midi):
        if not os.path.exists(GROUND_TRUTH):
            pytest.skip('ground_truth.mid not found')
        gt = mido.MidiFile(GROUND_TRUTH)
        gt_note_tracks = [i for i, t in enumerate(gt.tracks)
                          if any(m.type == 'note_on' and m.velocity > 0 for m in t)]
        request.cls.gt_rh = _extract_note_events(gt.tracks[gt_note_tracks[0]], M105_TICK)
        request.cls.gt_lh = _extract_note_events(gt.tracks[gt_note_tracks[1]], M105_TICK)
        request.cls.out_rh = _extract_note_events(midi.tracks[0], M105_TICK)
        request.cls.out_lh = _extract_note_events(midi.tracks[1], M105_TICK)

    def test_rh_note_count(self):
        """RH note count (m1–104) must match GT exactly."""
        assert len(self.out_rh) == len(self.gt_rh), \
            f'RH: output has {len(self.out_rh)} notes, GT has {len(self.gt_rh)}'

    def test_lh_note_count(self):
        """LH note count (m1–104) must match GT exactly."""
        assert len(self.out_lh) == len(self.gt_lh), \
            f'LH: output has {len(self.out_lh)} notes, GT has {len(self.gt_lh)}'

    def test_rh_notes_match_gt(self):
        """Every RH note (m1–104) must match GT in both pitch and tick."""
        mismatches = []
        for i, (gt_ev, out_ev) in enumerate(zip(self.gt_rh, self.out_rh)):
            gt_tick, gt_note = gt_ev
            out_tick, out_note = out_ev
            if gt_tick != out_tick or gt_note != out_note:
                nn = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                m_gt = gt_tick // (4 * GT_TPB) + 1
                m_out = out_tick // (4 * GT_TPB) + 1
                mismatches.append(
                    f'  note {i}: GT=(m{m_gt}, {gt_tick}, {nn[gt_note%12]}{gt_note//12-1}) '
                    f'vs OUT=(m{m_out}, {out_tick}, {nn[out_note%12]}{out_note//12-1})')
        assert not mismatches, \
            f'RH has {len(mismatches)} mismatches (m1–104):\n' + '\n'.join(mismatches[:20])

    def test_lh_notes_match_gt(self):
        """Every LH note (m1–104) must match GT in both pitch and tick."""
        mismatches = []
        for i, (gt_ev, out_ev) in enumerate(zip(self.gt_lh, self.out_lh)):
            gt_tick, gt_note = gt_ev
            out_tick, out_note = out_ev
            if gt_tick != out_tick or gt_note != out_note:
                nn = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                m_gt = gt_tick // (4 * GT_TPB) + 1
                m_out = out_tick // (4 * GT_TPB) + 1
                mismatches.append(
                    f'  note {i}: GT=(m{m_gt}, {gt_tick}, {nn[gt_note%12]}{gt_note//12-1}) '
                    f'vs OUT=(m{m_out}, {out_tick}, {nn[out_note%12]}{out_note//12-1})')
        assert not mismatches, \
            f'LH has {len(mismatches)} mismatches (m1–104):\n' + '\n'.join(mismatches[:20])
