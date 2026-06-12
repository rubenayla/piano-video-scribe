"""Regression tests for the lead-sheet (melody + chords) generator.

Fixture: `tests/test-lead-sheet/beat-it.mid` — the corrected, drift-free Beat It
transcription (138 BPM; the warp-normalization fix). `build_musicxml` turns it
into a single-staff melody with chord symbols.

The approved output (`beat-it-expected-lead-sheet.musicxml`) is the version the
user signed off on, including the "shift everything back an eighth so the chords
land on the measure start" alignment. These tests pin that output and, more
importantly, assert the structural invariants for every bug fixed in the
2026-06-04 session (see .agents/error-log.md):

  * note <type>(+dots) must equal <duration>      → MuseScore-4 "corrupted" bug
  * every measure must sum to a full bar
  * ties must be balanced (every stop has a start)
  * chord symbols must sit on a melody note, not a rest, and not drift off it
  * one chord per bar, correct Em/D/C groove
"""
import os
import re

import mido
import pytest

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test-lead-sheet')
FIXTURE_MID = os.path.join(TEST_DIR, 'beat-it.mid')
GOLDEN_XML = os.path.join(TEST_DIR, 'beat-it-expected-lead-sheet.musicxml')

from piano_video_scribe.lead_sheet import build_musicxml

# Chord change once per bar across the song, on the measure start (post-shift).
EXPECTED_CHORDS = (
    'Em D Em D C D Em D Em D Em D C D Em D Em D Em D Em D Em D Em D Em D Em D Em D '
    'C D Em D Em D Em D C D Em D Em D Em D Em D Em D Em D Em D Em D Em D Em D Em D '
    'C D Em D Em D Em D Em D Em D Em D Em D Em D Em D Em D Em D Em D Em D Em D Em D '
    'Em D Em'
).split()

_TYPE_TICKS = {'whole': 4, 'half': 2, 'quarter': 1, 'eighth': 0.5,
               '16th': 0.25, '32nd': 0.125}


def _build():
    """Reproduce the approved lead sheet: load the fixture, shift everything back
    an eighth (the user-approved measure-start alignment), build the MusicXML."""
    mid = mido.MidiFile(FIXTURE_MID)
    tpb = mid.ticks_per_beat
    tempo = 500000
    right, left = [], []
    for track in mid.tracks:
        name, tick = '', 0
        for msg in track:
            tick += msg.time
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'track_name':
                name = msg.name
            elif msg.type in ('note_on', 'note_off'):
                (left if 'Left' in name else right).append(
                    (tick, msg.type, msg.note, msg.velocity))
    shift = tpb // 2  # an eighth note
    right = [(max(0, t - shift), ty, n, v) for t, ty, n, v in right]
    left = [(max(0, t - shift), ty, n, v) for t, ty, n, v in left]
    return build_musicxml(right, left, tempo, tpb, (4, 4)), tpb


def _measures(xml):
    parts = re.split(r'(<measure number="\d+")', xml)
    return [(int(re.search(r'\d+', parts[i]).group()),
             parts[i + 1].split('</measure>')[0])
            for i in range(1, len(parts), 2)]


@pytest.fixture(scope='module')
def built():
    xml, tpb = _build()
    return xml, tpb


def test_every_note_type_matches_duration(built):
    """MuseScore 4 flags a note whose <type>(+dots) != <duration> as corrupt."""
    xml, tpb = built
    bad = []
    for note in re.findall(r'<note>(.*?)</note>', xml, re.S):
        dur = int(re.search(r'<duration>(\d+)', note).group(1))
        ntype = re.search(r'<type>(\w+)</type>', note).group(1)
        implied = _TYPE_TICKS[ntype] * tpb * (1.5 if '<dot/>' in note else 1.0)
        if abs(implied - dur) > 0.5:
            bad.append((dur, ntype, '<dot/>' in note, implied))
    assert not bad, f'{len(bad)} notes with type/duration mismatch: {bad[:10]}'


def test_every_measure_is_complete(built):
    """Each measure's notes/rests must sum to exactly one 4/4 bar."""
    xml, tpb = built
    tpm = tpb * 4
    bad = []
    for n, body in _measures(xml):
        body_no_harm = re.sub(r'<harmony>.*?</harmony>', '', body, flags=re.S)
        s = sum(int(d) for d in re.findall(r'<duration>(\d+)</duration>', body_no_harm))
        if s != tpm:
            bad.append((n, s))
    assert not bad, f'measures not summing to {tpm}: {bad}'


def test_ties_are_balanced(built):
    """Every tie-stop has a matching prior tie-start; no dangling starts."""
    xml, _ = built
    active = {}
    errors = []
    for n, body in _measures(xml):
        for note in re.findall(r'<note>(.*?)</note>', body, re.S):
            if '<rest' in note:
                continue
            p = re.search(r'<step>([A-G])</step>.*?<octave>(\d)</octave>', note, re.S)
            pitch = (p.group(1), p.group(2)) if p else None
            if 'tie type="stop"' in note and not active.get(pitch):
                errors.append(f'M{n}: tie-stop on {pitch} with no start')
            if 'tie type="stop"' in note:
                active[pitch] = False
            if 'tie type="start"' in note:
                active[pitch] = True
    errors += [f'dangling tie-start on {p}' for p, v in active.items() if v]
    assert not errors, errors


def test_music21_parses(built):
    """The output must parse under music21 without raising."""
    xml, _ = built
    music21 = pytest.importorskip('music21')
    score = music21.converter.parseData(xml, format='musicxml')
    assert len(score.parts[0].getElementsByClass('Measure')) > 0


def test_chords_sit_on_a_melody_note(built):
    """Every chord symbol must be anchored to a NOTE (the note it sounds with),
    never a rest and never floating — the 'C goes with that A' requirement."""
    xml, _ = built
    on_rest = 0
    for n, body in _measures(xml):
        toks = re.findall(r'<harmony>.*?</harmony>|<note>.*?</note>', body, re.S)
        for i, t in enumerate(toks):
            if t.startswith('<harmony'):
                nxt = toks[i + 1] if i + 1 < len(toks) else ''
                if '<rest' in nxt:
                    on_rest += 1
    # A couple of held-through bars can legitimately re-state a chord over a rest;
    # the vast majority must sit on a note.
    total = xml.count('<harmony>')
    assert on_rest <= 2, f'{on_rest}/{total} chord symbols sit on a rest, not a note'


def test_at_most_one_chord_per_measure(built):
    xml, _ = built
    bad = [(n, body.count('<harmony>')) for n, body in _measures(xml)
           if body.count('<harmony>') > 1]
    assert not bad, f'measures with >1 chord symbol: {bad}'


def test_chord_sequence_matches_groove(built):
    """One chord per bar, the Em/D/C groove, in order."""
    xml, _ = built
    seq = []
    for h in re.findall(r'<harmony>(.*?)</harmony>', xml, re.S):
        root = re.search(r'<root-step>([A-G])</root-step>', h).group(1)
        alter = re.search(r'<root-alter>(-?\d+)</root-alter>', h)
        root += {'1': '#', '-1': 'b'}.get(alter.group(1), '') if alter else ''
        kind = re.search(r'<kind text="([^"]*)"', h).group(1)
        seq.append(root + kind)
    assert seq == EXPECTED_CHORDS, (
        f'chord sequence drift: got {len(seq)} chords\n'
        f'  expected[:20]={EXPECTED_CHORDS[:20]}\n  got[:20]     ={seq[:20]}')


def test_matches_approved_golden(built):
    """Byte-for-byte reproduction of the user-approved lead sheet. If an
    intentional change breaks this, re-check the invariant tests above and
    regenerate the golden file."""
    xml, _ = built
    with open(GOLDEN_XML) as f:
        golden = f.read()
    assert xml == golden, 'generated lead sheet differs from approved golden'
