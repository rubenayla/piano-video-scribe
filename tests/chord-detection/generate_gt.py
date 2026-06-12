#!/usr/bin/env python3
"""Generate synthetic test MIDIs with known chord progressions for validation.

Each test has:
  - LH track: block chord voicings (whole notes, root position)
  - RH track: stepwise melody (quarter notes) or empty
  - Matching JSON ground truth in mir_eval format ('Root:quality')

Run this once to create the test fixtures:
    python tests/chord-detection/generate_gt.py
"""

import json
import os
import sys

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BPM = 120
TPB = 480
US_PER_BEAT = int(60_000_000 / BPM)
BEATS_PER_CHORD = 4  # each chord lasts one 4/4 measure

# Root → MIDI note in octave 3 (C3=48)
_ROOT_MIDI = {
    'C': 48, 'C#': 49, 'D': 50, 'D#': 51, 'E': 52,
    'F': 53, 'F#': 54, 'G': 55, 'G#': 56, 'A': 57, 'A#': 58, 'B': 59,
}

_INTERVALS = {
    'maj': [0, 4, 7],
    'min': [0, 3, 7],
    '7':   [0, 4, 7, 10],
    'min7': [0, 3, 7, 10],
}


def chord_notes(label):
    """Return MIDI notes for 'Root:quality' label at octave 3."""
    root, qual = label.split(':')
    base = _ROOT_MIDI[root]
    return [base + i for i in _INTERVALS[qual]]


def make_midi(lh_chords, rh_melody_notes=None):
    """
    lh_chords: list of 'Root:quality' labels, one per BEATS_PER_CHORD ticks
    rh_melody_notes: list of lists of MIDI note numbers (one list per chord,
                     played as quarter notes), or None for melody-free tests
    """
    mid = MidiFile(type=1, ticks_per_beat=TPB)

    # Conductor track
    conductor = MidiTrack()
    conductor.append(MetaMessage('set_tempo', tempo=US_PER_BEAT, time=0))
    conductor.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    mid.tracks.append(conductor)

    chord_ticks = BEATS_PER_CHORD * TPB  # ticks per chord block

    # RH track (melody)
    rh_track = MidiTrack()
    rh_track.name = 'Right Hand'
    cursor = 0
    if rh_melody_notes:
        for chord_idx, notes in enumerate(rh_melody_notes):
            q = TPB  # quarter note
            for i, note in enumerate(notes):
                rh_track.append(Message('note_on',  channel=0, note=note, velocity=70, time=0))
                rh_track.append(Message('note_off', channel=0, note=note, velocity=0,  time=q))
            # Rest of the chord block
            remaining = chord_ticks - len(notes) * q
            if remaining > 0:
                rh_track.append(Message('note_on',  channel=0, note=60, velocity=0, time=remaining))
    mid.tracks.append(rh_track)

    # LH track (block chords)
    lh_track = MidiTrack()
    lh_track.name = 'Left Hand'
    cursor = 0
    for label in lh_chords:
        notes = chord_notes(label)
        # note_on at start
        for i, note in enumerate(notes):
            lh_track.append(Message('note_on', channel=1, note=note, velocity=80, time=0))
        # note_off at end of chord block
        lh_track.append(Message('note_off', channel=1, note=notes[0], velocity=0, time=chord_ticks))
        for note in notes[1:]:
            lh_track.append(Message('note_off', channel=1, note=note, velocity=0, time=0))
    mid.tracks.append(lh_track)

    return mid


def make_beatch_midi():
    """
    Beat-It-like: E drone (E2+E3) in LH throughout.
    RH: Em melody notes (G4, B4) in odd measures, D melody notes (D4, F#4) in even ones.
    GT: Em – D – Em – D
    """
    mid = MidiFile(type=1, ticks_per_beat=TPB)
    conductor = MidiTrack()
    conductor.append(MetaMessage('set_tempo', tempo=US_PER_BEAT, time=0))
    mid.tracks.append(conductor)

    n_measures = 4
    chord_ticks = BEATS_PER_CHORD * TPB

    rh_track = MidiTrack()
    rh_track.name = 'Right Hand'
    rh_patterns = [
        [67, 71],   # G4, B4  → Em feel
        [62, 66],   # D4, F#4 → D feel
        [67, 71],
        [62, 66],
    ]
    for pattern in rh_patterns:
        q = TPB
        for note in pattern:
            rh_track.append(Message('note_on',  channel=0, note=note, velocity=70, time=0))
            rh_track.append(Message('note_off', channel=0, note=note, velocity=0,  time=q))
        remaining = chord_ticks - len(pattern) * q
        if remaining > 0:
            rh_track.append(Message('note_on',  channel=0, note=60, velocity=0, time=remaining))
    mid.tracks.append(rh_track)

    lh_track = MidiTrack()
    lh_track.name = 'Left Hand'
    e_drone = [40, 52]  # E2=40, E3=52
    total_ticks = n_measures * chord_ticks
    for note in e_drone:
        lh_track.append(Message('note_on', channel=1, note=note, velocity=80, time=0))
    lh_track.append(Message('note_off', channel=1, note=e_drone[0], velocity=0, time=total_ticks))
    for note in e_drone[1:]:
        lh_track.append(Message('note_off', channel=1, note=note, velocity=0, time=0))
    mid.tracks.append(lh_track)

    return mid


def gt_json(labels, beats_per_chord=BEATS_PER_CHORD):
    entries = []
    for i, label in enumerate(labels):
        entries.append({
            'start_beat': i * beats_per_chord,
            'end_beat':   (i + 1) * beats_per_chord,
            'chord': label,
        })
    return entries


TESTS = [
    {
        'name': 'test-cmaj',
        'desc': 'C major — pure block chords, no melody (easiest)',
        'chords': ['C:maj', 'F:maj', 'G:maj', 'C:maj'],
        'melody': None,
    },
    {
        'name': 'test-amin',
        'desc': 'A minor — block chords, no melody',
        'chords': ['A:min', 'F:maj', 'C:maj', 'G:maj'],
        'melody': None,
    },
    {
        'name': 'test-melody',
        'desc': 'A minor — block chords LH + scale melody RH',
        'chords': ['A:min', 'F:maj', 'C:maj', 'G:maj'],
        'melody': [
            [69, 71, 72, 74],   # A4 B4 C5 D5 over Am
            [65, 67, 69, 71],   # F4 G4 A4 B4 over F
            [60, 62, 64, 65],   # C4 D4 E4 F4 over C
            [55, 57, 59, 60],   # G3 A3 B3 C4 over G
        ],
    },
    {
        'name': 'test-dom7',
        'desc': 'Extended chords — G7 C Dm7 G7',
        'chords': ['G:7', 'C:maj', 'D:min7', 'G:7'],
        'melody': None,
    },
    {
        'name': 'test-beatch',
        'desc': 'Beat-It-like — E drone LH, Em/D melody RH (hardest)',
        'chords': ['E:min', 'D:maj', 'E:min', 'D:maj'],
        'melody': 'beatch',
    },
]


def main():
    for t in TESTS:
        mid_path = os.path.join(OUT_DIR, t['name'] + '.mid')
        gt_path  = os.path.join(OUT_DIR, t['name'] + '-gt.json')

        if t['melody'] == 'beatch':
            mid = make_beatch_midi()
        else:
            mid = make_midi(t['chords'], t['melody'])

        mid.save(mid_path)

        gt = gt_json(t['chords'])
        with open(gt_path, 'w') as f:
            json.dump(gt, f, indent=2)

        n_rh = sum(1 for track in mid.tracks for msg in track if msg.type == 'note_on' and msg.channel == 0)
        n_lh = sum(1 for track in mid.tracks for msg in track if msg.type == 'note_on' and msg.channel == 1)
        print(f"{t['name']:20s}  RH={n_rh:3d} notes  LH={n_lh:3d} notes  — {t['desc']}")

    print(f'\nWrote {len(TESTS)} test MIDIs to {OUT_DIR}')


if __name__ == '__main__':
    main()
