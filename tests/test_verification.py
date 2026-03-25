#!/usr/bin/env python3
"""Tests for output verification: out-of-key detection, note density, hand balance."""

import pytest

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Key signature pitch class sets
KEY_PCS = {
    'C':  {0, 2, 4, 5, 7, 9, 11},
    'D':  {2, 4, 6, 7, 9, 11, 1},
    'E':  {4, 6, 8, 9, 11, 1, 3},
    'F':  {5, 7, 9, 10, 0, 2, 4},
    'G':  {7, 9, 11, 0, 2, 4, 6},
    'A':  {9, 11, 1, 2, 4, 6, 8},
    'Bb': {10, 0, 2, 3, 5, 7, 9},
    'Eb': {3, 5, 7, 8, 10, 0, 2},
    'Am': {9, 11, 0, 2, 4, 5, 7},
    'Dm': {2, 4, 5, 7, 9, 10, 0},
    'Em': {4, 6, 7, 9, 11, 0, 2},
    'Bm': {11, 1, 2, 4, 6, 7, 9},
}


def find_out_of_key(midi_notes, key_sig):
    """Find notes outside the key signature.

    Args:
        midi_notes: list of MIDI note numbers (ints)
        key_sig: string key name, e.g. 'D', 'Am', 'Bb'

    Returns:
        list of (midi_note, note_name) tuples for out-of-key notes
    """
    if key_sig not in KEY_PCS:
        return []
    pcs = KEY_PCS[key_sig]
    violations = []
    for n in midi_notes:
        pc = n % 12
        if pc not in pcs:
            violations.append((n, NOTE_NAMES[pc] + str(n // 12 - 1)))
    return violations


def check_note_density(n_notes, duration_sec):
    """Check if note density is in a reasonable range.

    Returns:
        (density, status) where status is 'OK', 'WARNING', or 'ERROR'
    """
    if duration_sec <= 0:
        return 0, 'ERROR'
    density = n_notes / duration_sec
    if density < 0.1 or density > 15:
        return density, 'ERROR'
    if density < 0.5 or density > 8:
        return density, 'WARNING'
    return density, 'OK'


def check_hand_balance(rh_notes, lh_notes, melody_hand='right'):
    """Check if the melody hand has more notes than accompaniment.

    Returns:
        (ratio, status) where ratio is melody/total and status is 'OK' or 'WARNING'
    """
    total = rh_notes + lh_notes
    if total == 0:
        return 0, 'ERROR'
    melody_count = rh_notes if melody_hand == 'right' else lh_notes
    ratio = melody_count / total
    if ratio < 0.3:
        return ratio, 'WARNING'
    return ratio, 'OK'


# --- pytest tests ---

def test_d_major_all_diatonic():
    """All notes in D major scale should produce no violations."""
    # D E F# G A B C# in octave 4
    notes = [62, 64, 66, 67, 69, 71, 73]
    violations = find_out_of_key(notes, 'D')
    assert violations == []


def test_d_major_detects_g_sharp():
    """G# is not in D major — should be flagged."""
    notes = [62, 64, 68, 67]  # D E G# G
    violations = find_out_of_key(notes, 'D')
    assert len(violations) == 1
    assert violations[0][0] == 68
    assert 'G#' in violations[0][1]


def test_d_major_detects_a_sharp():
    """A# is not in D major — should be flagged."""
    notes = [62, 70]  # D A#
    violations = find_out_of_key(notes, 'D')
    assert len(violations) == 1
    assert violations[0][0] == 70


def test_c_major_no_accidentals():
    """C major: C D E F G A B — all natural."""
    notes = [60, 62, 64, 65, 67, 69, 71]
    violations = find_out_of_key(notes, 'C')
    assert violations == []


def test_c_major_detects_f_sharp():
    notes = [60, 66]  # C F#
    violations = find_out_of_key(notes, 'C')
    assert len(violations) == 1


def test_unknown_key_returns_empty():
    violations = find_out_of_key([60, 61, 62], 'X#')
    assert violations == []


def test_note_density_normal():
    density, status = check_note_density(300, 120)
    assert status == 'OK'
    assert 2.0 < density < 3.0


def test_note_density_too_sparse():
    density, status = check_note_density(5, 120)
    assert status in ('WARNING', 'ERROR')


def test_note_density_too_dense():
    density, status = check_note_density(2000, 120)
    assert status in ('WARNING', 'ERROR')


def test_hand_balance_normal():
    ratio, status = check_hand_balance(300, 130, 'right')
    assert status == 'OK'
    assert ratio > 0.5


def test_hand_balance_swapped():
    """Melody hand has very few notes — likely hands are swapped."""
    ratio, status = check_hand_balance(30, 300, 'right')
    assert status == 'WARNING'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
