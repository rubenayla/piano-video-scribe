#!/usr/bin/env python3
"""WCS scoring for chord detection against synthetic ground truth.

Usage:
    python chord-score.py [--config '{"self_loop_prob": 0.9}']
"""

import argparse
import json
import os
import sys

import mido
import mir_eval
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from piano_video_scribe.lead_sheet import detect_chords

TEST_DIR = os.path.join(REPO, 'tests', 'chord-detection')

TESTS = [
    'test-cmaj',
    'test-amin',
    'test-melody',
    'test-dom7',
    'test-beatch',
]

# Map our internal labels to mir_eval vocabulary
# mir_eval supports: 'C:maj', 'A:min', 'G:7', 'D:min7', 'N' etc.
# Our labels already use this format — no conversion needed.


def load_midi_events(mid_path):
    """Load right/left hand events from a 3-track MIDI (conductor, RH ch0, LH ch1)."""
    mid = mido.MidiFile(mid_path)
    tpb = mid.ticks_per_beat
    tempo_us = 500_000  # default 120 BPM

    right_events, left_events = [], []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type in ('note_on', 'note_off'):
                ev = (tick, msg.type, msg.note, msg.velocity)
                if msg.channel == 0:
                    right_events.append(ev)
                elif msg.channel == 1:
                    left_events.append(ev)

    return right_events, left_events, tpb, tempo_us


def gt_to_intervals(gt_entries, tempo_us, tpb):
    """Convert beat-based ground truth to time intervals (seconds) and labels."""
    sec_per_beat = tempo_us / 1_000_000
    ref_intervals = []
    ref_labels = []
    for entry in gt_entries:
        start_s = entry['start_beat'] * sec_per_beat
        end_s   = entry['end_beat']   * sec_per_beat
        ref_intervals.append([start_s, end_s])
        ref_labels.append(entry['chord'])
    return np.array(ref_intervals), ref_labels


def det_to_intervals(detected, tempo_us, tpb):
    """Convert (start_tick, end_tick, label) to time intervals (seconds)."""
    sec_per_tick = (tempo_us / 1_000_000) / tpb
    est_intervals = []
    est_labels = []
    for start_tick, end_tick, label in detected:
        est_intervals.append([start_tick * sec_per_tick, end_tick * sec_per_tick])
        est_labels.append(label)
    if not est_intervals:
        est_intervals = [[0, 1]]
        est_labels = ['N']
    return np.array(est_intervals), est_labels


def score_test(test_name, config=None):
    """Run detection on one test, return dict with WCS metrics."""
    mid_path = os.path.join(TEST_DIR, test_name + '.mid')
    gt_path  = os.path.join(TEST_DIR, test_name + '-gt.json')

    if not os.path.exists(mid_path):
        return None

    right_ev, left_ev, tpb, tempo_us = load_midi_events(mid_path)
    with open(gt_path) as f:
        gt_entries = json.load(f)

    detected = detect_chords(right_ev, left_ev, tpb, config)
    ref_iv, ref_lb = gt_to_intervals(gt_entries, tempo_us, tpb)
    est_iv, est_lb = det_to_intervals(detected, tempo_us, tpb)

    # Clip est to ref duration
    ref_dur = ref_iv[-1, 1]
    est_iv = np.clip(est_iv, 0, ref_dur)

    scores = mir_eval.chord.evaluate(ref_iv, ref_lb, est_iv, est_lb)
    return {
        'root':   float(scores['root']),
        'thirds': float(scores['thirds']),
        'majmin': float(scores['majmin']),
    }


def aggregate_wcs(config=None):
    """Run all tests, return average 'thirds' score (our primary metric)."""
    total, count = 0.0, 0
    for name in TESTS:
        result = score_test(name, config)
        if result:
            total += result['thirds']
            count += 1
    return total / count if count > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=json.loads, default=None,
                        help='JSON dict of config overrides')
    args = parser.parse_args()

    print(f'{"Test":<22}  {"root":>6}  {"thirds":>7}  {"majmin":>7}  detected')
    print('-' * 70)
    results = []
    for name in TESTS:
        result = score_test(name, args.config)
        if result is None:
            print(f'{name:<22}  MISSING')
            continue

        # Also print what was detected for inspection
        mid_path = os.path.join(TEST_DIR, name + '.mid')
        right_ev, left_ev, tpb, tempo_us = load_midi_events(mid_path)
        detected = detect_chords(right_ev, left_ev, tpb, args.config)
        det_str = ' → '.join(l for _, _, l in detected)

        print(f'{name:<22}  {result["root"]:6.3f}  {result["thirds"]:7.3f}  '
              f'{result["majmin"]:7.3f}  {det_str}')
        results.append(result['thirds'])

    if results:
        avg = sum(results) / len(results)
        print('-' * 70)
        print(f'{"AGGREGATE (thirds avg)":<22}  {"":6}  {avg:7.3f}')


if __name__ == '__main__':
    main()
