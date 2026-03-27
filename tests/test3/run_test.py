#!/usr/bin/env python3
"""Regression test for test2 (ya_no_mas): compare video output against ground truth.

Checks pitch classes (ignoring octave), timing, and hand assignment.
"""

import os
import sys
import mido

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH = os.path.join(TEST_DIR, "ground_truth.mid")
VIDEO_OUTPUT = os.path.join(TEST_DIR, "output-video.mid")

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def extract_notes(mid, track_idx):
    """Extract (onset_sec, pitch_class, midi_note) from a track."""
    # Find tempo from any track (conductor track may differ from note track)
    tempo = 500000  # default 120 BPM
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo != 500000:
            break
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


def match_notes(gt_notes, out_notes, time_tolerance=0.25):
    """Match output notes to ground truth by pitch class + timing.

    Sorts candidate pairs by time distance to find globally better matches.
    Returns (matched, missed_gt, extra_out).
    """
    # Build all possible (gt_idx, out_idx, dt) pairs
    candidates = []
    for i, (gt_t, gt_pc, gt_midi) in enumerate(gt_notes):
        for j, (out_t, out_pc, out_midi) in enumerate(out_notes):
            if out_pc != gt_pc:
                continue
            dt = abs(out_t - gt_t)
            if dt <= time_tolerance:
                candidates.append((dt, i, j))

    # Greedy match: best time distance first, no double-assignment
    candidates.sort()
    used_gt = set()
    used_out = set()
    matched = []

    for dt, i, j in candidates:
        if i in used_gt or j in used_out:
            continue
        gt_t, gt_pc, gt_midi = gt_notes[i]
        matched.append((gt_t, gt_pc, gt_midi, out_notes[j]))
        used_gt.add(i)
        used_out.add(j)

    missed_gt = [(t, pc, m) for i, (t, pc, m) in enumerate(gt_notes) if i not in used_gt]
    extra_out = [(t, pc, m) for j, (t, pc, m) in enumerate(out_notes) if j not in used_out]
    return matched, missed_gt, extra_out


def run_test():
    if not os.path.exists(VIDEO_OUTPUT):
        print(f"No video output found at {VIDEO_OUTPUT}")
        print("Run the pipeline first:")
        print("  python pianovideoscribe.py video.mp4 output-video.mid --bpm 81 --frame 5")
        return False

    gt = mido.MidiFile(GROUND_TRUTH)
    vo = mido.MidiFile(VIDEO_OUTPUT)

    # Ground truth: track 0 = RH, track 1 = LH (from mscz export)
    # Video output: track 0 = RH, track 1 = LH
    gt_rh = extract_notes(gt, 0)
    gt_lh = extract_notes(gt, 1)
    vo_rh = extract_notes(vo, 0)
    vo_lh = extract_notes(vo, 1)

    all_pass = True

    for hand, gt_notes, out_notes in [
        ("Right Hand", gt_rh, vo_rh),
        ("Left Hand", gt_lh, vo_lh),
    ]:
        matched, missed, extra = match_notes(gt_notes, out_notes)
        total_gt = len(gt_notes)
        match_pct = 100 * len(matched) / total_gt if total_gt else 0

        print(f"\n--- {hand} ---")
        print(f"  Ground truth: {total_gt} notes")
        print(f"  Video output: {len(out_notes)} notes")
        print(f"  Matched:      {len(matched)} ({match_pct:.0f}%)")
        print(f"  Missed:       {len(missed)}")
        print(f"  Extra:        {len(extra)}")

        if missed and len(missed) <= 10:
            print(f"  Missed notes:")
            for t, pc, midi in missed[:10]:
                print(f"    t={t:.2f}s  {NOTE_NAMES[pc]}")

        # Timing accuracy of matched notes
        if matched:
            timing_errors = [abs(gt_t - out[0]) for gt_t, _, _, out in matched]
            avg_err = sum(timing_errors) / len(timing_errors)
            max_err = max(timing_errors)
            within_8th = sum(1 for e in timing_errors if e < 0.18)  # half an 8th at 81 BPM
            print(f"  Timing: avg={avg_err*1000:.0f}ms  max={max_err*1000:.0f}ms  "
                  f"within_8th={within_8th}/{len(matched)} ({100*within_8th/len(matched):.0f}%)")

        # Check for pitch substitution errors: missed GT notes that have an
        # extra output note at the same time with adjacent pitch class (±1 semitone).
        # These indicate the detector is hitting the wrong key.
        pitch_subs = 0
        for m_t, m_pc, m_midi in missed:
            for e_t, e_pc, e_midi in extra:
                if abs(m_t - e_t) < 0.25 and abs(m_pc - e_pc) % 12 == 1:
                    pitch_subs += 1
                    break
        if pitch_subs > 0:
            print(f"  Pitch substitutions: {pitch_subs} (adjacent key errors)")

        # Thresholds
        threshold = 70 if "Right" in hand else 30
        if pitch_subs > len(gt_notes) * 0.03:  # >3% pitch errors = fail
            print(f"  FAIL: {pitch_subs} pitch substitutions > 3% of {total_gt}")
            all_pass = False
        elif match_pct < threshold:
            print(f"  FAIL: match rate {match_pct:.0f}% < {threshold}%")
            all_pass = False
        else:
            print(f"  PASS")

    print()
    if all_pass:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
    return all_pass


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
