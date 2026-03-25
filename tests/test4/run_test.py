#!/usr/bin/env python3
"""Regression test for test4 (End of Beginning — Djo, easy piano tutorial).

This video is a letterboxed Synthesia tutorial with:
- Green = left hand, Cyan = right hand (non-default)
- Keyboard from ~Bb1 to C#6 (not C-to-C)
- Title screen banner that caused false note detections
- Black key glow bleed that triggered false A#/G# notes

Pipeline command:
  python pianovideoscribe.py video.mp4 output_video.mid \\
    --bpm 80 --frame 80 --key D --green-hand left \\
    --right-hand monophonic --left-hand no-overlap
"""

import os
import subprocess
import sys
import mido

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
GROUND_TRUTH = os.path.join(TEST_DIR, "ground_truth.mid")
VIDEO_OUTPUT = os.path.join(TEST_DIR, "output_video.mid")
SETTINGS_JSON = os.path.join(TEST_DIR, "settings.json")

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# D major pitch classes
D_MAJOR_PCS = {2, 4, 6, 7, 9, 11, 1}  # D E F# G A B C#


def extract_notes(mid, track_idx):
    """Extract (onset_sec, pitch_class, midi_note) from a track."""
    tempo = 500000
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
    """Match output notes to ground truth by pitch class + timing."""
    candidates = []
    for i, (gt_t, gt_pc, gt_midi) in enumerate(gt_notes):
        for j, (out_t, out_pc, out_midi) in enumerate(out_notes):
            if out_pc != gt_pc:
                continue
            dt = abs(out_t - gt_t)
            if dt <= time_tolerance:
                candidates.append((dt, i, j))

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


def check_out_of_key(notes, key_pcs, hand_name):
    """Check for notes outside the key signature. Returns list of violations."""
    violations = []
    for t, pc, midi in notes:
        if pc not in key_pcs:
            violations.append((t, pc, midi))
    if violations:
        pct = 100 * len(violations) / len(notes) if notes else 0
        print(f"  Out-of-key notes: {len(violations)} ({pct:.1f}%)")
        for t, pc, midi in violations[:5]:
            print(f"    t={t:.2f}s  {NOTE_NAMES[pc]}{midi // 12 - 1}")
        if len(violations) > 5:
            print(f"    ... and {len(violations) - 5} more")
    return violations


def run_test():
    if not os.path.exists(VIDEO_OUTPUT):
        if not os.path.exists(SETTINGS_JSON):
            print(f"No video output found at {VIDEO_OUTPUT}")
            print("No settings.json found either. Run the pipeline first:")
            print("  cd ~/repos/PianoVideoScribe && python pianovideoscribe.py "
                  "tests/test4/video.mp4 tests/test4/output_video.mid "
                  "--bpm 80 --frame 80 --key D --green-hand left "
                  "--right-hand monophonic --left-hand no-overlap")
            return False
        print(f"No video output found — running pipeline using {SETTINGS_JSON} ...")
        script = os.path.join(REPO_DIR, "pianovideoscribe.py")
        video = os.path.join(TEST_DIR, "video.mp4")
        cmd = [sys.executable, script, video, VIDEO_OUTPUT,
               "--settings", SETTINGS_JSON]
        result = subprocess.run(cmd, cwd=REPO_DIR)
        if result.returncode != 0:
            print("Pipeline failed!")
            return False

    gt = mido.MidiFile(GROUND_TRUTH)
    vo = mido.MidiFile(VIDEO_OUTPUT)

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

        # Out-of-key check
        key_violations = check_out_of_key(out_notes, D_MAJOR_PCS, hand)

        if missed and len(missed) <= 10:
            print(f"  Missed notes:")
            for t, pc, midi in missed[:10]:
                print(f"    t={t:.2f}s  {NOTE_NAMES[pc]}")

        # Timing accuracy
        if matched:
            timing_errors = [abs(gt_t - out[0]) for gt_t, _, _, out in matched]
            avg_err = sum(timing_errors) / len(timing_errors)
            max_err = max(timing_errors)
            within_8th = sum(1 for e in timing_errors if e < 0.19)
            print(f"  Timing: avg={avg_err * 1000:.0f}ms  max={max_err * 1000:.0f}ms  "
                  f"within_8th={within_8th}/{len(matched)} "
                  f"({100 * within_8th / len(matched):.0f}%)")

        # Thresholds
        threshold = 70 if "Right" in hand else 30
        if match_pct < threshold:
            print(f"  FAIL: match rate {match_pct:.0f}% < {threshold}%")
            all_pass = False
        elif len(key_violations) > len(out_notes) * 0.05:
            print(f"  FAIL: >5% out-of-key notes")
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
