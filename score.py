#!/usr/bin/env python3
"""Unified scoring script for PianoVideoScribe autoresearch.

Runs the pipeline on all test videos with ground truth, computes F1 scores
per hand, and returns a single aggregate score (0-100).

Usage:
    python score.py              # run pipeline + score
    python score.py --score-only # score existing output_video.mid files
"""

import json
import os
import subprocess
import sys

import mido

REPO = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.join(REPO, "tests")

# Test configs: each test that has a ground_truth.mid
TEST_CONFIGS = [
    {
        "dir": "test2",
        "settings": {"bpm": 81, "frame": 5},
    },
    {
        "dir": "test3",
        "settings": {"bpm": 76, "frame": 5},
    },
    {
        "dir": "test4",
        "settings_file": "settings.json",
    },
]


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
    for i, (gt_t, gt_pc, _) in enumerate(gt_notes):
        for j, (out_t, out_pc, _) in enumerate(out_notes):
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
        matched.append((i, j, dt))
        used_gt.add(i)
        used_out.add(j)
    missed = len(gt_notes) - len(matched)
    extra = len(out_notes) - len(matched)
    return len(matched), missed, extra


def f1_score(matched, missed, extra):
    """Compute F1 from matched/missed/extra counts."""
    precision = matched / (matched + extra) if (matched + extra) > 0 else 0
    recall = matched / (matched + missed) if (matched + missed) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_pipeline(test_dir, settings):
    """Run pianovideoscribe on a test video."""
    video = os.path.join(test_dir, "video.mp4")
    output = os.path.join(test_dir, "output_video.mid")
    script = os.path.join(REPO, "pianovideoscribe.py")

    cmd = [sys.executable, script, video, output]

    # Check for settings.json file first
    settings_file = os.path.join(test_dir, "settings.json")
    if os.path.exists(settings_file):
        cmd += ["--settings", settings_file]
    else:
        if "bpm" in settings:
            cmd += ["--bpm", str(settings["bpm"])]
        if "frame" in settings:
            cmd += ["--frame", str(settings["frame"])]
        for key in ["key", "green_hand", "right_hand", "left_hand"]:
            if key in settings:
                cmd += [f"--{key.replace('_', '-')}", str(settings[key])]

    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Pipeline failed: {result.stderr[-200:]}", file=sys.stderr)
        return False
    return True


def score_test(test_dir):
    """Score a single test's output_video.mid against ground_truth.mid."""
    gt_path = os.path.join(test_dir, "ground_truth.mid")
    out_path = os.path.join(test_dir, "output_video.mid")

    if not os.path.exists(gt_path) or not os.path.exists(out_path):
        return None

    gt = mido.MidiFile(gt_path)
    vo = mido.MidiFile(out_path)

    results = {}
    for hand_idx, hand_name in [(0, "RH"), (1, "LH")]:
        gt_notes = extract_notes(gt, hand_idx)
        out_notes = extract_notes(vo, hand_idx)
        matched, missed, extra = match_notes(gt_notes, out_notes)
        f1 = f1_score(matched, missed, extra)

        # Timing accuracy for matched notes
        avg_timing_ms = 0
        if matched > 0:
            # Re-do matching to get timing errors
            candidates = []
            for i, (gt_t, gt_pc, _) in enumerate(gt_notes):
                for j, (out_t, out_pc, _) in enumerate(out_notes):
                    if out_pc != gt_pc:
                        continue
                    dt = abs(out_t - gt_t)
                    if dt <= 0.25:
                        candidates.append((dt, i, j))
            candidates.sort()
            used_gt, used_out = set(), set()
            timing_errors = []
            for dt, i, j in candidates:
                if i in used_gt or j in used_out:
                    continue
                timing_errors.append(dt)
                used_gt.add(i)
                used_out.add(j)
            avg_timing_ms = 1000 * sum(timing_errors) / len(timing_errors) if timing_errors else 0

        results[hand_name] = {
            "gt": len(gt_notes),
            "out": len(out_notes),
            "matched": matched,
            "missed": missed,
            "extra": extra,
            "f1": f1,
            "timing_ms": avg_timing_ms,
        }

    return results


def main():
    score_only = "--score-only" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    all_f1s = []

    for tc in TEST_CONFIGS:
        test_dir = os.path.join(TESTS_DIR, tc["dir"])
        gt_path = os.path.join(test_dir, "ground_truth.mid")
        if not os.path.exists(gt_path):
            continue

        name = tc["dir"]

        if not score_only:
            print(f"[{name}] Running pipeline...", end=" ", flush=True)
            settings = tc.get("settings", {})
            if run_pipeline(test_dir, settings):
                print("done")
            else:
                print("FAILED")
                continue

        results = score_test(test_dir)
        if results is None:
            continue

        for hand in ["RH", "LH"]:
            r = results[hand]
            f1_pct = r["f1"] * 100
            all_f1s.append(f1_pct)
            if verbose:
                print(f"  {name}/{hand}: F1={f1_pct:.1f}%  "
                      f"matched={r['matched']}/{r['gt']}  "
                      f"extra={r['extra']}  timing={r['timing_ms']:.0f}ms")

    if not all_f1s:
        print("No tests scored.")
        return

    aggregate = sum(all_f1s) / len(all_f1s)
    print(f"\nAggregate score: {aggregate:.1f}/100  "
          f"(avg F1 across {len(all_f1s)} hand tracks)")

    # Per-test breakdown
    idx = 0
    for tc in TEST_CONFIGS:
        test_dir = os.path.join(TESTS_DIR, tc["dir"])
        if not os.path.exists(os.path.join(test_dir, "ground_truth.mid")):
            continue
        results = score_test(test_dir)
        if results:
            rh, lh = results["RH"]["f1"] * 100, results["LH"]["f1"] * 100
            print(f"  {tc['dir']}: RH={rh:.1f}%  LH={lh:.1f}%")
            idx += 2


if __name__ == "__main__":
    main()
