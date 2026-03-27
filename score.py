#!/usr/bin/env python3
"""Unified scoring script for PianoVideoScribe autoresearch.

Runs the pipeline on all test videos with ground truth, computes F1 scores
per hand, and returns a single aggregate score (0-100).

Usage:
    python score.py              # run pipeline + score
    python score.py --score-only # score existing output-video.mid files
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
    {
        "dir": "test7",
        "settings_file": "settings.json",
    },
    {
        "dir": "test8",
        "settings_file": "settings.json",
    },
    {
        "dir": "test9",
        "settings_file": "settings.json",
    },
    {
        "dir": "test10",
        "settings_file": "settings.json",
    },
    {
        "dir": "test11",
        "settings_file": "settings.json",
    },
    {
        "dir": "test12",
        "settings_file": "settings.json",
    },
]


def _find_note_tracks(mid):
    """Return indices of the first two tracks that contain notes.

    Handles both 2-track (RH, LH) and 3-track (conductor, RH, LH) layouts.
    """
    note_tracks = []
    for i, track in enumerate(mid.tracks):
        has_notes = any(m.type == 'note_on' and m.velocity > 0 for m in track)
        if has_notes:
            note_tracks.append(i)
    return note_tracks[0], note_tracks[1] if len(note_tracks) >= 2 else (0, 1)


def extract_notes(mid, hand_idx):
    """Extract (onset_sec, pitch_class, midi_note) from a hand track.

    hand_idx: 0 = RH (first note track), 1 = LH (second note track).
    Automatically skips conductor tracks that have no notes.
    """
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo != 500000:
            break
    rh_idx, lh_idx = _find_note_tracks(mid)
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


def align_times(gt_notes, out_notes):
    """Align GT times to output times using linear scaling.

    Uses first and last note onsets to compute scale + offset.
    Only applies if the tempo ratio differs by >5%.
    """
    if len(gt_notes) < 2 or len(out_notes) < 2:
        return gt_notes

    gt_start, gt_end = gt_notes[0][0], gt_notes[-1][0]
    out_start, out_end = out_notes[0][0], out_notes[-1][0]

    gt_dur = gt_end - gt_start
    out_dur = out_end - out_start

    if gt_dur < 1.0 or out_dur < 1.0:
        return gt_notes

    scale = out_dur / gt_dur

    # Only apply if there's a meaningful tempo difference (>5%)
    if abs(scale - 1.0) < 0.05:
        return gt_notes
    if not (0.5 < scale < 2.0):
        return gt_notes

    offset = out_start - scale * gt_start
    return [(scale * t + offset, pc, m) for t, pc, m in gt_notes]


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
    output = os.path.join(test_dir, "output-video.mid")
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
    """Score a single test's output-video.mid against ground_truth.mid.

    Tries both hand orderings (normal and swapped) and picks the better one,
    since GT and pipeline may assign track 0/1 to different hands.
    """
    gt_path = os.path.join(test_dir, "ground_truth.mid")
    out_path = os.path.join(test_dir, "output-video.mid")

    if not os.path.exists(gt_path) or not os.path.exists(out_path):
        return None

    gt = mido.MidiFile(gt_path)
    vo = mido.MidiFile(out_path)

    # Try both orderings: normal (RH=0,LH=1) and swapped (RH=1,LH=0)
    def score_ordering(gt_rh_idx, gt_lh_idx, out_rh_idx, out_lh_idx):
        r = {}
        for hand_name, gi, oi in [("RH", gt_rh_idx, out_rh_idx), ("LH", gt_lh_idx, out_lh_idx)]:
            gn = align_times(extract_notes(gt, gi), extract_notes(vo, oi))
            on = extract_notes(vo, oi)
            m, mi, ex = match_notes(gn, on)
            r[hand_name] = {"matched": m, "missed": mi, "extra": ex, "gt": len(gn), "out": len(on)}
        total_matched = r["RH"]["matched"] + r["LH"]["matched"]
        return r, total_matched

    normal, normal_m = score_ordering(0, 1, 0, 1)
    swapped, swapped_m = score_ordering(0, 1, 1, 0)
    best = normal if normal_m >= swapped_m else swapped

    # Determine which output hand index maps to which GT hand
    if normal_m >= swapped_m:
        hand_map = {0: 0, 1: 1}
    else:
        hand_map = {0: 1, 1: 0}

    results = {}
    for hand_idx, hand_name in [(0, "RH"), (1, "LH")]:
        gt_notes = align_times(extract_notes(gt, hand_idx), extract_notes(vo, hand_map[hand_idx]))
        out_notes = extract_notes(vo, hand_map[hand_idx])
        matched, missed, extra = match_notes(gt_notes, out_notes)
        f1 = f1_score(matched, missed, extra)

        # Timing accuracy for matched notes
        avg_timing_ms = 0
        if matched > 0:
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
    json_output = "--json" in sys.argv

    all_f1s = []
    all_results = {}

    for tc in TEST_CONFIGS:
        test_dir = os.path.join(TESTS_DIR, tc["dir"])
        gt_path = os.path.join(test_dir, "ground_truth.mid")
        if not os.path.exists(gt_path):
            continue

        name = tc["dir"]

        if not score_only:
            if not json_output:
                print(f"[{name}] Running pipeline...", end=" ", flush=True)
            settings = tc.get("settings", {})
            if run_pipeline(test_dir, settings):
                if not json_output:
                    print("done")
            else:
                if not json_output:
                    print("FAILED")
                continue

        results = score_test(test_dir)
        if results is None:
            continue

        all_results[name] = results

        for hand in ["RH", "LH"]:
            r = results[hand]
            f1_pct = r["f1"] * 100
            all_f1s.append(f1_pct)
            if verbose and not json_output:
                print(f"  {name}/{hand}: F1={f1_pct:.1f}%  "
                      f"matched={r['matched']}/{r['gt']}  "
                      f"extra={r['extra']}  timing={r['timing_ms']:.0f}ms")

    if not all_f1s:
        if not json_output:
            print("No tests scored.")
        return

    aggregate = sum(all_f1s) / len(all_f1s)

    if json_output:
        output = {
            "aggregate": round(aggregate, 1),
            "n_tracks": len(all_f1s),
            "tests": {},
        }
        for name, results in all_results.items():
            output["tests"][name] = {
                hand: {k: round(v, 3) if isinstance(v, float) else v
                       for k, v in r.items()}
                for hand, r in results.items()
            }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nAggregate score: {aggregate:.1f}/100  "
              f"(avg F1 across {len(all_f1s)} hand tracks)")
        for name, results in all_results.items():
            rh, lh = results["RH"]["f1"] * 100, results["LH"]["f1"] * 100
            print(f"  {name}: RH={rh:.1f}%  LH={lh:.1f}%")


if __name__ == "__main__":
    main()
