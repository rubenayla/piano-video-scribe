#!/usr/bin/env python3
"""Scoring script for falling blocks detector autoresearch.

Runs the falling blocks pipeline on test videos with known ground truth,
computes F1 scores, and returns a single aggregate score.

Usage:
    python autoresearch-score.py
"""

import os
import sys

import mido

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from score import extract_notes, match_notes, f1_score, align_times


# Test configs: videos with ground truth that the falling blocks detector
# should be able to handle. Start with synthetic (midi2video) tests for
# guaranteed-correct GT, then add real videos as quality improves.
# RELIABLE TEST DATA: These videos were rendered by midi2video directly from
# the ground truth MIDI files. The GT is 100% correct by construction —
# the MIDI IS the source that generated the video. Any mismatch is a
# detector bug, not a GT issue.
#
# The detector must figure out how to read the video correctly regardless
# of keyboard layout, block positions, colors, etc.
TESTS = [
    {
        "name": "test11 (midi2video, scale+chords, 21 notes)",
        "video": "tests/test11/video.mp4",
        "gt": "tests/test11/ground_truth.mid",
    },
    # Add test12 once test11 reaches 95%+:
    # {
    #     "name": "test12 (midi2video, End of Beginning, 401 notes)",
    #     "video": "tests/test12/video.mp4",
    #     "gt": "tests/test12/ground_truth.mid",
    # },
]


def score_falling_blocks(video_path, gt_path):
    """Run falling blocks pipeline and score against GT."""
    from detect_falling_blocks import detect_falling_notes_pipeline
    import tempfile

    output_path = tempfile.mktemp(suffix='.mid')
    try:
        notes = detect_falling_notes_pipeline(video_path, output_path)
    except Exception as e:
        print(f"  CRASH: {e}", file=sys.stderr)
        return None

    if not os.path.exists(output_path):
        return None

    try:
        gt = mido.MidiFile(gt_path)
        out = mido.MidiFile(output_path)

        results = {}
        for hand_idx, hand_name in [(0, "RH"), (1, "LH")]:
            gt_notes = align_times(extract_notes(gt, hand_idx),
                                   extract_notes(out, hand_idx))
            out_notes = extract_notes(out, hand_idx)
            matched, missed, extra = match_notes(gt_notes, out_notes)
            f1 = f1_score(matched, missed, extra)
            results[hand_name] = {
                "f1": f1,
                "matched": matched,
                "missed": missed,
                "extra": extra,
                "gt": len(gt_notes),
                "out": len(out_notes),
            }
        return results
    except Exception as e:
        print(f"  SCORE ERROR: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def main():
    all_f1s = []

    for tc in TESTS:
        video = os.path.join(REPO, tc["video"])
        gt = os.path.join(REPO, tc["gt"])

        if not os.path.exists(video):
            print(f"  {tc['name']}: SKIP (no video)")
            continue

        print(f"\n--- {tc['name']} ---")
        results = score_falling_blocks(video, gt)

        if results is None:
            print(f"  FAILED")
            all_f1s.append(0.0)
            continue

        for hand in ["RH", "LH"]:
            r = results[hand]
            f1_pct = r["f1"] * 100
            all_f1s.append(f1_pct)
            print(f"  {hand}: F1={f1_pct:.1f}%  "
                  f"matched={r['matched']}/{r['gt']}  "
                  f"extra={r['extra']}")

    if not all_f1s:
        print("\nNo tests scored.")
        return

    aggregate = sum(all_f1s) / len(all_f1s)
    print(f"\n=== AGGREGATE SCORE: {aggregate:.1f}/100 ===")


if __name__ == "__main__":
    main()
