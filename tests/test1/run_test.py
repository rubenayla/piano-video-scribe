#!/usr/bin/env python3
"""Regression test for PianoVideoScribe using test1 video.

Runs the video-only extraction pipeline and checks the output against
known expected properties.
"""

import json
import os
import sys
import tempfile

# Add parent repo to path
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(TEST_DIR, "video.mp4")
EXPECTED = os.path.join(TEST_DIR, "expected.json")


def load_expected():
    with open(EXPECTED) as f:
        return json.load(f)


def run_pipeline(video_path, output_path, bpm, frame):
    """Run pianovideoscribe in video-only mode."""
    import cv2
    from pianovideoscribe import (
        detect_keyboard, build_note_x_map, extract_notes_from_video,
        classify_hand, load_config,
    )

    cfg = load_config(None)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    white_keys, black_keys, y_white, y_black = detect_keyboard(cap, frame_idx=frame)
    note_x_map = build_note_x_map(white_keys, black_keys, 21)

    y_top = y_white - cfg['sampling']['y_offset_top']
    y_bot = y_white - cfg['sampling']['y_offset_bot']
    half_w = cfg['sampling']['half_w']

    notes = extract_notes_from_video(
        cap, note_x_map, y_top, y_bot, half_w,
        fps, total_frames, green_is_right=True, colors=cfg['colors'],
        start_frame=frame, frame_step=1,
    )
    cap.release()

    return notes, white_keys, black_keys


def test_keyboard_detection(white_keys, black_keys, expected):
    """Check keyboard detection matches expected counts."""
    kb = expected["keyboard"]
    errors = []
    if len(white_keys) != kb["white_keys_expected"]:
        errors.append(
            f"White keys: got {len(white_keys)}, expected {kb['white_keys_expected']}")
    if len(black_keys) != kb["black_keys_expected"]:
        errors.append(
            f"Black keys: got {len(black_keys)}, expected {kb['black_keys_expected']}")
    return errors


def test_first_notes(notes, expected):
    """Check first note properties for each hand."""
    errors = []
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Split by hand
    right = [(p, h, on, off) for p, h, on, off in notes if h == 0]
    left = [(p, h, on, off) for p, h, on, off in notes if h == 1]

    if not left:
        errors.append("No left hand notes found")
    if not right:
        errors.append("No right hand notes found")

    if left and right:
        # Left hand should start before right hand
        if left[0][2] >= right[0][2]:
            errors.append(
                f"Left hand should start before right: "
                f"left={left[0][2]:.2f}s, right={right[0][2]:.2f}s")

    # Check first note pitch names
    fn = expected["first_notes"]
    for hand_name, hand_notes, hand_key in [
        ("left_hand", left, "left_hand"),
        ("right_hand", right, "right_hand"),
    ]:
        if not hand_notes:
            continue
        first_pitch = hand_notes[0][0]
        first_name = note_names[first_pitch % 12]
        expected_name = fn[hand_key]["pitch_name"]
        if first_name != expected_name:
            errors.append(
                f"{hand_name} first note: got {first_name} (MIDI {first_pitch}), "
                f"expected {expected_name}")

        # Check onset time range
        onset = hand_notes[0][2]
        lo, hi = fn[hand_key]["onset_sec_approx"]
        if not (lo <= onset <= hi):
            errors.append(
                f"{hand_name} onset: {onset:.2f}s not in expected range [{lo}, {hi}]")

    return errors


def test_right_hand_rhythm(notes, expected):
    """Check that right hand plays eighth notes (~0.33s intervals)."""
    errors = []
    right = [(p, h, on, off) for p, h, on, off in notes if h == 0]
    if len(right) < 5:
        errors.append(f"Too few right hand notes to check rhythm: {len(right)}")
        return errors

    # Check intervals between first 10 onsets
    onsets = [on for _, _, on, _ in right[:10]]
    intervals = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
    avg_interval = sum(intervals) / len(intervals)

    # Eighth note at 180/min = 0.333s
    if not (0.25 <= avg_interval <= 0.45):
        errors.append(
            f"Right hand avg interval: {avg_interval:.3f}s, "
            f"expected ~0.333s (eighth at 180/min)")

    return errors


def test_no_bleed(notes):
    """Check that black key bleed has been filtered."""
    errors = []
    BLACK = {1, 3, 6, 8, 10}
    bleed_count = 0

    for pitch, hand, on, off in notes:
        if pitch % 12 not in BLACK:
            continue
        # Check if adjacent white key overlaps
        for other_p, other_h, other_on, other_off in notes:
            if other_p % 12 in BLACK:
                continue
            if abs(other_p - pitch) == 1 and other_h == hand:
                if other_on < off and other_off > on:
                    bleed_count += 1
                    break

    if bleed_count > 0:
        errors.append(f"Found {bleed_count} possible black key bleed notes")

    return errors


def main():
    expected = load_expected()
    print(f"Running test1: {expected['description']}")
    print(f"Video: {VIDEO}")
    print()

    kb = expected["keyboard"]
    notes, white_keys, black_keys = run_pipeline(
        VIDEO, None, expected["bpm"], kb["frame"])

    print(f"\nExtracted {len(notes)} notes")
    print(f"Right: {sum(1 for _,h,_,_ in notes if h==0)}, "
          f"Left: {sum(1 for _,h,_,_ in notes if h==1)}")
    print()

    all_errors = []

    print("--- Keyboard detection ---")
    errs = test_keyboard_detection(white_keys, black_keys, expected)
    all_errors.extend(errs)
    print("  PASS" if not errs else "\n".join(f"  FAIL: {e}" for e in errs))

    print("--- First notes ---")
    errs = test_first_notes(notes, expected)
    all_errors.extend(errs)
    print("  PASS" if not errs else "\n".join(f"  FAIL: {e}" for e in errs))

    print("--- Right hand rhythm ---")
    errs = test_right_hand_rhythm(notes, expected)
    all_errors.extend(errs)
    print("  PASS" if not errs else "\n".join(f"  FAIL: {e}" for e in errs))

    print("--- No bleed ---")
    errs = test_no_bleed(notes)
    all_errors.extend(errs)
    print("  PASS" if not errs else "\n".join(f"  FAIL: {e}" for e in errs))

    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s)")
        sys.exit(1)
    else:
        print("ALL PASSED")


if __name__ == "__main__":
    main()
