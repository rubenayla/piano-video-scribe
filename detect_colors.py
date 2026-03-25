#!/usr/bin/env python3
"""Auto-detect Synthesia hand colors from a video.

Compares a neutral keyboard frame (no keys pressed) against frames where
notes are being played.  Pixels that jump from low saturation (white/black
keys) to high saturation (colored lit keys) reveal the hand colors.

Usage:
    python detect_colors.py video.mp4 [--frame-neutral 250] [--frame-playing 600]

If --frame-playing is omitted, the script scans forward from the neutral
frame until it finds enough saturated pixels in the key-face zone.
"""

import argparse
import cv2
import numpy as np


def find_key_face_zone(hsv_frame, quiet=False):
    """Find the y-range of the white key faces (bottom-up scan)."""
    h, w = hsv_frame.shape[:2]
    y_white = None
    for y in range(h - 5, h // 2, -1):
        row = hsv_frame[y, :]
        white_count = int(np.sum((row[:, 1] < 60) & (row[:, 2] > 180)))
        dark_count = int(np.sum(row[:, 2] < 80))
        if white_count > w * 0.3 and dark_count < w * 0.05:
            y_white = y
            break
    if y_white is None:
        if not quiet:
            print("Could not find keyboard.  Try a different --frame-neutral.")
        return None, None
    # Key face zone: 90px above white key row
    return y_white - 90, y_white


def find_playing_frame(cap, y_top, y_bot, start_frame, min_saturated=50):
    """Scan forward to find the first frame with enough lit keys."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in range(start_frame, total, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        region = hsv[y_top:y_bot, :, :]
        saturated = np.sum((region[:, :, 1] > 80) & (region[:, :, 2] > 100))
        if saturated >= min_saturated:
            return f
    return None


def detect_colors(video_path, frame_neutral=None, frame_playing=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- Step 1: Find a neutral frame with visible keyboard ---
    if frame_neutral is None:
        # Scan for first frame where keyboard is visible
        for f in range(0, min(total_frames, 500), 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            y_top, y_bot = find_key_face_zone(hsv, quiet=True)
            if y_top is not None:
                frame_neutral = f
                print(f"Found keyboard at frame {f} ({f/fps:.1f}s)")
                break
        if frame_neutral is None:
            print("Could not find keyboard in first 500 frames.")
            cap.release()
            return
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_neutral)
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        y_top, y_bot = find_key_face_zone(hsv)
        if y_top is None:
            cap.release()
            return

    print(f"Key face zone: y={y_top}..{y_bot}")

    # Capture neutral key-face region
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_neutral)
    ret, neutral_frame = cap.read()
    neutral_hsv = cv2.cvtColor(neutral_frame, cv2.COLOR_BGR2HSV)
    neutral_region = neutral_hsv[y_top:y_bot, :, :]

    # --- Step 2: Find frame(s) where notes are playing ---
    if frame_playing is None:
        frame_playing = find_playing_frame(cap, y_top, y_bot, frame_neutral + 30)
        if frame_playing is None:
            print("Could not find a frame with lit keys.")
            cap.release()
            return
        print(f"First playing frame: {frame_playing} ({frame_playing/fps:.1f}s)")

    # Scan playing frames spread across the entire video to catch both hands
    # (some videos start with only one hand for the first section)
    music_duration = total_frames - frame_playing
    if music_duration > 0:
        # Sample ~40 frames evenly across the whole playing range
        n_samples = min(40, music_duration // 5)
        step = max(1, music_duration // n_samples) if n_samples > 0 else 1
        sample_frames = list(range(frame_playing, total_frames, step))[:40]
    else:
        sample_frames = [frame_playing]

    # --- Step 3: Collect hue values of pixels that changed from neutral to saturated ---
    changed_hues = []

    for f in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        region = hsv[y_top:y_bot, :, :]

        # Neutral: low saturation (white or black key)
        was_neutral = (neutral_region[:, :, 1] < 50)
        # Now: high saturation AND bright (lit colored key)
        is_saturated = (region[:, :, 1] > 80) & (region[:, :, 2] > 100)
        # Changed pixels
        changed = was_neutral & is_saturated
        hues = region[:, :, 0][changed]
        changed_hues.extend(hues.tolist())

    if not changed_hues:
        print("No color changes detected.  The video might use unusual colors or timing.")
        cap.release()
        return

    hues = np.array(changed_hues)
    print(f"\nCollected {len(hues)} changed pixels across {len(sample_frames)} frames")

    # --- Step 4: Cluster hues to find the two hand colors ---
    # Build a hue histogram (0-179 in OpenCV HSV)
    hist, bin_edges = np.histogram(hues, bins=180, range=(0, 180))

    # Smooth the histogram to find clean peaks (gaussian kernel, no scipy needed)
    kernel = np.exp(-np.arange(-9, 10)**2 / (2 * 3**2))
    kernel /= kernel.sum()
    smoothed = np.convolve(hist.astype(float), kernel, mode='same')

    # Find peaks: local maxima above a threshold
    threshold = max(smoothed) * 0.1
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > threshold:
            peaks.append((i, smoothed[i]))

    # Sort by strength
    peaks.sort(key=lambda x: -x[1])

    cap.release()

    if not peaks:
        print("Could not identify distinct color peaks.")
        return

    # --- Step 5: Report results ---
    print(f"\n{'='*50}")
    print(f"Detected {min(len(peaks), 4)} color candidate(s):")
    print(f"{'='*50}")

    color_names = {
        (0, 10): "Red",
        (10, 25): "Orange",
        (25, 35): "Yellow",
        (35, 75): "Green",
        (75, 95): "Cyan",
        (95, 130): "Blue",
        (130, 155): "Purple",
        (155, 180): "Pink/Red",
    }

    for i, (hue, strength) in enumerate(peaks[:4]):
        name = "Unknown"
        for (lo, hi), n in color_names.items():
            if lo <= hue < hi:
                name = n
                break
        pct = 100 * strength / sum(s for _, s in peaks[:4])
        print(f"  {i+1}. H={hue:3d}  ({name:10s})  strength={strength:.0f}  ({pct:.0f}%)")

    if len(peaks) >= 2:
        h1, h2 = peaks[0][0], peaks[1][0]
        n1, n2 = _name(h1, color_names), _name(h2, color_names)
        print(f"\nSuggested config (assign right/left based on the video):")
        print(f'  "colors": {{')
        print(f'    "right": {{ "h_min": {max(0,h1-15)}, "h_max": {min(179,h1+15)}, "s_min": 80, "v_min": 80 }},  // {n1} H={h1}')
        print(f'    "left":  {{ "h_min": {max(0,h2-15)}, "h_max": {min(179,h2+15)}, "s_min": 80, "v_min": 80 }}   // {n2} H={h2}')
        print(f'  }}')
        print(f"\n  Swap right/left if the assignment is wrong for your video.")
    elif len(peaks) == 1:
        print(f"\nOnly ONE color detected — this video may use a single color for both hands.")
        print(f"Hand separation will rely on pitch-based fallback.")


def _name(hue, color_names):
    for (lo, hi), n in color_names.items():
        if lo <= hue < hi:
            return n
    return "Unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-detect Synthesia hand colors")
    parser.add_argument("video", help="Path to Synthesia video")
    parser.add_argument("--frame-neutral", type=int, default=None,
                        help="Frame index for neutral keyboard (auto-detected if omitted)")
    parser.add_argument("--frame-playing", type=int, default=None,
                        help="Frame index where notes are playing (auto-detected if omitted)")
    args = parser.parse_args()

    detect_colors(args.video, args.frame_neutral, args.frame_playing)
