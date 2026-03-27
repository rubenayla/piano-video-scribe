#!/usr/bin/env python3
"""Detect piano notes from falling colored blocks above the keyboard.

Works with videos where hands are visible on the keyboard (Rousseau, Kassia
style). Detects colored rectangles in the "waterfall" area above the keyboard,
maps them to pitches, and extracts timing from vertical position.

Usage:
    from detect_falling_blocks import detect_notes_from_blocks
    notes = detect_notes_from_blocks(cap, note_x_map, y_kb_top, fps,
                                      color_ranges, note_speed_pps)
"""

import cv2
import numpy as np


def detect_blocks_in_frame(hsv_frame, waterfall_top, waterfall_bottom,
                           note_x_map, color_ranges, min_area=50):
    """Detect colored note blocks in a single frame's waterfall region.

    Args:
        hsv_frame: Full frame in HSV color space.
        waterfall_top: y-coordinate of the top of the waterfall region.
        waterfall_bottom: y-coordinate of the bottom (keyboard top).
        note_x_map: dict mapping MIDI pitch → x pixel center.
        color_ranges: dict with 'right' and 'left' keys, each containing
            h_min, h_max, s_min, v_min for HSV thresholding.
        min_area: minimum pixel area for a valid block.

    Returns:
        list of dicts: [{pitch, hand, bottom_y, top_y, width, height, area}]
    """
    waterfall = hsv_frame[waterfall_top:waterfall_bottom, :]
    wf_h = waterfall_bottom - waterfall_top

    if wf_h <= 0:
        return []

    blocks = []
    for hand_name, color in color_ranges.items():
        h_min = color.get('h_min', 0)
        h_max = color.get('h_max', 180)
        s_min = color.get('s_min', 80)
        v_min = color.get('v_min', 80)

        # HSV threshold
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, 255, 255])
        mask = cv2.inRange(waterfall, lower, upper)

        # Morphological cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        for i in range(1, n_labels):  # skip background (label 0)
            x, y, w, h, area = stats[i]
            if area < min_area or h < 3:
                continue

            x_center = x + w // 2
            # Map to nearest pitch
            best_pitch = min(note_x_map.keys(),
                             key=lambda p: abs(note_x_map[p] - x_center))
            if abs(note_x_map[best_pitch] - x_center) > w * 2:
                continue  # too far from any key

            blocks.append({
                'pitch': best_pitch,
                'hand': 0 if hand_name == 'right' else 1,
                'bottom_y': waterfall_top + y + h,  # in full-frame coords
                'top_y': waterfall_top + y,
                'width': w,
                'height': h,
                'area': area,
            })

    return blocks


def estimate_note_speed(cap, frame_idx_a, frame_idx_b, waterfall_top,
                        waterfall_bottom, note_x_map, color_ranges):
    """Estimate note speed (pixels/second) by tracking blocks between two frames.

    Args:
        cap: cv2.VideoCapture
        frame_idx_a, frame_idx_b: two frame indices to compare
        Others: same as detect_blocks_in_frame

    Returns:
        float: estimated speed in pixels per second, or None on failure
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = (frame_idx_b - frame_idx_a) / fps

    frames = []
    for fi in [frame_idx_a, frame_idx_b]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            return None
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    blocks_a = detect_blocks_in_frame(frames[0], waterfall_top, waterfall_bottom,
                                       note_x_map, color_ranges)
    blocks_b = detect_blocks_in_frame(frames[1], waterfall_top, waterfall_bottom,
                                       note_x_map, color_ranges)

    # Match blocks by pitch and approximate y-position
    displacements = []
    for ba in blocks_a:
        for bb in blocks_b:
            if ba['pitch'] != bb['pitch'] or ba['hand'] != bb['hand']:
                continue
            dy = ba['bottom_y'] - bb['bottom_y']
            # Block should have moved DOWN (increasing y = toward keyboard)
            if dy < 0:
                continue
            # Sanity: displacement should be reasonable
            expected_dy = dt * 200  # rough guess: ~200 px/s
            if 0.2 * expected_dy < dy < 5 * expected_dy:
                displacements.append(dy)

    if not displacements:
        return None

    median_dy = sorted(displacements)[len(displacements) // 2]
    speed = median_dy / dt
    return speed


def detect_notes_from_blocks(cap, note_x_map, waterfall_top, waterfall_bottom,
                              fps, total_frames, color_ranges,
                              note_speed_pps=None, frame_step=3,
                              start_frame=0):
    """Extract notes from falling blocks across all video frames.

    Args:
        cap: cv2.VideoCapture
        note_x_map: dict mapping MIDI pitch → x pixel center
        waterfall_top: y of waterfall region top
        waterfall_bottom: y of waterfall region bottom (keyboard top)
        fps: video frame rate
        total_frames: total number of frames
        color_ranges: dict with 'right'/'left' HSV ranges
        note_speed_pps: pixels per second (auto-detected if None)
        frame_step: process every Nth frame (3 = every 3rd frame)
        start_frame: first frame to process

    Returns:
        list of (pitch, hand, onset_sec, offset_sec) tuples
    """
    # Auto-detect note speed if not provided
    if note_speed_pps is None:
        # Use two frames 0.5s apart from middle of video
        mid = total_frames // 2
        gap = int(fps * 0.5)
        note_speed_pps = estimate_note_speed(
            cap, mid, mid + gap, waterfall_top, waterfall_bottom,
            note_x_map, color_ranges)
        if note_speed_pps is None:
            print("WARNING: Could not auto-detect note speed, using default 200 px/s")
            note_speed_pps = 200
        else:
            print(f"  Note speed: {note_speed_pps:.0f} px/s")

    # Collect detections from all frames
    all_detections = []  # (pitch, hand, onset_sec, offset_sec, area)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = 0

    for f_idx in range(start_frame, total_frames, frame_step):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip intermediate frames
        if frame_step > 1:
            for _ in range(frame_step - 1):
                cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_time = f_idx / fps

        blocks = detect_blocks_in_frame(hsv, waterfall_top, waterfall_bottom,
                                         note_x_map, color_ranges)

        for b in blocks:
            # Convert y-position to time
            # bottom_y is where the note starts (onset)
            # The keyboard line is at waterfall_bottom
            # Distance from bottom_y to keyboard = how far in the future
            onset_offset_px = waterfall_bottom - b['bottom_y']
            onset_sec = frame_time + onset_offset_px / note_speed_pps
            duration_sec = b['height'] / note_speed_pps
            offset_sec = onset_sec + duration_sec

            all_detections.append((
                b['pitch'], b['hand'], onset_sec, offset_sec, b['area']
            ))

        processed += 1
        if processed % 100 == 0:
            progress = f_idx / total_frames * 100
            print(f"  Frame {f_idx}/{total_frames} ({progress:.0f}%) — "
                  f"{len(all_detections)} raw detections")

    print(f"  {len(all_detections)} raw detections from {processed} frames")

    # Merge overlapping detections (same pitch, overlapping time)
    notes = merge_detections(all_detections)
    return notes


def merge_detections(detections, time_tolerance=0.1):
    """Merge overlapping detections of the same pitch into single notes.

    Groups detections by (pitch, hand), sorts by onset, and merges
    overlapping or near-overlapping entries. Similar to NMS in object
    detection but in (pitch, time) space.

    Args:
        detections: list of (pitch, hand, onset_sec, offset_sec, area)
        time_tolerance: max gap in seconds to still merge

    Returns:
        list of (pitch, hand, onset_sec, offset_sec) tuples
    """
    from collections import defaultdict

    # Group by (pitch, hand)
    groups = defaultdict(list)
    for pitch, hand, onset, offset, area in detections:
        groups[(pitch, hand)].append((onset, offset, area))

    notes = []
    for (pitch, hand), dets in groups.items():
        dets.sort()  # sort by onset

        # Merge overlapping detections
        merged = []
        current_onset, current_offset, total_area = dets[0]
        count = 1

        for onset, offset, area in dets[1:]:
            if onset <= current_offset + time_tolerance:
                # Overlapping or close — extend
                current_offset = max(current_offset, offset)
                total_area += area
                count += 1
            else:
                # Gap — emit current and start new
                merged.append((current_onset, current_offset))
                current_onset, current_offset = onset, offset
                total_area = area
                count = 1

        merged.append((current_onset, current_offset))

        for onset, offset in merged:
            notes.append((pitch, hand, onset, offset))

    # Sort by onset time
    notes.sort(key=lambda n: n[2])
    return notes
