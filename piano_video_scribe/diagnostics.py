"""Diagnostic visualization for falling-blocks detector.

Generates annotated frames showing exactly what the detector sees:
detected blocks, color classification, pitch assignment, scan region bounds.
"""

import cv2
import numpy as np

from piano_video_scribe.constants import NOTE_NAMES, midi_to_name


def diagnostic_frame(video_path, timestamp, note_map, keyboard_y, y_min, y_max,
                     colors, fall_speed=None, output_path=None):
    """Generate a diagnostic image for a specific video timestamp.

    Args:
        video_path: Path to video file.
        timestamp: Time in seconds, or frame number if int > 1000.
        note_map: dict {midi_note: x_position} from keyboard detector.
        keyboard_y: y-coordinate of keyboard top.
        y_min, y_max: Scan region bounds.
        colors: Color definitions from auto_detect_colors().
        fall_speed: Pixels/frame (optional, for onset projection lines).
        output_path: Where to save. Defaults to /tmp/diag_{timestamp}.png.

    Returns:
        Path to saved image.
    """
    from piano_video_scribe.detectors.falling_blocks import detect_blocks_in_frame
    from piano_video_scribe.color import make_color_mask

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if isinstance(timestamp, float) or (isinstance(timestamp, (int, float)) and timestamp < 1000):
        frame_num = int(timestamp * fps)
    else:
        frame_num = int(timestamp)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_num}")

    h, w = frame.shape[:2]
    t_sec = frame_num / fps
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Make a copy for drawing
    vis = frame.copy()

    # --- Draw scan region bounds ---
    cv2.line(vis, (0, y_min), (w, y_min), (255, 0, 255), 1)
    cv2.line(vis, (0, y_max), (w, y_max), (255, 0, 255), 1)
    cv2.line(vis, (0, keyboard_y), (w, keyboard_y), (0, 255, 255), 2)
    cv2.putText(vis, f"scan y={y_min}", (5, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    cv2.putText(vis, f"scan y={y_max}", (5, y_max - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    cv2.putText(vis, f"keyboard y={keyboard_y}", (5, keyboard_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # --- Draw note grid lines on keyboard ---
    white_semi = {0, 2, 4, 5, 7, 9, 11}
    for midi, x in sorted(note_map.items()):
        is_black = midi % 12 not in white_semi
        color = (100, 100, 100) if is_black else (60, 60, 60)
        cv2.line(vis, (x, y_min), (x, keyboard_y), color, 1)
        # Label every C
        if midi % 12 == 0:
            label = f"C{midi // 12 - 1}"
            cv2.putText(vis, label, (x - 8, keyboard_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- Detect blocks and annotate ---
    blocks = detect_blocks_in_frame(hsv, y_min, y_max, colors=colors)

    # Build x_to_pitch
    pitches_sorted = sorted(note_map.items(), key=lambda kv: kv[1])
    x_min_kb = pitches_sorted[0][1] - 20
    x_max_kb = pitches_sorted[-1][1] + 20
    boundaries = []
    for i, (midi, x) in enumerate(pitches_sorted):
        x_lo = (pitches_sorted[i - 1][1] + x) / 2.0 if i > 0 else x - 20
        x_hi = (x + pitches_sorted[i + 1][1]) / 2.0 if i < len(pitches_sorted) - 1 else x + 20
        boundaries.append((midi, x_lo, x_hi, x))

    def x_to_pitch(cx):
        if cx < x_min_kb or cx > x_max_kb:
            return None
        for midi, x_lo, x_hi, _ in boundaries:
            if x_lo <= cx <= x_hi:
                return midi
        return min(boundaries, key=lambda b: abs(b[3] - cx))[0]

    # Color map for drawing
    color_bgr = {}
    for c in colors:
        if c['h_center'] < 60:  # green-ish
            color_bgr[c['name']] = (0, 255, 0)
        elif c['h_center'] > 90:  # blue-ish
            color_bgr[c['name']] = (255, 150, 0)
        else:
            color_bgr[c['name']] = (0, 200, 200)

    px_per_sec = fall_speed * fps if fall_speed else None

    for cx, bw, bh, y_bot, color_name in blocks:
        pitch = x_to_pitch(cx)
        y_top = y_bot - bh
        bgr = color_bgr.get(color_name, (200, 200, 200))

        # Draw block outline
        x1 = int(cx - bw / 2)
        x2 = int(cx + bw / 2)
        cv2.rectangle(vis, (x1, y_top), (x2, y_bot), bgr, 2)

        # Label with pitch and color
        if pitch is not None:
            name = midi_to_name(pitch)
            label = f"{name} [{color_name}]"
        else:
            label = f"?? [{color_name}]"
        cv2.putText(vis, label, (x1, y_top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, bgr, 1)

        # Draw onset projection line if fall_speed known
        if pitch is not None and px_per_sec:
            onset_sec = t_sec + max(0, keyboard_y - y_bot) / px_per_sec
            onset_label = f"onset={onset_sec:.2f}s"
            cv2.putText(vis, onset_label, (x1, y_bot + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, bgr, 1)
            # Draw projection line from block bottom to keyboard
            cv2.line(vis, (int(cx), y_bot), (int(cx), keyboard_y), bgr, 1)

    # --- Info box ---
    info = f"t={t_sec:.2f}s  frame={frame_num}  blocks={len(blocks)}"
    cv2.rectangle(vis, (0, 0), (350, 20), (0, 0, 0), -1)
    cv2.putText(vis, info, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Color legend
    for i, c in enumerate(colors):
        bgr = color_bgr.get(c['name'], (200, 200, 200))
        y_leg = 35 + i * 18
        cv2.rectangle(vis, (5, y_leg - 10), (20, y_leg + 2), bgr, -1)
        cv2.putText(vis, f"{c['name']}: H={c['h_min']}-{c['h_max']}",
                    (25, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.35, bgr, 1)

    if output_path is None:
        output_path = f"/tmp/diag_{t_sec:.2f}s.png"
    cv2.imwrite(output_path, vis)
    return output_path


def diagnose(video_path, timestamp, settings_path=None):
    """One-call convenience: set up detector and generate diagnostic frame.

    Args:
        video_path: Path to video file.
        timestamp: Time in seconds.
        settings_path: Optional path to settings.json.

    Returns:
        Path to saved diagnostic image.
    """
    from piano_video_scribe.keyboard import detect_keyboard, build_note_x_map
    from piano_video_scribe.detectors.falling_blocks import (
        find_keyboard_y, auto_detect_colors, measure_fall_speed
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Keyboard detection
    result = detect_keyboard(cap, frame_idx=5)
    wk, bk = result[0], result[1]
    note_map = build_note_x_map(wk, bk, 21)

    # Falling blocks setup
    kb_y = find_keyboard_y(cap)

    # Auto-detect y_min (skip UI bar)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(150, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
    ret, frame = cap.read()
    y_min = 5
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for _y in range(0, kb_y // 2):
            _row = hsv[_y, :, :]
            _sat = ((_row[:, 1] > 50) & (_row[:, 2] > 50)).sum()
            if _sat > w * 0.05:
                y_min = _y
                break
    y_max = max(y_min + 10, kb_y - 50)

    colors = auto_detect_colors(cap, y_min, y_max)
    fall_speed = measure_fall_speed(cap, y_min, kb_y)
    cap.release()

    return diagnostic_frame(video_path, timestamp, note_map, kb_y,
                            y_min, y_max, colors, fall_speed)
