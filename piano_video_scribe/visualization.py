"""Visualization utilities for piano-video-scribe."""

import cv2
import numpy as np

from piano_video_scribe.constants import NOTE_NAMES


def generate_summary_image(cap, frame_idx, white_keys, black_keys, y_white,
                           y_black, y_kb_top, note_x_map, det_regions, bpm,
                           key_sig, green_is_right, rh_count, lh_count,
                           output_path):
    """Generate a visual summary image showing keyboard detection + settings.

    Crops to the keyboard area with some context above, so the detectors
    and key labels are clearly visible regardless of keyboard size.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return

    h, w = frame.shape[:2]
    BLACK_SEMI = {1, 3, 6, 8, 10}

    # Colors for detector overlays
    WHITE_DET_COLOR = (0, 255, 0)    # green overlay for white key detectors
    BLACK_DET_COLOR = (0, 0, 255)    # red overlay for black key detectors

    # Draw detector regions as semi-transparent overlays
    overlay = frame.copy()
    for pitch, (x1, x2, y1, y2) in det_regions.items():
        is_black = (pitch % 12) in BLACK_SEMI
        color = BLACK_DET_COLOR if is_black else WHITE_DET_COLOR
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    # Draw borders
    for pitch, (x1, x2, y1, y2) in det_regions.items():
        is_black = (pitch % 12) in BLACK_SEMI
        color = BLACK_DET_COLOR if is_black else WHITE_DET_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Draw scan lines
    cv2.line(frame, (0, y_white), (w, y_white), (255, 255, 0), 1)
    if y_black is not None:
        cv2.line(frame, (0, y_black), (w, y_black), (0, 255, 255), 1)

    # Label C notes on the keyboard
    for pitch, x_center in note_x_map.items():
        if pitch % 12 == 0:
            octave = pitch // 12 - 1
            cv2.putText(frame, f"C{octave}", (x_center - 10, y_white + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- Info panel at top ---
    rh_color_name = "green" if green_is_right else "blue"
    lh_color_name = "blue" if green_is_right else "green"
    rh_bgr = (0, 200, 0) if green_is_right else (255, 150, 0)
    lh_bgr = (255, 150, 0) if green_is_right else (0, 200, 0)

    panel_h = 80
    frame[:panel_h, :] = (30, 30, 30)

    # Line 1: settings
    y_line1 = 22
    info_line = (f"BPM: {bpm}    "
                 f"Key: {key_sig or 'auto'}    "
                 f"Keyboard: {len(white_keys)} white + {len(black_keys)} black keys    "
                 f"Detection frame: {frame_idx}")
    cv2.putText(frame, info_line, (10, y_line1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Line 2: hand info with color swatches
    y_line2 = 45
    cv2.rectangle(frame, (10, y_line2 - 12), (26, y_line2 + 2), rh_bgr, -1)
    cv2.putText(frame, f"Right hand ({rh_color_name}): {rh_count} notes", (32, y_line2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, rh_bgr, 1)
    x2 = 320
    cv2.rectangle(frame, (x2, y_line2 - 12), (x2 + 16, y_line2 + 2), lh_bgr, -1)
    cv2.putText(frame, f"Left hand ({lh_color_name}): {lh_count} notes", (x2 + 22, y_line2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, lh_bgr, 1)

    # Line 3: legend
    y_line3 = 68
    cv2.rectangle(frame, (10, y_line3 - 10), (22, y_line3), WHITE_DET_COLOR, -1)
    cv2.putText(frame, "White key detector", (28, y_line3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.rectangle(frame, (190, y_line3 - 10), (202, y_line3), BLACK_DET_COLOR, -1)
    cv2.putText(frame, "Black key detector", (208, y_line3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.line(frame, (370, y_line3 - 5), (390, y_line3 - 5), (255, 255, 0), 1)
    cv2.putText(frame, "White key scan", (395, y_line3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    if y_black is not None:
        cv2.line(frame, (530, y_line3 - 5), (550, y_line3 - 5), (0, 255, 255), 1)
        cv2.putText(frame, "Black key scan", (555, y_line3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Crop to keyboard area: show from some context above the keys to below y_white.
    # This ensures the keyboard is prominent even on videos with huge falling-notes areas.
    kb_top = y_kb_top if y_kb_top is not None and y_kb_top > 0 else y_black - 50
    context_above = max(30, (y_white - kb_top))  # show ~1 keyboard height above
    crop_top = max(0, kb_top - context_above)
    crop_bot = min(h, y_white + 30)
    # Keep the info panel (top 80px) — paste it onto the cropped image
    panel = frame[:panel_h, :].copy()
    cropped = frame[crop_top:crop_bot, :]
    # Stack: panel on top, cropped keyboard below
    result = np.vstack([panel, cropped])

    cv2.imwrite(output_path, result)
    print(f"Summary image: {output_path}")
