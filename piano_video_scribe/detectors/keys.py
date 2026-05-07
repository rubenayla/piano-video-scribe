"""Key-press detector — extracts notes by watching keyboard key state changes.

Scans video frames for saturation changes on the keyboard face to detect
note-on (sudden saturation increase) and note-off (sudden decrease) events.
Robust against falling note bars (gradual color) vs actual key presses (sudden).
"""

import time as _time

import cv2
import numpy as np

from piano_video_scribe.keyboard import build_detector_regions
from piano_video_scribe.color import _classify_hand_from_hue


def extract_notes_from_video(cap, note_x_map, y_sample_top, y_sample_bot,
                             half_w, fps, total_frames, green_is_right, colors,
                             start_frame=0, frame_step=1, white_keys=None, cfg=None,
                             y_black=None, start_time=None, y_kb_top=None,
                             y_bk_bottom=None):
    """Extract notes from the video using frame-to-frame saturation delta.

    Instead of checking absolute color, detects the *moment* a key changes
    from neutral to colored (sudden saturation increase = note on, sudden
    decrease = note off).  This is robust against falling note bars (which
    create gradual color changes) vs actual key presses (sudden changes).

    Args:
        cap: OpenCV VideoCapture object.
        note_x_map: dict mapping MIDI pitch → x pixel center.
        y_sample_top, y_sample_bot: vertical bounds of the sampling zone.
        half_w: horizontal half-width for sampling (fallback).
        fps: video frame rate.
        total_frames: total number of frames.
        green_is_right: True if green = right hand (0).
        colors: color config dict (or None for defaults).
        start_frame: first frame to scan (skip intro).
        frame_step: scan every Nth frame (1 = every frame, 2 = every other, etc.)
        white_keys: list of white key x-centers (for computing detector bounds).

    Returns:
        List of (pitch, hand, onset_sec, offset_sec) tuples.
    """
    SAT_ON = 70       # saturation threshold for white keys: above = key pressed
    SAT_ON_BLACK = 90 # higher threshold for black keys to reject glow bleed
    SAT_OFF = 40      # saturation threshold: below = key released
    BLACK_SEMITONES_SET = {1, 3, 6, 8, 10}

    # Build detector regions
    if white_keys is not None:
        det_regions = build_detector_regions(note_x_map, white_keys,
                                            y_sample_bot, cfg=cfg, y_black=y_black,
                                            y_kb_top=y_kb_top, y_bk_bottom=y_bk_bottom)
    else:
        det_regions = {p: (x - half_w, x + half_w, y_sample_top, y_sample_bot)
                       for p, x in note_x_map.items()}

    pitches = sorted(note_x_map.keys())

    # Pre-compute detector slices for bulk numpy extraction
    # Group by y-zone to minimize per-pitch overhead
    pitch_slices = []
    for pitch in pitches:
        x_left, x_right, yt, yb = det_regions[pitch]
        pitch_slices.append((pitch, max(0, x_left), x_right, max(0, yt), yb))

    # Active notes: pitch → (hand, onset_frame)
    active = {}
    notes = []
    # Debounce: track consecutive frames above SAT_ON per pitch.
    # A note-on requires 2+ consecutive frames to filter single-frame
    # glows from adjacent key presses bleeding into black key zones.
    pending_on = {}  # pitch → (frame_idx, hand)

    # Auto-detect the first playing frame: scan forward from start_frame
    # looking for saturated pixels on the keyboard face (where keys light up
    # when pressed).  Uses a band 30px above y_white to 5px below — this
    # catches key presses but avoids falling note bars higher up.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    play_start = start_frame
    face_yt = max(0, y_sample_bot - 30)
    face_yb = min(y_sample_bot + 5, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 1)
    x_lo = min(x for _, x, _, _, _ in pitch_slices) if pitch_slices else 0
    x_hi = max(x for _, _, x, _, _ in pitch_slices) if pitch_slices else 1920
    face_yt, face_yb = int(face_yt), int(face_yb)
    if start_time is not None:
        # Manual override: skip auto-detection (allows scanning before keyboard frame)
        play_start = int(start_time * fps)
        play_start = max(0, min(play_start, total_frames - 1))
        print(f"  Using manual start time: frame {play_start} ({play_start/fps:.1f}s)")
    else:
        for f_scan in range(start_frame, min(start_frame + int(fps * 30), total_frames), 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_scan)
            ret, frame = cap.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            region = hsv[face_yt:face_yb, x_lo:x_hi, 1]
            n_saturated = int(np.sum(region > SAT_ON))
            if n_saturated > 200:  # enough saturated pixels = key actually pressed
                play_start = f_scan
                break
        if play_start > start_frame:
            print(f"  Skipping intro: first notes at frame {play_start} "
                  f"({play_start/fps:.1f}s)")

    # Seek once to play_start, then read sequentially (much faster than
    # seeking every frame — avoids H.264 keyframe backtracking)
    cap.set(cv2.CAP_PROP_POS_FRAMES, play_start)
    start_frame = play_start
    t0 = _time.time()

    for f_idx in range(start_frame, total_frames, frame_step):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if frame_step > 1 (read but don't process)
        if frame_step > 1:
            for _ in range(frame_step - 1):
                cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]  # extract saturation once
        val_channel = hsv[:, :, 2]  # brightness — needed to gate dark pixels

        confirmed_this_frame = []  # track new onsets to detect bar artifacts

        # First pass: compute saturation for all keys this frame
        frame_sat = {}
        for pitch, x1, x2, y1, y2 in pitch_slices:
            if y1 >= y2 or x1 >= x2:
                frame_sat[pitch] = 0.0
                continue
            V_MIN = 30
            region_s = sat_channel[y1:y2, x1:x2]
            region_v = val_channel[y1:y2, x1:x2]
            bright_mask = region_v >= V_MIN
            if np.any(bright_mask):
                frame_sat[pitch] = float(np.mean(region_s[bright_mask]))
            else:
                frame_sat[pitch] = 0.0

        for pitch, x1, x2, y1, y2 in pitch_slices:
            if y1 >= y2 or x1 >= x2:
                continue
            sat = frame_sat[pitch]

            # Glow bleed suppression for black keys
            is_black = (pitch % 12) in BLACK_SEMITONES_SET
            if is_black and sat > 0:
                # 1. Neighbor ratio: suppress if well below adjacent white key
                neighbor_sat = max(frame_sat.get(pitch - 1, 0),
                                   frame_sat.get(pitch + 1, 0))
                if neighbor_sat > 0 and sat < neighbor_sat * 0.6:
                    sat = 0.0
                # 2. Coverage check: a real key press illuminates the full
                #    detector width. A falling note bar edge clips only 1-2
                #    pixels on one side.  Require >=60% of columns to have
                #    saturated pixels.
                elif (x2 - x1) > 0 and (x2 - x1) <= 8:
                    col_region_s = sat_channel[y1:y2, x1:x2]
                    col_region_v = val_channel[y1:y2, x1:x2]
                    n_cols = x2 - x1
                    lit_cols = 0
                    for cx in range(n_cols):
                        col_s = col_region_s[:, cx]
                        col_v = col_region_v[:, cx]
                        bright = col_v >= 30
                        if np.any(bright) and float(np.mean(col_s[bright])) > SAT_ON_BLACK:
                            lit_cols += 1
                    if lit_cols < n_cols * 0.6:
                        sat = 0.0

            thresh = SAT_ON_BLACK if is_black else SAT_ON
            if pitch not in active and sat > thresh:
                # Debounce: require 2 consecutive frames above threshold
                # to filter single-frame glow bleed from adjacent keys.
                if pitch in pending_on and f_idx - pending_on[pitch][0] <= frame_step:
                    # Second consecutive frame — confirm note on
                    hand = pending_on.pop(pitch)[1]
                    active[pitch] = (hand, f_idx - frame_step)
                    confirmed_this_frame.append(pitch)
                else:
                    # First frame — classify hand and store as pending
                    region = hsv[y1:y2, x1:x2]
                    h_ch = region[:, :, 0].flatten().astype(float)
                    s_ch = region[:, :, 1].flatten().astype(float)
                    mask = s_ch > 50
                    hue = float(np.mean(h_ch[mask])) if np.sum(mask) >= 3 else None
                    hand = _classify_hand_from_hue(hue, green_is_right, colors)
                    if hand is not None:
                        pending_on[pitch] = (f_idx, hand)
                    # If hue doesn't match any configured color, skip —
                    # it's likely a UI element, border glow, or artifact.

            elif sat <= thresh and pitch in pending_on:
                # Saturation dropped before second frame — cancel pending
                del pending_on[pitch]

            note_off = False
            if pitch in active and sat < SAT_OFF:
                note_off = True
            elif pitch in active and sat > thresh:
                # Saturation is high — check if the hue still matches a
                # configured hand color.  If not, a non-musical overlay
                # (e.g. phone nav bar) has replaced the note bar.
                region_h = hsv[y1:y2, x1:x2, 0]
                region_s = hsv[y1:y2, x1:x2, 1]
                s_mask = region_s > 50
                hue = float(np.mean(region_h[s_mask])) if np.sum(s_mask) >= 3 else None
                if _classify_hand_from_hue(hue, green_is_right, colors) is None:
                    note_off = True
            if note_off and pitch in active:
                hand, onset_frame = active.pop(pitch)
                onset_sec = onset_frame / fps
                offset_sec = f_idx / fps
                if offset_sec - onset_sec > 0.02:
                    notes.append((pitch, hand, onset_sec, offset_sec))

        if f_idx % 500 == 0 and f_idx > start_frame:
            elapsed = _time.time() - t0
            rate = (f_idx - start_frame) / elapsed if elapsed > 0 else 0
            eta = (total_frames - f_idx) / rate if rate > 0 else 0
            print(f"  Frame {f_idx}/{total_frames} "
                  f"({100*f_idx/total_frames:.0f}%) — {len(notes)} notes "
                  f"— {rate:.0f} fps, ETA {eta:.0f}s")

    # Close remaining active notes
    for pitch, (hand, onset_frame) in active.items():
        onset_sec = onset_frame / fps
        offset_sec = (total_frames - 1) / fps
        if offset_sec - onset_sec > 0.02:
            notes.append((pitch, hand, onset_sec, offset_sec))

    notes.sort(key=lambda n: n[2])
    return notes
