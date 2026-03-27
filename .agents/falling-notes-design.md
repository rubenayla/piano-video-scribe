<!-- reference — read when relevant -->
# Falling Notes Detector — Design

## Goal
Detect piano notes from falling colored blocks in videos where hands are
visible on the keyboard (Rousseau, Kassia style). Complements the existing
keyboard-face detector which fails when hands obscure the keys.

## Architecture
New module: `detect_falling_blocks.py` — same output format as the keyboard
detector (list of `(pitch, hand, onset_sec, offset_sec)` tuples).

The pipeline selects the detector based on video type:
- Keyboard detector: Synthesia-style (no hands, colored keys)
- Falling blocks detector: hands visible, colored blocks above keyboard
- Audio detector: fallback (existing librosa-based)

## Algorithm

### Phase 1: Calibration (single frame)
1. Detect keyboard with existing `detect_keyboard()`
2. Build `note_x_map` (pitch → x pixel)
3. Find the "waterfall region" = area above keyboard top
4. Auto-detect note speed (pixels/second) from two frames

### Phase 2: Per-frame block detection
1. Crop to waterfall region (above keyboard)
2. HSV threshold per hand color → binary mask
3. Morphological cleanup (close gaps, remove noise)
4. `cv2.connectedComponentsWithStats()` → bounding boxes
5. Map each box: x_center → pitch, height → duration, bottom_y → onset offset

### Phase 3: Cross-frame merging
- Group detections by pitch + overlapping time window
- Merge: median onset, max duration (like NMS in object detection)
- Handles the same block being detected in multiple consecutive frames

## Key Parameters
- `note_speed`: pixels per second (auto-detected or manual)
- `color_ranges`: HSV ranges for each hand color
- `min_block_area`: minimum pixel area to count as a note
- `merge_tolerance`: time window for merging duplicate detections (e.g., 50ms)

## Advantages Over Keyboard Detector
- Works when hands cover keys
- Gets note duration directly from block height
- Can see future notes (above keyboard line)
- Single-frame detection possible (multi-frame improves accuracy)
