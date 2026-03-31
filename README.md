# piano-video-scribe

Convert [Synthesia](https://synthesiagame.com) piano videos into hand-separated MIDI and PDF sheet music. Pure computer vision — no ML, no audio transcription.

## Quick start

```bash
git clone https://github.com/rubenayla/piano-video-scribe
cd piano-video-scribe
pip install -r requirements.txt

# Download video
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" "https://youtu.be/VIDEO_ID" -o video.mp4

# Run pipeline (BPM auto-detected if omitted)
python pianovideoscribe.py video.mp4 output.mid

# Open in MuseScore (Preferences > Import > MIDI > Shortest note: 16th)
open -a "MuseScore 4" output.mid
```

## How it works

The pipeline reads falling colored blocks from the video to extract notes, then quantizes them to a musical grid.

### 1. Keyboard detection — [`piano_video_scribe/keyboard.py`](piano_video_scribe/keyboard.py)

Find the keyboard in a reference frame to build a pitch map (x-pixel → MIDI note).

- `detect_keyboard()` — Scan for regularly-spaced white keys and black keys using HSV thresholding.
- `find_first_c()` — Identify which white key is C by matching the black key grouping pattern (2-3-2-3...).
- `build_note_x_map()` — Assign MIDI note numbers to x-positions based on keyboard size and octave heuristics.

### 2. Note detection (falling blocks) — [`piano_video_scribe/detectors/falling_blocks.py`](piano_video_scribe/detectors/falling_blocks.py)

Track colored blocks as they fall through the waterfall area, collecting onset predictions from every frame.

**Block detection** (per frame):
1. `auto_detect_colors()` — Sample frames to find the two hand colors via hue histogram clustering.
2. `make_color_mask()` — HSV threshold: pixels matching hue range + saturation minimum → binary mask. Saturation threshold is configurable per color via `s_min_per_color` in settings.json (needed to separate rapid repeated notes whose gaps have lower saturation).
3. Morphological erosion (horizontal 1×5 kernel) breaks bridges between adjacent keys.
4. `cv2.findContours()` traces connected white regions → bounding rectangles.
5. Size/shape filters reject UI elements, progress bars, and noise.

**Block tracking** across frames:
- Each block is tracked by matching (pitch, color, expected y-position) across consecutive frames.
- Every observation produces an independent onset prediction: `onset = t + (keyboard_y - y_bot) / fall_speed`.
- The **median** of all predictions for a track gives the final onset — naturally rejecting outliers from clipped or glow-corrupted frames.
- `measure_fall_speed()` — Measure fall rate (pixels/frame) from block displacement across frame pairs.

**Configurable parameters** (via `settings.json`):
- `glow_margin` — Pixels above keyboard to exclude from scan (avoids key-press glow artifacts, default 50).
- `s_min_per_color` — Per-color saturation threshold, e.g. `{"h46": 145, "h108": 80}` for different block shades.

### 3. Hand separation — [`piano_video_scribe/color.py`](piano_video_scribe/color.py)

- Colors are auto-assigned: the color with higher average pitch = right hand, lower = left hand.
- Synthesia videos use two shades per hand (e.g. bright green + light green). Both share the same hue, differing only in saturation/brightness.

### 4. Quantization — [`piano_video_scribe/quantization.py`](piano_video_scribe/quantization.py)

Snap raw onset times to a musical grid (16th-note resolution).

- `quantize_onsets_viterbi()` — Viterbi DP finds the globally optimal grid assignment minimizing interval + absolute-position error.
- `quantize_onsets_adaptive()` — Two-pass adaptive tempo tracking: first Viterbi pass estimates a smooth local BPM curve via `estimate_local_bpm()`, then `warp_onsets()` compensates for drift before a second Viterbi pass. Handles tempo changes mid-piece (e.g. ritardando, fermata).
- Chord notes within 50ms are grouped so the quantizer places them simultaneously.

### 5. Post-processing — [`piano_video_scribe/midi_output.py`](piano_video_scribe/midi_output.py)

- `make_monophonic()` — Keep only the highest (RH) or lowest (LH) note at each time.
- `remove_overlaps()` — Cut held notes at the next onset while preserving chords.
- Transpose, key signature metadata (auto-detected via music21).

### 6. Output

Two-track Type-1 MIDI (right hand + left hand) via `build_track()`.

## CLI arguments

| Argument | Description |
|---|---|
| `video` | Path to Synthesia MP4 |
| `midi` | *(optional)* Input MIDI — if provided, uses video only for hand separation |
| `output` | Path for output MIDI |
| `--bpm N` | BPM (auto-detected from audio if omitted) |
| `--detector` | `falling-blocks` (default) or `keys` (key-lighting fallback) |
| `--right-hand` | `no-overlap` (default), `monophonic`, or `normal` |
| `--left-hand` | `no-overlap` (default), `monophonic`, or `normal` |
| `--green-hand` | Which hand is green: `right` (default) or `left` |
| `--start-beat N` | Beat position of first note (auto-detected if omitted) |
| `--time-sig` | Time signature, e.g. `6/8`, `3/4` (default: 4/4) |
| `--key` | Key signature for MIDI metadata, e.g. `G`, `Em` (auto-detected if omitted) |
| `--transpose N` | Transpose by N semitones |
| `--settings` | Path to settings.json (CLI flags override) |
| `--frame N` | Frame index for keyboard detection (default: 5) |
| `--config` | Path to color config JSON |
| `--triplet` | Use combined grid (16ths + triplets) |
| `--dry-run` | Detect keyboard only, print stats, exit |

Settings are auto-saved as `settings.json` next to the output MIDI for reproducibility.

## Detectors

**Falling blocks** (default): Reads colored block contours above the keyboard. Handles repeated notes, 16th-note runs, and re-attacks that the key-press detector misses (keys stay lit between re-attacks, but blocks are visually distinct).

**Key-press**: Scans key saturation frame-by-frame. Fallback for videos without falling blocks (if any exist). Use `--detector keys`.

## Using with Claude Code

The repo includes a `/PianoVideoScribe` slash command. In any Claude Code session inside this repo:

> /PianoVideoScribe https://youtu.be/VIDEO_ID

The agent reads `.agents/pipeline.md` and handles everything: download, detection, quantization, and output.

## Tests

```bash
python -m pytest tests/ --ignore=tests/test1 -v
```

Tests run the full pipeline on test videos and check output MIDI properties (note counts, grid alignment, pitch correctness). ~45 seconds.
