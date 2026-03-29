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

### 1. Keyboard detection

Find the keyboard in a reference frame to build a pitch map (x-pixel -> MIDI note).

- [`detect_keyboard()`](pianovideoscribe.py#L63) — Scan for regularly-spaced white keys and black keys above them using HSV thresholding.
- [`find_first_c()`](pianovideoscribe.py#L356) — Identify which white key is C by matching the black key grouping pattern (2-3-2-3...).
- [`build_note_x_map()`](pianovideoscribe.py#L404) — Assign MIDI note numbers to x-positions.

### 2. Note detection (falling blocks)

Detect colored block contours above the keyboard and project their positions to onset/offset times.

- [`auto_detect_colors()`](detect_falling_blocks.py#L45) — Sample frames to find the two hand colors via hue histogram clustering.
- [`detect_blocks_in_frame()`](detect_falling_blocks.py#L136) — HSV color masking + morphological erosion + OpenCV contour detection per frame.
- [`measure_fall_speed()`](detect_falling_blocks.py#L607) — Track blocks across frame pairs to measure pixels/frame fall rate.
- [`extract_notes_contour()`](detect_falling_blocks.py#L806) — Collect block observations across sampled frames, project onset times via `t = t_now + (keyboard_y - block_y) / speed`, chain-merge observations of the same block, deduplicate overlapping same-pitch notes.
- [`_calibrate_note_map_with_blocks()`](detect_falling_blocks.py#L1104) — Affine correction aligning block x-positions to keyboard x-positions.

### 3. Hand separation

- Colors are auto-assigned: the color with higher average pitch = right hand, lower = left hand.
- For the key-press detector: [`classify_hand()`](pianovideoscribe.py#L703) reads HSV color at the key position per frame.

### 4. Quantization

Snap raw onset times to a musical grid (16th-note resolution).

- [`quantize_onsets_viterbi()`](pianovideoscribe.py#L1128) — Viterbi DP finds the globally optimal grid assignment minimizing interval + absolute-position error.
- [`quantize_onsets_adaptive()`](pianovideoscribe.py#L1268) — Two-pass adaptive tempo tracking: first Viterbi pass estimates a smooth local BPM curve via [`estimate_local_bpm()`](pianovideoscribe.py#L1201), then [`warp_onsets()`](pianovideoscribe.py#L1250) compensates for drift before a second Viterbi pass. Handles videos where tempo drifts even slightly (e.g., 89 vs 90 BPM).
- Chord notes within 50ms are grouped so the quantizer places them simultaneously.
- Each hand's onsets are shifted to start at t=0 before quantizing to prevent absolute-position cost from distorting intervals.

### 5. Post-processing

- [`make_monophonic()`](pianovideoscribe.py#L1449) — Keep only the highest (RH) or lowest (LH) note at each time.
- [`remove_overlaps()`](pianovideoscribe.py#L1414) — Cut held notes at the next onset while preserving chords.
- Auto-detect hand start offset when hands enter at different times.
- Transpose, key signature metadata (auto-detected via music21).

### 6. Output

Two-track Type-1 MIDI (right hand + left hand) via [`build_track()`](pianovideoscribe.py#L1508).

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
