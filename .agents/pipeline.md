# AGENT.md — PianoVideoScribe

AI agent reference. Everything here is verified to work. No further research needed.

---

## Full pipeline: YouTube URL → PDF sheet music

### 1. Download video

```bash
pip install yt-dlp
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
  "https://youtu.be/VIDEO_ID" -o video.mp4
```

### 2. Extract audio

```bash
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 44100 audio.wav
```

### 3. Transcribe audio → MIDI

```bash
pip install piano_transcription_inference librosa
```

Model download (~165 MB, one-time, auto on first run):
```python
from piano_transcription_inference import PianoTranscription, sample_rate
import librosa
audio, _ = librosa.load('audio.wav', sr=sample_rate, mono=True)
transcriptor = PianoTranscription(device='cpu')
transcriptor.transcribe(audio, 'transcription.mid')
```

Output: `transcription.mid` — 120 BPM, 384 TPB, raw audio timestamps (not quantized).

### 4. Separate hands

```bash
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm BPM --left-hand no-overlap
```

Replace `BPM` with the actual BPM of the song (see "Finding the BPM" below).

Each hand can be processed independently with `--right-hand MODE` and `--left-hand MODE`:

- `normal` (default) — no processing, keep all notes as transcribed
- `no-overlap` — cut held notes when the next onset arrives, but keep chords (simultaneous notes)
  intact. Recommended for most songs: piano players naturally hold notes slightly into the next for
  legato/pedal sustain, creating overlapping voices in the notation. This cleans it up while
  preserving chordal voicing.
- `monophonic` — strictly one note at a time (keeps highest for right hand, lowest for left).
  Good when you want a single clean melody/bass line with no chords at all.

Example with both hands processed:
```bash
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm BPM --right-hand monophonic --left-hand no-overlap
```

### 5. Export to PDF and open for the user

```bash
"/Applications/MuseScore 4.app/Contents/MacOS/mscore" output.mid -o sheet.pdf
```

MuseScore 4 prints QML type warnings on startup — these are harmless, ignore them.

For MuseScore 3:
```bash
"/Applications/MuseScore 3.app/Contents/MacOS/mscore" output.mid -o sheet.pdf
```

**Always open the result in MuseScore for the user** — do not just tell them the file path:
```bash
open -a "MuseScore 4" output.mid
```

This saves the user from having to switch apps manually.

---

## Finding the BPM

If the user doesn't provide the BPM, detect it with librosa and **ask the user to confirm**:

```python
import librosa, numpy as np
audio, sr = librosa.load('audio.wav', sr=None, mono=True)
oenv = librosa.onset.onset_strength(y=audio, sr=sr)
tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
ac = np.mean(tg, axis=1)
bpms = librosa.tempo_frequencies(tg.shape[0], sr=sr)
# Filter to reasonable range and show top candidates
mask = (bpms >= 40) & (bpms <= 200)
top_idx = np.argsort(ac[mask])[::-1][:3]
candidates = [(bpms[mask][i], ac[mask][i]) for i in top_idx]
for bpm, strength in candidates:
    print(f"  {bpm:.0f} BPM  (strength: {strength:.2f})")
```

Present the top 2–3 candidates to the user. Common pitfalls:
- Librosa often picks **double-time** (e.g. 152 instead of 76). If the top two candidates
  are in a 2:1 ratio with similar strength, the slower one is usually correct.
- Do not try to look it up online — most songs processed here are too niche for reliable
  online BPM databases.

---

## Config files

Use `--config path/to/config.json` to load channel/video-specific settings (colors, sampling zone,
keyboard frame). Configs live in `configs/` and can be reused across videos from the same source.

```bash
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 67 --config configs/synthesia_cyan_blue.json
```

Config JSON structure:
```json
{
  "name": "Description (informational only)",
  "colors": {
    "green": { "h_min": 40, "h_max": 65, "s_min": 100, "v_min": 80 },
    "blue":  { "h_min": 85, "h_max": 125, "s_min": 80, "v_min": 80 }
  },
  "sampling": {
    "y_offset_top": 90,
    "y_offset_bot": 0,
    "half_w": 10
  },
  "keyboard": {
    "frame": 140
  }
}
```

- **colors**: HSV thresholds for green (right hand) and blue (left hand) detection
- **sampling**: y_offset_top/bot are offsets from the detected white key row; half_w is the horizontal sample width
- **keyboard.frame**: which video frame to use for keyboard detection (should be note-free)
- Quantization and BPM are per-song, not in the config — pass them via CLI flags

Available configs:
- `configs/default.json` — Standard Synthesia colors (blue H=100–120)
- `configs/synthesia_cyan_blue.json` — Synthesia videos using cyan left hand (H≈93, range 85–125)

CLI `--frame` overrides the config's keyboard.frame when explicitly provided.

---

## Calibration guide

### Keyboard detection frame

Default is frame 5. Use `--dry-run` to test without writing output:

```bash
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm BPM --frame N --dry-run
```

Choose N such that:
- The keyboard is fully visible (no notes falling on top of it yet)
- `White keys detected` ≈ expected number (e.g. 52 for full 88-key, 26–35 for partial)
- `Black keys detected` ≈ expected number (e.g. 36 for full, 15–22 for partial)

Early frames (1–10) are usually safe. Avoid frames > ~100 — notes will be visible on the keys.

### Color thresholds

Defaults in `classify_hand()` — overridable via `--config`:

```
Green (right hand): H 40–65, S > 100, V > 80
Blue  (left hand):  H 100–120, S > 80, V > 80
```

To override, create a config JSON (see "Config files" above) or pass a matching existing one.

If your video uses non-default colors, inspect a frame in any image editor or with:
```python
import cv2
frame = cv2.VideoCapture('video.mp4')
frame.set(cv2.CAP_PROP_POS_FRAMES, 300)  # frame with notes visible
ret, img = frame.read()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Click on a green/blue note pixel and read HSV values
```

### Y-sample zone

Relative to detected white key row (`y_white`). Defaults: `y_offset_top=90`, `y_offset_bot=0`,
sampling from `y_white - 90` to `y_white` (the lit key faces). Overridable via `--config`.

---

## Known quirks & fixes

### Tick formula (critical)

**Wrong formula** (produces ticks ~1.82× too small at BPM=81):
```python
out_tick = t_sec * (OUT_US_PER_BEAT / 1e6) * OUT_TPB   # DO NOT USE
```

**Correct formula:**
```python
out_tick = t_sec * OUT_TPB * OUT_BPM / 60
```

At BPM=81, the wrong formula shrinks all ticks by ~0.74, causing adjacent 16th notes to
merge into chords in the sheet music.

### Keyboard detection: use early frames

Use frame 5 (or similar) for keyboard detection, not frame 300+.
Colored note bars landing on keys corrupt white key position detection.

### Dark pixel artifact in color sampling

When finding the most-saturated pixel in a region, dark pixels (V < 80) produce
artificially high saturation values (S=255) from noise. Always filter them first:
```python
s_bright = s_ch.copy()
s_bright[v_ch <= 80] = 0
best = np.argmax(s_bright)
```
This is already done in `sample_color()`.

### Black key artifact at x < 20

Dark edge artifacts at the left border of the frame can be detected as black keys.
Filter any candidate with `center <= 20`.
Already handled in `detect_keyboard()`.

### MuseScore: separate instruments instead of grand staff

If the two hand tracks use different MIDI channels (e.g. ch 0 and ch 1), MuseScore treats
them as separate instruments instead of one piano grand staff. Fix: use `channel=0` for
both hands. MuseScore assigns treble/bass clef based on pitch range automatically.
Already done in `build_track()`. A conductor track (Track 0) is not needed.

### MuseScore: stale cache

After saving a MIDI externally, MuseScore may show the old version. Do not just re-save —
fully close and reopen the file.

### MuseScore: import panel tempo

Do not try to set tempo in MuseScore 3's import panel (Edit → Preferences → Import → MIDI).
It resets when you click Apply. Set tempo via `set_tempo` MetaMessage in the MIDI itself.
Already done by the script.

### MuseScore 3 and music21

Do not use `music21 → MusicXML → MuseScore 3`. MuseScore 3 crashes on music21-generated
MusicXML. Use MIDI directly.

### MuseScore 4 CLI warnings

```
QML type used in C++ is not a QML element type: ...
```
These are harmless. The export still works.

---

## Alternative MIDI sources

If you don't have a Synthesia video or audio transcription fails:

- **Basic Pitch (Spotify):** `pip install basic-pitch` then `basic-pitch ./out/ audio.wav`
  Lighter than piano_transcription, instrument-agnostic, polyphonic.
- **MIDI repositories:** MuseScore.com, Musescore.org, midi.world, or search
  "[song name] MIDI download".
- **MIDI2HANDS:** ML model for hand separation from existing MIDI. Note: verify channel
  assignments before using with MuseScore — it may produce the 4-stave issue.
- **Manual transcription:** Signal MIDI editor (https://signal.vercel.app/edit) — free,
  browser-based, no install.
