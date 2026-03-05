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
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm BPM --monophonic-left
```

Replace `BPM` with the actual BPM of the song (see "Finding the BPM" below).

`--monophonic-left` is recommended for most songs: piano players naturally hold bass notes slightly
into the next note for a legato feel or because of pedal sustain. This overlap is musically a single
voice but creates multiple simultaneous voices in the notation, producing a cluttered staff. The flag
cuts each left-hand note when the next begins, giving MuseScore a clean single-voice bass line.

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

- Check the song's metadata on MusicBrainz, Songsterr, or a MIDI repository.
- Use a tap-tempo tool (search "tap tempo" in any browser): tap the beat for ~10 seconds.
- Listen for the kick drum or bass pulse and count beats per minute.
- As a sanity check: at BPM=81 and 16th quantization, the smallest note lasts 0.185s.
  If notes at that spacing are still merging in MuseScore, the BPM is likely wrong.

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

Hardcoded in `classify_hand()` in `pianovideoscribe.py`:

```python
# Green (right hand by default): H 40–65, S > 100, V > 80
is_green = 40 <= h <= 65 and s > 100 and v > 80

# Blue (left hand by default): H 100–120, S > 80, V > 80
is_blue = 100 <= h <= 120 and s > 80 and v > 80
```

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

Hardcoded in `main()` as `y_sample_top = 480`, `y_sample_bot = 600`.
This covers the falling note bar and the lit key region in a typical 1280×720 Synthesia video.
If using a different resolution or zoom level, adjust these values.

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

### MuseScore: 4-stave issue

If each MIDI track uses 2+ channels (e.g. ch 0 and ch 1), MuseScore creates 4 staves
instead of 2. Fix: use only `channel=0` in all `note_on` / `note_off` messages.
Already done in `build_track()`.

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
