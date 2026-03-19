<!-- reference — read when relevant -->
# PianoVideoScribe — Agent Pipeline Reference

AI agent reference. Everything here is verified to work.

---

## Full pipeline: YouTube URL → PDF sheet music

### 1. Download video

```bash
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
  "https://youtu.be/VIDEO_ID" -o video.mp4
```

### 2. Extract notes from video (primary method)

**If the video shows a keyboard with colored note bars** (Synthesia or similar), extract
notes directly from the video — no audio transcription needed. The `midi` argument is
optional; omit it for video-only mode:

```bash
python pianovideoscribe.py video.mp4 output.mid --bpm BPM --frame FRAME --right-hand monophonic --left-hand no-overlap
```

This scans every frame, detects which keys light up (using frame-to-frame saturation
delta), classifies each as right hand (green) or left hand (blue), and outputs a
two-track MIDI.

- `--bpm` — **quarter-note BPM**. For 6/8 with dotted-quarter=60, use `--bpm 90`
  (quarter=90 means eighth=180, dotted-quarter=60).
- `--frame` — frame index for keyboard detection. Must show the keyboard clearly with
  no notes on the keys. Videos with intros need a later frame (e.g. `--frame 250`).
- `--right-hand` / `--left-hand` — processing mode per hand:
  - `normal` — keep all notes as detected
  - `no-overlap` — cut held notes at next onset, keep chords. Best for accompaniment.
  - `monophonic` — single voice only (highest for right, lowest for left). Best for melody.

### 3. Export to PDF and open for the user

```bash
"/Applications/MuseScore 4.app/Contents/MacOS/mscore" output.mid -o sheet.pdf
```

MuseScore 4 prints QML type warnings — harmless, ignore them.

**Always open the result in MuseScore for the user:**
```bash
open -a "MuseScore 4" output.mid
```

### Important: MuseScore overrides MIDI tempo

MuseScore 4 ignores `set_tempo` from MIDI files and defaults to quarter=60. The tempo
must be changed manually by the user in MuseScore after import.
Do NOT use music21 → MusicXML → MuseScore — it crashes.

---

## Fallback: Audio transcription (no visible keyboard)

Only use this when the video does NOT show a keyboard (concert recordings, audio-only
files, etc.). **Never use audio transcription when the video shows a keyboard.**

```bash
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 44100 audio.wav
```

```python
from piano_transcription_inference import PianoTranscription, sample_rate
import librosa
audio, _ = librosa.load('audio.wav', sr=sample_rate, mono=True)
transcriptor = PianoTranscription(device='cpu')
transcriptor.transcribe(audio, 'transcription.mid')
```

Then pass the MIDI to the script for hand separation:
```bash
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm BPM --left-hand no-overlap
```

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
mask = (bpms >= 40) & (bpms <= 200)
top_idx = np.argsort(ac[mask])[::-1][:3]
candidates = [(bpms[mask][i], ac[mask][i]) for i in top_idx]
for bpm, strength in candidates:
    print(f"  {bpm:.0f} BPM  (strength: {strength:.2f})")
```

Common pitfalls:
- Librosa often picks **double-time** (e.g. 152 instead of 76).
- For 6/8 time: the BPM flag is always quarter-note BPM. Dotted-quarter=60 → quarter=90.

---

## Auto-detect colors

Use `detect_colors.py` to automatically detect hand colors from the video:

```bash
python detect_colors.py video.mp4
```

This compares a neutral keyboard frame against frames with notes playing, finds pixels
that changed from unsaturated to saturated, and clusters the hues to identify the two
hand colors. No config file needed — it suggests HSV ranges automatically.

---

## How the video detector works

The note extraction uses **frame-to-frame saturation delta** detection:

1. **Keyboard detection**: scans a clean frame for white/black key x-positions
2. **Detector regions**: each key gets a tight sampling rectangle:
   - White keys: inner 50% of key width, at the key face level (y_white-30 to y_white+5)
   - Black keys: ±3px from center, at the key body level (y_white-80 to y_white-40)
   - The y-zone separation prevents white key glow from bleeding into black key detectors
3. **Delta detection**: for each frame, computes avg saturation in each detector region.
   - Saturation increase > 30 AND absolute saturation > 80 → note on
   - Saturation decrease > 30 → note off
   - This rejects falling note bars (gradual change) and detects key presses (sudden change)
4. **Hand classification**: average hue at note-on: green (H 40-65) = right, blue (H 85-125) = left
5. **Smart quantization**: snaps to 8th note grid by default (40% tolerance), falls through
   to 16th grid for notes clearly between two 8ths

---

## Config files

Use `--config path/to/config.json` for custom color thresholds and sampling zones.

Available configs:
- `configs/default.json` — Standard Synthesia colors (blue H=100–120)
- `configs/synthesia_cyan_blue.json` — Synthesia videos using cyan left hand (H≈93, range 85–125)

---

## Known quirks & fixes

### MuseScore: separate instruments instead of grand staff

If the two hand tracks use different MIDI channels, MuseScore treats them as separate
instruments. Fix: use `channel=0` for both. Already done in `build_track()`.

### MuseScore: stale cache

After saving a MIDI externally, fully close and reopen the file in MuseScore.

### MuseScore 4 and music21

Do NOT use `music21 → MusicXML → MuseScore 4`. MuseScore crashes on music21-generated
MusicXML. Use MIDI directly.

### Keyboard detection: use early frames

Use a frame where the keyboard is visible but no notes are falling on the keys.
Videos with intros may need frame 200+ instead of the default frame 5.

---

## Alternative MIDI sources

If you don't have a Synthesia video or video detection fails:

- **Basic Pitch (Spotify):** `pip install basic-pitch` then `basic-pitch ./out/ audio.wav`
- **MIDI repositories:** MuseScore.com, midi.world, or search "[song name] MIDI download"
- **Manual transcription:** Signal MIDI editor (https://signal.vercel.app/edit)
