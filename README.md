# PianoVideoScribe

Separate a Synthesia piano video into right-hand and left-hand MIDI tracks.
Works by reading the video's color coding: green notes go to one track, blue notes to the other.

## What is this

Synthesia displays each hand in a different color (green = right, blue = left by default).
This tool reads those colors frame-by-frame and uses them to split a MIDI file — transcribed
from the same video's audio — into two tracks that MuseScore can render as standard sheet music.

## Quick start

```bash
git clone https://github.com/yourusername/PianoVideoScribe
cd PianoVideoScribe
pip install -r requirements.txt
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120
```

---

## Step 0: Get the Synthesia video

**Option A — YouTube:**
```bash
pip install yt-dlp
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
  "https://youtu.be/VIDEO_ID" -o video.mp4
```

**Option B — manual:** Right-click the video in your browser → Save video as.

---

## Step 1: Get the MIDI

You need a MIDI file that represents the notes in the video.
The video and the MIDI must be for the same recording.

### Option A (best): transcribe the audio with `piano_transcription_inference`

```bash
# Install (requires Python ≥ 3.8, downloads a ~165 MB model on first run)
pip install piano_transcription_inference librosa

# Extract audio from the video
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 44100 audio.wav

# Transcribe
python - <<'EOF'
from piano_transcription_inference import PianoTranscription, sample_rate
import librosa
audio, _ = librosa.load('audio.wav', sr=sample_rate, mono=True)
transcriptor = PianoTranscription(device='cpu')
transcriptor.transcribe(audio, 'transcription.mid')
EOF
```

This produces a MIDI at 120 BPM / 384 TPB with raw audio timestamps (not quantized) —
`pianovideoscribe.py` handles the quantization.

### Option B: use any existing MIDI

If you already have a MIDI that matches the video, use it directly.

---

## Step 2: Run pianovideoscribe.py

```bash
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120
```

**All arguments:**

| Argument | Description |
|---|---|
| `video` | Path to the Synthesia MP4 |
| `midi` | Path to the input MIDI |
| `output` | Path for the output MIDI |
| `--bpm` | **Required.** Real BPM of the song (e.g. `--bpm 81`) |
| `--frame` | Frame used for keyboard detection (default: 5) |
| `--green-hand` | Which hand is green: `right` (default) or `left` |
| `--dry-run` | Detect keyboard only, print stats, exit (use for calibration) |

---

## Step 3: Open in MuseScore

1. Download [MuseScore 3](https://musescore.org/en/download) (free)
2. Open the output MIDI: **File → Open**
3. Before or after opening, set: **Edit → Preferences → Import → MIDI → Shortest note: 16th**
4. You should see two staves: treble (right hand) + bass (left hand)

---

## Calibration

If the hands look swapped:
```bash
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --green-hand left
```

If detection is low (< 70%), try a different frame for keyboard detection:
```bash
python pianovideoscribe.py video.mp4 transcription.mid output.mid --bpm 120 --frame 10 --dry-run
```
Look for a frame index where the keyboard is fully visible with no colored note bars on top of it.

---

## Limitations

- Requires Synthesia's **default color scheme** (green/blue). Custom colors need hardcoded HSV
  threshold adjustments in the script.
- **BPM must be known.** Wrong BPM = notes merged into chords in the sheet music.
- **Partial keyboard is fine** — the script auto-detects the visible key range.
- Very high or very low notes outside the visible keyboard region fall back to pitch-proximity
  assignment and are usually still correct.

---

## How it works

1. Reads an early video frame to detect white and black key positions via HSV thresholding.
2. Builds a MIDI note → x-pixel map using black key grouping to find octave alignment.
3. For each `note_on` event in the input MIDI, seeks to the corresponding video frame and
   samples the HSV color at the note's x position (y ≈ 480–600, covering both the falling
   bar and the lit key).
4. Classifies green → right hand, blue → left hand. Undetected notes fall back to
   pitch-proximity assignment.
5. Re-quantizes all note timestamps to a 16th-note grid at the given BPM and writes a
   2-track Type-1 MIDI (channel 0 only, to avoid MuseScore creating 4 staves).
