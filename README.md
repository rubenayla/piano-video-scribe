# piano-video-scribe

Convert a [Synthesia](https://synthesiagame.com) piano video into a clean, hand-separated MIDI and PDF sheet music.
Works by reading the video's color coding (green/blue per hand) to split notes into right and left hand tracks, then quantizes the result to a musical grid. Includes a Claude Code slash command so an AI agent can run the full pipeline — YouTube URL to sheet music PDF — autonomously.

## What is this

[Synthesia](https://synthesiagame.com) is a piano learning app that displays falling colored notes — one color per hand. It displays each hand in a different color (green = right, blue = left by default).
This tool reads those colors frame-by-frame and uses them to split a MIDI file — transcribed
from the same video's audio — into two tracks that MuseScore can render as standard sheet music.

## Quick start

```bash
git clone https://github.com/rubenayla/piano-video-scribe
cd piano-video-scribe
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
| `--monophonic-left` | **Recommended for most songs.** Force left hand to one note at a time. |
| `--dry-run` | Detect keyboard only, print stats, exit (use for calibration) |

### About `--monophonic-left`

When playing piano, it's natural to keep a bass note slightly pressed while the next note begins — it sounds better, especially with the sustain pedal. But technically that creates two simultaneous notes, which notation software counts as two separate voices. The result is a cluttered staff with rests and stems going in both directions that is hard to read.

If the left hand part is melodically a single voice (one note at a time), `--monophonic-left` cuts each note exactly when the next one starts, producing a clean single-voice staff. The music sounds identical (the overlap was inaudible in the sheet anyway).

---

## Step 3: Open in MuseScore

On macOS with MuseScore 4 installed, you can open the output directly from the terminal:

```bash
open -a "MuseScore 4" output.mid
```

Otherwise:
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

## Using with an AI agent

If you're using a Claude Code agent (or similar), you can delegate the full pipeline with a single prompt. Clone this repo first, then say:

> Use the piano-video-scribe repo (`~/repos/piano-video-scribe`) to transcribe this Synthesia video: `https://youtu.be/VIDEO_ID`. The BPM is 120. Give me the final MIDI and MuseScore instructions.

The agent will read `AGENT.md` for the full pipeline and handle everything — downloading, transcription, hand separation, and output.

If you don't know the BPM, say so and the agent will figure it out.

This repo includes a `.claude/commands/PianoVideoScribe.md` file, so the `/PianoVideoScribe` slash command is available automatically in any Claude Code session opened inside the repo directory. (The command file keeps its original name for compatibility.)

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
