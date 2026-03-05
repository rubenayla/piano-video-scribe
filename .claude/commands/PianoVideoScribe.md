---
description: Full pipeline from Synthesia YouTube video to hand-separated MIDI (and optionally PDF). Reads ~/repos/PianoVideoScribe/AGENT.md for all steps.
---

Use `~/repos/PianoVideoScribe` to run the full pipeline for this Synthesia video:

$ARGUMENTS

Read `~/repos/PianoVideoScribe/AGENT.md` for the exact commands. Run every step:
1. Download the video with yt-dlp
2. Extract audio with ffmpeg
3. Transcribe audio → MIDI with piano_transcription_inference
4. Run `pianovideoscribe.py` to separate hands
5. Export to PDF with MuseScore CLI

If BPM is not provided in the arguments, figure it out before running step 4.

Work in a song-specific subdirectory, e.g. `~/piano/songs/<song-name>/`.
At the end, tell the user where the MIDI and PDF are, and give MuseScore import instructions.
