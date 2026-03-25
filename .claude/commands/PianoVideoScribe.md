---
description: Full pipeline from Synthesia YouTube video to hand-separated MIDI (and optionally PDF). Reads ~/repos/piano-video-scribe/.agents/pipeline.md for all steps.
---

Use `~/repos/piano-video-scribe` to run the full pipeline for this Synthesia video:

$ARGUMENTS

Read `~/repos/piano-video-scribe/.agents/pipeline.md` for the exact commands. Run every step:
1. Download the video with yt-dlp
2. Extract audio with ffmpeg
3. Transcribe audio → MIDI with piano_transcription_inference
4. Run `pianovideoscribe.py --monophonic-left` to separate hands (use `--monophonic-left` by default)
5. Export to PDF with MuseScore CLI
6. Open the result in MuseScore 4 for the user: `open -a "MuseScore 4" output.mid`

If BPM is not provided in the arguments, figure it out before running step 4.

Work in a song-specific subdirectory, e.g. `~/piano/songs/<song-name>/`.
