<!-- consult selectively — grep, never read in full -->
# Error Log

## 2026-03-19 — Used audio transcription instead of video-based note extraction for Synthesia video

**What happened:** Agent ran the full pipeline using audio transcription (`piano_transcription_inference`) to get MIDI notes from a Synthesia video that had a clearly visible keyboard with colored note bars. The audio transcriber produced wrong pitches (e.g., F# instead of G, E instead of B), leading to incorrect sheet music. The video was only used for hand separation (green/blue color detection), not for note extraction.

**Root cause:** The pipeline.md describes audio transcription as step 3 without clarifying that it's a **fallback** for when video-based extraction isn't possible. The agent followed the pipeline steps linearly without applying common sense: if the video shows a piano keyboard with colored keys being pressed, extract notes directly from the video — that's why the keyboard detector exists.

**Prevention:**
- **Video-based extraction is the primary approach** for Synthesia videos (or any video with a visible keyboard). The keyboard detector + color detector already identify which keys are pressed and by which hand — use this to extract notes directly.
- **Audio transcription is a fallback** only for cases where the video doesn't show a keyboard (e.g., audio-only files, concert recordings, non-Synthesia tutorials).
- The pipeline.md and PianoVideoScribe skill instructions need to be updated to make this hierarchy explicit.

**Status:** Video-based note extraction not yet implemented. To be built as a replacement for the audio transcription step in keyboard-visible videos.
