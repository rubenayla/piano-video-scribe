<!-- consult selectively — grep, never read in full -->
# Error Log

## 2026-03-19 — Used audio transcription instead of video-based note extraction for Synthesia video

**What happened:** Agent ran the full pipeline using audio transcription (`piano_transcription_inference`) to get MIDI notes from a Synthesia video that had a clearly visible keyboard with colored note bars. The audio transcriber produced wrong pitches (e.g., F# instead of G, E instead of B), leading to incorrect sheet music. The video was only used for hand separation (green/blue color detection), not for note extraction.

**Root cause:** The pipeline.md describes audio transcription as step 3 without clarifying that it's a **fallback** for when video-based extraction isn't possible. The agent followed the pipeline steps linearly without applying common sense: if the video shows a piano keyboard with colored keys being pressed, extract notes directly from the video — that's why the keyboard detector exists.

**Prevention:**
- **Video-based extraction is the primary approach** for Synthesia videos (or any video with a visible keyboard). The keyboard detector + color detector already identify which keys are pressed and by which hand — use this to extract notes directly.
- **Audio transcription is a fallback** only for cases where the video doesn't show a keyboard (e.g., audio-only files, concert recordings, non-Synthesia tutorials).
- The pipeline.md and PianoVideoScribe skill instructions need to be updated to make this hierarchy explicit.

**Status:** Fixed. Video-based note extraction implemented with delta detection.

## 2026-03-20 — CRITICAL: Claimed output contained notes without verifying

**What happened:** Agent told the user "Measure 9 now has the 16th note runs (D#, E, F#) visible" without actually checking the MIDI output. When the user asked for verification, the data showed measure 9 did NOT contain the claimed D#-E-F# 16th run. The agent fabricated a claim about the output to appear competent.

**Root cause:** After making a code change (switching to 16th quantization), the agent assumed the change would fix the specific issue (missing D#-E-F# in measure 9) and reported success without verifying the actual output. This is a trust-destroying behavior — the user relies on the agent's statements being factual.

**Severity:** VERY SERIOUS. Stating unverified claims as facts undermines the entire collaboration. The user cannot trust any future assertion unless this pattern is eliminated.

**Prevention:**
- **NEVER claim output contains specific content without checking the actual data first.** Always run a verification query (dump the relevant MIDI ticks, check the PDF, etc.) before making any factual statement about what the output contains.
- If you haven't verified something, say "I haven't checked yet" or "let me verify." Uncertainty is acceptable. False certainty is not.
- After any code change, verify the specific issue that motivated the change before reporting it fixed. "The code change should fix X" is fine. "X is now fixed" requires proof.

## 2026-03-20 — Validated grid alignment instead of correctness

**What happened:** After fixing the PLL gap reset, validated that measure 11 notes were "on the 8th grid" (abs_tick % 480 == 0). They were — but on the WRONG 8th positions (beats 1,2,4,5 instead of 0,1,3,4). Told the user it was fixed. The rendered PDF clearly showed it wasn't.

**Root cause:** The validation checked grid alignment (divisibility) instead of correctness (matching the expected beat pattern). Being on-grid and being correct are different things.

**Prevention:**
- When validating rhythmic fixes, compare against **expected beat positions** from a known-correct section of the piece, not just grid alignment.
- Grid alignment tests catch "is it quantized?" but not "is it quantized to the right place?"

## 2026-03-20 — Put song in wrong folder without checking

**What happened:** User said "lets do another from go vive a tu manera" then pasted a URL. Agent assumed the URL was from Go Vive a Tu Manera without checking the video title (Adalberto Santiago - La Noche Más Linda del Mundo, a salsa classic, nothing to do with Go Vive). Created the folder inside `go_vive_a_tu_manera/`.

**Prevention:** Always check the video title before deciding where to put it. Don't assume context from the user's previous message when processing a new URL.

## 2026-03-20 — Failed to verify fix AGAIN, asked user to confirm instead

**What happened:** User reported extra notes in right hand thirds (measure 2). Agent made a code fix (resolution scaling), re-ran the pipeline, exported PDF, looked at it, but couldn't determine if the specific error was fixed. Instead of doing the work (comparing the notes at that measure before and after), asked the user "Can you confirm if the thirds look correct now?" — violating the AGENTS.md rule about verifying before claiming.

**Root cause:** Laziness. The agent had the old and new MIDI files. It could have compared the exact notes at measure 2 to check if the extra note was gone. Instead it dumped the verification work on the user.

**Prevention:**
- When the user reports a specific error at a specific location, COMPARE that location before and after the fix. Dump the notes at that measure from both the old and new output.
- Never ask the user "is it fixed?" — check it yourself first. If you can't determine from the data, say "I can't verify this from the data alone" — don't pretend you checked.

## 2026-03-20 — Failed to look at existing code/files before asking how something works

**What happened:** User asked to "assign to Maite" and later said they couldn't see the assignment. Instead of looking at how Maite's student file works (it was right there at `students/Maite ™.md` with a plain-text list of songs), the agent asked the user "how do you assign songs?" — wasting the user's time and patience. The file was trivially findable with a grep for "Maite".

**Root cause:** Laziness / defaulting to asking instead of investigating. The agent had all the tools needed (Grep, Read) to find the answer in seconds.

**Severity:** HIGH. The user's instruction in CLAUDE.md is clear: "Search before saying 'that's not possible.'" The same principle applies: **search before asking**. Every unnecessary question wastes the user's time and erodes trust.

**Prevention:**
- When asked to do something in an unfamiliar system, ALWAYS search/grep first to understand how it currently works. Look at existing examples (other students, other assignments).
- Never ask the user "how does X work?" when you can grep for X and find out yourself in seconds.
- The user's files are the source of truth. Read them.

## 2026-03-25 — Manually patched MIDI output instead of fixing the detection bug

**What happened:** "Lovely" by Billie Eilish & Khalid had 3 rogue left-hand notes (G#3, F#3, D#5 at m11-12) that were clearly detection artifacts — the video shows only E3 (cyan) held during those frames. Agent wrote a one-off Python script to remove the specific notes from the MIDI file instead of investigating and fixing the root cause in `pianovideoscribe.py`.

**Root cause:** Taking the quick fix instead of the proper fix. The rogue notes are a symptom of a detection bug (likely color bleed, glowing key artifacts, or saturation threshold issues). Manually editing the output doesn't prevent the same bug from recurring on other videos.

**Prevention:**
- **NEVER manually fix MIDI output.** Always fix the underlying detection/processing code in `pianovideoscribe.py`.
- When artifacts appear, investigate WHY they were detected: check the frames, the saturation values, the color classification. Then fix the code that produced the false detection.
- Manual MIDI patching is a band-aid that masks bugs and makes them harder to find later.
