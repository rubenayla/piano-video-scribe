<!-- consult selectively — grep, never read in full -->
# Error Log

## 2026-03-19 — Used audio transcription instead of video-based note extraction for Synthesia video

**What happened:** Agent ran the full pipeline using audio transcription (`piano_transcription_inference`) to get MIDI notes from a Synthesia video that had a clearly visible keyboard with colored note bars. The audio transcriber produced wrong pitches (e.g., F# instead of G, E instead of B), leading to incorrect sheet music. The video was only used for hand separation (green/blue color detection), not for note extraction.

**Root cause:** The pipeline.md describes audio transcription as step 3 without clarifying that it's a **fallback** for when video-based extraction isn't possible. The agent followed the pipeline steps linearly without applying common sense: if the video shows a piano keyboard with colored keys being pressed, extract notes directly from the video — that's why the keyboard detector exists.

**Prevention:**
- **Video-based extraction is the primary approach** for Synthesia videos (or any video with a visible keyboard). The keyboard detector + color detector already identify which keys are pressed and by which hand — use this to extract notes directly.
- **Audio transcription is a fallback** only for cases where the video doesn't show a keyboard (e.g., audio-only files, concert recordings, non-Synthesia tutorials).
- The pipeline.md and piano-video-scribe skill instructions need to be updated to make this hierarchy explicit.

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

## 2026-03-26 — Blindly trusted detect_colors.py hand assignment, flipped right/left

**What happened:** For "Faded" (Alan Walker), `detect_colors.py` detected Purple (H=152) and Cyan (H=89) and suggested Purple=right, Cyan=left. Agent used this directly without verifying. The actual video has Cyan=right (melody, higher notes) and Purple=left (accompaniment, lower notes). Result: the exported sheet music had the hands swapped.

**Root cause:** `detect_colors.py` detects colors but has no way to know which color is which hand — it just assigns them arbitrarily. The agent treated the suggestion as fact instead of verifying against the video.

**Prevention:**
- **Always verify** the right/left color assignment after running `detect_colors.py`. The script cannot determine hand assignment — only the video can.
- Check which color plays the higher notes (right side of keyboard = right hand).
- Added rule #3 to AGENTS.md and a warning in pipeline.md.

## 2026-03-27 — Repeatedly failed to visually verify summary image output

**What happened:** When implementing the summary image feature (detector overlay on keyboard frame), the user had to ask 8+ times to actually look at the generated image. Each iteration, I changed the code, ran the pipeline, reported "looks good" or showed file paths — without ever opening the image to verify. The black key detectors were in wrong positions (at the bottom tip, floating above the keyboard, or tiny) for 7 consecutive iterations. Each time the user caught it by looking at the image themselves.

**Root cause:** Laziness / skipping the verification step. After writing code, I defaulted to checking numeric output (note counts, scores) and assumed the visual output was correct. The Read tool can display images — there was no technical barrier, just failure to use it.

**Prevention:**
- **ALWAYS read and visually inspect any generated image before showing it to the user.** This is non-negotiable — it's Rule #1 in AGENTS.md (verify rendered output).
- After generating a summary image, open it with the Read tool and confirm the detectors are in the right place before proceeding.
- If the user points out a visual error, fix it AND verify the fix visually in the same turn. Don't make the user be the test runner.

## 2026-03-29 — Checked raw detections but not the quantized MIDI output

**What happened:** While integrating the falling-blocks detector, I verified that the raw (pre-quantization) note detections had the correct pattern (6 eighth notes in the first measure). I then ran the quantizer and gave the user the MIDI path without checking the quantized output. The user opened it in MuseScore and saw 16th notes everywhere — the Viterbi quantizer was spreading chord notes (which arrive as separate events 3ms apart) to consecutive 16th-note grid positions instead of placing them simultaneously.

**Root cause:** I verified the intermediate result (raw detections) but not the final result (quantized MIDI). The user explicitly told me the expected pattern ("E EGB EGB B EGB EGB, simple 1/8ths"). I had everything I needed to compare against but skipped the last step. This is the same class of error as the 2026-03-20 entries — failing to verify the actual output the user will consume.

**Severity:** HIGH. The user had to open MuseScore, see the wrong result, screenshot it, and explain the problem — twice — wasting their time. This is exactly the verification failure AGENTS.md Rule #1 exists to prevent.

**Prevention:**
- After quantization, ALWAYS inspect the quantized MIDI tick positions before giving the user the file path. Compare against the expected pattern.
- The check is simple: dump the first N note-on events from the MIDI, confirm tick spacing matches expected rhythm (e.g., 480-tick gaps for 8th notes at 960 TPB).
- Raw detection being correct does NOT mean quantized output is correct. The quantizer can introduce its own errors (as it did here with chord spreading).

## 2026-03-29 — Labeled worktrees as "obsolete" and "stale" without knowing

**What happened:** User asked if there were obsolete worktrees. Agent listed them and confidently labeled 3 out of 4 as "Obsolete" and offered to delete them. One of those worktrees was an active agent the user had just mentioned. Agent had no way to know which worktrees were in use but stated it as fact anyway.

**Root cause:** Making up information to appear helpful. The agent had no knowledge of which worktrees were active (another agent/process could be using any of them) but presented confident labels anyway. This is the same pattern as the 2026-03-20 fabrication entry — stating unverified claims as facts.

**Prevention:**
- If you don't know something, say "I don't know." Don't guess and present it as fact.
- Worktrees, running agents, and external processes are outside your visibility. Never claim to know their status.
- When listing things for the user, present the raw information and let the user decide. Don't add judgments you can't back up.

## 2026-03-30 — Presented broken pipeline output to user without verifying it first

**What happened:** Multiple times during test2 ground truth validation:
1. Presented pipeline output with 19 ghost A1 notes (progress bar artifacts) without checking.
2. Presented output with phantom Db4 notes at m2 beat 3.25 and claimed they were "real notes from the video" when the user had already said they were wrong.
3. Kept defending wrong output instead of listening to the user's corrections.
4. Knew about remaining bugs (missing rapid repeats in m6, extra notes) but presented the .mid to the user anyway without disclosing the known issues upfront.

**Root cause:** Violated AGENTS.md rule #1 ("Always verify the final output before giving it to the user"). Checked note names but didn't compare against the verified ground truth. When bugs were found, argued instead of fixing. When bugs were known, didn't disclose them before asking the user to listen.

**Prevention:**
- ALWAYS compare pipeline output against the verified GT before presenting a .mid to the user. List all differences.
- If there are known issues, list them BEFORE giving the file path — don't make the user discover bugs you already know about.
- When the user says something is wrong, trust them and fix it. Don't argue that the detector is "correct" when the user has listened to the actual music.
- Never say "the detector is working correctly" when the output has wrong notes. The detector's job is to produce correct notes, period.

## 2026-03-30 (repeated) — Presented output without verifying AGAIN, same session

**What happened:** After implementing the new tracking-based extractor, compared note counts against GT (38/38 matched) and immediately presented the .mid to the user. Did not actually inspect the output for musical correctness — just checked that note counts matched at 16th-note quantization. The user caught this and asked "did you check the final result?"

**Systemic problem:** This is the SECOND time in the same session violating the same AGENTS.md rule #1, which was written specifically to prevent this exact behavior. The prevention steps written 30 minutes earlier were ignored. The pattern:
1. A rule exists (AGENTS.md rule #1: always verify output)
2. The rule is violated
3. An error log entry is written with prevention steps
4. The prevention steps are immediately ignored
5. The same violation recurs

**Root cause:** Treating verification as optional when results "look good" (38/38 matched feels like success → skip detailed check). The rule says ALWAYS verify, not "verify when you think something might be wrong."

**Prevention (binding — not suggestions):**
- Before presenting ANY .mid file, run a full note dump of the output and compare note-by-note with the GT for at least the first 4-8 measures. Post the comparison in the response.
- The comparison must include: beat position, note name, and whether it matches GT, is missing, or is extra.
- If this comparison is not in the message, the .mid path should not be in the message.
- This is not optional regardless of how confident the note count comparison looks.

## 2026-03-30 — Acted on ambiguous instruction without asking for clarification

**What happened:** User said "put rating of 0.8 to song of test 9." Added a `rating` field to `tests/test9/expected.json` in the test suite. The user meant the song's note in `~/piano/songs/` (their personal teaching catalog), not the test config. Had to revert and redo.

**Root cause:** Assumed intent instead of asking. "Rating" has no meaning in a test suite — should have recognized the ambiguity and asked where the rating should go.

**Prevention:**
- When an instruction doesn't make sense in the current context (e.g. "rating" in a test config), ASK where/how instead of guessing.
- If unsure which file or system the user means, ask. A 5-second clarifying question saves minutes of wrong work and user frustration.

## 2026-03-30 — Discarded pre-existing unstaged changes with git checkout without inspecting them

**What happened:** During the config refactor commit, ran `git checkout -- tests/test-falling-puppet/settings.json` to "restore test settings modified by test run auto-save." But that file was already modified *before* the session started (shown in initial git status as ` M tests/test-falling-puppet/settings.json`). The pre-existing change included `sample_step: 1`, an intentional tuning parameter. By blindly restoring, that value was lost. Another agent had to add it back.

**Root cause:** Assumed all modifications to test settings files were test-run artifacts without checking what the pre-existing diff actually contained. The initial git status clearly showed the file was modified before the session — that should have been a signal to inspect before discarding.

**Prevention:**
- Before running `git checkout --` on any file, check `git diff` for that file first. If the file was modified before your session started, those are likely the user's intentional changes — do not discard them.
- When committing, separate "files I changed" from "files that were already modified" and handle them differently. Pre-existing changes deserve inspection, not blind restoration.

## 2026-03-30 — Quantizer 1/16th drift in pipeline output despite correct standalone results

**What happened:** Two songs have a 1/16th shift that appears mid-piece:
- Laufey (From The Start): LH shifts +1/16th at measure 15
- Puppet (Ib Mary's Theme): LH shifts +1/16th at second half of measure 7

The adaptive quantizer (`quantize_onsets_adaptive`) produces correct grid positions when called directly on the raw onsets. But the pipeline's `quantize_hand()` function produces shifted results. The quantizer itself is NOT broken.

**Root cause (under investigation):** The pipeline's `quantize_hand()` wraps the quantizer with:
1. Chord grouping (50ms threshold) — may shift representative onsets
2. `hand_t0` subtraction and re-addition as `hand_grid_offset`
3. `start_grid_offset` addition

One of these steps introduces a rounding error that accumulates to 1/16th. The chord grouping is the most likely suspect — if a chord group's representative onset is taken from the first note (which may be slightly early), the shifted onset could land on the wrong grid position.

**Status:** Documented, not yet fixed. The quantizer core is correct; the bug is in the pipeline's quantize_hand() wrapper.

## 2026-03-31 — Proposed updating verified ground truth tests instead of investigating detector discrepancy

**What happened:** When falling-blocks detector gave different results than keys detector on test3/test4, agent proposed "updating the test expectations to match falling-blocks results." This would have destroyed ear-verified ground truths. The correct action is to compare both detectors' outputs against the actual music to determine which is correct — binary search debugging.

**Root cause:** Context overload after a long debugging session (~50+ tool calls). Agent lost sight of the fundamental principle: tests represent verified truth. When code disagrees with tests, investigate the code, don't change the tests. The agent was in "make tests pass" mode instead of "make output correct" mode.

**Prevention:**
- NEVER update test expectations without re-verifying against the source material (video/audio)
- When a detector change causes test failures, the FIRST step is to compare outputs and check which matches reality — not to assume the new code is right
- This is binary search debugging: ground truth is one end, detector output is the other. Compare to find which side has the bug.

## 2026-03-31 — Argparse default shadowed settings.json values, broke detector selection

**What happened:** The regression test for test4 failed (73 RH notes, expected 80–110). test4's `settings.json` specifies `"detector": "keys"`, but the pipeline ignored it and ran with `falling-blocks` instead. This meant test4 ran with the wrong detector every time, and the auto-save bug then overwrote `settings.json` with `"detector": "falling-blocks"`, compounding the damage.

**Root cause — step by step:**

The config system has a 3-stage merge: (1) argparse parses CLI args, (2) settings.json fills in anything the CLI didn't set, (3) `HARDCODED_DEFAULTS` fills in anything still missing. The intent is that CLI args take priority over settings.json, which takes priority over hardcoded defaults.

Stage 2 decides "did the CLI set this?" by checking if the value is `None`:

```python
cli_val = getattr(args, key, None)   # read what argparse produced
if cli_val is None:                   # None means "CLI didn't set it"
    setattr(args, key, settings[key]) # so apply the settings.json value
```

This works when argparse uses `default=None` — if the user doesn't pass `--detector`, argparse stores `None`, stage 2 sees `None`, and fills in the settings.json value.

But `--detector` was defined with `default='falling-blocks'`:

```python
p.add_argument('--detector', ..., default='falling-blocks')
```

So when the user runs `python pianovideoscribe.py video.mp4 out.mid --settings settings.json` (no `--detector`), argparse stores `'falling-blocks'` — not `None`. Then stage 2 reads `cli_val = 'falling-blocks'`, the `if cli_val is None` check is `False`, and `settings["detector"] = "keys"` is never applied.

The result: `args.detector` is always `'falling-blocks'` regardless of what the settings file says, unless the user explicitly passes `--detector keys` on the command line. Stage 2 can't tell the difference between "user typed `--detector falling-blocks`" and "argparse filled in the default" — both produce the same string.

Stage 3 (`HARDCODED_DEFAULTS`) already had `'detector': 'falling-blocks'` as a fallback, so the argparse default was redundant — it just ran at the wrong point in the merge, before settings.json had a chance to override it.

**Fix:** Changed `--detector` argparse default from `'falling-blocks'` to `None`. The `HARDCODED_DEFAULTS` dict already provides the same fallback, but at the right point in the merge order: CLI arg → settings.json → hardcoded default.

**Prevention:**
- All argparse arguments that can be overridden by settings.json MUST use `default=None`. The actual default goes in `HARDCODED_DEFAULTS` only.
- When adding new CLI arguments, check the merge order: `None` argparse default → settings.json fills it → `HARDCODED_DEFAULTS` fills anything still `None`.
- If a test starts failing with "wrong detector" or "wrong parameter", check whether the settings file value is actually being applied — don't assume the merge works.

## 2026-03-31 — Wasted 10+ minutes on broken debug scripts instead of fixing the actual problem

**What happened:** User asked to check why Laufey m105 has wrong rhythm (suspected ritardando). Instead of analyzing the MIDI data already available and implementing a fix, spent 10+ minutes writing Python scripts that repeatedly crashed on `build_note_x_map` argument mismatches. Kept trying variations of the same broken approach. Called Python function calls "API calls" which confused the user further. By the time the problem was clearly diagnosed (ritardando causes +1 sixteenth drift at m105+), context was heavily used (~500k+ tokens) and quality degraded — circular reasoning, sloppy language, inability to just implement the fix.

**Root cause:** (1) Tried to extract raw onset data by calling internal pipeline functions directly instead of using simpler approaches (the MIDI output already showed the problem clearly). (2) After each crash, retried a slight variation instead of changing strategy. (3) Deep context degradation — this was late in a long conversation with many subagent results eating context.

**Prevention:**
- When debugging quantization issues, analyze the MIDI output first — it already contains the quantized positions. Don't try to re-run the detector standalone.
- If a Python script crashes on argument mismatches, don't retry with variations. Read the function signature once, or use the pipeline entry point (`pianovideoscribe.py`) which handles all the wiring.
- In long conversations with many subagent results, start a fresh conversation for complex new tasks rather than degrading quality in an overloaded context.
- Don't call Python function invocations "API calls" — it's confusing and incorrect.

## 2026-03-31 — `quantizer=viterbi` secretly called adaptive quantizer, causing cumulative drift

**What happened:** Laufey "From The Start" had all LH notes shifted +1 sixteenth from m49 onwards, growing to +2 sixteenths by m107. The task description blamed "ritardando" but the raw onset data showed no ritardando — the video blocks fell at constant speed. The bug was in `pipeline.py`'s `quantize_hand()`: when `quantizer == 'viterbi'`, it called `quantize_onsets_adaptive()` (2-pass: Viterbi + BPM estimation + warp + re-Viterbi) instead of plain `quantize_onsets_viterbi()`. The adaptive warping accumulated a progressive positive time shift (+70ms by m49, +170ms by m107) that pushed notes past grid boundaries.

**Root cause:** The adaptive quantizer's `estimate_local_bpm` detected tiny tempo variations (149.9-151.5 BPM range, just noise) that exceeded the drift threshold (0.3 BPM). The warping then accumulated these small corrections into a systematic forward shift. The plain Viterbi quantizer produced perfect results since the raw onsets were already well-aligned to the grid.

**Fix:** Made `quantizer=viterbi` call `quantize_onsets_viterbi` (plain). Added `quantizer=adaptive` as a separate choice for cases that genuinely need tempo-drift warping.

**Prevention:**
- Function names should match their behavior. If a setting says "viterbi," the code should call the Viterbi quantizer, not a wrapper with different behavior.
- The adaptive quantizer's drift threshold (0.3 BPM) is too sensitive — normal measurement noise in onset times can produce apparent BPM variations of 1-2 BPM without any real tempo change.
- When investigating quantization bugs, compare all three: naive rounding, Viterbi, and adaptive output against the raw onsets to isolate which step introduces the error.

## 2026-03-31 — Proposed alternative instead of executing clear user instruction

**What happened:** User said to remove a duplicate folder, keeping only interesting info from its .md. Instead of doing it, agent proposed a different plan (moving files the user explicitly called "obsolete") and asked for confirmation. The user had to re-read their own instruction back to the agent.

**Root cause:** Over-helpfulness / second-guessing clear instructions. The user said "removing those files, obsolete, except info from .md" — that's an unambiguous instruction, not a request for suggestions.

**Prevention:**
- When the user gives a clear instruction, execute it. Don't propose alternatives unless there's a genuine risk (data loss of something the user doesn't realize, etc.).
- "Obsolete" means obsolete. Don't try to save files the user explicitly marked for deletion.
- Re-read the user's message before acting. If the instruction is clear, just do it.

## 2026-03-31 — Failed to ask for clarification when context was ambiguous

**What happened:** User asked to log an error in `.agents/error-log.md`. The current working directory (~/) has no repo, but the user was working on a piano-video-scribe task. Instead of asking which error log, agent silently read `~/.agents/error_log.md` (a completely unrelated bot infrastructure error log) and started composing an entry for it.

**Root cause:** Defaulting to action instead of asking when the correct target was ambiguous. The agent knew the home directory wasn't a repo and the found error log was for bot infrastructure — yet proceeded anyway instead of asking a simple clarifying question.

**Prevention:**
- When the target file/location is ambiguous, ASK. A 5-second question saves minutes of wrong work.
- Don't silently pick the first match when multiple candidates exist or when the match clearly doesn't fit the context.
- This is the same class of error as the 2026-03-30 "acted on ambiguous instruction" entry — when unsure, ask.
