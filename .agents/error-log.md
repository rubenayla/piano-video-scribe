<!-- consult selectively — grep, never read in full -->
# Error Log

## 2026-04-22 — Shipped PDF without comparing first measures to video (Naughty / Matilda)

**What happened:** Ran the pipeline on a Synthesia video of "Naughty" from Matilda the Musical and reported the PDF/MIDI as done after only checking macro stats (note count, per-octave histogram, RH/LH ranges). The histogram looked "reasonable" (RH centered on oct 3-4, LH on oct 2-3) so I declared success. User pushed back and asked me to verify m1. A frame-by-frame check revealed the first LH chord (should be A2-C3-F3 simultaneous) was detected as F#3, C#3, C3, F3 spread over ~1 second, and many RH notes were garbage bass notes. The sheet was wrong.

**Root cause:** Two failures compounded:
1. I used the *default* `--detector falling-blocks`, whose x→pitch calibration was off by ~20px on this video. That shifted every white-key note a half-step right (A→A#, C→C#, F→F#) and spread chord onsets across the beat. `--detector keys` (with `y_offset_top: 20` in config) produced clean output.
2. I accepted note-count and octave-histogram sanity checks as "verification." They are not. Octave histograms can look plausible even when every note is wrong. Only a direct comparison of video frames to MIDI at specific timestamps proves correctness.

**Prevention:**
- Always end with a **frame-vs-MIDI comparison table** at ≥4 timestamps (first real keypress plus 3 more spread across the song). Columns: `video_time | video_notes | midi_notes | match`. If that table is not in the final message, the task is not done.
- Automatic sanity checks (histogram, hand-range median, accidental ratio, chord simultaneity, intro-noise count) are first-pass filters for gross errors — never substitutes for the frame check.
- Heuristic: if `--key C` is set and >30% of output notes are black keys, the x-calibration is off — try `--detector keys` or a different `--frame`.
- Heuristic: if a visually simultaneous chord in the video is spread over more than ~50ms in the MIDI, either BPM is wrong or the detector is mis-tracking — do not accept "close enough."
- Added rule 1a to `AGENTS.md` and a mandatory "Verification" section to `~/.claude/commands/PianoVideoScribe.md` (command file, not a SKILL.md).

**Status:** Fixed for this video by switching to `--detector keys` + `config.json` with `y_offset_top: 20`, `start-time: 10.5`. Verification table posted in the response.

## 2026-04-22 — 31-minute stall waiting on piped Python script (no buffering flush)

**What happened:** Launched `python ~/repos/piano-video-scribe/pianovideoscribe.py ... > /tmp/log 2>&1 &` in the background and polled the log file to detect completion. The log stayed empty for ~31 minutes while the Python process was actually running fine — stdout was block-buffered because it was piped, so nothing was flushed until the process ended (or was killed). I burned wall-clock (and cache) polling an invisible process instead of diagnosing the buffer.

**Root cause:** CPython block-buffers stdout when it's not a TTY. `python script.py > file` accumulates output in an internal 4–8 KB buffer per stream and only flushes on exit or explicit `sys.stdout.flush()`. Running in background with log redirection hit this path.

**Prevention:**
- When running a long script in the background and watching its log, always pass `python -u` (unbuffered) or set `PYTHONUNBUFFERED=1`.
- For ffmpeg/OpenCV-heavy scripts, also consider `stdbuf -oL -eL <cmd>` as a wrapper if `-u` isn't available (e.g. via a non-Python entry point).
- If a log file is empty for longer than one expected iteration of the script, the problem is almost always buffering or a hang before the first print — not slow progress. Check `ps` for the process and strace/py-spy it rather than waiting longer.
- Do not `Monitor` or `TaskOutput(block=true)` with timeouts in the tens of minutes hoping output will appear. Kill the run, add `-u`, and restart.

**Status:** Fixed by switching to `python -u ... > /tmp/log 2>&1` for all subsequent runs. Progress visible in seconds.

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

## 2026-05-26 — keys detector silently missed all black key notes (Beat It)

**What happened:** Ran the pipeline on a Synthesia video of "Beat It" (Michael Jackson) with `--detector keys`. Output had 680 notes but 0 F# notes, even though F# is clearly visible in the video. Key auto-detected as C major instead of E minor. User noticed F# missing.

**Root cause:** `detect_keyboard` found `y_kb_top=1033` instead of the true ~745. It scanned upward from y_white-5=1052 and stopped at a thin decorative 4-pixel dark horizontal border at y=1029–1032 (part of the white key face graphic). This is above y_bk_bottom=952, making `bk_h = y_bk_bottom - y_kb_top = 952 - 1033 = -81`. Negative `bk_h` in `build_detector_regions` collapsed the black key detector zone to 1px at y=950:951, where the bar's V≈2 (below V_MIN=30), so all black keys read `frame_sat=0` → never detected.

**Fix (keyboard.py):**
1. Changed y_kb_top scan to start from y_black-5 (above the black keys) rather than y_white-5, skipping the white-key face area where decorative borders live.
2. Added consecutive-dark requirement (≥3 rows) to skip single-pixel artifacts.
3. Added sanity check in `build_detector_regions`: if `bk_h < avg_white_w * 0.5`, fall back to `y_black - avg_white_w * 1.5` as bk_top. Ensures the detector always gets a meaningful zone even if y_kb_top is wrong.

**Prevention:**
- When verifying output, check `key auto-detected` in pipeline output — if it says C major on a clearly Em/Am/Fm song, all black keys were missed.
- After any run, check the note pitch-class distribution (count by note name) and verify black keys are present if the song has any.
- The sanity check fallback in `build_detector_regions` is the defensive last line; the y_kb_top scan fix is the root-cause fix.

## 2026-06-04 — Claimed lead-sheet chords "on the downbeat" without zooming the render (Beat It)

**What happened:** Rebuilt the lead-sheet chord engine (measure-synchronous detection) and told the user the chords now "sit on the downbeats", based on (a) an XML tick analysis showing every `<harmony>` was the first child of its measure and (b) a *thumbnail* render. The user immediately spotted that the chord symbols were rendered ~a 16th-note to the RIGHT of the downbeat on many measures — obvious at normal zoom, invisible in a thumbnail. Wasted the user's time by presenting an unverified "fixed".

**Root cause (two layers):**
1. *Process:* verified the data (XML beats) but not the actual rendered artifact at readable zoom. "Harmony is first child in the XML" is necessary but NOT sufficient — the renderer can still place it elsewhere. AGENTS.md rule 1 requires verifying the real output; a thumbnail is not verification of symbol-to-note alignment.
2. *Technical:* MuseScore does NOT anchor a chord symbol to a note that is *tied over* the barline (a tie-stop on the downbeat). It drifts the symbol to the next onset. Beat It's riff is syncopated, so melody notes routinely tie across the bar exactly where chords change → ~53/99 chord symbols rendered a 16th late. Confirmed with an A/B render: fresh-onset downbeat anchors correctly, tie-stop downbeat drifts. Negative `<offset>` on the harmony is ignored by MuseScore (tested, no effect).

**Fix (lead_sheet.py, build_musicxml):** re-strike the melody at chord-change downbeats — split any melody note that spans a chord tick at that tick, with no tie, so the downbeat is a fresh onset the chord can anchor to. Only notes that actually cross a change are touched; every measure still sums to a full bar. Verified by zooming the render on the previously-broken measures.

**Prevention:**
- After generating any notation, render AND zoom to readable size on the specific feature you're claiming is correct (chord-to-note alignment, accidentals, ties). Never claim placement correctness off a thumbnail or off XML structure alone.
- Renderer-specific gotcha: a chord symbol over a tied-over (tie-stop) downbeat drifts right in MuseScore. Keep downbeats at chord changes as fresh onsets.

### 2026-06-04 (addendum) — wrong mental model: chords belong at the bar downbeat

After fixing the tie-stop anchoring (above), I still snapped each chord symbol to the **bar downbeat**. User: "that blue C should go with that blue A note — they go together in reality." The data proved it: the left-hand C triad and the melody A4 are both struck at offset 720 (beat 1.75), but I placed the C on beat 1, tearing it off the note it sounds with. This is also what "the melody is shifted relative to the chords" meant in the very first message — I'd misread it as chord *lag* and never checked the actual strike ticks.

**Root cause:** assumed a lead-sheet convention (chords on downbeats) instead of reading where the chords are actually played. Beat It's chords are syncopated and the strike offset varies per bar (0/480/720/960/…), so there is no single downbeat to snap to.

**Fix (lead_sheet.py `_strike_tick`):** anchor each chord at the melody (right-hand) onset nearest the left-hand chord attack, not the barline. 95/99 chords now land exactly on a melody note (was 45). Detection (one chord per bar) is unchanged; only placement moved.

**Prevention:** for alignment between two streams (chords vs melody), read the actual onset ticks of BOTH and check they coincide — don't assume either belongs on a metric grid. When the user says two things "go together," verify by comparing their tick values, not by eye on a thumbnail.

### 2026-06-04 — Beat It lead-sheet chords drift ~1.5 beats across the song — the QUANTIZER WARP, not the BPM

**Symptom:** chords wander relative to the barlines across the song (beat 1.75 early, drifting to 1.5/2.0/…). User: "is the song even properly timed?"

**FIRST (WRONG) DIAGNOSIS — recorded as a lesson:** I fit a constant bar period to the chord-strike ticks *of the output MIDI* and got 3854 ticks/bar = 137.5 BPM (conc 0.981 vs 0.784 at 138). Concluded the real tempo was 137.5 and `--bpm 138` was slightly too fast. Rewrote `_refine_bpm_for_grid` to a bar-phase objective + stopped integer-rounding BPM, "validated" on a simulation and on the *quantized* onsets (which gave 137.55). **All of that was diagnosing the symptom as the cause.** Re-running the real pipeline still drifted, and capturing the RAW (pre-quantization) onsets showed the truth: raw chord strikes sit at **138.0** (conc 0.990) — the performance is metronomic at 138. The 137.5 only appeared in the already-corrupted *output*. The refiner change was reverted.

**ACTUAL ROOT CAUSE:** the local-tempo **warp** applies a constant global stretch. `estimate_local_bpm` reads ~138.3–138.5 for steady-138 data (per-note 16th spacing reads slightly fast; with frame jitter, `BPM = k/interval` is also convex so it biases high). `build_shared_warp_lookup` then multiplies every interval by `local_bpm/effective_bpm ≈ 1.003` and integrates — a steady stretch that compounds to ~1.5 beats over the song, dragging clean 138 strikes onto a 137.5-drifting grid. Proof: raw strikes pure-rounded at 138 → conc@3840 = **0.989** (no drift); the same strikes through the warp → **0.787** (drift).

**FIX (`quantization.py`, `build_shared_warp_lookup`):** the warp must only *redistribute* timing locally (fix rush/drag), never change the *global* rate — that is `effective_bpm`'s job. After building the cumulative warp, rescale it so the warped span equals the raw span. A constant-tempo warp becomes the identity; genuine local corrections survive. Beat It strikes: conc@3840 0.787 → **0.989**, bar = 138.0. All 14 `test_quantization.py` still pass. Re-transcribed end-to-end: drift gone, chords now land at consistent groove positions (downbeat / "and of 1") instead of smearing.

**Prevention:**
- **Diagnose tempo from RAW onsets, never the quantized output** — the quantizer can manufacture the very drift you're measuring. Add a raw-onset dump (env-gated) and measure there.
- A warp/local-tempo correction must be span-preserving; verify a constant-tempo input is a no-op through it.
- When a "fix" validates on a simulation/proxy but not on a real re-run, the proxy is wrong — get the real intermediate data before claiming success.

### 2026-06-04 — MuseScore 4 "file is corrupted" — note <type> not matching <duration>

**Symptom:** MS4 dialog "File … is corrupted / contains errors" on open. MuseScore 3 and music21 parse it fine; MS4's stricter validator rejects it. MS4 CLI export crashed *without* producing output.

**Cause:** `lead_sheet._note_xml` wrote a single `<note>` using `_nearest_type(duration)` — the closest representable value. For durations that aren't a clean note value (2.5 beats = 2400 ticks, 2160, 2640, 3120, 3360 …) the emitted `<type>` implied a different length than `<duration>` (e.g. duration 2400 but `type=half`+dot → 2880). 18 such notes in the Beat It sheet. A 2.5-beat span is not one note; it must be a half **tied** to an eighth.

**Fix:** `_decompose_duration()` greedily splits any duration into representable components (whole … 32nd, dotted variants); `_note_xml` emits them as tied notes (pitched) or consecutive rests, chaining the outer tie flags across components so every note has `<type>` consistent with `<duration>`. Verified: 0 type/duration mismatches, MS4 now exports a valid PDF, MS3 renders cleanly.

**Prevention:** every emitted `<note>`'s `<type>`(+dots) must equal its `<duration>`; assert this in any MusicXML writer. To validate a MusicXML for MS4 specifically: check per-note type==duration and per-measure duration sum — music21/MS3 parsing success is NOT sufficient (they're lenient where MS4 is strict).

## 2026-06-11 — falling-blocks detector mapped every note to a white key (Billie Jean, 360p)
- **Symptom:** transcribing Billie Jean (F# minor) from a PlutaX tutorial yielded 0–1% black keys — musically impossible for F# minor (F#, C#, G# are all black). The bassline came out all naturals (E/B instead of F#/C#/E).
- **Root cause:** the default `falling-blocks` detector snapped block x-centers to white keys only on this video. The keyboard detection itself was fine (23 black keys found, note_map included F#2/C#2/G# etc.) — the failure was in the falling-blocks sampling, not the pitch map. Low resolution (360p; YouTube signature challenge blocked higher formats in yt-dlp 2026.02.04) made it worse.
- **Fix:** `--detector keys`. Immediately gave 62%/49% black keys and the correct bassline (F#2 C#2 E2 B1...). 
- **Prevention:** sanity-check black-key ratio against the key signature right after extraction. Near-0% black in a sharp/flat key (or >30% black in a natural key) means the detector is mis-mapping — switch detector or reframe before exporting. Don't trust note-count/octave-histogram checks alone.

## 2026-06-11 — "verified" a transcription whose rhythm was wrong (Billie Jean), and the quantizer drifts the tempo on steady patterns
- **Software failure:** the default `simple` quantizer + `_refine_bpm_for_grid` shifted Billie Jean's grid 117→116 BPM and snapped onsets independently. A pure straight-eighth intro (must be 64 identical 480-tick intervals) came out with 4 defects: three onsets a sixteenth early (480→240) + one 720 gap. Identical repeating patterns are NOT rendered identically → reads as the tempo "shifting." Full brief + repro + fix plan: `.agents/investigations/rhythm-quantization-pattern-drift.md`.
- **Agent failure (mine):** I checked pitch names at 4 timestamps + black-key ratio + song structure, then declared the output "verified." None of those look at inter-onset intervals, so the rhythm bug was invisible to my checks. I handed over a PDF and called it correct. The user caught the drift immediately by eye.
- **Root cause of the agent failure:** treated "spot-checked some notes" as verification. Pitch correctness is necessary but not sufficient — a transcription can have every note right and every duration wrong.
- **Prevention:** verifying a transcription MUST include a rhythm check. Concretely: take the most obviously-repeating passage, compute its inter-onset intervals, and confirm they're identical where the music is identical (put the histogram in the response). A repeating ostinato whose intervals aren't all equal = fail, regardless of how right the pitches are. Never call a sheet "verified" off pitch/black-key checks alone again.
