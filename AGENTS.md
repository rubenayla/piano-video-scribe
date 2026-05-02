# Agent Quick Reference

## Output Convention
When finishing a task that produces files (MIDI, PDF, images), always list the output file paths so the user can access them in one click. Example:
```
Output: /path/to/output.mid
```

## Rules

1. **Always verify the final output before giving it to the user — NO EXCEPTIONS.** Before presenting any .mid path, dump the output note-by-note for at least m1-8 and compare against the GT. Post the comparison in the response showing beat, note, and match/miss/extra status. **If this comparison is not in the message, the .mid path must not be in the message.** Note count matches are NOT sufficient — you must check actual note positions. This rule has been violated twice in one session (see `.agents/error-log.md` 2026-03-30 entries) — it is not optional regardless of how confident the results look.
1a. **When there is no GT MIDI, verify against the video directly.** Find the first real keypress timestamp (via `ffmpeg -ss <t> ... -frames:v 1`), read the notes/color/chord off the frame, and compare against the MIDI at that timestamp. Repeat at ≥3 more timestamps spread across the song. Include the comparison table in the final message (video_time | video_notes | midi_notes | match?). Automatic checks (note count, octave histogram, hand range) can catch gross errors but DO NOT substitute for the frame comparison. Common failure modes this catches: intro-title false positives (fix with `--start-time`), swapped hand colors (fix with `--green-hand`/config swap), keyboard x-calibration off by a half-step producing all-black-key output (fix with a different `--frame`), wrong BPM spreading chords across a beat.
1b. **When the user says a note is wrong, trust them and fix it.** Don't argue that the detector is "correct" when the output has wrong notes. The user has listened to the music and knows the piece. The detector's job is to produce correct notes — if it doesn't, it's a bug regardless of what the raw pixels show.
2. **Never open files in the user's face.** Do not run `open -a "MuseScore 4"` or similar. Just give the file path so the user can click it themselves.
3. **Verify hand-color assignment after `detect_colors.py`.** The script cannot determine which color is right vs left — it just guesses. Watch the video to check which color plays the higher notes (right hand). Don't blindly use its suggested assignment.
4. **Use kebab-case for data/config filenames.** `output-video.mid`, not `output_video.mid`. Applies to MIDI, JSON, configs, and `.agents/` docs. Python source files follow PEP 8 (snake_case). The pipeline output filename is a CLI argument — always pass `output-video.mid`.

## Key Files
1. **`.agents/pipeline.md`** — Full pipeline: YouTube URL → PDF sheet music (commands, calibration, known quirks)
2. **`.agents/notes.md`** — Experiment findings, technical insights
3. **`.agents/error-log.md`** — Mistakes and lessons learned
4. **`.agents/tasks.md`** — Task board (TODO / In Progress / Done)

See `.agents/README.md` for the full index.
