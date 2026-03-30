# Agent Quick Reference

## Output Convention
When finishing a task that produces files (MIDI, PDF, images), always list the output file paths so the user can access them in one click. Example:
```
Output: /path/to/output.mid
```

## Rules

1. **Always verify the final output before giving it to the user.** After generating a MIDI file, inspect the actual tick positions and compare against expected values before sharing the path. Don't just check raw detections — check the *quantized* output that the user will actually hear/see. If the user told you the expected pattern (e.g., "6 eighth notes: E EGB EGB B EGB EGB"), compare the MIDI against that *exactly*. (See `.agents/error-log.md` 2026-03-20 and 2026-03-29 entries.)
2. **Never open files in the user's face.** Do not run `open -a "MuseScore 4"` or similar. Just give the file path so the user can click it themselves.
3. **Verify hand-color assignment after `detect_colors.py`.** The script cannot determine which color is right vs left — it just guesses. Watch the video to check which color plays the higher notes (right hand). Don't blindly use its suggested assignment.
4. **Use kebab-case for data/config filenames.** `output-video.mid`, not `output_video.mid`. Applies to MIDI, JSON, configs, and `.agents/` docs. Python source files follow PEP 8 (snake_case). The pipeline output filename is a CLI argument — always pass `output-video.mid`.

## Key Files
1. **`.agents/pipeline.md`** — Full pipeline: YouTube URL → PDF sheet music (commands, calibration, known quirks)
2. **`.agents/notes.md`** — Experiment findings, technical insights
3. **`.agents/error-log.md`** — Mistakes and lessons learned
4. **`.agents/tasks.md`** — Task board (TODO / In Progress / Done)

See `.agents/README.md` for the full index.
