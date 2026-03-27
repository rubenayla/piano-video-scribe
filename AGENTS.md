# Agent Quick Reference

## Rules

1. **Always verify rendered output before confirming a fix.** Export the PDF and visually check it — MIDI data alone is not sufficient. Never tell the user something is fixed without seeing the rendered result. (See `.agents/error-log.md` 2026-03-20 entry.)
2. **Never open files in the user's face.** Do not run `open -a "MuseScore 4"` or similar. Just give the file path so the user can click it themselves.
3. **Verify hand-color assignment after `detect_colors.py`.** The script cannot determine which color is right vs left — it just guesses. Watch the video to check which color plays the higher notes (right hand). Don't blindly use its suggested assignment.
4. **Use kebab-case for data/config filenames.** `output-video.mid`, not `output_video.mid`. Applies to MIDI, JSON, configs, and `.agents/` docs. Python source files follow PEP 8 (snake_case). The pipeline output filename is a CLI argument — always pass `output-video.mid`.

## Key Files
1. **`.agents/pipeline.md`** — Full pipeline: YouTube URL → PDF sheet music (commands, calibration, known quirks)
2. **`.agents/notes.md`** — Experiment findings, technical insights
3. **`.agents/error-log.md`** — Mistakes and lessons learned
4. **`.agents/tasks.md`** — Task board (TODO / In Progress / Done)

See `.agents/README.md` for the full index.
