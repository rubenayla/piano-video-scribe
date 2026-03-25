# Agent Quick Reference

## Rules

1. **Always verify rendered output before confirming a fix.** Export the PDF and visually check it — MIDI data alone is not sufficient. Never tell the user something is fixed without seeing the rendered result. (See `.agents/error-log.md` 2026-03-20 entry.)
2. **Never open files in the user's face.** Do not run `open -a "MuseScore 4"` or similar. Just give the file path so the user can click it themselves.

## Key Files
1. **`.agents/pipeline.md`** — Full pipeline: YouTube URL → PDF sheet music (commands, calibration, known quirks)
2. **`.agents/notes.md`** — Experiment findings, technical insights
3. **`.agents/error-log.md`** — Mistakes and lessons learned

See `.agents/README.md` for the full index.
