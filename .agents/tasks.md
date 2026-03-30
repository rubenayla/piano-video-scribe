<!-- consult selectively — grep, never read in full -->
# Tasks

## TODO

- **Fix 1/16th quantizer drift at m15 (Laufey) and m7 (Puppet)**
  - Symptom: LH notes shift +1/16th mid-piece. Laufey m15 at beat 0.25 instead of 0.00. Puppet m7 LH at 18.25 instead of 18.00.
  - Root cause: the tracker's fall speed measurement is ~0.5% off. Each onset prediction is `t + (keyboard_y - y_bot) / (fall_speed * fps)`. A 0.5% speed error compounds to ~25ms over 40s → 1/4 of a 16th at BPM=151.
  - The simple quantizer (round-to-nearest) exposes this because it has no inter-note correction.
  - `warp_onsets` + `estimate_local_bpm` (the adaptive drift correction) doesn't fix it because the drift is linear fall-speed error, not BPM variation.
  - **Approach to try**: compute per-track fall speed from the tracking data (each block is tracked across many frames — `dy/dt` gives local speed). Use median of per-track speeds instead of global `measure_fall_speed()`. An agent was started for this but may not have finished.
  - Test: `python pianovideoscribe.py tests/test10/video.mp4 /tmp/test.mid --settings tests/test10/settings.json` then check LH m15 beats. Should be 0.00, 1.00, 2.00, 3.00.
  - Test: `python pianovideoscribe.py tests/test-falling-puppet/video.mp4 /tmp/test.mid --settings tests/test-falling-puppet/settings.json` then check LH m7 beats. Should be 18.00, 18.50, 19.00 etc.
  - Files: `piano_video_scribe/quantization.py` (quantize_onsets_simple), `piano_video_scribe/detectors/falling_blocks.py` (extract_notes_tracking, measure_fall_speed)

- **Regenerate Puppet and Laufey PDFs once drift is fixed**
  - Puppet: `~/piano/songs/ib-marys-theme-puppet/` — for student Maite
  - Laufey: `~/piano/songs/from-the-start-laufey/` — for student Maite
  - Both MIDIs sound mostly right but have the 1/16th shift issue above

## In Progress

## Done (2026-03-30)

- **Codebase restructuring**: Monolithic files split into `piano_video_scribe/` package (keyboard.py, color.py, quantization.py, midi_output.py, visualization.py, config.py, pipeline.py, detectors/)
- **Ground truth verification**: All GTs verified by ear — test2, test3, test4, test7, test8, test9, test10. test11/12 are synthetic.
- **Tracking-based block extractor**: `extract_notes_tracking()` replaces `extract_notes_contour()`. Tracks blocks across frames, median onset prediction. Fixes ghost notes, glow artifacts.
- **Ghost note fix**: Progress bar (A1) and glow artifacts eliminated via auto-detecting scan region top (skip UI bar) and configurable `glow_margin`
- **Per-color s_min**: `s_min_per_color` in settings.json allows different saturation thresholds per hand color (needed to separate rapid repeats vs. light-blue blocks)
- **Hand alignment fix**: Both hands quantized on shared timeline with common `start_grid_offset`
- **Diagnostic tool**: `piano_video_scribe/diagnostics.py` — `diagnose(video, timestamp)` generates annotated frames showing detected blocks, pitch assignments, scan bounds
- **Audio re-added to test videos**: test2, test3, test7, test9, test10 re-downloaded with audio for easier manual verification
- **Simple quantizer**: `quantize_onsets_simple()` as default (round-to-nearest + always-on drift correction). Viterbi available via `--quantizer viterbi`.
- **Verify-midi skill**: `.claude/skills/verify-midi.md` — must be called before presenting .mid files
