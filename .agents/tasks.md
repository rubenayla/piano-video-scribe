<!-- consult selectively — grep, never read in full -->
# Tasks

## TODO

## In Progress

- **Improve `estimate_local_bpm` convergence**
  - Currently uses median filter (window=7) + rate limiter to reject garbage BPM estimates from mis-rounded grid positions
  - Removing them entirely causes regressions (test3, test4) because garbage values pass through
  - The real fix: don't depend on initial grid positions for BPM estimation (chicken-and-egg problem)
  - Possible approach: use Viterbi grid positions (more robust), or estimate BPM from intervals without grid_step

## Done (2026-03-31)

- [2026-03-31] **Fix Laufey m105+ wrong rhythm**: Root cause was `quantizer=viterbi` in pipeline called `quantize_onsets_adaptive` (2-pass with BPM warping) instead of plain `quantize_onsets_viterbi`. The adaptive warping accumulated +70ms by m49 and +170ms by m107, shifting notes +1 to +2 sixteenths. Fix: `viterbi` setting now calls plain Viterbi; added separate `adaptive` option. Regenerated output MIDI and PDF. m1-m48 unchanged; m49+ now correct.
- [2026-03-31] **Make settings.json read-only**: Removed auto-save block from pipeline.py. Settings are now read-only input; CLI flags override in memory only. Reverted corrupted test settings files.
- [2026-03-31] **Puppet GT committed as regression test**: `ground_truth.mid` + 4 exact-match tests (pitch AND tick, zero tolerance). 96 RH, 247 LH notes. User confirmed by ear.
- [2026-03-31] **Regenerate Puppet PDF for Maite**: Re-ran pipeline with current code. Output verified correct. Files at `~/piano/songs/ib-marys-theme-puppet/`.
- [2026-03-31] **Regenerate Laufey PDF for Maite (partial)**: m1-m104 perfect. m105+ has wrong rhythm — see TODO. Files at `~/piano/songs/from-the-start-laufey/`.

## Done (2026-03-30)

- [2026-03-31] **Fix Puppet LH 8th-note grid alignment**: Added `_refine_bpm_for_grid()` drift correction to `quantize_onsets_simple()`. Root cause: falling-blocks detector's fall speed implies ~89.07 BPM vs nominal 90 BPM, causing progressive phase drift (+0.5 grid16 over the piece). The quantizer now searches ±1.5% of nominal BPM (0.01 steps) to maximize 8th-grid alignment before rounding. Result: 0/247 LH notes off-grid (was 87/247). All 7 puppet tests pass.
- [2026-03-31] **Fix falling-blocks pitch mapping**: Keyboard detector now used first for pitch mapping (reliable octave), with block-based as fallback. Calibration rejected when max residual > min_gap/2 to prevent semitone errors. Also fixed argparse `--detector` default shadowing settings file values. test3: E6/D#6/B5 correct (was D4/B3/G#3). test4: C#5 correct (was C5).
- [2026-03-31] **Compare keys vs falling-blocks detector on test3/test4**: Keys detector is correct; falling-blocks has critical pitch mapping bugs. See details below.
  - **test3**: Keys detector gets correct pitches (E6, D#6, B5, G#5 matching GT exactly). Falling-blocks maps same notes to D4, B3, G#3 (~2 octaves too low). Root cause: `identify_c_position` with `base_octave=2` places C at wrong grid position; grid fit has max_residual=38px (nearly equal to spacing=43px), meaning the grid itself is unreliable. Falling-blocks: 0% exact pitch match. Keys: 98% LH, 49% RH (RH limited by progressive timing drift, not pitch errors).
  - **test4**: Keys detector matches GT perfectly (C#5, D5, A5...). Falling-blocks falls back to keyboard detector (too few blocks for grid fit) but still gets pitch errors (C5 instead of C#5). Root cause: block x-center slightly off, landing in wrong pitch boundary. Falling-blocks: 6% RH exact match. Keys: 32% RH, 90% LH.
  - **Conclusion**: `"detector": "keys"` should remain the setting for test3/test4. The falling-blocks detector has two bugs: (1) octave assignment via `base_octave` is unreliable when the keyboard range is ambiguous, and (2) x-center precision in contour detection can cause semitone errors.
  - Restored corrupted test settings (test3, test5, test7) where auto-save had overwritten `"detector": "keys"` with `"falling-blocks"`.

- [2026-03-31] **BPM precision fix**: `detect_bpm` and `--bpm` now use float (was int). BPM refinement searches in 0.1 steps using grid-alignment metric (notes on 8th grid) instead of interval-fractional error. Laufey auto-detects 149.9 BPM → all 154 LH notes on 8th grid.
- [2026-03-31] **Key signature enharmonic fix**: Auto-detection now maps C#→Db, G#→Ab, etc. (prefer fewer accidentals). Laufey was C# major (7 sharps + double sharps) → now Db major (5 flats).
- [2026-03-31] **Default detector → falling-blocks**: Changed from None (fell through to keys) to `falling-blocks` in argparse defaults.
- [2026-03-30] **Laufey drift investigation**: Root cause was BPM mismatch (video at ~149.9 BPM, nominal 151). The fix is precise BPM detection (float), not a quantizer warp. Global warp approach was tried and reverted — it overcorrects the second half of the piece.

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
