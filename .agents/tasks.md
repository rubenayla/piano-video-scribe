<!-- consult selectively — grep, never read in full -->
# Tasks

## TODO


- **Fix Puppet LH 8th-note grid alignment (m7 drift)**
  - Puppet LH still has 87/247 notes on off-8th positions (test expects ≤10)
  - The BPM is correct (90), but raw onsets have a constant phase offset (+0.1-0.2 grid16)
  - This is NOT a BPM drift — it's a detection bias (fall speed or keyboard_y calibration)
  - Might need a phase-offset correction step in the quantizer
  - **Also**: LH measure 5 has wrong rhythm — investigate why

- **Improve `estimate_local_bpm` convergence**
  - Currently uses median filter (window=7) + rate limiter to reject garbage BPM estimates from mis-rounded grid positions
  - Removing them entirely causes regressions (test3, test4) because garbage values pass through
  - The real fix: don't depend on initial grid positions for BPM estimation (chicken-and-egg problem)
  - Possible approach: use Viterbi grid positions (more robust), or estimate BPM from intervals without grid_step

- **Regenerate Laufey and Puppet PDFs for student Maite**
  - Laufey: BPM auto-detection now finds 149.9 automatically. Key=Db. Regenerate and verify m1-m32.
  - Puppet: `~/piano/songs/ib-marys-theme-puppet/`
  - Pipeline saves settings on each run — be careful not to overwrite test settings files

- **Fix falling-blocks pitch mapping bugs (octave + semitone errors)**
  - test3: `identify_c_position` + `base_octave=2` places notes ~2 octaves too low. Grid fit residual (38px) nearly equals spacing (43px) — unreliable.
  - test4: block x-center off by a few pixels causes C5 instead of C#5 (semitone errors at pitch boundaries).
  - Possible fix: use the keyboard detector's pitch map as ground truth when the video has visible key labels, and only use block-based mapping when keyboard detection fails.
  - Until fixed, test3/test4/test5/test7 must use `"detector": "keys"`.

- **Pipeline saves settings.json on every run — can corrupt test fixtures**
  - The pipeline auto-saves settings (including auto-detected values like `detector`) to the settings file
  - This overwrites committed test settings (e.g., changing `"detector": "keys"` to `"falling-blocks"`)
  - Fix: don't auto-save to test directories, or save to a separate output path

## In Progress

## Done (2026-03-30)

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
