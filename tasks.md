<!-- consult selectively — grep, never read in full -->
# Tasks

Done items do not stay on the board. When one closes, move it — with its date and closing note —
to `tasks/done-archive.md`, which holds nothing actionable. Exception: a done step of a task that is
still open stays put, since archiving it strips the remaining open step of the context saying what
was already settled. A cluster moves to the archive whole, once its last step closes.

## TODO

- **test3 regression tests ERROR: mido rejects key `'Dbm'`** — `tests/test3/settings.json` has `"key": "Dbm"`, but D♭ minor is not a valid MIDI key signature (mido raises `ValueError: invalid key 'Dbm'` when writing the meta message). Either map enharmonic keys to valid ones before writing (`Dbm` → `C#m`) in midi_output, or fix the settings file. Pre-existing, surfaced 2026-06-11 during the quantizer work; NOT related to quantization.

- **Keys detector merges rapid repeated strikes & holds noise tails (Magic School Bus)**
  - Symptoms on `~/piano/songs/the-magic-school-bus-theme-original-1994/video-1080p.mp4` (BPM 119, key E):
    - RH E4 at video t=7.97s reported as one 0.76s held note; the video shows 4 separate E strikes (16th-note run). Detector merges them.
    - LH B2 at video t=10.49s reported as 2.02s held; user states it's not held that long musically (the key does flip on/off in the actual video — user reviewed t≈8s frame and confirmed separate strikes are visible in the falling-bar pattern).
    - RH F#5 at video t=18.69s reported (0.13s) but visually only G5 (immediately right) is lit; F#5's narrow black-key sample column is picking up adjacent G5 white-key glow. Likely a separate, simpler bug.
  - Root cause attempt #1 (commit f97b904 — currently on main, needs revisiting): added in-loop re-strike rising-edge detection + relative-peak floor truncation in `piano_video_scribe/detectors/keys.py`. Heuristic shape is correct but **does not fire on real signal** because `KeySampler` in `piano_video_scribe/backend.py` uses `v_min=30` to mask dim pixels then averages bright ones — the resulting per-key signal is flat (e.g. E4 stayed at 146.1 for the whole 0.76s span, B2 stayed at 84.0 for the whole 2.02s span). All within-note variation is filtered out by KeySampler before keys.py ever sees it.
  - Standalone proof that the variation IS in the video: a narrow-column raw-mean trace (no v_min mask, x±4 strip) at the same pixels shows clear valleys/restrikes for E4 (peaks ~140 with dips to ~98–108 between strikes) and a brief spike-then-decay for B2 (sat 83→50→30 within ~100ms of strike, never re-rising). See `/tmp/fix_midi.py` (the post-hoc patcher that successfully split E4 into 3 strikes and trimmed B2 from 2.02s to 1.33s).
  - Two candidate fixes (decide in plan mode):
    1. **Second sampling channel in `KeySampler`**: expose an unfiltered narrow-column mean alongside the existing v_min-filtered mean. keys.py uses unfiltered for re-strike + peak-floor; filtered for on/off thresholding.
    2. **Post-extraction refinement pass**: after `extract_notes_from_video`, for each note with dur > 0.30s, re-trace its pitch's narrow column over [onset, offset] using a raw mean (no v_min) and apply the rising-edge / peak-floor heuristics to split or trim. Slower but isolates the change to a single new module without touching the hot loop.
  - Pre-existing 2 failing tests (`test-falling-laufey`) are unrelated to this and live on main already.
  - For F#5 specifically: separate root cause (black-key column overlapping adjacent white-key glow). Likely needs stricter neighbor-ratio suppression or shifting F#5's sample x by half a pitch to dodge G5's edge.

- **Improve `estimate_local_bpm` convergence**
  - Currently uses median filter (window=7) + rate limiter to reject garbage BPM estimates from mis-rounded grid positions
  - Removing them entirely causes regressions (test3, test4) because garbage values pass through
  - The real fix: don't depend on initial grid positions for BPM estimation (chicken-and-egg problem)
  - Possible approach: use Viterbi grid positions (more robust), or estimate BPM from intervals without grid_step

## Done (2026-06-11)

- **[HIGH] Identical repeating patterns render with shifting/wrong rhythm — FIXED.** Default quantizer changed `simple` → `viterbi` fed the shared *effective* BPM (pipeline.py dispatch). Billie Jean intro gate passes end-to-end (`{480: 64}`, zero exceptions); Beat It output byte-identical to the validated simple baseline (syncopation intact); Laufey RH now 0 GT mismatches (was 2 with simple, 18 with adaptive). Regression test added: `test_billie_jean_intro_equal_intervals_viterbi` in tests/test_quantization.py (real frozen onsets). Resolution details in `.agents/investigations/rhythm-quantization-pattern-drift.md`. The user's Billie Jean MIDI + PDF in `~/piano/songs/billie-jean-michael-jackson/` regenerated and visually verified.

## Done (2026-03-31)

- [2026-03-31] **Laufey GT committed as regression test**: `ground_truth.mid` + 4 exact-match tests (m1–104 only, 517 RH + 564 LH notes). m105–108 excluded: ritardando not encoded as tempo changes, arpeggios approximate. User hand-corrected in MuseScore.
- [2026-03-31] **Separate viterbi/adaptive quantizer options**: `viterbi` now calls plain Viterbi DP; `adaptive` calls 2-pass with BPM warping. Laufey uses `adaptive` in settings.
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
