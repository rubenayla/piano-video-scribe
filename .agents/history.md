<!-- consult selectively — grep, never read in full -->
# History

## 2026-05-02 — Damirón "Piano Merengue" debug session: pitch calibration, hand sync, sub-bass artifacts

Worked through a full pipeline run on a Synthesia of Damirón's *Piano Merengue* (Marcos Burbano transcription, ♩=244, G major, 4/4 with 2-sixteenth pickup). The first MIDI was clearly wrong; iterated through several distinct bugs.

**What was wrong on the first pass:**
- First-event pitches were off by ~1 white key (script said `LH=A4 + RH=C5` but the bars at video t=2.0s were centered ~18 px left of where keyboard.py thought A4/C5 were).
- LH drifted out of sync with RH starting at the 5th sixteenth of m.1 (LH events landed 1 sixteenth-grid late), then accumulated more drift through the song.
- 8 super-low notes (A#0/B0/D1) scattered through the MIDI at moments matching when the YouTube SUBSCRIBE button overlay appeared.
- Drift came back later in the song even after the early sync fix.

**Root causes (in order of fixing):**
1. **`detect_falling_notes_pipeline` ignored `--frame`.** It hardcoded `frame_idx=None` and re-ran keyboard auto-detect on whatever frame the scanner landed on (frame 0 here), getting a keyboard map polluted by intro animation. Fix: pass `args.frame` through `fb_kwargs` in `pipeline.py` and accept `frame_idx` in the pipeline.
2. **Black-key x-positions used the noisy detected list.** ±19 px error (~ half a white-key width) was enough to drop the black-key detector zone inside an adjacent white key's bar path → phantom black-key notes. Fix: midpoint-only positioning in both `keyboard.py::build_note_x_map` and `detectors/falling_blocks.py::build_pitch_map`.
3. **Black-key sampling y-zone covered the full body.** Falling bars travel through the upper part of the body before striking. Fix: tightened to bottom 25% via a new `black_bottom_frac` knob in `build_detector_regions`.
4. **`quantize_onsets_simple` ran drift correction independently per hand.** Each hand picked its own effective BPM (sometimes 243.55 vs 244.0) → desync by one 16th by the second half of m.1. Fix: compute `shared_effective_bpm` and `shared_local_bpm` once over the union of all onsets, pass them in.
5. **`warp_onsets` integrated the local-BPM curve piecewise per-hand even with a shared BPM lookup.** Sparser LH integrated differently from denser RH → drift returned by mid-song. Fix: new `build_shared_warp_lookup(onset_secs, bpm)` builds a continuous t→warped_t function over the union of both hands' onsets; each hand looks up its onsets in the same map.
6. **Per-hand quantize_hand call structure was a foot-gun.** Even with shared inputs, future changes could re-introduce per-hand drift. Refactored: one combined sorted call to the chosen quantizer, then split grid positions back per hand. `RIGHT_SUB != LEFT_SUB` now hard-errors.
7. **Sub-bass artifacts from over-detected leading "white keys" before C1 + static UI overlays.** Fixes:
   - Drop notes with MIDI < 28 (below E1) at the end of `extract_notes_tracking`.
   - Reject tracks whose y_bot doesn't descend ≥ 30% of `fall_speed × frame_span`, OR which span ≥ 6 frames yet move < 25 px.

**Verification (final state on the Damirón video, output-anacrusis.mid, `--quantizer adaptive`):**
- First-note pitches match the visible bar centroids (verified via overlay: BLUE on A4, GREEN on C5 at f60).
- Both hands stay locked to a single 16th-grid for the whole song. Mid-song sample at t=80–83 s: every event within ±0.3 ms of the exact grid. Late-song sample at t=120–122 s: same.
- Pickup placement: with `--start-beat 4.0`, first 2 events land in m.0 at beats 4.0 and 4.5 (= 4 sixteenths of pickup); m.1 starts on its downbeat at event #3.
- 0 sub-bass artifacts (was 9 before the filter).
- Black-key fraction 14.1% — consistent with G major + chromatic ornaments shown in the published score.
- User confirmed "all the errors are gone, this is working really well" after the adaptive quantizer pass.

**Open follow-up:** user reported small timing drift past m.27 in an earlier pass. Switching from `simple` to `adaptive` quantizer resolved it for them; the underlying cause is likely the local-BPM warp's 7-window median smoothing missing tempo micro-changes when the music thins out (gap visible at video t≈30 s where no keys are being struck). If it recurs, try `--quantizer adaptive` first; if still off, dig into raw onset spacing in the gap window.

**Files touched (this session):**
- `piano_video_scribe/keyboard.py` — black-key midpoint positioning + `black_bottom_frac` knob
- `piano_video_scribe/detectors/falling_blocks.py` — `frame_idx` kwarg, midpoint-only black keys, static-blob rejection, sub-bass filter
- `piano_video_scribe/quantization.py` — `quantize_onsets_simple` accepts shared bpm/lookup; new `build_shared_local_bpm_lookup` and `build_shared_warp_lookup`
- `piano_video_scribe/pipeline.py` — propagate `--frame` to falling-blocks; precompute shared bpm + warp; combined single-pass quantization

Commits: `3e309c5`, `2630518`, `eaf6805`, plus this commit.
