<!-- consult selectively — grep, never read in full -->
# History

## 2026-05-06 — Magic School Bus run: perf wins, detector switch, hand classification fix

Picking up from 2026-05-05. Three changes landed in the repo and dropped wall time from ~12 min to **62 s** for the 62 s Magic School Bus video (~220 fps sustained, torch-mps backend).

**Code changes (in this repo):**
- `piano_video_scribe/backend.py` — new module. Auto-selects torch (MPS / CUDA / CPU) or vectorized NumPy at import. `PVS_BACKEND=numpy` env var forces NumPy. PyTorch is optional; non-torch users fall back automatically. Exports `KeySampler` which pre-computes a flat `(pixel_index, key_id)` map and reduces a whole HSV frame to per-key mean saturation in one bincount / scatter_add — replaces the per-key Python loop.
- `piano_video_scribe/detectors/keys.py` — uses `KeySampler` for the inner per-frame reduction. Was the per-key Python loop (~325k iterations for a 1-min video).
- `piano_video_scribe/video_io.py` — new `ResizingCapture` wrapping `cv2.VideoCapture`, returns frames downscaled to a max height (default 1080). Drop-in: `cap.get(CAP_PROP_*)` reports the resized dims so all downstream pixel math stays consistent.
- `piano_video_scribe/pipeline.py` + `detectors/falling_blocks.py` — both wrap their `cv2.VideoCapture` in `ResizingCapture` when `--max-height > 0` and source height exceeds it. Falling-blocks accepts `max_height` kwarg now.
- `piano_video_scribe/config.py` — `--max-height N` flag (default 1080, 0 disables).

**Decisions / preferences (binding):**
- **Never stride frames.** User explicitly rejected `--sample-step` shortcuts: striding skips fast notes and breaks timing precision. The `keys` detector processes every frame; this is a hard rule. Striding is only conceivably defensible for a falling-blocks detector watching long-lived bars — never for keyboard-glow detection where the signal is ~1 frame wide.
- **Pre-scaling the source video with ffmpeg is the preferred resize path**, not per-frame `cv2.resize`. `ffmpeg -hwaccel videotoolbox -i src -vf scale=-2:1080 -c:v h264_videotoolbox -b:v 8M -an out.mp4` — runs at ~3× realtime on Apple Silicon, one-shot. `cv2.resize` per-frame in `ResizingCapture` is the fallback; on the Magic Bus video it actually made wall time worse (12 min vs 5 min) because the falling-blocks detector doesn't gain enough from smaller frames to offset the per-frame resize cost.
- **`--detector keys` is the right detector** for clean Synthesia videos. The pipeline default is currently `falling-blocks`; that should likely flip, or at least docs should steer users toward `keys`. Falling-blocks defaults to `--sample-step 10` which silently violates the no-strides rule.
- **GPU is a real win for `keys` detector** (220 fps with torch-mps vs ~5 fps Python loop). For `falling-blocks` it doesn't help — work isn't pixel-bound.

**How to apply:**
- When proposing perf work: per-frame Python loops are the first target. Don't suggest striding. Don't suggest `cv2.resize` per-frame as the primary scale path — pre-scale the source first.
- When running this pipeline on user videos: default `--detector keys`, pre-scale to 1080p with ffmpeg if source is 4K.

**Magic School Bus run results:**
- 243 notes (vs 87 with falling-blocks). RH 104 / LH 139, ranges A3–G5 / E2–G3 — clean register split. Hand color classification works as expected when `keys` is the detector.
- Frame-vs-MIDI verification at t=5.5s: RH G#4 matches lit green key; LH may be off by a few semitones (keyboard x-calibration to confirm in MuseScore). Frame check is the only way to catch calibration errors — note counts and ranges look fine in isolation.

**Vault note (separate concern, but recurring):**
- The user's vault had `the-magic-school-bus-theme-original-1994-easy.pdf` which was actually a *different song* misnamed. User deleted it. **Do not offer to restore it from its MuseScore source URL** — the source is the wrong song. Only `-medium.pdf` and `-intermediate.pdf` (both arr. Leo the Tiger, score `21159355`) are legitimate for this folder.

---

## 2026-05-05 — User preferences: performance vs accuracy, Magic School Bus run

Discussion during a Magic School Bus theme run (62s video, ~5–10 min wall time).

**Decisions / preferences:**
- **Do not stride frames** (don't sample every Nth frame as a perf shortcut). User explicitly rejected this — concerned about timing precision and missed keys. Detection must process every frame.
- **Decode is the bottleneck.** OpenCV `VideoCapture` on macOS uses software FFmpeg decode, single-threaded. The mp4 is H.264; Apple Silicon has VideoToolbox (hardware decoder) sitting idle. Plan: pipe `ffmpeg -hwaccel videotoolbox` raw frames into the detector — offloads decode to silicon, no accuracy change.
- **GPU for per-key sampling is on the table** if hardware decode alone isn't enough. Best Mac fit is Metal/MPS via PyTorch or Apple's MLX (no CUDA on Apple Silicon). Reformulate per-frame HSV conversion + per-key reductions as tensor ops. Real rewrite, but accuracy-preserving.
- **`cv2.cuda` is not an option** on macOS — no NVIDIA.

**How to apply:** when proposing perf work on this repo, never suggest temporal subsampling. Decode-side and GPU-compute optimizations only.

**Magic School Bus run notes (separate issue from perf):**
- Per-pitch rainbow color labels in this video, not per-hand. Detector's hue→hand mapping doesn't separate hands; resulting MIDI had bass notes on RH and treble on LH. Hand classification by color alone is unreliable for videos like this.
- First pass: BPM auto-detected 105 (actual 119–120 per MuseScore source `21159355`), key auto-detected B major (actual E major), only 90 notes for 62s. Fixed run uses `--bpm 119 --key E --start-time 5.5 --frame 0`.
- Trusted score downloaded from MuseScore (`21159355`) — saved as `the-magic-school-bus-theme-original-1994-intermediate.pdf` in the song folder. Pipeline output for this video is the secondary artifact pending the iteration.

---

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
