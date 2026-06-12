<!-- reference — read in full when working this issue -->
# Investigation: identical repeating patterns render with shifting/wrong rhythm

**Status:** RESOLVED 2026-06-11 — default quantizer changed `simple` → `viterbi` fed the shared effective BPM. See §0 for the resolution; the original brief is kept below for the record.

## 0. Resolution (2026-06-11)

**Fix shipped:** `pipeline.py` dispatch now defaults to `quantize_onsets_viterbi(sorted_onsets, shared_effective_bpm, subdivisions=SUB)`. Viterbi gained a `subdivisions` param (was hardcoded to 16ths; would have silently broken `--triplet` mode as default) plus the sub-12 valid-position snap copied from PLL.

**Key findings that changed the §3 hypotheses:**
1. **The 117→116 refine was RIGHT, not the bug.** Raw detector onsets measured: LH intro mean interval 0.2589 s = 115.95 BPM. The real tempo IS ~116; `simple`'s defect is purely its independent per-onset snapping under ±14 ms frame jitter (hypothesis 2, not hypothesis 1).
2. **Detector layer is clean** (§4 step 1 done): all 64 raw LH intro intervals within a quarter of the mean. Quantizer-only bug, confirmed.
3. **PLL is a trap.** It passes the Billie Jean gate (0/64) but DESTROYS Beat It: 220 of 415 RH onsets pushed to false sixteenth offsets, 22 notes lost outright. Never judge a quantizer on one repro. The earlier note "swapping simple↔pll moves defects around" was directionally right for the wrong reason.
4. **Viterbi's single bake-off defect (1/64) was the nominal-vs-effective BPM gap.** The pipeline computed `shared_effective_bpm` but only fed it to `simple`. `viterbi @ 116` = 0/64 clean. Viterbi at the effective BPM also reproduces Beat It's validated output byte-identically and beats simple on the Laufey GT (RH 0 vs 2 mismatches).
5. **Onset-warping (the local-BPM warp `simple` uses) HURTS when fed to viterbi** on steady-tempo jittery video: warp+viterbi = 2–3 BJ defects. The warp's local-BPM estimate wobbles on noise; viterbi's DP handles drift better unwarped. Don't "improve" the default by adding the warp back without re-running all three gates.
6. **`adaptive` is broken as shipped** — its internal `warp_onsets` lacks the global-span rescale that `build_shared_warp_lookup` has, so a constant local-BPM bias accumulates (Beat It: 704 positions shifted). `tests/test-falling-laufey/settings.json` was pinned to `adaptive`; switched to `viterbi`.

**Verification (the §5 gates):**
- Primary: end-to-end Billie Jean with NO `--quantizer` flag → LH first-64 intervals `{480: 64}`. PDF re-engraved; bars 1–8 visually identical. ✓
- Regression: end-to-end Beat It with no flag → byte-identical MIDI to the validated `simple` baseline. ✓
- Suite: 67 passed. Pre-existing failures, both unrelated to quantization: laufey LH 149 GT mismatches (detector WIP, already in tasks.md), test3 ERROR `invalid key 'Dbm'` (mido key-signature, new tasks.md entry).
- Regression test added: `test_billie_jean_intro_equal_intervals_viterbi` (tests/test_quantization.py) with the real frozen detector onsets.

**Tools added:** `PVS_DUMP_ONSETS=<path>` env var dumps pre-quantization onsets/hands from the pipeline — lets quantizers be compared offline in seconds instead of re-running video extraction per candidate.

**git-blame answer (§ "why was simple default"):** commit cf8cef7 (2026-03-30) introduced `simple` and made it default in the same commit, with no recorded bake-off. There was no hidden reason to preserve it.

---

The original brief follows (pre-fix state).

**Owner of report:** user, 2026-06-11 (frustrated — this has recurred and been declared "fixed" before without being fixed). Treat "looks right in a spot-check" as NOT done. Success is defined numerically below.

---

## 1. The symptom (what the user sees)

A repeating ostinato that is *obviously* identical every bar (e.g. the Billie Jean bass intro — a straight stream of eighth notes for ~16 bars) is engraved with **non-identical measures**: beam groups compress/expand, notes land off the beat, the rhythm appears to "shift" or "speed up and slow down" even though the source is metronomically steady. The pitches are correct; only the *timing/notation* is wrong.

User's exact framing: *"the pattern should obviously repeat exactly the same, and the tempo is shifting wrong, again."*

## 2. Hard evidence (Billie Jean, this is the canonical repro)

Working dir: `~/piano/songs/billie-jean-michael-jackson/`
Output analysed: `billie-jean-piano-from-plutax.mid` (ticks_per_beat=960, so an eighth note = 480 ticks).

The intro left hand is pure straight eighths → **every** inter-onset interval must be exactly 480 ticks. Measured first 64 LH intervals:

```
delta histogram (ticks): {480: 60, 240: 3, 720: 1}
non-480 intervals: 4 of 64
positions (interval#, delta): (15, 240), (40, 240), (43, 720), (45, 240)
```

So 3 onsets got snapped **one sixteenth early** (480→240) and one interval absorbed the slack (720). Each defect breaks the visual repetition exactly as the user describes. A correct transcription of this intro is **64 × 480 with zero exceptions**.

Reproduce the measurement:
```python
import mido
m=mido.MidiFile('billie-jean-piano-from-plutax.mid'); tr=m.tracks[1]
t=0; ons=[]
for x in tr:
    t+=x.time
    if x.type=='note_on' and x.velocity>0: ons.append(t)
d=[ons[i]-ons[i-1] for i in range(1,len(ons))]
from collections import Counter; print(Counter(d[:64]))   # want {480: 64}
```

Regenerate the MIDI:
```
cd ~/repos/piano-video-scribe
python3 pianovideoscribe.py \
  ~/piano/songs/billie-jean-michael-jackson/video.mp4 \
  /tmp/bj.mid --config ~/piano/songs/billie-jean-michael-jackson/config.json \
  --bpm 117 --frame 80 --detector keys --right-hand no-overlap --left-hand no-overlap --key F#m
```
Run output includes: `Drift correction: nominal BPM 117.0 → effective 116.0 (grid RMS 0.2426)` — see §3.

## 3. Root-cause analysis (where the bug lives)

Dispatch: `piano_video_scribe/pipeline.py:392` → `quantizer = getattr(args, 'quantizer', None) or 'simple'`.
**The default quantizer is `simple`.** It is `quantize_onsets_simple` in `piano_video_scribe/quantization.py:14`.

`simple` does two things, both implicated:

1. **`_refine_bpm_for_grid` (`quantization.py:153`)** searches ±a small window around the nominal BPM to minimise RMS distance of *all* onsets to the 16th grid, then commits to one "effective BPM". For Billie Jean it shifted **117 → 116**. Over a long steady passage a 1-BPM grid error accumulates phase; once accumulated phase exceeds half a sixteenth, the *nearest-grid* rounding flips an onset to the adjacent sixteenth slot → the 480→240 jumps above. This is the "tempo shifting wrong" the user names. The refine step is optimising a global RMS that is happy to trade a little error on every note for the appearance of fit, instead of recognising the passage is exactly on an integer-BPM eighth grid.

2. **Independent per-onset nearest-grid snapping.** `simple` snaps each onset to the nearest grid slot in isolation. There is **no constraint that equal real intervals map to equal quantized intervals**, so identical input patterns are NOT guaranteed identical output. Jitter of a few frames (≈33 ms/frame at 30 fps, and this video is only 360p — worse onset jitter) is enough to push a borderline onset over a slot boundary.

### Why this keeps getting declared "fixed"
- Spot-checking pitch names and black-key ratios (what was done last time, 2026-06-11) does NOT look at inter-onset intervals. The bug is invisible to every check except "are the intervals of a known-steady passage all equal."
- The PLL quantizer (`quantize_onsets_pll`, line 232) has its own drift/phase-reset machinery (`alpha`, `alpha_period`, gap resets) that produces *different* but still-present misplacements — swapping simple↔pll moves the defects around rather than removing them.

### Measured bake-off (2026-06-11, the §2 intro gate, all with `--detector keys`)
```
simple   (default): {480: 60, 240: 3, 720: 1}   non-480 = 4/64   ← shipped output, worst
viterbi          : {480: 63, 720: 1}            non-480 = 1/64
pll              : {480: 64}                     non-480 = 0/64   ← CLEAN on this repro
adaptive         : {480: 61, 240: 3}            non-480 = 3/64
```
So **PLL passes the primary gate on Billie Jean and the default `simple` is the worst option.** This flatly contradicts the earlier assumption that "swapping simple↔pll just moves defects around" — on this repro PLL removes them. `viterbi` (docstring claims 100% vs 97% PLL on *its* test set) is nearly clean here (1/64, a single 720 = one missed/extra onset, possibly a detector miss not a quantizer error).

**Recommended direction (validate, then commit):**
1. The quick win is changing the default quantizer off `simple`. PLL (0/64) and viterbi (1/64) both beat it. Decide between them by running BOTH against the regression set, not just Billie Jean.
2. Before flipping the default: re-transcribe a genuinely syncopated song (Beat It — its off-beat chords are REAL, see history.md) with the new default and confirm the syncopation survives. `simple` was presumably chosen for some reason; find out why before removing it (git blame `pipeline.py:392`).
3. The 1 remaining viterbi defect (720 gap) and any residual: run §4 step 1 (dump pre-quantization onset seconds) to tell whether it's a detector onset miss vs a quantizer error — don't assume.
4. Add the regression test in §5 and wire whichever quantizer wins as the default in `pipeline.py:392`.

## 4. Investigation / fix plan (binary-search, guarantees progress)

1. **Confirm the layer.** The onset *seconds* feeding the quantizer are the dividing line. Dump the pre-quantization `sorted_onsets` (seconds) for the intro and compute their inter-onset intervals.
   - If the raw intervals are already steady (≈ constant seconds) → the bug is 100% in the quantizer. Fix there. (Expected outcome.)
   - If the raw intervals are themselves jittery beyond ±half a sixteenth → the detector onset timing is the problem; quantizer can't recover it. Different fix (detector/onset refinement).
   This single test tells you which half to work in. Do it first.

2. **Quantizer bake-off on the intro.** Run all four (`simple`, `pll`, `viterbi`, `adaptive`) on Billie Jean and print the §2 histogram for each. Pick the one (if any) that gives `{480: 64}`.

3. **If none is clean**, the missing capability is *pattern-aware* quantization: detect that a sub-sequence of onsets is periodic and force-equal its quantized intervals. Options to evaluate (decide in plan mode, don't guess):
   - a. Round the refined BPM to the nearest integer when the RMS-vs-integer-BPM penalty is within ε (kills the 117→116 drift at the source).
   - b. After quantizing, a "snap-to-self" pass: find runs of near-equal *real* intervals and reassign them the modal quantized interval.
   - c. Constrain `_refine_bpm_for_grid` to prefer the BPM that makes the *most common* interval land exactly on an 8th/16th, not the one with lowest global RMS.

4. **Re-measure after every change** with the §2 script. Progress = monotonic drop in non-480 count toward 0.

## 5. Definition of done (numeric — do not hand back without these)

- **Primary gate:** Billie Jean intro LH (first 64 intervals) histogram == `{480: 64}`. Zero exceptions.
- **Regression gate:** the existing test suite passes (`pytest`), AND a known-syncopated song (Beat It, see tasks.md) is re-transcribed and *still* shows its real syncopation (don't "fix" by flattening everything to the downbeat — see the Beat It note in history.md: that syncopation is real).
- **Verification protocol that actually catches this (the protocol that was missing):** for any transcription, pick the most obviously-repeating passage, extract its inter-onset intervals, and assert they are identical where the music is identical. Put this in the response as a histogram, not as "I spot-checked a few notes." A pitch/black-key check is necessary but NOT sufficient.
- Add a regression test fixture: Billie Jean intro onsets → assert all-480 after quantization.

## 6. Files
- `piano_video_scribe/quantization.py` — all quantizers; `quantize_onsets_simple` (default), `_refine_bpm_for_grid`, `quantize_onsets_pll`, `quantize_onsets_viterbi`.
- `piano_video_scribe/pipeline.py:392` — quantizer dispatch / default selection.
- `piano_video_scribe/config.py:140` — `--quantizer` CLI (default None → 'simple').
- Repro asset: `~/piano/songs/billie-jean-michael-jackson/{video.mp4,config.json}` (BPM 117, key F#m, `--detector keys`).
