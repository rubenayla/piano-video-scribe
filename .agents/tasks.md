<!-- consult selectively — grep, never read in full -->
# Tasks

## TODO

- **Auto-detect hand start offset**: The Viterbi quantizer places each hand's first note at grid 0, losing the time gap between hands. Currently requires `--start-beat` to manually specify where the RH enters. A robust fix would auto-detect the gap and compute `target_start_grid` per hand. Challenge: determining the "correct" beat position from a raw time gap is ambiguous — the gap between LH and RH onsets (e.g. 3.9s) doesn't cleanly map to a beat boundary without knowing the measure structure (time sig, pickup bars, etc.). Possible approaches: snap the gap to the nearest whole-measure boundary given the time signature, or use onset clustering to detect phrase boundaries. Needs more thought.

- **falling-blocks-detector**: Detect notes from falling colored blocks (branch: falling-notes-detector). Prototype works but output is poor quality. Needs:
  - Better keyboard detection on dark videos (currently finds only 7 of ~36 keys)
  - Correct pitch mapping (C position wrong due to missing black keys)
  - BPM detection from block speed (currently hardcoded)
  - Reference videos: https://youtu.be/kvSNAgwa-O8 (Hedwig's Theme), https://youtu.be/NPBCbTZWnq0 (Rousseau)
  - Hard test: https://youtu.be/e8jN1-HC-Vc (complex, many effects)

## In Progress

## Done
