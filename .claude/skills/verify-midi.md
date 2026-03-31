---
description: Verify a MIDI file against a ground truth before presenting it to the user. MUST be called before sharing any .mid file path.
user-invocable: true
---

# verify-midi

Compare a pipeline output MIDI against a ground truth file, note-by-note.

## Usage

```
/verify-midi <output.mid> <ground_truth.mid> [measures]
```

- `output.mid`: the file to verify
- `ground_truth.mid`: the reference file
- `measures`: number of measures to compare (default: 8)

If no ground truth exists, dump the first 8 measures of both hands so the user can visually inspect.

## Steps

Run this Python script (adjust paths from arguments):

```python
import mido, sys

out_path = "$OUTPUT_MID"
gt_path = "$GROUND_TRUTH_MID"
measures = $MEASURES  # default 8

NOTE_NAMES = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
def nn(midi): return f'{NOTE_NAMES[midi % 12]}{midi // 12 - 1}'

def get_onsets(mid, ti):
    t = mid.tracks[ti]; tick = 0; tpb = mid.ticks_per_beat; notes = []
    for msg in t:
        tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append((round(tick / tpb * 4) / 4, msg.note))
    return notes

out = mido.MidiFile(out_path)
gt = mido.MidiFile(gt_path) if gt_path else None
max_beat = measures * 4

for hand, gt_ti, out_ti in [("RH", 1, 0), ("LH", 2, 1)]:
    out_notes = [(b, n) for b, n in get_onsets(out, out_ti) if b <= max_beat]

    if gt:
        gt_notes = [(b, n) for b, n in get_onsets(gt, gt_ti) if b <= max_beat]
        gt_set = set(gt_notes)
        out_set = set(out_notes)
        all_notes = sorted(gt_set | out_set)

        print(f"\n{'='*55}")
        print(f"{hand} m1-{measures} ({len(gt_set)} GT, {len(out_set)} out)")
        print(f"{'='*55}")
        print(f"{'beat':>6} {'m':>2} {'pos':>5} {'note':>5} {'status':>8}")
        for b, n in all_notes:
            m = int(b // 4) + 1
            in_gt = (b, n) in gt_set
            in_out = (b, n) in out_set
            status = "  OK" if (in_gt and in_out) else "  MISS" if in_gt else "  EXTRA"
            print(f"{b:6.2f} m{m} {b%4:5.2f} {nn(n):>5} {status}")

        matched = len(gt_set & out_set)
        missing = len(gt_set - out_set)
        extra = len(out_set - gt_set)
        print(f"\n  {matched} matched, {missing} missing, {extra} extra")
    else:
        print(f"\n{hand} m1-{measures} (no GT, showing output):")
        for b, n in out_notes:
            m = int(b // 4) + 1
            print(f"  m{m} beat {b%4:.2f}  {nn(n):>5}")
```

## After running

- If there are MISS or EXTRA notes, list them clearly and note as known issues
- Only then provide the file paths to the user
- If the output looks wrong (many missing/extra), do NOT present it — fix first
