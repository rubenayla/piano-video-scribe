---
description: "Autonomous improvement loop for the falling blocks detector. Iteratively modifies detect_falling_blocks.py, scores against test videos with known ground truth, keeps improvements, reverts regressions. Never stops."
---

# Autoresearch: Falling Blocks Detector

You are an autonomous research agent improving the falling blocks piano note detector.

## The Loop

Repeat forever:

1. **Read current score**: `python autoresearch-score.py 2>&1 | tail -5`
2. **Analyze**: What's the biggest gap? Which test has the lowest F1? What kind of errors (missed notes, extra notes, wrong pitch, wrong timing)?
3. **Hypothesize**: Think of ONE small, targeted change to improve the score.
4. **Implement**: Modify `detect_falling_blocks.py` (ONLY this file).
5. **Commit**: `git add detect_falling_blocks.py && git commit -m "experiment: <description>"`
6. **Score**: `python autoresearch-score.py 2>&1 | tail -5`
7. **Decide**:
   - If score IMPROVED: **keep**. Log to `autoresearch-results.tsv`. Continue.
   - If score EQUAL or WORSE: **revert**. `git reset --hard HEAD~1`. Log as "discard". Continue.
8. **NEVER STOP**. Go back to step 1.

## Scoring

The score is the aggregate F1 from `autoresearch-score.py`, which:
- Runs the falling blocks pipeline on test videos with known-correct ground truth
- Compares output against GT using pitch class + timing matching
- Returns a single number 0-100 (higher = better)

## Rules

- Only modify `detect_falling_blocks.py`. Nothing else.
- Every experiment gets ONE commit. Small, focused changes.
- If an experiment crashes, revert and try something different.
- Log every experiment to `autoresearch-results.tsv` (tab-separated: commit, score, status, description).
- Read the results log before each experiment to avoid repeating failed ideas.
- Removing code and getting equal results? Keep it (simpler is better).
- A 0.1% improvement that adds 50 lines of hacky code? Probably not worth it.
- Prefer robust, general improvements over video-specific hacks.

## Context

The falling blocks detector (`detect_falling_blocks.py`) detects piano notes from colored rectangles falling in tutorial videos. It:
1. Auto-detects block colors (HSV thresholding)
2. Builds pitch mapping (keyboard detector or block clustering)
3. Detects blocks via connected components per frame
4. Projects onset time from y-position + fall speed
5. Merges cross-frame observations into notes

Current issues to investigate:
- Chain-merge can over-merge or under-merge (tolerance tuning)
- Fall speed measurement accuracy
- Block detection filters (min area, aspect ratio)
- Color detection sensitivity
- Hand assignment accuracy
