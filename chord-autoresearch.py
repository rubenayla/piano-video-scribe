#!/usr/bin/env python3
"""Autoresearch: sweep config parameters, score each against validation set.

Pattern mirrors autoresearch-score.py: defines a grid of configs, runs
chord-score.py's aggregate_wcs(), prints ranked results, emits best config.

Usage:
    python chord-autoresearch.py
"""

import itertools
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# Import via importlib to handle hyphenated filename
import importlib.util
_spec = importlib.util.spec_from_file_location('chord_score', os.path.join(REPO, 'chord-score.py'))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
aggregate_wcs = _mod.aggregate_wcs

# ── parameter grid ────────────────────────────────────────────────────────────
# Each list is the values to try for that parameter.
# Total configs = product of all list lengths.

GRID = {
    'lh_weight':            [1.0],            # left-hand chroma weight (fixed: LH = harmony)
    'rh_weight':            [0.3, 0.5, 0.7],  # right-hand weight (lower = melody pollutes less)
    'bass_bonus':           [0.10, 0.20, 0.30],  # root-matches-bass bonus
    'chord_change_penalty': [1.5, 2.5, 4.0, 6.0, 10.0],  # Viterbi smoothing
}


def main():
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f'Running {total} configs...\n')
    results = []

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        try:
            score = aggregate_wcs(config)
        except Exception as e:
            print(f'  [{i+1}/{total}] ERROR: {e}')
            score = 0.0
        results.append((score, config))
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f'  [{i+1}/{total}]  best so far: {max(r[0] for r in results):.3f}')

    results.sort(key=lambda r: r[0], reverse=True)

    print('\n─── Top 10 configs ────────────────────────────────────────────')
    print(f'{"rank":>4}  {"thirds":>7}  config')
    for rank, (score, cfg) in enumerate(results[:10], 1):
        print(f'{rank:4d}  {score:7.3f}  {json.dumps(cfg)}')

    best_score, best_cfg = results[0]
    print(f'\n═══ WINNER  thirds={best_score:.3f} ═══')
    print(json.dumps(best_cfg, indent=2))
    print('\nPaste into lead_sheet.py _DEFAULT_CONFIG to apply.')


if __name__ == '__main__':
    main()
