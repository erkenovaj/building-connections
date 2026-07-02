"""Empirical board-difficulty ordering from win@k probe results.

``train/probe_wink.py --out-boards`` writes one JSONL record per board with
the model's observed win rate; this module turns that file into a seed list
ordered easiest-first. Curriculum stages train on the easiest slice of the
probed pool (board selection, not in-run ordering — the TRL sampler
shuffles dataset rows).
"""

from __future__ import annotations

import json


def load_seed_order(path: str) -> list[int]:
    """Return board seeds ordered easiest-first by probed win rate.

    Ties break by lower mean mistakes, then by seed for determinism.
    """
    with open(path) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    records.sort(key=lambda r: (-r["win_rate"], r["mean_mistakes"], r["seed"]))
    return [r["seed"] for r in records]
