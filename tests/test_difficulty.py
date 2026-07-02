"""Tests for empirical difficulty ordering of board seeds."""

import json

from train.difficulty import load_seed_order


def _write(tmp_path, records):
    """Write records as one-per-line JSONL and return the path."""
    path = tmp_path / "boards.jsonl"
    with open(path, "w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    return str(path)


def test_orders_easiest_first(tmp_path):
    path = _write(tmp_path, [
        {"seed": 1, "episodes": 8, "wins": 0, "win_rate": 0.0, "mean_mistakes": 4.0},
        {"seed": 2, "episodes": 8, "wins": 6, "win_rate": 0.75, "mean_mistakes": 1.0},
        {"seed": 3, "episodes": 8, "wins": 2, "win_rate": 0.25, "mean_mistakes": 2.5},
    ])
    assert load_seed_order(path) == [2, 3, 1]


def test_ties_break_by_mistakes_then_seed(tmp_path):
    path = _write(tmp_path, [
        {"seed": 5, "episodes": 8, "wins": 4, "win_rate": 0.5, "mean_mistakes": 2.0},
        {"seed": 4, "episodes": 8, "wins": 4, "win_rate": 0.5, "mean_mistakes": 1.0},
        {"seed": 3, "episodes": 8, "wins": 4, "win_rate": 0.5, "mean_mistakes": 1.0},
    ])
    assert load_seed_order(path) == [3, 4, 5]
