"""Tests for train/probe_wink.py --out-boards incremental flushing."""

import json
import sys
from types import SimpleNamespace

import pytest

from train import probe_wink


def test_out_boards_flushed_per_board(tmp_path, monkeypatch):
    """Records for finished boards must be on disk even if a later board is interrupted."""
    out = tmp_path / "boards.jsonl"
    calls = {"n": 0}

    def fake_play_episode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 4:  # boards 1-2 done (2 episodes each), interrupt on board 3
            raise KeyboardInterrupt
        return SimpleNamespace(won=True, mistakes=1, tool_calls=3, truncations=0)

    monkeypatch.setattr(probe_wink, "load_policy", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(probe_wink, "_pick_device", lambda: "cpu")
    monkeypatch.setattr(probe_wink, "ConnectionsEnv", lambda **k: object())
    monkeypatch.setattr(probe_wink, "play_episode", fake_play_episode)
    monkeypatch.setattr(sys, "argv", [
        "probe_wink.py", "--num-boards", "8", "--num-episodes", "2",
        "--out-boards", str(out),
    ])

    with pytest.raises(KeyboardInterrupt):
        probe_wink.main()

    assert out.exists(), "out-boards file missing after interrupt"
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["seed"] == 1_000_000
    assert lines[0]["wins"] == 2
    assert lines[1]["seed"] == 1_000_001
