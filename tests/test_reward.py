"""Tests for the per-guess reward function (plan §3 reward table)."""

from __future__ import annotations

import unittest

from connections_gym.reward import reward

GROUPS = {
    "easy": ["A1", "A2", "A3", "A4"],
    "medium": ["B1", "B2", "B3", "B4"],
    "hard": ["C1", "C2", "C3", "C4"],
    "harder": ["D1", "D2", "D3", "D4"],
}


class RewardTableTest(unittest.TestCase):
    def test_exact_group_k4(self):
        r, info = reward(["A1", "A2", "A3", "A4"], GROUPS, set())
        self.assertAlmostEqual(r, 1.0)
        self.assertEqual(info["k"], 4)
        self.assertEqual(info["matched_group_key"], "easy")

    def test_one_away_k3(self):
        r, info = reward(["A1", "A2", "A3", "B1"], GROUPS, set())
        self.assertAlmostEqual(r, 0.25)
        self.assertEqual(info["k"], 3)
        self.assertIsNone(info["matched_group_key"])

    def test_k2(self):
        r, info = reward(["A1", "A2", "B1", "B2"], GROUPS, set())
        self.assertAlmostEqual(r, 0.10)
        self.assertEqual(info["k"], 2)

    def test_k1_spread(self):
        r, info = reward(["A1", "B1", "C1", "D1"], GROUPS, set())
        self.assertAlmostEqual(r, -0.5)
        self.assertEqual(info["k"], 1)

    def test_info_carries_ground_truth(self):
        _, info = reward(["A1", "A2", "A3", "A4"], GROUPS, set())
        self.assertEqual(info["ground_truth_groups"], GROUPS)


class RewardInvalidTest(unittest.TestCase):
    def test_wrong_length_invalid(self):
        r, info = reward(["A1", "A2", "A3"], GROUPS, set())
        self.assertAlmostEqual(r, -1.0)
        self.assertIsNone(info["k"])

    def test_duplicate_invalid(self):
        r, info = reward(["A1", "A1", "A2", "A3"], GROUPS, set())
        self.assertAlmostEqual(r, -1.0)
        self.assertIsNone(info["k"])

    def test_term_not_remaining_invalid(self):
        r, _ = reward(["A1", "A2", "A3", "ZZ"], GROUPS, set())
        self.assertAlmostEqual(r, -1.0)


class RewardUnsolvedOnlyTest(unittest.TestCase):
    def test_solved_group_terms_are_invalid(self):
        r, info = reward(["A1", "A2", "A3", "A4"], GROUPS, {"easy"})
        self.assertAlmostEqual(r, -1.0)
        self.assertIsNone(info["k"])

    def test_overlap_counts_only_unsolved(self):
        r, info = reward(["B1", "B2", "B3", "C1"], GROUPS, {"easy"})
        self.assertAlmostEqual(r, 0.25)
        self.assertEqual(info["k"], 3)


if __name__ == "__main__":
    unittest.main()
