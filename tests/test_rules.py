"""Tests for the Python port of the Connections rules engine.

Parity reference: tests/game-rules.test.js ("Making a guess").
"""

from __future__ import annotations

import unittest

from connections_gym.rules import (
    MAX_MISTAKES,
    evaluate_guess,
    game_status,
    remaining_terms,
)

GROUPS = {
    "easy": ["A1", "A2", "A3", "A4"],
    "medium": ["B1", "B2", "B3", "B4"],
    "hard": ["C1", "C2", "C3", "C4"],
    "harder": ["D1", "D2", "D3", "D4"],
}
BOARD = [m for members in GROUPS.values() for m in members]


class EvaluateGuessTest(unittest.TestCase):
    def test_correct_group_returns_matched_key(self):
        self.assertEqual(evaluate_guess(GROUPS, ["A1", "A2", "A3", "A4"]), (True, "easy"))

    def test_correct_ignores_order(self):
        self.assertEqual(evaluate_guess(GROUPS, ["C4", "C1", "C3", "C2"]), (True, "hard"))

    def test_mixed_terms_incorrect(self):
        self.assertEqual(evaluate_guess(GROUPS, ["A1", "A2", "A3", "B1"]), (False, None))

    def test_wrong_length_three_incorrect(self):
        self.assertEqual(evaluate_guess(GROUPS, ["A1", "A2", "A3"]), (False, None))

    def test_wrong_length_five_incorrect(self):
        self.assertEqual(evaluate_guess(GROUPS, ["A1", "A2", "A3", "A4", "B1"]), (False, None))

    def test_duplicates_incorrect(self):
        self.assertEqual(evaluate_guess(GROUPS, ["A1", "A1", "A2", "A3"]), (False, None))


class RemainingTermsTest(unittest.TestCase):
    def test_no_solved_returns_full_board(self):
        self.assertEqual(remaining_terms(BOARD, GROUPS, set()), BOARD)

    def test_solved_group_removed_preserving_order(self):
        self.assertEqual(
            remaining_terms(BOARD, GROUPS, {"medium"}),
            ["A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4"],
        )

    def test_all_solved_returns_empty(self):
        self.assertEqual(remaining_terms(BOARD, GROUPS, set(GROUPS)), [])


class GameStatusTest(unittest.TestCase):
    def test_all_solved_under_max_is_won(self):
        self.assertEqual(game_status(0, 4, 4), "won")

    def test_max_mistakes_is_lost(self):
        self.assertEqual(game_status(MAX_MISTAKES, 1, 4), "lost")

    def test_partial_is_playing(self):
        self.assertEqual(game_status(1, 2, 4), "playing")


if __name__ == "__main__":
    unittest.main()
