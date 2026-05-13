"""Contract tests for the CP-SAT Connections sampler."""

from __future__ import annotations

import json
from io import BytesIO
import os
import unittest

from api.game.sample import handler as sample_handler
from connections_sampler import (
    GameMode,
    SamplingParameters,
    build_config_index,
    completed_categories,
    sample_basic_game,
    sample_game,
    verify_unique_solution,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CATEGORY_TEMPLATES_PATH = os.path.join(
    REPO_ROOT, "configs", "category-templates-new.json"
)
OVERLAP_TEMPLATES_PATH = os.path.join(
    REPO_ROOT, "configs", "category-templates-test.json"
)


def _load(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _parameters(seed):
    return SamplingParameters(
        seed=seed,
        max_attempts=20,
        per_solve_timeout_seconds=0.5,
        total_timeout_seconds=5.0,
    )


class SamplerAssertions(unittest.TestCase):
    def assert_valid_game(self, result, *, categories, items_per_category=4):
        self.assertTrue(result.ok, result.to_dict())
        game = result.game
        self.assertIsNotNone(game)
        self.assertEqual(len(game.groups), categories)
        expected_board_size = (
            16 if categories == 3 else categories * items_per_category
        )
        self.assertEqual(len(game.board), expected_board_size)
        self.assertEqual(len(set(game.board)), len(game.board))
        group_members = []
        for group in game.groups:
            self.assertEqual(len(group.members), items_per_category)
            group_members.extend(group.members)
        self.assertEqual(len(group_members), len(set(group_members)))
        self.assertTrue(set(group_members).issubset(game.board))


class TestModeContracts(SamplerAssertions):
    @classmethod
    def setUpClass(cls):
        cls.config = _load(CATEGORY_TEMPLATES_PATH)
        cls.overlap_config = _load(OVERLAP_TEMPLATES_PATH)

    def test_easy_uses_globally_disjoint_category_templates(self):
        result = sample_game(
            self.config, mode=GameMode.EASY, parameters=_parameters(11)
        )
        self.assert_valid_game(result, categories=4)
        index = build_config_index(self.config)
        selected = [
            set(index.category_to_members[group.name])
            for group in result.game.groups
        ]
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1 :]:
                self.assertTrue(left.isdisjoint(right))

    def test_basic_supports_intersecting_category_templates(self):
        config = [
            {"name": "shared", "tags": ["A", "B"]},
            {"name": "a1", "tags": ["A"]},
            {"name": "a2", "tags": ["A"]},
            {"name": "b1", "tags": ["B"]},
            {"name": "b2", "tags": ["B"]},
        ]
        result = sample_basic_game(
            config,
            num_categories=2,
            items_per_category=2,
            parameters=_parameters(12),
        )
        self.assert_valid_game(result, categories=2, items_per_category=2)
        self.assertEqual({group.name for group in result.game.groups}, {"A", "B"})
        self.assertNotIn("shared", result.game.board)

    def test_basic_overlap_heavy_config_has_one_full_partition(self):
        result = sample_game(
            self.overlap_config,
            mode=GameMode.BASIC,
            parameters=_parameters(13),
        )
        self.assert_valid_game(result, categories=4)
        index = build_config_index(self.overlap_config)
        report = verify_unique_solution(
            index,
            board=result.game.board,
            intended_groups=[
                (group.name, group.members) for group in result.game.groups
            ],
            num_categories=4,
            items_per_category=4,
            mode=GameMode.BASIC,
        )
        self.assertTrue(report.is_unique, report)

    def test_advanced_all_non_solution_categories_are_incomplete(self):
        result = sample_game(
            self.overlap_config,
            mode=GameMode.ADVANCED,
            parameters=_parameters(14),
        )
        self.assert_valid_game(result, categories=3)
        index = build_config_index(self.overlap_config)
        complete = completed_categories(index, result.game.board, 4)
        self.assertEqual(
            set(complete),
            {group.name for group in result.game.groups},
        )
        solution_members = {
            member for group in result.game.groups for member in group.members
        }
        self.assertEqual(len(set(result.game.board) - solution_members), 4)

    def test_same_seed_is_reproducible(self):
        first = sample_game(
            self.overlap_config,
            mode=GameMode.BASIC,
            parameters=_parameters("room-123-round-1"),
        )
        second = sample_game(
            self.overlap_config,
            mode=GameMode.BASIC,
            parameters=_parameters("room-123-round-1"),
        )
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_ambiguous_config_is_not_returned(self):
        config = [
            {"name": "x1", "tags": ["A", "B"]},
            {"name": "x2", "tags": ["A", "B"]},
        ]
        result = sample_basic_game(
            config,
            num_categories=1,
            items_per_category=2,
            parameters=_parameters(15),
        )
        self.assertFalse(result.ok)
        self.assertGreaterEqual(
            result.diagnostics.get("ambiguousCandidatesRejected", 0),
            1,
        )

    def test_invalid_dimensions_return_a_structured_failure(self):
        result = sample_basic_game(
            self.config,
            num_categories=0,
            items_per_category=4,
        )
        self.assertEqual(result.status, "invalid_config")
        self.assertIsNone(result.game)


class _FakeHttpRequest:
    def __init__(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.headers = {"content-length": str(len(body))}
        self.rfile = BytesIO(body)
        self.wfile = BytesIO()
        self.status = None
        self.response_headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.response_headers[name] = value

    def end_headers(self):
        pass


class TestSamplerHttpApi(unittest.TestCase):
    def request(self, payload):
        request = _FakeHttpRequest(payload)
        sample_handler.do_POST(request)
        return request.status, json.loads(request.wfile.getvalue())

    def test_advanced_endpoint(self):
        status, payload = self.request(
            {"mode": "advanced", "seed": "api-test"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["game"]["numCategories"], 3)
        self.assertEqual(len(payload["game"]["board"]), 16)

    def test_generic_basic_endpoint(self):
        status, payload = self.request(
            {
                "mode": "basic",
                "seed": "api-generic",
                "numCategories": 2,
                "itemsPerCategory": 3,
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["game"]["numCategories"], 2)
        self.assertEqual(payload["game"]["itemsPerCategory"], 3)
        self.assertEqual(len(payload["game"]["board"]), 6)


if __name__ == "__main__":
    unittest.main()
