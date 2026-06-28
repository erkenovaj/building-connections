"""Tests for the Gymnasium-style ConnectionsEnv (full-episode driver)."""

from __future__ import annotations

import unittest

from connections_gym.env import ConnectionsEnv

LIGHT_SAMPLING = {
    "max_attempts": 20,
    "per_solve_timeout_seconds": 0.5,
    "total_timeout_seconds": 5.0,
}


def make_env():
    return ConnectionsEnv(sampling_params=LIGHT_SAMPLING)


class ResetTest(unittest.TestCase):
    def test_reset_is_reproducible(self):
        a = make_env().reset(7)
        b = make_env().reset(7)
        self.assertEqual(a["remaining"], b["remaining"])

    def test_different_seed_differs(self):
        a = make_env().reset(7)
        b = make_env().reset(8)
        self.assertNotEqual(a["remaining"], b["remaining"])

    def test_fresh_obs_shape(self):
        obs = make_env().reset(7)
        self.assertEqual(len(obs["remaining"]), 16)
        self.assertEqual(obs["num_categories"], 4)
        self.assertEqual(obs["history"], [])


class StepTest(unittest.TestCase):
    def test_correct_guess_removes_group(self):
        env = make_env()
        env.reset(7)
        key, members = next(iter(env.groups.items()))
        obs, reward, done, info = env.step(members)
        self.assertAlmostEqual(reward, 1.0)
        self.assertFalse(done)
        self.assertEqual(info["status"], "playing")
        for term in members:
            self.assertNotIn(term, obs["remaining"])
        self.assertEqual(len(obs["remaining"]), 12)

    def test_wrong_guess_tracks_mistake_and_history(self):
        env = make_env()
        env.reset(7)
        spread = [members[0] for members in env.groups.values()]
        obs, reward, done, info = env.step(spread)
        self.assertAlmostEqual(reward, -0.5)
        self.assertEqual(info["mistakes"], 1)
        self.assertEqual(len(obs["history"]), 1)
        self.assertFalse(done)

    def test_win_terminates_with_bonus(self):
        env = make_env()
        env.reset(7)
        rewards = []
        for members in list(env.groups.values()):
            _, reward, done, info = env.step(members)
            rewards.append(reward)
        self.assertTrue(done)
        self.assertEqual(info["status"], "won")
        self.assertAlmostEqual(rewards[-1], 3.0)  # 1.0 exact + 2.0 win bonus

    def test_four_mistakes_loses(self):
        env = make_env()
        env.reset(7)
        spread = [members[0] for members in env.groups.values()]
        done = False
        for _ in range(4):
            _, _, done, info = env.step(spread)
        self.assertTrue(done)
        self.assertEqual(info["status"], "lost")
        self.assertEqual(info["mistakes"], 4)


if __name__ == "__main__":
    unittest.main()
