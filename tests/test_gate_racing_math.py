import unittest

import torch

from q250_uzh.gate_racing_math import (
    GateRacingRewardCfg,
    advance_gate_indices,
    compute_gate_racing_reward,
    detect_gate_crossing,
    gate_curriculum,
    signed_gate_distance,
)


class TestGateRacingMath(unittest.TestCase):
    def test_gate_curriculum_progresses_from_large_single_to_three_gates(self):
        s0 = gate_curriculum(0)
        s1 = gate_curriculum(800)
        s2 = gate_curriculum(2400)
        self.assertEqual((s0.stage, s0.gate_count, s0.width_m, s0.height_m), (0, 1, 3.0, 3.0))
        self.assertEqual((s1.stage, s1.gate_count, s1.width_m, s1.height_m), (1, 1, 1.5, 1.5))
        self.assertEqual((s2.stage, s2.gate_count, s2.width_m, s2.height_m), (2, 3, 1.5, 1.5))

    def test_signed_gate_distance_is_negative_before_and_positive_after_plane(self):
        center = torch.tensor([[3.0, 0.0, 1.5]])
        normal = torch.tensor([[1.0, 0.0, 0.0]])
        before = signed_gate_distance(torch.tensor([[2.0, 0.0, 1.5]]), center, normal)
        after = signed_gate_distance(torch.tensor([[4.0, 0.0, 1.5]]), center, normal)
        self.assertTrue(torch.allclose(before, torch.tensor([-1.0])))
        self.assertTrue(torch.allclose(after, torch.tensor([1.0])))

    def test_gate_crossing_inside_opening_counts_as_pass(self):
        passed, missed, crossed = detect_gate_crossing(
            torch.tensor([-0.10]),
            torch.tensor([0.05]),
            torch.tensor([0.20]),
            torch.tensor([-0.30]),
            half_width_m=0.75,
            half_height_m=0.75,
        )
        self.assertTrue(bool(crossed.item()))
        self.assertTrue(bool(passed.item()))
        self.assertFalse(bool(missed.item()))

    def test_gate_crossing_outside_opening_counts_as_miss(self):
        passed, missed, crossed = detect_gate_crossing(
            torch.tensor([-0.10]),
            torch.tensor([0.05]),
            torch.tensor([0.90]),
            torch.tensor([0.00]),
            half_width_m=0.75,
            half_height_m=0.75,
        )
        self.assertTrue(bool(crossed.item()))
        self.assertFalse(bool(passed.item()))
        self.assertTrue(bool(missed.item()))

    def test_no_plane_crossing_is_neither_pass_nor_miss(self):
        passed, missed, crossed = detect_gate_crossing(
            torch.tensor([-0.50]),
            torch.tensor([-0.20]),
            torch.tensor([0.00]),
            torch.tensor([0.00]),
            half_width_m=0.75,
            half_height_m=0.75,
        )
        self.assertFalse(bool(crossed.item()))
        self.assertFalse(bool(passed.item()))
        self.assertFalse(bool(missed.item()))

    def test_advance_gate_indices_finishes_only_after_last_gate(self):
        idx = torch.tensor([0, 0, 1, 2], dtype=torch.long)
        counts = torch.tensor([1, 3, 3, 3], dtype=torch.long)
        passed = torch.tensor([True, True, True, True])
        new_idx, finished = advance_gate_indices(idx, counts, passed)
        self.assertTrue(torch.equal(new_idx, torch.tensor([0, 1, 2, 2])))
        self.assertTrue(torch.equal(finished, torch.tensor([True, False, False, True])))

    def test_gate_reward_rewards_progress_and_pass_and_penalizes_crash(self):
        cfg = GateRacingRewardCfg()
        reward, parts = compute_gate_racing_reward(
            gate_progress_m=torch.tensor([0.10, -0.10]),
            gate_passed=torch.tensor([True, False]),
            race_finished=torch.tensor([False, False]),
            crashed=torch.tensor([False, True]),
            actions=torch.zeros((2, 4)),
            cfg=cfg,
        )
        self.assertGreater(float(parts["progress"][0]), 0.0)
        self.assertAlmostEqual(float(parts["gate"][0]), cfg.gate_bonus, places=6)
        self.assertAlmostEqual(float(parts["crash"][1]), cfg.crash_penalty, places=6)
        self.assertGreater(float(reward[0]), float(reward[1]))

    def test_finish_bonus_is_larger_than_single_gate_bonus(self):
        cfg = GateRacingRewardCfg()
        reward, parts = compute_gate_racing_reward(
            gate_progress_m=torch.tensor([0.0]),
            gate_passed=torch.tensor([True]),
            race_finished=torch.tensor([True]),
            crashed=torch.tensor([False]),
            actions=torch.zeros((1, 4)),
            cfg=cfg,
        )
        self.assertGreater(float(parts["finish"].item()), float(parts["gate"].item()))
        self.assertGreater(float(reward.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
