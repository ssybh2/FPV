import unittest

import torch

from q250_uzh.fly_to_point_math import FlyToPointRewardCfg, compute_fly_to_point_reward, curriculum_bounds


class TestFlyToPointMath(unittest.TestCase):
    def test_progress_toward_target_is_positive(self):
        cfg = FlyToPointRewardCfg(progress_scale=5.0, action_penalty_scale=0.0)
        total, parts = compute_fly_to_point_reward(
            torch.tensor([1.0]), torch.tensor([1.2]), torch.tensor([False]), torch.tensor([False]), torch.zeros((1, 4)), cfg
        )
        self.assertGreater(float(parts["progress"][0]), 0.0)
        self.assertGreater(float(total[0]), 0.0)

    def test_moving_away_is_negative(self):
        cfg = FlyToPointRewardCfg(progress_scale=5.0, action_penalty_scale=0.0)
        total, parts = compute_fly_to_point_reward(
            torch.tensor([1.3]), torch.tensor([1.0]), torch.tensor([False]), torch.tensor([False]), torch.zeros((1, 4)), cfg
        )
        self.assertLess(float(parts["progress"][0]), 0.0)
        self.assertLess(float(total[0]), 0.0)

    def test_success_and_crash_bonuses_apply(self):
        cfg = FlyToPointRewardCfg(success_bonus=10.0, crash_penalty=-10.0, progress_scale=0.0, action_penalty_scale=0.0)
        total, parts = compute_fly_to_point_reward(
            torch.tensor([0.1, 2.0]), torch.tensor([0.1, 2.0]), torch.tensor([True, False]), torch.tensor([False, True]), torch.zeros((2, 4)), cfg
        )
        self.assertEqual(float(parts["success"][0]), 10.0)
        self.assertEqual(float(parts["crash"][1]), -10.0)
        self.assertEqual(float(total[0]), 10.0)
        self.assertEqual(float(total[1]), -10.0)

    def test_action_penalty_discourages_large_actions(self):
        cfg = FlyToPointRewardCfg(progress_scale=0.0, action_penalty_scale=0.1)
        total, parts = compute_fly_to_point_reward(
            torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([False]), torch.tensor([False]), torch.ones((1, 4)), cfg
        )
        self.assertLess(float(parts["action"][0]), 0.0)
        self.assertLess(float(total[0]), 0.0)

    def test_curriculum_has_three_expanding_stages(self):
        s0 = curriculum_bounds(0); s1 = curriculum_bounds(1000); s2 = curriculum_bounds(3000)
        self.assertEqual((s0.stage, s1.stage, s2.stage), (0, 1, 2))
        self.assertLess(s0.xy_extent_m, s1.xy_extent_m); self.assertLess(s1.xy_extent_m, s2.xy_extent_m)
        self.assertLess(s0.z_max_m, s1.z_max_m); self.assertLess(s1.z_max_m, s2.z_max_m)


if __name__ == "__main__":
    unittest.main()
