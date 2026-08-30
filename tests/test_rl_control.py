import math
import unittest

import torch

from q250_uzh.config import Q250
from q250_uzh.rl_control import CTBRActionCfg, TorchBodyRatePID, TorchMotorAllocator, map_actions_to_ctbr


class TestRLControl(unittest.TestCase):
    def test_zero_action_maps_to_hover_and_zero_rates(self):
        actions = torch.zeros((2, 4), dtype=torch.float32)
        thrust, rates = map_actions_to_ctbr(actions, CTBRActionCfg())
        expected = Q250.mass_kg * Q250.gravity_m_s2
        self.assertTrue(torch.allclose(thrust, torch.full((2,), expected), atol=1e-6))
        self.assertTrue(torch.allclose(rates, torch.zeros((2, 3)), atol=1e-7))

    def test_collective_action_has_asymmetric_hover_centered_range(self):
        actions = torch.tensor([[-1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float32)
        thrust, _ = map_actions_to_ctbr(actions, CTBRActionCfg())
        weight = Q250.mass_kg * Q250.gravity_m_s2
        self.assertTrue(math.isclose(float(thrust[0]), 0.30 * weight, rel_tol=1e-6))
        self.assertTrue(math.isclose(float(thrust[1]), 2.50 * weight, rel_tol=1e-6))

    def test_rate_actions_map_to_expected_limits(self):
        actions = torch.tensor([[0.0, 1.0, -1.0, 1.0]], dtype=torch.float32)
        _, rates = map_actions_to_ctbr(actions, CTBRActionCfg())
        deg = torch.rad2deg(rates[0])
        self.assertTrue(torch.allclose(deg, torch.tensor([200.0, -200.0, 100.0]), atol=1e-5))

    def test_vectorized_pid_produces_positive_roll_torque_for_positive_error(self):
        pid = TorchBodyRatePID(num_envs=3, device="cpu")
        cmd = torch.zeros((3, 3)); cmd[:, 0] = 1.0
        measured = torch.zeros((3, 3))
        torque = pid.update(cmd, measured, 1.0 / 240.0)
        self.assertTrue(torch.all(torque[:, 0] > 0.0))
        self.assertTrue(torch.allclose(torque[:, 1:], torch.zeros((3, 2)), atol=1e-8))

    def test_vectorized_pid_partial_reset_clears_selected_state(self):
        pid = TorchBodyRatePID(num_envs=2, device="cpu")
        pid.update(torch.ones((2, 3)), torch.zeros((2, 3)), 0.01)
        self.assertTrue(torch.any(pid.integral != 0.0))
        pid.reset(torch.tensor([1]))
        self.assertTrue(torch.all(pid.integral[1] == 0.0))
        self.assertTrue(torch.any(pid.integral[0] != 0.0))
        self.assertFalse(bool(pid.has_previous[1]))

    def test_allocator_hover_is_equal_on_all_motors(self):
        allocator = TorchMotorAllocator(device="cpu")
        thrust = torch.tensor([Q250.mass_kg * Q250.gravity_m_s2])
        omega, saturated = allocator.allocate(thrust, torch.zeros((1, 3)))
        self.assertFalse(bool(saturated[0]))
        self.assertTrue(torch.allclose(omega[0], torch.full((4,), Q250.hover_omega_rad_s), rtol=1e-5, atol=1e-4))

    def test_allocator_positive_roll_increases_left_motors(self):
        allocator = TorchMotorAllocator(device="cpu")
        thrust = torch.tensor([Q250.mass_kg * Q250.gravity_m_s2])
        omega, _ = allocator.allocate(thrust, torch.tensor([[0.05, 0.0, 0.0]]))
        self.assertGreater(float(omega[0, 0]), float(omega[0, 1]))
        self.assertGreater(float(omega[0, 3]), float(omega[0, 2]))

    def test_allocator_reports_saturation_for_impossible_command(self):
        allocator = TorchMotorAllocator(device="cpu")
        thrust = torch.tensor([10.0 * Q250.mass_kg * Q250.gravity_m_s2])
        _, saturated = allocator.allocate(thrust, torch.zeros((1, 3)))
        self.assertTrue(bool(saturated[0]))


if __name__ == "__main__":
    unittest.main()
