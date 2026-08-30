import unittest

from q250_uzh.allocator import MotorAllocator
from q250_uzh.config import Q250
from q250_uzh.mixer import wrench_from_motor_omega


class TestMotorAllocator(unittest.TestCase):
    def setUp(self):
        self.alloc = MotorAllocator()
        self.hover_total = Q250.mass_kg * Q250.gravity_m_s2

    def test_hover_wrench_allocates_equal_hover_speed(self):
        result = self.alloc.allocate(self.hover_total, (0.0, 0.0, 0.0))
        for w in result.omega_cmd_rad_s:
            self.assertAlmostEqual(w, Q250.hover_omega_rad_s, places=6)
        self.assertFalse(result.saturated)

    def test_positive_roll_torque_increases_left_pair(self):
        result = self.alloc.allocate(self.hover_total, (0.05, 0.0, 0.0))
        w1, w2, w3, w4 = result.omega_cmd_rad_s
        self.assertGreater(w1, w2)
        self.assertGreater(w4, w3)

    def test_requested_unsaturated_wrench_is_reconstructed(self):
        requested_torque = (0.04, -0.03, 0.01)
        result = self.alloc.allocate(self.hover_total, requested_torque)
        force, torque = wrench_from_motor_omega(result.omega_cmd_rad_s)
        self.assertAlmostEqual(force[2], self.hover_total, places=6)
        for actual, requested in zip(torque, requested_torque, strict=True):
            self.assertAlmostEqual(actual, requested, places=6)

    def test_impossible_command_reports_saturation_and_nonnegative_motor_speed(self):
        result = self.alloc.allocate(self.hover_total, (10.0, 0.0, 0.0))
        self.assertTrue(result.saturated)
        self.assertTrue(all(w >= 0.0 for w in result.omega_cmd_rad_s))


if __name__ == "__main__":
    unittest.main()
