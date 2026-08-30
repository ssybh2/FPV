import unittest

from q250_uzh.config import Q250
from q250_uzh.mixer import wrench_from_motor_omega


class TestMixer(unittest.TestCase):
    def test_equal_motor_speed_has_zero_body_moment(self):
        w = (Q250.hover_omega_rad_s,) * 4
        force, moment = wrench_from_motor_omega(w)
        self.assertAlmostEqual(force[0], 0.0, places=12)
        self.assertAlmostEqual(force[1], 0.0, places=12)
        self.assertAlmostEqual(force[2], Q250.mass_kg * Q250.gravity_m_s2, places=6)
        self.assertAlmostEqual(moment[0], 0.0, places=10)
        self.assertAlmostEqual(moment[1], 0.0, places=10)
        self.assertAlmostEqual(moment[2], 0.0, places=10)

    def test_left_pair_more_thrust_gives_positive_roll_in_flu(self):
        base = Q250.hover_omega_rad_s
        w = (base * 1.02, base, base, base * 1.02)  # M1 + M4 are left
        _, moment = wrench_from_motor_omega(w)
        self.assertGreater(moment[0], 0.0)

    def test_front_pair_more_thrust_gives_negative_pitch_in_flu(self):
        base = Q250.hover_omega_rad_s
        w = (base * 1.02, base * 1.02, base, base)  # M1 + M2 are front
        _, moment = wrench_from_motor_omega(w)
        self.assertLess(moment[1], 0.0)

    def test_cw_pair_more_thrust_gives_positive_yaw_reaction(self):
        base = Q250.hover_omega_rad_s
        w = (base, base * 1.02, base, base * 1.02)  # M2 + M4 are CW
        _, moment = wrench_from_motor_omega(w)
        self.assertGreater(moment[2], 0.0)


if __name__ == "__main__":
    unittest.main()
