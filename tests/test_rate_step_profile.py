import math
import unittest

from q250_uzh.rate_step_profile import RateStepProfile


class TestRateStepProfile(unittest.TestCase):
    def test_roll_pulse_is_zero_then_positive_then_zero(self):
        profile = RateStepProfile(axis="roll", rate_rad_s=math.radians(100.0), start_s=0.5, end_s=0.9)
        self.assertEqual(profile.command_at(0.2), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(profile.command_at(0.7)[0], math.radians(100.0), places=12)
        self.assertEqual(profile.command_at(1.1), (0.0, 0.0, 0.0))

    def test_pitch_and_yaw_map_to_correct_axis(self):
        pitch = RateStepProfile(axis="pitch", rate_rad_s=1.0, start_s=0.0, end_s=1.0)
        yaw = RateStepProfile(axis="yaw", rate_rad_s=-2.0, start_s=0.0, end_s=1.0)
        self.assertEqual(pitch.command_at(0.5), (0.0, 1.0, 0.0))
        self.assertEqual(yaw.command_at(0.5), (0.0, 0.0, -2.0))

    def test_invalid_axis_rejected(self):
        with self.assertRaises(ValueError):
            RateStepProfile(axis="bad", rate_rad_s=1.0)


if __name__ == "__main__":
    unittest.main()
