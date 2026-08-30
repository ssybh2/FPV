import math
import unittest

from q250_uzh.motor_model import MotorModel, load_default_lut


class TestMotorModel(unittest.TestCase):
    def test_lut_hits_measured_point(self):
        lut = load_default_lut()
        model = MotorModel(lut=lut, tau_s=0.10)
        rpm = model.pwm_to_rpm(1400.0)
        self.assertAlmostEqual(rpm, 12908.033333333333, places=6)

    def test_lut_interpolates_linearly(self):
        lut = load_default_lut()
        model = MotorModel(lut=lut, tau_s=0.10)
        r0 = model.pwm_to_rpm(1400.0)
        r1 = model.pwm_to_rpm(1450.0)
        mid = model.pwm_to_rpm(1425.0)
        self.assertAlmostEqual(mid, 0.5 * (r0 + r1), places=9)

    def test_motor_lag_matches_exact_first_order_solution(self):
        lut = load_default_lut()
        model = MotorModel(lut=lut, tau_s=0.10)
        omega0 = 0.0
        omega_cmd = 1000.0
        dt = 0.01
        omega1 = model.first_order_step(omega0, omega_cmd, dt)
        expected = omega_cmd * (1.0 - math.exp(-dt / 0.10))
        self.assertAlmostEqual(omega1, expected, places=12)

    def test_thrust_and_reaction_torque_use_identified_coefficients(self):
        lut = load_default_lut()
        model = MotorModel(lut=lut, tau_s=0.10)
        omega = 1400.0
        self.assertAlmostEqual(model.thrust_from_omega(omega), 1.3287717252618608e-6 * omega**2, places=12)
        self.assertAlmostEqual(model.torque_from_omega(omega), 1.772957417327994e-8 * omega**2, places=12)


if __name__ == "__main__":
    unittest.main()
