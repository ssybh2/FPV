import math
import unittest

from q250_uzh.rate_controller import AxisPIDGains, BodyRatePID, BodyRatePIDConfig


class TestBodyRatePID(unittest.TestCase):
    def test_positive_roll_rate_error_commands_positive_roll_torque(self):
        cfg = BodyRatePIDConfig(
            roll=AxisPIDGains(kp=0.10, ki=0.0, kd=0.0),
            pitch=AxisPIDGains(kp=0.10, ki=0.0, kd=0.0),
            yaw=AxisPIDGains(kp=0.10, ki=0.0, kd=0.0),
            torque_limit_nm=(1.0, 1.0, 1.0),
        )
        ctrl = BodyRatePID(cfg)
        tau = ctrl.update((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), dt_s=0.01)
        self.assertGreater(tau[0], 0.0)
        self.assertAlmostEqual(tau[1], 0.0, places=12)
        self.assertAlmostEqual(tau[2], 0.0, places=12)

    def test_integrator_is_clamped(self):
        cfg = BodyRatePIDConfig(
            roll=AxisPIDGains(kp=0.0, ki=1.0, kd=0.0, integral_limit=0.2),
            pitch=AxisPIDGains(kp=0.0, ki=0.0, kd=0.0),
            yaw=AxisPIDGains(kp=0.0, ki=0.0, kd=0.0),
            torque_limit_nm=(1.0, 1.0, 1.0),
        )
        ctrl = BodyRatePID(cfg)
        for _ in range(100):
            ctrl.update((10.0, 0.0, 0.0), (0.0, 0.0, 0.0), dt_s=0.01)
        self.assertAlmostEqual(ctrl.integral[0], 0.2, places=12)

    def test_torque_output_is_saturated(self):
        cfg = BodyRatePIDConfig(
            roll=AxisPIDGains(kp=10.0, ki=0.0, kd=0.0),
            pitch=AxisPIDGains(kp=10.0, ki=0.0, kd=0.0),
            yaw=AxisPIDGains(kp=10.0, ki=0.0, kd=0.0),
            torque_limit_nm=(0.2, 0.3, 0.1),
        )
        ctrl = BodyRatePID(cfg)
        tau = ctrl.update((10.0, -10.0, 10.0), (0.0, 0.0, 0.0), dt_s=0.01)
        self.assertEqual(tau, (0.2, -0.3, 0.1))

    def test_default_controller_accepts_degrees_per_second_conversion(self):
        ctrl = BodyRatePID()
        cmd = math.radians(100.0)
        tau = ctrl.update((cmd, 0.0, 0.0), (0.0, 0.0, 0.0), dt_s=1.0 / 240.0)
        self.assertGreater(tau[0], 0.0)


if __name__ == "__main__":
    unittest.main()
