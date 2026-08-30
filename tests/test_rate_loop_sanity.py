import math
import unittest

from q250_uzh.allocator import MotorAllocator
from q250_uzh.config import Q250
from q250_uzh.mixer import wrench_from_motor_omega
from q250_uzh.motor_model import default_motor_model
from q250_uzh.rate_controller import BodyRatePID
from q250_uzh.rate_step_profile import RateStepProfile


class TestRateLoopSanity(unittest.TestCase):
    def test_default_roll_loop_tracks_and_stops_in_simple_rigid_body_model(self):
        dt = 1.0 / 240.0
        profile = RateStepProfile("roll", math.radians(100.0), 0.5, 0.9)
        controller = BodyRatePID()
        allocator = MotorAllocator()
        motor = default_motor_model()
        omega = [Q250.hover_omega_rad_s] * 4
        rates = [0.0, 0.0, 0.0]
        peak_roll = 0.0

        for step in range(int(2.0 / dt)):
            t = step * dt
            cmd = profile.command_at(t)
            torque_cmd = controller.update(cmd, rates, dt)
            allocation = allocator.allocate(Q250.mass_kg * Q250.gravity_m_s2, torque_cmd)
            omega = [
                motor.first_order_step(w, wc, dt)
                for w, wc in zip(omega, allocation.omega_cmd_rad_s, strict=True)
            ]
            _, torque = wrench_from_motor_omega(omega)
            rates[0] += torque[0] / Q250.inertia_x_kg_m2 * dt
            rates[1] += torque[1] / Q250.inertia_y_kg_m2 * dt
            rates[2] += torque[2] / Q250.inertia_z_kg_m2 * dt
            peak_roll = max(peak_roll, rates[0])

        self.assertGreater(peak_roll, math.radians(65.0))
        self.assertLess(abs(rates[0]), math.radians(8.0))
        self.assertLess(abs(rates[1]), math.radians(1.0))
        self.assertLess(abs(rates[2]), math.radians(1.0))


if __name__ == "__main__":
    unittest.main()
