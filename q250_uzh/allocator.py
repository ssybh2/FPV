from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .config import Q250
from .motor_model import load_default_lut


@dataclass(frozen=True)
class AllocationResult:
    omega_cmd_rad_s: tuple[float, float, float, float]
    motor_thrust_n: tuple[float, float, float, float]
    saturated: bool


class MotorAllocator:
    """Closed-form X-quad allocator for the project's FLU/Z-up Q250 layout.

    It solves requested [Fz, tau_x, tau_y, tau_z] into individual motor thrusts,
    clips each motor to the measured operating envelope, then converts thrust to
    omega through T = Kt * omega^2.
    """

    def __init__(self, max_omega_rad_s: float | None = None):
        if max_omega_rad_s is None:
            max_omega_rad_s = load_default_lut()[-1].omega_rad_s
        if max_omega_rad_s <= 0.0:
            raise ValueError("max_omega_rad_s must be positive.")
        self.max_omega_rad_s = float(max_omega_rad_s)
        self.max_motor_thrust_n = Q250.kt_n_per_rad_s_sq * self.max_omega_rad_s**2

    def allocate(
        self,
        collective_thrust_n: float,
        body_torque_nm: Sequence[float],
    ) -> AllocationResult:
        if len(body_torque_nm) != 3:
            raise ValueError("body_torque_nm must contain roll, pitch, yaw torque.")

        f = max(0.0, float(collective_thrust_n))
        tx, ty, tz = (float(v) for v in body_torque_nm)
        a = Q250.arm_xy_m
        c = Q250.kq_nm_per_rad_s_sq / Q250.kt_n_per_rad_s_sq

        # Hadamard inverse of the Q250 X mixer, expressed directly in motor thrust.
        x = tx / a
        y = ty / a
        z = tz / c
        requested = (
            0.25 * (f + x - y - z),  # M1 front-left  CCW
            0.25 * (f - x - y + z),  # M2 front-right CW
            0.25 * (f - x + y - z),  # M3 rear-right CCW
            0.25 * (f + x + y + z),  # M4 rear-left   CW
        )

        clipped = tuple(_clamp(t, 0.0, self.max_motor_thrust_n) for t in requested)
        saturated = any(abs(a_ - b_) > 1e-12 for a_, b_ in zip(requested, clipped, strict=True))
        omega = tuple(
            math.sqrt(t / Q250.kt_n_per_rad_s_sq) if t > 0.0 else 0.0 for t in clipped
        )
        return AllocationResult(
            omega_cmd_rad_s=(omega[0], omega[1], omega[2], omega[3]),
            motor_thrust_n=(clipped[0], clipped[1], clipped[2], clipped[3]),
            saturated=saturated,
        )


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))
