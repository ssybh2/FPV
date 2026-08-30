from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class AxisPIDGains:
    """PID gains for one body-rate axis.

    Units when the command/measurement is rad/s and output is N*m:
      kp: N*m / (rad/s)
      ki: N*m / rad
      kd: N*m / (rad/s^2)
    """

    kp: float
    ki: float
    kd: float
    integral_limit: float = 0.6


@dataclass(frozen=True)
class BodyRatePIDConfig:
    """Initial Q250 body-rate gains for the identified inertia + provisional motor lag.

    These are conservative simulation starting gains, not gains measured on the real vehicle.
    They are intentionally kept in one config object so later tuning does not touch the
    controller implementation.
    """

    roll: AxisPIDGains = field(
        default_factory=lambda: AxisPIDGains(kp=0.075, ki=0.080, kd=0.0060, integral_limit=0.50)
    )
    pitch: AxisPIDGains = field(
        default_factory=lambda: AxisPIDGains(kp=0.070, ki=0.080, kd=0.0056, integral_limit=0.50)
    )
    yaw: AxisPIDGains = field(
        default_factory=lambda: AxisPIDGains(kp=0.105, ki=0.040, kd=0.0085, integral_limit=0.35)
    )
    torque_limit_nm: tuple[float, float, float] = (0.30, 0.30, 0.08)


class BodyRatePID:
    """Three-axis body-rate PID with derivative-on-measurement and anti-windup.

    Inputs and outputs use the project's FLU/Z-up convention:
      rate = (p, q, r) in rad/s
      torque = (tau_x, tau_y, tau_z) in N*m
    """

    def __init__(self, config: BodyRatePIDConfig | None = None):
        self.config = config or BodyRatePIDConfig()
        self.integral = [0.0, 0.0, 0.0]
        self._previous_measurement: list[float] | None = None

    def reset(self) -> None:
        self.integral[:] = [0.0, 0.0, 0.0]
        self._previous_measurement = None

    def update(
        self,
        rate_cmd_rad_s: Sequence[float],
        rate_measured_rad_s: Sequence[float],
        dt_s: float,
    ) -> tuple[float, float, float]:
        if len(rate_cmd_rad_s) != 3 or len(rate_measured_rad_s) != 3:
            raise ValueError("Body-rate command and measurement must each have three axes.")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")

        cmd = [float(v) for v in rate_cmd_rad_s]
        measured = [float(v) for v in rate_measured_rad_s]
        gains = (self.config.roll, self.config.pitch, self.config.yaw)
        output: list[float] = []

        for axis, gain in enumerate(gains):
            error = cmd[axis] - measured[axis]
            self.integral[axis] += error * dt_s
            self.integral[axis] = _clamp(
                self.integral[axis], -gain.integral_limit, gain.integral_limit
            )

            if self._previous_measurement is None:
                measurement_derivative = 0.0
            else:
                measurement_derivative = (
                    measured[axis] - self._previous_measurement[axis]
                ) / dt_s

            raw = (
                gain.kp * error
                + gain.ki * self.integral[axis]
                - gain.kd * measurement_derivative
            )
            limit = self.config.torque_limit_nm[axis]
            saturated = _clamp(raw, -limit, limit)

            # Simple conditional-integration anti-windup: if saturation and the current
            # error would push farther into saturation, undo this sample's integration.
            if saturated != raw and error * raw > 0.0:
                self.integral[axis] -= error * dt_s
                self.integral[axis] = _clamp(
                    self.integral[axis], -gain.integral_limit, gain.integral_limit
                )

            output.append(saturated)

        self._previous_measurement = measured
        return (output[0], output[1], output[2])


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))
