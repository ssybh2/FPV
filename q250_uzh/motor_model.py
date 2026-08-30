from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from importlib.resources import files
from typing import Iterable

from .config import Q250


@dataclass(frozen=True)
class MotorLUTRow:
    pwm_us: float
    rpm: float
    omega_rad_s: float
    thrust_n: float
    torque_nm: float
    voltage_v: float
    current_a: float


def load_default_lut() -> tuple[MotorLUTRow, ...]:
    path = files("q250_uzh.data").joinpath("motor_lut.csv")
    rows: list[MotorLUTRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(MotorLUTRow(**{key: float(value) for key, value in row.items()}))
    if len(rows) < 2:
        raise RuntimeError("Motor LUT must contain at least two rows.")
    return tuple(rows)


def _interp(x: float, xs: Iterable[float], ys: Iterable[float]) -> float:
    x_list = tuple(xs)
    y_list = tuple(ys)
    if len(x_list) != len(y_list) or len(x_list) < 2:
        raise ValueError("Interpolation inputs must have equal length >= 2.")
    if x <= x_list[0]:
        return y_list[0]
    if x >= x_list[-1]:
        return y_list[-1]
    i = bisect_right(x_list, x) - 1
    x0, x1 = x_list[i], x_list[i + 1]
    y0, y1 = y_list[i], y_list[i + 1]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@dataclass
class MotorModel:
    lut: tuple[MotorLUTRow, ...]
    tau_s: float = Q250.motor_tau_s
    kt: float = Q250.kt_n_per_rad_s_sq
    kq: float = Q250.kq_nm_per_rad_s_sq

    def __post_init__(self) -> None:
        if self.tau_s <= 0.0:
            raise ValueError("Motor time constant must be positive.")

    @property
    def pwm_axis(self) -> tuple[float, ...]:
        return tuple(r.pwm_us for r in self.lut)

    @property
    def rpm_axis(self) -> tuple[float, ...]:
        return tuple(r.rpm for r in self.lut)

    def pwm_to_rpm(self, pwm_us: float) -> float:
        return _interp(float(pwm_us), self.pwm_axis, self.rpm_axis)

    def rpm_to_pwm(self, rpm: float) -> float:
        return _interp(float(rpm), self.rpm_axis, self.pwm_axis)

    def pwm_to_measured_thrust(self, pwm_us: float) -> float:
        value = _interp(float(pwm_us), self.pwm_axis, (r.thrust_n for r in self.lut))
        return max(0.0, value)

    def pwm_to_omega(self, pwm_us: float) -> float:
        return self.pwm_to_rpm(pwm_us) * 2.0 * math.pi / 60.0

    def first_order_step(self, omega: float, omega_cmd: float, dt_s: float) -> float:
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative.")
        alpha = math.exp(-dt_s / self.tau_s)
        return float(omega_cmd) + (float(omega) - float(omega_cmd)) * alpha

    def thrust_from_omega(self, omega_rad_s: float) -> float:
        omega = max(0.0, float(omega_rad_s))
        return self.kt * omega * omega

    def torque_from_omega(self, omega_rad_s: float) -> float:
        omega = max(0.0, float(omega_rad_s))
        return self.kq * omega * omega


def default_motor_model() -> MotorModel:
    return MotorModel(load_default_lut())
