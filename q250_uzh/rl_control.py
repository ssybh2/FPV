from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import Q250
from .motor_model import load_default_lut
from .rate_controller import BodyRatePIDConfig


@dataclass(frozen=True)
class CTBRActionCfg:
    """Mapping from normalized PPO actions to collective thrust + body-rate commands."""

    min_collective_weight_ratio: float = 0.30
    max_collective_weight_ratio: float = 2.50
    max_roll_rate_rad_s: float = math.radians(200.0)
    max_pitch_rate_rad_s: float = math.radians(200.0)
    max_yaw_rate_rad_s: float = math.radians(100.0)


def map_actions_to_ctbr(actions: torch.Tensor, cfg: CTBRActionCfg | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Map normalized actions [-1, 1]^4 to hover-centered CTBR commands.

    action[:, 0] is asymmetric around hover so zero action produces exactly mg:
      -1 -> 0.30 mg, 0 -> 1.00 mg, +1 -> 2.50 mg.
    action[:, 1:4] map linearly to roll/pitch/yaw body-rate limits.
    """

    cfg = cfg or CTBRActionCfg()
    if actions.ndim != 2 or actions.shape[-1] != 4:
        raise ValueError("actions must have shape (num_envs, 4)")

    a = actions.clamp(-1.0, 1.0)
    collective_action = a[:, 0]
    downward_span = 1.0 - cfg.min_collective_weight_ratio
    upward_span = cfg.max_collective_weight_ratio - 1.0
    weight_ratio = 1.0 + torch.where(
        collective_action < 0.0,
        collective_action * downward_span,
        collective_action * upward_span,
    )
    weight = Q250.mass_kg * Q250.gravity_m_s2
    collective = weight_ratio * weight

    rate_limits = torch.tensor(
        [cfg.max_roll_rate_rad_s, cfg.max_pitch_rate_rad_s, cfg.max_yaw_rate_rad_s],
        dtype=actions.dtype,
        device=actions.device,
    )
    rates = a[:, 1:4] * rate_limits
    return collective, rates


class TorchBodyRatePID:
    """GPU-vectorized equivalent of the validated scalar body-rate PID."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        config: BodyRatePIDConfig | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config or BodyRatePIDConfig()

        gains = (self.config.roll, self.config.pitch, self.config.yaw)
        self.kp = torch.tensor([g.kp for g in gains], dtype=dtype, device=self.device)
        self.ki = torch.tensor([g.ki for g in gains], dtype=dtype, device=self.device)
        self.kd = torch.tensor([g.kd for g in gains], dtype=dtype, device=self.device)
        self.integral_limit = torch.tensor(
            [g.integral_limit for g in gains], dtype=dtype, device=self.device
        )
        self.torque_limit = torch.tensor(self.config.torque_limit_nm, dtype=dtype, device=self.device)

        self.integral = torch.zeros((self.num_envs, 3), dtype=dtype, device=self.device)
        self.previous_measurement = torch.zeros_like(self.integral)
        self.has_previous = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.integral.zero_()
            self.previous_measurement.zero_()
            self.has_previous.zero_()
            return
        self.integral[env_ids] = 0.0
        self.previous_measurement[env_ids] = 0.0
        self.has_previous[env_ids] = False

    def update(self, rate_cmd_rad_s: torch.Tensor, rate_measured_rad_s: torch.Tensor, dt_s: float) -> torch.Tensor:
        if rate_cmd_rad_s.shape != (self.num_envs, 3) or rate_measured_rad_s.shape != (self.num_envs, 3):
            raise ValueError(f"rate tensors must both have shape ({self.num_envs}, 3)")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        error = rate_cmd_rad_s - rate_measured_rad_s
        previous_integral = self.integral.clone()
        candidate_integral = torch.clamp(
            previous_integral + error * float(dt_s),
            min=-self.integral_limit,
            max=self.integral_limit,
        )

        derivative = (rate_measured_rad_s - self.previous_measurement) / float(dt_s)
        derivative = torch.where(self.has_previous[:, None], derivative, torch.zeros_like(derivative))

        raw = self.kp * error + self.ki * candidate_integral - self.kd * derivative
        saturated = torch.clamp(raw, min=-self.torque_limit, max=self.torque_limit)

        # Conditional integration anti-windup: reject this integral sample on axes that
        # are saturated and where the error would push further into saturation.
        reject_integral = (torch.abs(raw) > self.torque_limit) & (error * raw > 0.0)
        self.integral = torch.where(reject_integral, previous_integral, candidate_integral)
        self.previous_measurement.copy_(rate_measured_rad_s)
        self.has_previous.fill_(True)
        return saturated


class TorchMotorAllocator:
    """Vectorized X-quad allocation from [Fz, tau_x, tau_y, tau_z] to motor omega."""

    def __init__(self, device: str | torch.device, max_omega_rad_s: float | None = None, dtype=torch.float32):
        if max_omega_rad_s is None:
            max_omega_rad_s = load_default_lut()[-1].omega_rad_s
        if max_omega_rad_s <= 0.0:
            raise ValueError("max_omega_rad_s must be positive")
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_omega_rad_s = float(max_omega_rad_s)
        self.max_motor_thrust_n = Q250.kt_n_per_rad_s_sq * self.max_omega_rad_s**2

    def allocate(self, collective_thrust_n: torch.Tensor, body_torque_nm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if collective_thrust_n.ndim != 1:
            raise ValueError("collective_thrust_n must have shape (num_envs,)")
        if body_torque_nm.shape != (collective_thrust_n.shape[0], 3):
            raise ValueError("body_torque_nm must have shape (num_envs, 3)")

        f = collective_thrust_n.clamp_min(0.0)
        tx, ty, tz = body_torque_nm.unbind(dim=-1)
        a = Q250.arm_xy_m
        c = Q250.kq_nm_per_rad_s_sq / Q250.kt_n_per_rad_s_sq
        x = tx / a
        y = ty / a
        z = tz / c

        requested = torch.stack(
            (
                0.25 * (f + x - y - z),
                0.25 * (f - x - y + z),
                0.25 * (f - x + y - z),
                0.25 * (f + x + y + z),
            ),
            dim=-1,
        )
        clipped = requested.clamp(0.0, self.max_motor_thrust_n)
        saturated = torch.any(torch.abs(requested - clipped) > 1.0e-6, dim=-1)
        omega = torch.sqrt(clipped / Q250.kt_n_per_rad_s_sq)
        return omega, saturated
