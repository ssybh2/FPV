from __future__ import annotations

import math

import torch

from q250_uzh.config import Q250


class Q250TorchMotorBank:
    """Vectorized four-motor first-order model for Isaac Lab.

    State shape: (num_envs, 4). This class intentionally uses omega commands,
    leaving PWM/RPM LUT conversion outside the physics inner loop for now.
    """

    def __init__(self, num_envs: int, device: str | torch.device, tau_s: float = Q250.motor_tau_s):
        self.tau_s = float(tau_s)
        self.omega = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        self.motor_pos = torch.tensor(Q250.motor_positions_m, dtype=torch.float32, device=device)
        self.reaction_sign = torch.tensor(Q250.rotor_reaction_sign, dtype=torch.float32, device=device)

    def reset(self, env_ids=None, omega_rad_s: float = 0.0) -> None:
        if env_ids is None:
            self.omega.fill_(float(omega_rad_s))
        else:
            self.omega[env_ids] = float(omega_rad_s)

    def step_omega_command(self, omega_cmd: torch.Tensor, dt_s: float) -> torch.Tensor:
        omega_cmd = omega_cmd.clamp_min(0.0)
        alpha = math.exp(-float(dt_s) / self.tau_s)
        self.omega = omega_cmd + (self.omega - omega_cmd) * alpha
        return self.omega

    def wrench(self) -> tuple[torch.Tensor, torch.Tensor]:
        omega_sq = self.omega.square()
        thrust = Q250.kt_n_per_rad_s_sq * omega_sq
        reaction = Q250.kq_nm_per_rad_s_sq * omega_sq * self.reaction_sign

        force = torch.zeros((self.omega.shape[0], 1, 3), dtype=self.omega.dtype, device=self.omega.device)
        torque = torch.zeros_like(force)
        force[:, 0, 2] = thrust.sum(dim=1)

        x = self.motor_pos[:, 0]
        y = self.motor_pos[:, 1]
        torque[:, 0, 0] = (thrust * y).sum(dim=1)
        torque[:, 0, 1] = (-thrust * x).sum(dim=1)
        torque[:, 0, 2] = reaction.sum(dim=1)
        return force, torque
