from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GateCurriculum:
    stage: int
    gate_count: int
    width_m: float
    height_m: float
    y_extent_m: float
    z_min_m: float
    z_max_m: float


@dataclass(frozen=True)
class GateRacingRewardCfg:
    progress_scale: float = 4.0
    gate_bonus: float = 5.0
    finish_bonus: float = 12.0
    crash_penalty: float = -15.0
    action_penalty_scale: float = 0.001
    time_penalty: float = -0.01
    max_progress_per_step_m: float = 0.50


def gate_curriculum(common_policy_step: int) -> GateCurriculum:
    """Three-stage gate curriculum matched to the v0.3.0 PPO training cadence."""
    step = int(common_policy_step)
    if step < 800:
        return GateCurriculum(
            stage=0,
            gate_count=1,
            width_m=3.0,
            height_m=3.0,
            y_extent_m=0.50,
            z_min_m=1.2,
            z_max_m=2.0,
        )
    if step < 2400:
        return GateCurriculum(
            stage=1,
            gate_count=1,
            width_m=1.5,
            height_m=1.5,
            y_extent_m=1.25,
            z_min_m=1.0,
            z_max_m=2.4,
        )
    return GateCurriculum(
        stage=2,
        gate_count=3,
        width_m=1.5,
        height_m=1.5,
        y_extent_m=1.6,
        z_min_m=0.9,
        z_max_m=2.7,
    )


def signed_gate_distance(position_w: torch.Tensor, center_w: torch.Tensor, normal_w: torch.Tensor) -> torch.Tensor:
    """Signed distance to a gate plane. Negative is before the gate, positive is after."""
    return torch.sum((position_w - center_w) * normal_w, dim=-1)


def detect_gate_crossing(
    previous_signed_distance: torch.Tensor,
    current_signed_distance: torch.Tensor,
    lateral_y: torch.Tensor,
    lateral_z: torch.Tensor,
    *,
    half_width_m: float | torch.Tensor,
    half_height_m: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect forward plane crossing and classify it as a clean pass or a missed opening."""
    crossed = (previous_signed_distance < 0.0) & (current_signed_distance >= 0.0)
    inside = (torch.abs(lateral_y) <= half_width_m) & (torch.abs(lateral_z) <= half_height_m)
    passed = crossed & inside
    missed = crossed & (~inside)
    return passed, missed, crossed


def advance_gate_indices(
    current_indices: torch.Tensor,
    gate_counts: torch.Tensor,
    passed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance passed environments and report which ones just completed the final gate."""
    last_gate = current_indices >= (gate_counts - 1)
    finished = passed & last_gate
    advance = passed & (~last_gate)
    new_indices = current_indices + advance.to(dtype=current_indices.dtype)
    return new_indices, finished


def compute_gate_racing_reward(
    gate_progress_m: torch.Tensor,
    gate_passed: torch.Tensor,
    race_finished: torch.Tensor,
    crashed: torch.Tensor,
    actions: torch.Tensor,
    cfg: GateRacingRewardCfg | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = cfg or GateRacingRewardCfg()
    progress = torch.clamp(
        gate_progress_m,
        min=-cfg.max_progress_per_step_m,
        max=cfg.max_progress_per_step_m,
    ) * cfg.progress_scale
    gate_reward = gate_passed.to(gate_progress_m.dtype) * cfg.gate_bonus
    finish_reward = race_finished.to(gate_progress_m.dtype) * cfg.finish_bonus
    crash_reward = crashed.to(gate_progress_m.dtype) * cfg.crash_penalty
    action_reward = -cfg.action_penalty_scale * torch.sum(actions.square(), dim=-1)
    time_reward = torch.full_like(gate_progress_m, cfg.time_penalty)
    parts = {
        "progress": progress,
        "gate": gate_reward,
        "finish": finish_reward,
        "crash": crash_reward,
        "action": action_reward,
        "time": time_reward,
    }
    total = torch.stack(tuple(parts.values()), dim=0).sum(dim=0)
    return total, parts
