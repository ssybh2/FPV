from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlyToPointRewardCfg:
    progress_scale: float = 5.0
    success_bonus: float = 10.0
    crash_penalty: float = -10.0
    action_penalty_scale: float = 0.002


@dataclass(frozen=True)
class CurriculumBounds:
    stage: int
    xy_extent_m: float
    z_min_m: float
    z_max_m: float


def curriculum_bounds(common_policy_step: int) -> CurriculumBounds:
    """Three-stage target curriculum keyed by DirectRLEnv common_step_counter."""
    step = int(common_policy_step)
    if step < 800:
        return CurriculumBounds(stage=0, xy_extent_m=1.0, z_min_m=1.0, z_max_m=2.0)
    if step < 2400:
        return CurriculumBounds(stage=1, xy_extent_m=2.0, z_min_m=0.8, z_max_m=2.7)
    return CurriculumBounds(stage=2, xy_extent_m=4.0, z_min_m=0.6, z_max_m=3.5)


def compute_fly_to_point_reward(
    current_distance: torch.Tensor,
    previous_distance: torch.Tensor,
    success: torch.Tensor,
    crashed: torch.Tensor,
    actions: torch.Tensor,
    cfg: FlyToPointRewardCfg | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = cfg or FlyToPointRewardCfg()
    progress = (previous_distance - current_distance) * cfg.progress_scale
    success_reward = success.to(current_distance.dtype) * cfg.success_bonus
    crash_reward = crashed.to(current_distance.dtype) * cfg.crash_penalty
    action_reward = -cfg.action_penalty_scale * torch.sum(actions.square(), dim=-1)
    parts = {
        "progress": progress,
        "success": success_reward,
        "crash": crash_reward,
        "action": action_reward,
    }
    total = torch.stack(tuple(parts.values()), dim=0).sum(dim=0)
    return total, parts
