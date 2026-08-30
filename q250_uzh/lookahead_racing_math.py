from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class LookAheadCurriculum:
    stage: int
    gate_count: int
    width_m: float
    height_m: float
    y_extent_m: float
    z_min_m: float
    z_max_m: float
    yaw_jitter_deg: float
    pitch_jitter_deg: float
    min_spacing_m: float
    max_spacing_m: float


def lookahead_curriculum(common_policy_step: int) -> LookAheadCurriculum:
    """Curriculum for v0.5.0 look-ahead racing.

    Stage 0 preserves a simple three-gate vertical track so the transferred
    v0.4.0 policy can adapt to the larger observation without losing its skill.
    Stage 1 introduces gate yaw/pitch. Stage 2 expands to five oriented gates.
    """
    step = int(common_policy_step)
    if step < 1000:
        return LookAheadCurriculum(
            stage=0,
            gate_count=3,
            width_m=1.8,
            height_m=1.8,
            y_extent_m=1.3,
            z_min_m=1.0,
            z_max_m=2.6,
            yaw_jitter_deg=0.0,
            pitch_jitter_deg=0.0,
            min_spacing_m=2.3,
            max_spacing_m=2.8,
        )
    if step < 3000:
        return LookAheadCurriculum(
            stage=1,
            gate_count=3,
            width_m=1.5,
            height_m=1.5,
            y_extent_m=1.8,
            z_min_m=0.9,
            z_max_m=2.9,
            yaw_jitter_deg=20.0,
            pitch_jitter_deg=10.0,
            min_spacing_m=2.2,
            max_spacing_m=2.9,
        )
    return LookAheadCurriculum(
        stage=2,
        gate_count=5,
        width_m=1.5,
        height_m=1.5,
        y_extent_m=2.3,
        z_min_m=0.8,
        z_max_m=3.2,
        yaw_jitter_deg=35.0,
        pitch_jitter_deg=15.0,
        min_spacing_m=2.1,
        max_spacing_m=3.0,
    )


def gate_basis_from_yaw_pitch(yaw_rad: torch.Tensor, pitch_rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gate normal/right/up unit vectors in world frame.

    The local +X axis is the forward gate normal. Yaw rotates about world +Z;
    positive pitch raises the normal's +Z component. Gate roll is intentionally
    zero in v0.5.0.
    """
    cy, sy = torch.cos(yaw_rad), torch.sin(yaw_rad)
    cp, sp = torch.cos(pitch_rad), torch.sin(pitch_rad)
    normal = torch.stack((cp * cy, cp * sy, sp), dim=-1)
    right = torch.stack((-sy, cy, torch.zeros_like(cy)), dim=-1)
    up = torch.stack((-sp * cy, -sp * sy, cp), dim=-1)
    return normal, right, up


def gate_quat_wxyz_from_yaw_pitch(yaw_rad: torch.Tensor, pitch_rad: torch.Tensor) -> torch.Tensor:
    """Quaternion rotating a gate's local +X normal to yaw/pitch in world frame."""
    hy = 0.5 * yaw_rad
    hp = 0.5 * pitch_rad
    cy, sy = torch.cos(hy), torch.sin(hy)
    cp, sp = torch.cos(hp), torch.sin(hp)
    # Rz(yaw) * Ry(-pitch), wxyz.
    return torch.stack((cy * cp, sy * sp, -cy * sp, sy * cp), dim=-1)


def gate_local_coordinates(
    position_w: torch.Tensor,
    center_w: torch.Tensor,
    normal_w: torch.Tensor,
    right_w: torch.Tensor,
    up_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = position_w - center_w
    signed = torch.sum(delta * normal_w, dim=-1)
    lateral = torch.sum(delta * right_w, dim=-1)
    vertical = torch.sum(delta * up_w, dim=-1)
    return signed, lateral, vertical


def quat_rotate_inverse_wxyz(quat_wxyz: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors into a quaternion's local/body frame."""
    w = quat_wxyz[..., :1]
    qv = -quat_wxyz[..., 1:]
    t = 2.0 * torch.cross(qv, vec_w, dim=-1)
    return vec_w + w * t + torch.cross(qv, t, dim=-1)


def build_lookahead_observation(
    current_gate_b: torch.Tensor,
    lin_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    next_gate_b: torch.Tensor,
    current_normal_b: torch.Tensor,
    next_normal_b: torch.Tensor,
) -> torch.Tensor:
    """Build the 21-D observation while preserving the old 12-D prefix exactly."""
    return torch.cat(
        (
            current_gate_b,
            lin_vel_b,
            projected_gravity_b,
            ang_vel_b,
            next_gate_b,
            current_normal_b,
            next_normal_b,
        ),
        dim=-1,
    )


def lookahead_alignment_reward(
    velocity_w: torch.Tensor,
    current_center_w: torch.Tensor,
    next_center_w: torch.Tensor,
    current_distance_m: torch.Tensor,
    has_next: torch.Tensor,
    *,
    scale: float = 0.05,
    distance_scale_m: float = 1.5,
) -> torch.Tensor:
    """Small near-gate shaping term encouraging exit velocity toward the next gate."""
    desired = next_center_w - current_center_w
    desired = desired / desired.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    speed = velocity_w.norm(dim=-1).clamp_min(1e-6)
    velocity_dir = velocity_w / speed[:, None]
    cosine = torch.sum(velocity_dir * desired, dim=-1).clamp(-1.0, 1.0)
    near_weight = torch.exp(-current_distance_m / distance_scale_m)
    return has_next.to(velocity_w.dtype) * scale * near_weight * cosine
