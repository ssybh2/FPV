from __future__ import annotations

from collections.abc import Sequence

import torch


def corrected_warm_start_root_state(
    root_state: torch.Tensor,
    position_xyz: Sequence[float] = (0.0, 0.0, 1.5),
) -> torch.Tensor:
    """Return a corrected 13-D root state for the first controlled physics step.

    Isaac/PhysX initialization can advance gravity while the asset is being reset/initialized.
    For a hover validation this can leave a small downward velocity before the first motor wrench
    is ever applied.  Rewriting the desired pose and zero 6-D velocity after ``sim.reset()`` makes
    the warm-start test measure force balance rather than initialization transients.

    State layout: ``[x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]``.
    The input is never modified in place.
    """
    if root_state.shape[-1] != 13:
        raise ValueError(f"root_state must have 13 values per body, got shape {tuple(root_state.shape)}")
    if len(position_xyz) != 3:
        raise ValueError("position_xyz must contain exactly three values")

    corrected = root_state.clone()
    corrected[..., 0] = float(position_xyz[0])
    corrected[..., 1] = float(position_xyz[1])
    corrected[..., 2] = float(position_xyz[2])
    corrected[..., 7:13] = 0.0
    return corrected
