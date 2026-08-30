from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class TransferReport:
    copied_exact: int
    expanded_input_layers: int
    kept_new: int
    skipped_noise: int
    source_iteration: int | None = None


def extract_model_state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint does not contain model_state_dict/state_dict")


def _is_noise_key(key: str) -> bool:
    lowered = key.lower()
    return lowered == "std" or lowered.endswith(".std") or "log_std" in lowered or "noise_std" in lowered


def expand_state_dict_observation_input(
    old_state: Mapping[str, torch.Tensor],
    new_state: Mapping[str, torch.Tensor],
    *,
    old_obs_dim: int = 12,
    new_obs_dim: int = 21,
) -> tuple[dict[str, torch.Tensor], TransferReport]:
    """Copy a 12-D actor/critic into a 21-D network without changing old behavior.

    Exact-shape parameters are copied except exploration-noise parameters. Any
    2-D layer with shape [H, 12] -> [H, 21] is expanded by copying the old 12
    columns and setting the new nine columns to zero.
    """
    merged = {k: v.clone() for k, v in new_state.items()}
    copied_exact = 0
    expanded = 0
    kept_new = 0
    skipped_noise = 0

    for key, new_value in new_state.items():
        old_value = old_state.get(key)
        if old_value is None:
            kept_new += 1
            continue
        if _is_noise_key(key):
            skipped_noise += 1
            continue
        if tuple(old_value.shape) == tuple(new_value.shape):
            merged[key] = old_value.to(device=new_value.device, dtype=new_value.dtype).clone()
            copied_exact += 1
            continue
        if (
            old_value.ndim == 2
            and new_value.ndim == 2
            and old_value.shape[0] == new_value.shape[0]
            and old_value.shape[1] == old_obs_dim
            and new_value.shape[1] == new_obs_dim
        ):
            value = torch.zeros_like(new_value)
            value[:, :old_obs_dim] = old_value.to(device=new_value.device, dtype=new_value.dtype)
            merged[key] = value
            expanded += 1
            continue
        kept_new += 1

    return merged, TransferReport(
        copied_exact=copied_exact,
        expanded_input_layers=expanded,
        kept_new=kept_new,
        skipped_noise=skipped_noise,
    )


def resolve_runner_policy_module(runner):
    alg = getattr(runner, "alg", None)
    for candidate in (
        getattr(alg, "policy", None),
        getattr(alg, "actor_critic", None),
        getattr(runner, "policy", None),
    ):
        if candidate is not None and hasattr(candidate, "state_dict"):
            return candidate
    raise AttributeError("Could not locate the RSL-RL policy module on runner")


def transfer_checkpoint_to_runner(
    runner,
    checkpoint_path: str | Path,
    *,
    old_obs_dim: int = 12,
    new_obs_dim: int = 21,
) -> TransferReport:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        checkpoint = torch.load(path, map_location="cpu")
    old_state = extract_model_state_dict(checkpoint)
    policy = resolve_runner_policy_module(runner)
    new_state = policy.state_dict()
    merged, report = expand_state_dict_observation_input(
        old_state, new_state, old_obs_dim=old_obs_dim, new_obs_dim=new_obs_dim
    )
    policy.load_state_dict(merged, strict=True)
    source_iteration = checkpoint.get("iter") if isinstance(checkpoint, Mapping) else None
    return TransferReport(
        copied_exact=report.copied_exact,
        expanded_input_layers=report.expanded_input_layers,
        kept_new=report.kept_new,
        skipped_noise=report.skipped_noise,
        source_iteration=source_iteration,
    )


def save_transfer_snapshot(runner, path: str | Path, *, source: str | Path) -> Path:
    """Save the pre-training transferred policy without depending on runner logger initialization.

    ``OnPolicyRunner.save()`` assumes ``learn()`` has already initialized
    ``logger_type``/``writer``. Transfer happens before ``learn()``, so this
    helper serializes the minimal RSL-RL-compatible checkpoint directly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = resolve_runner_policy_module(runner)
    optimizer = getattr(getattr(runner, "alg", None), "optimizer", None)
    saved = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
        "iter": int(getattr(runner, "current_learning_iteration", 0)),
        "infos": {"source": str(source)},
    }
    torch.save(saved, path)
    return path
