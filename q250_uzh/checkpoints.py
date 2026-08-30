from __future__ import annotations

import re
from pathlib import Path

_MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def find_latest_checkpoint(log_root: str | Path) -> Path | None:
    """Return newest run's highest-numbered RSL-RL model checkpoint."""
    root = Path(log_root)
    if not root.exists():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    for run in runs:
        candidates: list[tuple[int, Path]] = []
        for path in run.rglob("model_*.pt"):
            match = _MODEL_RE.search(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]
    return None
