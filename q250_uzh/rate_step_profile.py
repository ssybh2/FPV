from __future__ import annotations

from dataclasses import dataclass


_AXIS_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}


@dataclass(frozen=True)
class RateStepProfile:
    axis: str
    rate_rad_s: float
    start_s: float = 0.5
    end_s: float = 0.9

    def __post_init__(self) -> None:
        if self.axis not in _AXIS_INDEX:
            raise ValueError(f"axis must be one of {tuple(_AXIS_INDEX)}, got {self.axis!r}")
        if self.start_s < 0.0:
            raise ValueError("start_s must be non-negative.")
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s.")

    def command_at(self, time_s: float) -> tuple[float, float, float]:
        cmd = [0.0, 0.0, 0.0]
        if self.start_s <= float(time_s) < self.end_s:
            cmd[_AXIS_INDEX[self.axis]] = float(self.rate_rad_s)
        return (cmd[0], cmd[1], cmd[2])
