from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    print("Python executable:", sys.executable)
    print("Python version   :", sys.version.replace("\n", " "))
    required = ("isaaclab", "isaacsim", "torch", "isaaclab_rl", "rsl_rl")
    missing = []
    for name in required:
        spec = importlib.util.find_spec(name)
        print(f"{name:14s}:", spec.origin if spec else "NOT FOUND")
        if spec is None:
            missing.append(name)
    if missing:
        print("\nERROR: missing required modules:", ", ".join(missing))
        print("Use the same Isaac Lab Python environment that you used for the successful hover/rate tests.")
        return 2
    print("\nEnvironment probe passed. This Python has Isaac Lab + RSL-RL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
