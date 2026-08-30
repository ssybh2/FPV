# Q250 Fly-to-Point RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trainable Isaac Lab DirectRLEnv Fly-to-Point task using the validated Q250 CTBR/rate-PID/motor dynamics and RSL-RL PPO.

**Architecture:** Add pure-Torch vectorized inner-loop/control-allocation helpers and task math, then connect them to a `RigidObject`-based DirectRLEnv. Add RSL-RL PPO configuration plus standalone train/play/smoke scripts so the project remains external to the Isaac Lab source tree.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab 2.3.x / Isaac Sim 5.1, Gymnasium, RSL-RL 3.x, PowerShell.

**Spec:** `docs/superpowers/specs/2026-08-30-fly-to-point-rl-design.md`

## Global Constraints
- Keep the validated Q250 mass, inertia, 250 mm geometry, Kt/Kq, rotor directions, and 0.10 s provisional motor lag unchanged.
- Physics timestep is 1/240 s; policy decimation is 4.
- Policy action is normalized CTBR, not direct motor command.
- Policy observation is 12-dimensional privileged state; no camera in v0.3.0.
- Do not modify the user's Isaac Lab source tree.

---

### Task 1: Vectorized CTBR control path
**Files:** Create `q250_uzh/rl_control.py`; create tests `tests/test_rl_control.py`.
**Interfaces:** Produces `map_actions_to_ctbr`, `TorchBodyRatePID`, `TorchMotorAllocator` for the RL environment.
- [ ] Write failing tests for hover-centered action mapping, vectorized PID reset/update, allocator hover balance, roll sign, and saturation.
- [ ] Run the tests and verify they fail because the module does not exist.
- [ ] Implement minimal vectorized Torch code.
- [ ] Run tests and verify pass.

### Task 2: Fly-to-Point task math
**Files:** Create `q250_uzh/fly_to_point_math.py`; create `tests/test_fly_to_point_math.py`.
**Interfaces:** Produces `FlyToPointRewardCfg`, `compute_fly_to_point_reward`, `curriculum_bounds`.
- [ ] Write failing tests for progress reward sign, success/crash bonus, action penalty, and three curriculum stages.
- [ ] Run tests and verify failure.
- [ ] Implement task math with no Isaac dependency.
- [ ] Run tests and verify pass.

### Task 3: Isaac Lab DirectRLEnv
**Files:** Create `q250_uzh/tasks/fly_to_point_env.py`, `q250_uzh/tasks/__init__.py`.
**Interfaces:** Produces `FlyToPointEnvCfg`, `FlyToPointEnv`, Gym id `Isaac-Q250-FlyToPoint-Direct-v0`.
- [ ] Implement the scene with per-env Q250 rigid objects and terrain.
- [ ] Map 4-D actions in `_pre_physics_step`; run PID/allocator/motor lag in `_apply_action` every 240-Hz physics step.
- [ ] Implement 12-D observations, reward, termination, curriculum target reset, motor/PID reset, and target debug marker.
- [ ] Run Python compilation checks because Isaac Lab is unavailable in the packaging container.

### Task 4: RSL-RL PPO and launch scripts
**Files:** Create `q250_uzh/agents/rsl_rl_ppo_cfg.py`, `scripts/train_fly_to_point.py`, `scripts/play_fly_to_point.py`, `scripts/smoke_fly_to_point.py`, PowerShell launchers.
**Interfaces:** Default 512 envs / 300 iterations; logs under `logs/rsl_rl/q250_fly_to_point`.
- [ ] Add PPO config based on Isaac Lab 2.3.0's official Direct Quadcopter RSL-RL pattern.
- [ ] Add standalone training script that does not require editing Isaac Lab.
- [ ] Add checkpoint discovery and play script.
- [ ] Add zero-action smoke test and PowerShell launchers.
- [ ] Run compile checks.

### Task 5: Documentation, regression, package
**Files:** Modify `README.md`, `QUICKSTART_WINDOWS.md`, `PROJECT_ROADMAP.md`, `CHANGELOG.md`, `setup.py`; create ZIP.
- [ ] Document exact install/smoke/train/TensorBoard/play commands.
- [ ] Run all pure tests, compile all Python, validate ZIP contents/integrity.
- [ ] Package as `Q250_UZH_Racing_v0.3.0.zip` without overwriting v0.2.0.
