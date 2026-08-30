# Changelog

## v0.4.0 — Gate Racing Curriculum

Added:

- `Isaac-Q250-GateRacing-Direct-v0`
- large-gate -> standard-gate -> three-gate curriculum
- mathematically explicit forward plane-crossing gate detector
- missed-opening crash termination
- current-gate-center 12-D privileged observation (same shape as Fly-to-Point)
- gate progress, pass, race finish, crash, action and time rewards
- per-episode gate completion and race success metrics
- corrected allocator saturation metric accumulated across policy steps
- orange gate-frame and green current-gate UI markers
- dedicated gate smoke/train/play scripts and PowerShell wrappers
- lower PPO exploration (`noise_std=0.5`, `entropy_coef=0.003`)
- gate geometry/reward/curriculum unit tests

Deliberately deferred:

- gate yaw/pitch randomization
- next-gate preview observation
- physical gate-frame collisions
- camera perception
- domain randomization

## v0.3.0 — Fly-to-Point RL

Added 12-D privileged Fly-to-Point DirectRLEnv, 4-D CTBR PPO control, vectorized rate PID/allocator, target curriculum and train/play tooling.

## v0.2.0 — body-rate inner loop

Added three-axis PID, closed-form X-quad allocator, PhysX rate-step validation, CSV logging and plots.

## v0.1.1 — hover warm-start fix

Rewrote pose after reset, zeroed root velocity, enabled external-force iteration handling.

## v0.1.0 — first physics package

Q250 mass/inertia/geometry, identified Kt/Kq and motor LUT, provisional motor lag, PhysX hover validation.
