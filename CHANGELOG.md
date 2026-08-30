# Changelog

## v0.3.0 — Fly-to-Point RL

Added:

- `Isaac-Q250-FlyToPoint-Direct-v0` DirectRLEnv implementation
- 12-D privileged observation: target position in body frame, body velocity, projected gravity, body rates
- 4-D hover-centered CTBR action mapping
- GPU-vectorized body-rate PID
- GPU-vectorized Q250 motor allocator
- existing identified motor lag + Kt/Kq dynamics inside every 240-Hz physics step
- 60-Hz PPO action rate
- minimal progress/success/crash/action reward
- success/crash termination
- three-stage randomized target curriculum
- RSL-RL PPO configuration
- standalone train/play/smoke scripts
- automatic newest-checkpoint discovery
- PowerShell wrappers for smoke, train, TensorBoard and play
- RL control/reward/checkpoint unit tests

No camera, gates, domain randomization, attitude controller or position PID are introduced in this release.

## v0.2.0 — body-rate inner loop

Added three-axis PID, closed-form X-quad allocator, PhysX rate-step validation, CSV logging and plots.

## v0.1.1 — hover warm-start fix

Rewrote pose after reset, zeroed root velocity, enabled external-force iteration handling.

## v0.1.0 — first physics package

Q250 mass/inertia/geometry, identified Kt/Kq and motor LUT, provisional motor lag, PhysX hover validation.
