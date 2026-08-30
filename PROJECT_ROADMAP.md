# Project roadmap

## Phase 0 — physics sanity — PASSED

Measured Q250 mass/inertia, geometry, motor/prop Kt/Kq + LUT, provisional motor lag, PhysX hover equilibrium.

## Phase 1 — conventional body-rate inner loop — PASSED

Body-rate PID, constrained motor allocation, motor dynamics, roll/pitch/yaw step-response validation. User-provided 100 deg/s logs showed stable tracking with no allocator saturation.

## Phase 2 — Fly-to-Point DirectRLEnv — CURRENT (v0.3.0)

Implemented:

- 12-D privileged observation
- 4-D normalized CTBR action
- vectorized GPU rate PID + motor allocator
- 240 Hz physics / 60 Hz PPO policy
- progress/success/crash/action reward
- 3-stage target curriculum
- RSL-RL PPO train/play scripts
- TensorBoard workflow

Exit criterion: policy consistently reaches randomized 3-D targets with high success rate and low crash rate.

## Phase 3 — gate racing

- one large gate
- normal gate
- 3-gate sequence
- plane-crossing gate detector
- gate progress reward
- track curriculum

## Phase 4 — high-speed + robustness

Increase CTBR/rate envelopes and randomize mass, inertia, Kt/Kq, motor lag, delay, drag, battery/thrust scale, sensor noise, and gate pose.

## Phase 5 — perception + Sim2Real

Camera + IMU, teacher/student or perception front-end, latency/noise model, PX4 NED/FRD conversion, hardware replay and real-flight validation.
