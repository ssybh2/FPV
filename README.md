# Q250 UZH-style Drone Racing Workspace — v0.4.0

External Windows + Isaac Lab workspace for a Q250 racing quadrotor. The project does **not** modify the Isaac Lab source tree.

## Current status

- Phase 0: Q250 physics + motor model + PhysX hover — **validated on Windows**
- Phase 1: body-rate PID + motor allocator — **validated from roll/pitch/yaw step logs**
- Phase 2: Fly-to-Point RL — **passed; model around iteration 200 visually validated**
- Phase 3: Gate Racing curriculum — **implemented in v0.4.0**

## Physical model kept unchanged

- mass: `1.0006 kg`
- inertia: `Ix=0.00517`, `Iy=0.00484`, `Iz=0.00750 kg m^2`
- Q250 diagonal motor spacing: `250 mm`
- M1 front-left CCW, M2 front-right CW, M3 rear-right CCW, M4 rear-left CW
- `Kt = 1.3287717252618608e-6 N/(rad/s)^2`
- `Kq = 1.772957417327994e-8 N m/(rad/s)^2`
- measured motor LUT: `q250_uzh/data/motor_lut.csv`
- motor first-order lag: `tau = 0.10 s` (provisional)

## v0.4.0 architecture

The 12-D observation and 4-D CTBR action interface are intentionally preserved from v0.3.0.

```text
12-D privileged observation
[current_gate_center_b, v_b, gravity_b, omega_b]
                  |
                  v
               PPO policy
                  |
       normalized CTBR [-1,1]^4
                  |
                  v
       [T, p_cmd, q_cmd, r_cmd]
                  |
                  v
        Body-rate PID @ 240 Hz
                  |
                  v
            Motor allocator
                  |
                  v
       motor lag + Kt*w^2/Kq*w^2
                  |
                  v
              Isaac PhysX
```

Policy frequency remains 60 Hz (`dt=1/240`, `decimation=4`).

## What counts as a gate pass

A gate is not a waypoint. The Q250 must:

1. approach from the negative side of the gate plane;
2. cross the plane in the forward direction;
3. have its vehicle center inside the rectangular opening at the crossing instant.

Crossing the plane outside the opening is a **miss** and terminates the episode.

In v0.4.0 gate frames are visual markers, while this plane-crossing geometry is the authoritative training rule. Physical gate-frame collision is intentionally deferred until the racing policy is reliable.

## Curriculum

The curriculum uses the same global policy-step timing style as v0.3.0:

| Stage | Global policy steps | Task |
|---|---:|---|
| 0 | `< 800` | one large `3.0 x 3.0 m` gate |
| 1 | `800..2399` | one standard `1.5 x 1.5 m` gate |
| 2 | `>= 2400` | three sequential `1.5 x 1.5 m` gates |

All v0.4.0 gates are vertical and face `+X`. Gate centers randomize laterally and vertically. This is deliberate: the next version will add gate yaw/pitch, next-gate preview and true corner-cutting behavior.

## Reward

- progress toward the **current gate center** (keeps the successful Fly-to-Point shaping)
- gate-pass bonus
- larger race-finish bonus
- crash/missed-gate penalty
- small action penalty
- small time penalty to encourage faster completion

## PPO changes from v0.3.0

The first Fly-to-Point run showed late-training exploration noise and allocator saturation rising strongly. Gate PPO therefore starts with lower exploration:

- `init_noise_std = 0.5`
- `entropy_coef = 0.003`
- `learning_rate = 4e-4`
- default `400` iterations

## Fastest deployment

```powershell
cd E:\IsaacWork\Q250_UZH_Racing_v0.4.0
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\smoke_gate.ps1
.\train_gate.ps1 -NumEnvs 512 -MaxIterations 400
```

If 512 environments is too heavy, use 256. If GPU/VRAM has room, try 1024.

Watch training:

```powershell
.\tensorboard.ps1
```

Play the newest gate checkpoint in the hardest 3-gate stage:

```powershell
.\play_gate.ps1 -Stage 2 -Duration 0 -RealTime
```

Stage-specific playback:

```powershell
.\play_gate.ps1 -Stage 0 -Duration 0 -RealTime
.\play_gate.ps1 -Stage 1 -Duration 0 -RealTime
.\play_gate.ps1 -Stage 2 -Duration 0 -RealTime
```

In the UI:

- orange frames = gates
- green cube = current gate center

## TensorBoard metrics to watch

Priority order:

1. `Metrics/race_success_rate` — complete all currently active gates
2. `Metrics/gate_completion_fraction` — fraction of the track completed
3. `Metrics/missed_gate_rate` — crossed a gate plane outside the opening
4. `Metrics/allocator_saturation_rate`
5. `Metrics/episode_time_s` — once success is high, lower is faster
6. `Episode_Reward/progress`, `gate`, `finish`, `crash`
7. `Curriculum/stage`

For Stage 2, a strong exit criterion is roughly:

- race success consistently `> 80%`
- gate completion `> 90%`
- missed-gate rate low
- no collapse in allocator saturation

## New v0.4.0 files

```text
q250_uzh/gate_racing_math.py
q250_uzh/tasks/gate_racing_env.py
scripts/smoke_gate_racing.py
scripts/train_gate_racing.py
scripts/play_gate_racing.py
smoke_gate.ps1
train_gate.ps1
play_gate.ps1
```

The Fly-to-Point environment and its scripts remain in the workspace for regression and comparison.

## Coordinate convention

Internal convention remains FLU / Z-up: +x forward, +y left, +z up. Do not mix it with PX4 FRD/NED without an explicit conversion layer.
