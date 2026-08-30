# Q250 UZH-style Drone Racing Workspace — v0.3.0

External Windows + Isaac Lab workspace for a Q250 racing quadrotor. The project does **not** modify the Isaac Lab source tree.

## Current status

- Phase 0: Q250 physics + motor model + PhysX hover — **validated on the user's Windows machine**
- Phase 1: body-rate PID + motor allocator — **validated from roll/pitch/yaw step logs**
- Phase 2: Fly-to-Point RL — **implemented in v0.3.0; run smoke test, then PPO training**

## Physical model kept unchanged

- mass: `1.0006 kg`
- inertia: `Ix=0.00517`, `Iy=0.00484`, `Iz=0.00750 kg m^2`
- Q250 diagonal motor spacing: `250 mm`
- M1 front-left CCW, M2 front-right CW, M3 rear-right CCW, M4 rear-left CW
- `Kt = 1.3287717252618608e-6 N/(rad/s)^2`
- `Kq = 1.772957417327994e-8 N m/(rad/s)^2`
- measured motor LUT: `q250_uzh/data/motor_lut.csv`
- motor first-order lag: `tau = 0.10 s` (provisional)

## v0.3.0 control architecture

```text
12-D privileged observation
[target_b, v_b, gravity_b, omega_b]
             |
             v
          PPO policy
             |
 normalized CTBR action [-1,1]^4
             |
             v
 [T, p_cmd, q_cmd, r_cmd]
             |
             v
 vectorized BodyRate PID @ 240 Hz
             |
             v
 vectorized Q250 motor allocator
             |
             v
 motor lag + Kt*w^2 / Kq*w^2
             |
             v
         Isaac PhysX
```

Policy frequency is 60 Hz (`dt=1/240`, `decimation=4`). The conventional inner loop still runs every physics step.

## RL action mapping

- action 0 = 0 -> exactly hover thrust `mg`
- action 0 = -1 -> `0.30 mg`
- action 0 = +1 -> `2.50 mg`
- roll rate: +/-200 deg/s
- pitch rate: +/-200 deg/s
- yaw rate: +/-100 deg/s

RL does **not** command individual motors.

## Reward

Minimal first task:

- progress toward target
- +10 success bonus at 0.25 m radius
- -10 crash/out-of-bounds penalty
- small action penalty

Targets expand through a 3-stage curriculum.

## Quickest route

```powershell
cd E:\IsaacWork\Q250_UZH_Racing_v0.3.0
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\smoke_rl.ps1
.\train_rl.ps1 -NumEnvs 512 -MaxIterations 300
```

Watch training:

```powershell
.\tensorboard.ps1
```

Play newest checkpoint:

```powershell
.\play_rl.ps1
```

If 512 environments is too heavy, use 256. If GPU utilization is low and VRAM is comfortable, try 1024.

## Main new files

```text
q250_uzh/rl_control.py                 vectorized CTBR/PID/allocator
q250_uzh/fly_to_point_math.py          reward + curriculum math
q250_uzh/tasks/fly_to_point_env.py     Isaac DirectRLEnv
q250_uzh/agents/rsl_rl_ppo_cfg.py      PPO config
scripts/smoke_fly_to_point.py          environment smoke test
scripts/train_fly_to_point.py          RSL-RL training
scripts/play_fly_to_point.py           checkpoint playback
smoke_rl.ps1 / train_rl.ps1 / play_rl.ps1 / tensorboard.ps1
```

## Coordinate convention

Internal convention remains FLU / Z-up: +x forward, +y left, +z up. Do not mix it with PX4 FRD/NED without an explicit conversion layer.
