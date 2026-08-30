# Q250 UZH-style Drone Racing Workspace — v0.5.0

Windows + Isaac Lab workspace for a Q250 racing quadrotor. This release is the **Look-Ahead Racing** stage built on the validated v0.4.0 three-gate policy.

## Status

- Phase 0: Q250 physics + identified motor model + PhysX hover — validated
- Phase 1: body-rate PID + motor allocator — validated
- Phase 2: Fly-to-Point RL — validated (`model_200` milestone)
- Phase 3: three-gate racing — validated (`model_399` milestone)
- Phase 4: **Look-Ahead Racing — v0.5.0**

The physical Q250 model and CTBR inner-loop architecture are unchanged.

## Main v0.5.0 idea

v0.4.0 only saw the current gate. v0.5.0 sees both the current and next gate, plus their normals, so the policy can begin turning **before** crossing the current gate.

The old 12-D observation is preserved as the first 12 columns:

```text
0:3    current gate position in body frame
3:6    body linear velocity
6:9    projected gravity
9:12   body angular velocity
12:15  next gate position in body frame        NEW
15:18  current gate normal in body frame       NEW
18:21  next gate normal in body frame          NEW
```

Total: **21-D**.

Action remains the same 4-D CTBR command:

```text
[collective thrust, roll-rate cmd, pitch-rate cmd, yaw-rate cmd]
```

The control stack remains:

```text
21-D privileged observation
          |
          v
       PPO policy
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
identified motor lag + Kt*w^2/Kq*w^2
          |
          v
      Isaac PhysX Q250
```

## v0.4 -> v0.5 weight transfer

This release does **not** start from scratch.

`model_399.pt` has a 12-D actor and critic input. The transfer utility expands both first layers from:

```text
12 -> 128
```

to:

```text
21 -> 128
```

Transfer rule:

- old first 12 columns: copied exactly;
- new 9 columns: initialized to exactly zero;
- all compatible deeper actor/critic weights: copied exactly;
- PPO optimizer: fresh;
- exploration-noise parameter: intentionally reset to the lower v0.5 value instead of copying late v0.4 noise.

Therefore the new policy begins with nearly the same behavior as v0.4, then learns how to use the nine new look-ahead features.

## Curriculum

| Stage | Global policy steps | Task |
|---|---:|---|
| 0 | `<1000` | 3 vertical gates, 1.8 m opening; transfer adaptation |
| 1 | `1000..2999` | 3 gates, 1.5 m opening, yaw/pitch randomized |
| 2 | `>=3000` | 5 oriented 1.5 m gates, larger lateral/vertical variation |

Stage 0 intentionally resembles v0.4.0. Stage 1 introduces orientation. Stage 2 is the first five-gate racing task.

## Gate orientation

Each gate has a real plane basis:

- forward normal `n`
- horizontal/right axis `r`
- vertical/up axis `u`

A pass requires forward plane crossing and the Q250 center to lie inside the rotated rectangular opening. The authoritative pass rule is geometric; the orange frame is visualization.

## Reward changes

v0.5.0 keeps the stable v0.4 reward components and adds a small look-ahead shaping term near the current gate:

- progress to current gate
- gate pass bonus
- larger race-finish bonus
- crash/missed-gate penalty
- small action penalty
- stronger time pressure
- **small near-gate exit-velocity alignment toward the next gate**

The look-ahead term is deliberately small so it cannot make skipping the current gate profitable.

## Fast deployment

Extract to:

```text
E:\IsaacWork\Q250_UZH_Racing_v0.5.0
```

Then:

```powershell
cd E:\IsaacWork\Q250_UZH_Racing_v0.5.0
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

### 1. Import the validated v0.4 model_399

If your v0.4 workspace is still at the normal path, this is automatic:

```powershell
.\import_v04_checkpoint.ps1
```

If not:

```powershell
.\import_v04_checkpoint.ps1 -Source "E:\full\path\to\model_399.pt"
```

### 2. Smoke-test the new 21-D environment

```powershell
.\smoke_lookahead.ps1
```

Expected essentials:

```text
observation    : (32, 21)
action         : (32, 4)
finite tensors : True
unit normals   : True
```

### 3. Verify the 12D -> 21D weight transfer

```powershell
.\verify_transfer.ps1
```

The critical line is:

```text
input layers expanded : 2
```

or more. Two means both actor and critic input layers were expanded successfully.

### 4. Train

```powershell
.\train_lookahead.ps1 -NumEnvs 512 -MaxIterations 450
```

The transfer is automatic. The training script also saves `model_transfer_init.pt` before PPO modifies the transferred policy.

### 5. TensorBoard

```powershell
.\tensorboard.ps1
```

Watch, in priority order:

1. `Metrics/race_success_rate`
2. `Metrics/gate_completion_fraction`
3. `Metrics/missed_gate_rate`
4. `Metrics/episode_time_s`
5. `Metrics/gates_per_second`
6. `Metrics/mean_speed_m_s`
7. `Metrics/allocator_saturation_rate`
8. `Episode_Reward/lookahead`
9. `Curriculum/stage`

### 6. UI playback

Hardest stage, newest v0.5 checkpoint:

```powershell
.\play_lookahead.ps1 -Stage 2 -Duration 0 -RealTime
```

UI colors:

- orange frame = gates
- green cube = current gate
- cyan cube = next gate

The behavior to look for is not merely higher success. The key qualitative sign is **pre-turning**: before crossing the green gate, the Q250 should already begin shaping its velocity toward the cyan next gate.

## Recommended success criterion

For Stage 2 five-gate racing, do not expect the v0.4 three-gate success number immediately. A useful v0.5 milestone is:

- clear pre-turn / look-ahead behavior in UI;
- gate completion trending upward through Stage 2;
- five-gate race success becoming stable rather than collapsing;
- episode time or gates/second improving without a large rise in missed gates.

## Coordinate convention

Internal convention remains FLU / Z-up: +x forward, +y left, +z up. PX4 FRD/NED must be handled by an explicit conversion layer.
