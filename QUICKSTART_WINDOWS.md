# Windows Quick Start — v0.4.0 Gate Racing

Assumed Isaac Lab Python:

```text
E:\IsaacWork\env_isaaclab\python.exe
```

## 1. Extract

Recommended directory:

```text
E:\IsaacWork\Q250_UZH_Racing_v0.4.0
```

## 2. Setup

```powershell
cd E:\IsaacWork\Q250_UZH_Racing_v0.4.0
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

## 3. Gate environment smoke test

```powershell
.\smoke_gate.ps1
```

Expected essentials:

```text
observation    : (32, 12)
action         : (32, 4)
finite tensors : True
```

Zero action is still hover-centered, so a 2-second smoke test should not produce a violent crash.

## 4. Train

```powershell
.\train_gate.ps1 -NumEnvs 512 -MaxIterations 400
```

Training is headless by default.

## 5. TensorBoard

In another PowerShell:

```powershell
.\tensorboard.ps1
```

Watch:

- `Metrics/race_success_rate`
- `Metrics/gate_completion_fraction`
- `Metrics/missed_gate_rate`
- `Metrics/allocator_saturation_rate`
- `Metrics/episode_time_s`
- `Curriculum/stage`

## 6. UI playback

Hardest three-gate stage:

```powershell
.\play_gate.ps1 -Stage 2 -Duration 0 -RealTime
```

One standard gate:

```powershell
.\play_gate.ps1 -Stage 1 -Duration 0 -RealTime
```

Use a specific checkpoint:

```powershell
.\play_gate.ps1 -Checkpoint "E:\path\to\model_200.pt" -Stage 2 -Duration 0 -RealTime
```

Orange = gate frames. Green cube = current gate center.
