# Windows quick start — v0.3.0 Fly-to-Point RL

Assumed Isaac Lab Python:

```text
E:\IsaacWork\env_isaaclab\python.exe
```

## 1. Extract

Recommended:

```text
E:\IsaacWork\Q250_UZH_Racing_v0.3.0
```

## 2. Install and run pure tests

```powershell
cd E:\IsaacWork\Q250_UZH_Racing_v0.3.0
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

The environment probe must find: `isaaclab`, `isaacsim`, `torch`, `isaaclab_rl`, and `rsl_rl`.

## 3. Fast RL environment smoke test

```powershell
.\smoke_rl.ps1
```

Expected essentials:

```text
observation  : (32, 12)
action       : (32, 4)
finite tensors : True
mean z       : close to 1.5 m for zero CTBR action
```

If this fails, stop here and send the complete terminal output.

## 4. Start PPO training

Recommended first run:

```powershell
.\train_rl.ps1 -NumEnvs 512 -MaxIterations 300
```

Faster debugging run:

```powershell
.\train_rl.ps1 -NumEnvs 128 -MaxIterations 20
```

More parallel environments if your GPU has room:

```powershell
.\train_rl.ps1 -NumEnvs 1024 -MaxIterations 300
```

Training is headless by default for speed. Logs/checkpoints appear in:

```text
logs\rsl_rl\q250_fly_to_point\<timestamp>\
```

## 5. TensorBoard

Open another PowerShell in the workspace:

```powershell
.\tensorboard.ps1
```

Then open `http://localhost:6006`.

Useful curves:

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Episode_Reward/progress`
- `Episode_Reward/success`
- `Episode_Reward/crash`
- `Metrics/final_distance_m`
- `Metrics/success_rate`
- `Curriculum/stage`

## 6. Play the newest model

```powershell
.\play_rl.ps1
```

The script automatically finds the newest `model_*.pt` checkpoint. To use an explicit checkpoint:

```powershell
.\play_rl.ps1 -Checkpoint "E:\IsaacWork\Q250_UZH_Racing_v0.3.0\logs\rsl_rl\q250_fly_to_point\2026-08-30_12-00-00\model_299.pt"
```

## 7. Existing validation tools remain available

```powershell
.\run_hover.ps1
.\run_rate_step.ps1 -Axis roll -Rate 100
.\run_rate_step.ps1 -Axis pitch -Rate 100
.\run_rate_step.ps1 -Axis yaw -Rate 100
```
