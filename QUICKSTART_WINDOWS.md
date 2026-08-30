# v0.5.0 Windows Quick Start

```powershell
cd E:\IsaacWork\Q250_UZH_Racing_v0.5.0
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\import_v04_checkpoint.ps1
.\smoke_lookahead.ps1
.\verify_transfer.ps1
.\train_lookahead.ps1 -NumEnvs 512 -MaxIterations 450
```

TensorBoard:

```powershell
.\tensorboard.ps1
```

Playback after training:

```powershell
.\play_lookahead.ps1 -Stage 2 -Duration 0 -RealTime
```

If `import_v04_checkpoint.ps1` cannot auto-find `model_399.pt`:

```powershell
.\import_v04_checkpoint.ps1 -Source "E:\IsaacWork\Q250_UZH_Racing_v0.4.0\logs\rsl_rl\q250_gate_racing\2026-08-30_14-43-08\model_399.pt"
```
