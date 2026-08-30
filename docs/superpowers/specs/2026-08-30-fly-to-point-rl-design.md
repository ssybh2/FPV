# Q250 Fly-to-Point RL Design

## Goal
Add the first reinforcement-learning task to the validated Q250 workspace: fly from a stable reset pose to randomized 3-D targets using a 4-D CTBR action that sits above the existing body-rate inner loop and identified motor/rigid-body dynamics.

## Architecture
The PPO policy sees 12 privileged state observations: target position in body frame (3), body linear velocity (3), projected gravity (3), and body angular velocity (3). It outputs normalized actions in [-1, 1]^4 which map to collective thrust and commanded body rates [T, p_cmd, q_cmd, r_cmd]. A vectorized GPU body-rate PID, vectorized X-quad allocator, and existing four-motor lag model run at every physics step before body wrench is applied to the Q250 rigid object.

## Timing
Physics runs at 240 Hz (dt=1/240 s), preserving the validated dynamics. Policy decimation is 4, so PPO acts at 60 Hz while the PID/motor model continue at 240 Hz.

## Action mapping
- action[0] = 0 maps exactly to hover thrust mg.
- action[0] = -1 maps to 0.30 mg.
- action[0] = +1 maps to 2.50 mg.
- roll and pitch rate commands: +/-200 deg/s.
- yaw rate command: +/-100 deg/s.

## Rewards and termination
Reward is deliberately minimal: progress toward target, one-time success bonus (because success terminates), crash/out-of-bounds penalty, and a small action penalty. Success radius is 0.25 m. Episode timeout is 8 s. Ground strike, excessive altitude, or excessive displacement from the environment origin terminate as failure.

## Curriculum
Target sampling expands with the DirectRLEnv common policy-step counter. Stage 0 uses nearby targets, Stage 1 expands the horizontal/vertical range, and Stage 2 uses the full Fly-to-Point volume. Reset pose remains level at local (0,0,1.5) for this first RL phase so the agent learns translational flight before harder reset randomization.

## Training workflow
Use RSL-RL PPO with an external-workspace training script rather than modifying Isaac Lab source. Default is 512 parallel environments and 300 iterations, with CLI overrides. Logs/checkpoints live under `logs/rsl_rl/q250_fly_to_point/`. A play script loads an explicit checkpoint or automatically finds the newest one.

## Non-goals for v0.3.0
No camera, VIO, gates, racing track, domain randomization, position PID, attitude PID, or Sim2Real. Those belong to later phases after Fly-to-Point works.
