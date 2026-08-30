# Milestone: Q250 Fly-to-Point RL v0.3.0

Date: 2026-08-30

## Status

Fly-to-Point reinforcement learning phase completed successfully.

Architecture:

RL Policy
    -> CTBR command [Collective Thrust, Roll Rate, Pitch Rate, Yaw Rate]
    -> Body Rate PID
    -> Motor Allocator
    -> Identified Motor Dynamics
    -> Q250 PhysX Dynamics

## Q250 parameters

- Mass: 1.0006 kg
- Ix: 0.00517 kg m^2
- Iy: 0.00484 kg m^2
- Iz: 0.00750 kg m^2
- Wheelbase: 250 mm
- Motor layout: X
- Front-left / rear-right: CCW
- Front-right / rear-left: CW

## RL

Observation dimension: 12

- target position in body frame
- body linear velocity
- projected gravity
- body angular velocity

Action dimension: 4

- collective thrust
- roll rate command
- pitch rate command
- yaw rate command

Curriculum reached Stage 2.

Training completed for 300 PPO iterations.

Best observed training region:
approximately iteration 180-210.

Selected playback checkpoint:

checkpoints/fly_to_point/model_200.pt

The model_200 policy has been visually validated in Isaac Sim UI
and successfully performs aggressive 3-D fly-to-point maneuvers.

## Next milestone

Gate Racing:

Single large gate
    -> standard gate
    -> three-gate sequence
    -> randomized gates
    -> high-speed racing
