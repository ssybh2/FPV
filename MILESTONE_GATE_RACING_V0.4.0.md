# Milestone: Q250 Three-Gate Racing v0.4.0

Date: 2026-08-30

## Status

Q250 Gate Racing curriculum completed.

Pipeline:

PPO Policy
    -> CTBR [Collective Thrust, Roll Rate, Pitch Rate, Yaw Rate]
    -> Body Rate PID
    -> Motor Allocator
    -> Identified Motor Dynamics
    -> Q250 PhysX Dynamics

## Curriculum

Stage 0:
- Single 3 x 3 m large gate

Stage 1:
- Single 1.5 x 1.5 m standard gate

Stage 2:
- Three consecutive 1.5 x 1.5 m gates

## Final Stage-2 performance

Selected checkpoint:

checkpoints/gate_racing/model_399.pt

Approximately:
- Full three-gate race success: ~59%
- Mean gate completion: ~80%
- Mean gates completed: ~2.39 / 3

The model has been visually validated in Isaac Sim.

## Current limitation

The policy only observes the current gate.
It cannot observe the next gate before crossing the current gate.

## Next milestone

v0.5.0 Look-Ahead Racing

Observation will include:
- current gate position
- current gate normal
- next gate position
- next gate normal
- body velocity
- projected gravity
- body angular velocity

Goal:
Learn anticipatory cornering and racing lines.
