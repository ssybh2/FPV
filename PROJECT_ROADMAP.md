# Project roadmap

## Phase 0 — physics sanity — PASSED

Measured Q250 mass/inertia, geometry, motor/prop Kt/Kq + LUT, provisional motor lag, PhysX hover equilibrium.

## Phase 1 — conventional body-rate inner loop — PASSED

Body-rate PID, constrained motor allocation, motor dynamics, roll/pitch/yaw step-response validation.

## Phase 2 — Fly-to-Point DirectRLEnv — PASSED

The 300-iteration run reached Stage 2 and produced a visually strong policy around iteration 200. The task established that CTBR PPO on top of the conventional rate loop can command the identified Q250 model successfully.

## Phase 3 — Gate Racing Curriculum — CURRENT (v0.4.0)

Implemented:

- one large 3 x 3 m gate
- one standard 1.5 x 1.5 m gate
- sequential 3-gate track
- forward gate-plane crossing detector
- missed-opening termination
- current-gate-centered 12-D observation
- progress + gate + finish + crash + action + time reward
- gate/race completion TensorBoard metrics
- gate markers for UI playback
- lower PPO exploration than v0.3.0

Exit criterion: Stage 2 three-gate race success is consistently high and UI playback shows controlled sequential gate traversal.

## Phase 4 — true racing geometry

Next upgrade after v0.4.0 passes:

- expose current gate normal/orientation
- preview the next gate before current-gate crossing
- randomized gate yaw and pitch
- curved/zig-zag tracks
- physical gate-frame collision or contact-aware crash logic
- reward lap time / track progress rather than point-to-point stopping behavior

This is where anticipatory cornering and line choice should start to emerge.

## Phase 5 — high-speed + robustness

Increase CTBR/rate envelopes and randomize mass, inertia, Kt/Kq, motor lag, delay, drag, battery/thrust scale, sensor noise, and gate pose.

## Phase 6 — perception + Sim2Real

Camera + IMU, teacher/student or perception front-end, latency/noise model, PX4 NED/FRD conversion, hardware replay and real-flight validation.
