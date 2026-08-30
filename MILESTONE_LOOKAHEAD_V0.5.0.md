# v0.5.0 Look-Ahead Racing

Source milestone: v0.4.0 `model_399.pt`.

This release expands the observation from 12D to 21D while retaining the old 12D feature order. Actor and critic first layers are expanded from 12 to 21 inputs; old columns are copied exactly and the nine new columns start at zero. Deeper compatible weights are copied. PPO optimizer and exploration noise are fresh.

New information:

- next gate position in body frame
- current gate normal in body frame
- next gate normal in body frame

Training curriculum:

1. 3 vertical 1.8 m gates (transfer adaptation)
2. 3 oriented 1.5 m gates
3. 5 oriented 1.5 m gates

Goal: learn anticipatory cornering and racing-line behavior instead of waiting until a gate is crossed before reacting to the next one.

## Selected milestone checkpoint

Selected checkpoint:

checkpoints/lookahead_racing/model_449.pt

Training source:
- v0.4.0 model_399.pt
- transferred from 12D observation to 21D observation

The checkpoint has been visually replayed in Isaac Sim and the result is acceptable.

Stage 2 task:
- 5 oriented gates
- current gate look-ahead
- next gate position
- current and next gate normals
- anticipatory cornering / racing-line learning

This checkpoint is selected as the v0.5.0 milestone model.
