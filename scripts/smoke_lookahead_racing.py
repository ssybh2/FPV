from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test Q250 v0.5.0 Look-Ahead Racing")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--duration", type=float, default=2.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from q250_uzh.lookahead_racing_math import lookahead_curriculum
    from q250_uzh.tasks.lookahead_racing_env import LookAheadRacingEnv, LookAheadRacingEnvCfg

    cfg = LookAheadRacingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    if args.device is not None:
        cfg.sim.device = args.device

    env = LookAheadRacingEnv(cfg)
    obs, _ = env.reset()
    actions = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    steps = max(1, int(args.duration / env.step_dt))
    stage = lookahead_curriculum(env.common_step_counter)

    print("\n=== Look-Ahead Racing environment smoke test ===")
    print(f"num_envs       : {env.num_envs}")
    print(f"observation    : {tuple(obs['policy'].shape)}")
    print(f"action         : {tuple(actions.shape)}")
    print(f"policy rate    : {1.0 / env.step_dt:.1f} Hz")
    print(f"curriculum     : stage {stage.stage}, gates={stage.gate_count}, size={stage.width_m:.1f}x{stage.height_m:.1f} m")

    for _ in range(steps):
        obs, reward, terminated, truncated, _ = env.step(actions)

    finite = torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
    normals_ok = torch.allclose(
        env._gate_normals_w.norm(dim=-1),
        torch.ones_like(env._gate_normals_w[..., 0]),
        atol=1e-4,
    )
    print(f"finite tensors : {bool(finite)}")
    print(f"unit normals   : {bool(normals_ok)}")
    print(f"mean z         : {float(env._robot.data.root_pos_w[:, 2].mean()):.3f} m")
    print(f"terminated     : {int(terminated.sum())}/{env.num_envs}")
    print(f"timed out      : {int(truncated.sum())}/{env.num_envs}")

    env.close()
    simulation_app.close()
    if not bool(finite) or not bool(normals_ok):
        raise RuntimeError("Look-ahead smoke test failed")


if __name__ == "__main__":
    main()
