from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the Q250 Fly-to-Point DirectRLEnv")
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
    from q250_uzh.tasks.fly_to_point_env import FlyToPointEnv, FlyToPointEnvCfg

    cfg = FlyToPointEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    if args.device is not None:
        cfg.sim.device = args.device

    env = FlyToPointEnv(cfg)
    obs, _ = env.reset()
    actions = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    steps = max(1, int(args.duration / env.step_dt))

    print("\n=== Fly-to-Point environment smoke test ===")
    print(f"num_envs     : {env.num_envs}")
    print(f"observation  : {tuple(obs['policy'].shape)}")
    print(f"action       : {tuple(actions.shape)}")
    print(f"policy rate  : {1.0 / env.step_dt:.1f} Hz")

    for _ in range(steps):
        obs, reward, terminated, truncated, _ = env.step(actions)

    distance = torch.linalg.norm(env._desired_pos_w - env._robot.data.root_pos_w, dim=-1)
    finite = torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
    print(f"finite tensors : {bool(finite)}")
    print(f"mean distance  : {float(distance.mean()):.3f} m")
    print(f"mean z         : {float(env._robot.data.root_pos_w[:, 2].mean()):.3f} m")
    print(f"terminated     : {int(terminated.sum())}/{env.num_envs}")
    print(f"timed out      : {int(truncated.sum())}/{env.num_envs}")

    env.close()
    simulation_app.close()
    if not bool(finite):
        raise RuntimeError("Smoke test produced NaN/Inf values")


if __name__ == "__main__":
    main()
