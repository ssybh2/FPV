from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained Q250 Fly-to-Point policy")
    parser.add_argument("--checkpoint", type=str, default="", help="model_*.pt; empty = newest checkpoint")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds; <=0 runs until window closes")
    parser.add_argument("--real_time", action="store_true", help="pace simulation to wall-clock time")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from q250_uzh.agents.rsl_rl_ppo_cfg import Q250FlyToPointPPORunnerCfg
    from q250_uzh.checkpoints import find_latest_checkpoint
    from q250_uzh.tasks.fly_to_point_env import FlyToPointEnv, FlyToPointEnvCfg

    project_root = Path(__file__).resolve().parents[1]
    log_root = project_root / "logs" / "rsl_rl" / "q250_fly_to_point"
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else find_latest_checkpoint(log_root)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint found. Train first or pass --checkpoint. Searched: {log_root}"
        )
    checkpoint = checkpoint.resolve()

    env_cfg = FlyToPointEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.debug_vis = True
    if args.device is not None:
        env_cfg.sim.device = args.device

    agent_cfg = Q250FlyToPointPPORunnerCfg()
    if args.device is not None:
        agent_cfg.device = args.device

    env = FlyToPointEnv(env_cfg)
    env.sim.set_camera_view(eye=(5.0, 5.0, 4.0), target=(0.0, 0.0, 1.5))
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)

    print("\n=== Q250 Fly-to-Point PLAY ===")
    print(f"checkpoint : {checkpoint}")
    print("yellow/marker cube = current target")
    print("The target changes whenever the Q250 succeeds or an episode resets.\n")

    obs = wrapped_env.get_observations()
    elapsed = 0.0
    step_count = 0
    while simulation_app.is_running() and (args.duration <= 0.0 or elapsed < args.duration):
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = wrapped_env.step(actions)
        step_count += 1
        elapsed += env.step_dt

        if step_count % 30 == 0:
            dist = torch.linalg.norm(env._desired_pos_w - env._robot.data.root_pos_w, dim=-1)
            print(
                f"t={elapsed:6.2f}s  distance={float(dist[0]):5.2f}m  "
                f"z={float(env._robot.data.root_pos_w[0, 2]):5.2f}m  "
                f"action={actions[0].detach().cpu().numpy().round(3).tolist()}"
            )

        if args.real_time:
            sleep_s = env.step_dt - (time.time() - start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
