from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained Q250 v0.5.0 Look-Ahead Racing policy")
    parser.add_argument("--checkpoint", type=str, default="", help="model_*.pt; empty = newest v0.5 checkpoint")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=2)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _stage_to_counter(stage: int) -> int:
    return (0, 1200, 5000)[stage]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from q250_uzh.agents.rsl_rl_ppo_cfg import Q250LookAheadRacingPPORunnerCfg
    from q250_uzh.checkpoints import find_latest_checkpoint
    from q250_uzh.tasks.lookahead_racing_env import LookAheadRacingEnv, LookAheadRacingEnvCfg

    root = Path(__file__).resolve().parents[1]
    log_root = root / "logs" / "rsl_rl" / "q250_lookahead_racing"
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else find_latest_checkpoint(log_root)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"No v0.5 checkpoint found. Searched: {log_root}")
    checkpoint = checkpoint.resolve()

    cfg = LookAheadRacingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = True
    if args.device is not None:
        cfg.sim.device = args.device
    agent_cfg = Q250LookAheadRacingPPORunnerCfg()
    if args.device is not None:
        agent_cfg.device = args.device

    env = LookAheadRacingEnv(cfg)
    env.common_step_counter = _stage_to_counter(args.stage)
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    env.sim.set_camera_view(eye=(7.0, 12.0, 7.0), target=(6.0, 0.0, 1.7))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)

    print("\n=== Q250 Look-Ahead Racing PLAY v0.5.0 ===")
    print(f"checkpoint : {checkpoint}")
    print(f"stage      : {args.stage}")
    print("orange frame = gates; green = current gate; cyan = next gate")

    obs = wrapped.get_observations()
    elapsed = 0.0
    steps = 0
    while simulation_app.is_running() and (args.duration <= 0.0 or elapsed < args.duration):
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)
        elapsed += env.step_dt
        steps += 1
        if steps % 30 == 0:
            print(
                f"t={elapsed:6.2f}s gate={int(env._current_gate_idx[0])+1}/{int(env._gate_count[0])} "
                f"speed={float(torch.linalg.norm(env._robot.data.root_lin_vel_w[0])):5.2f}m/s "
                f"action={actions[0].detach().cpu().numpy().round(3).tolist()}"
            )
        if args.real_time:
            sleep_s = env.step_dt - (time.time() - start)
            if sleep_s > 0:
                time.sleep(sleep_s)

    wrapped.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
