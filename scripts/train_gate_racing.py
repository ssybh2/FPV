from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Q250 Gate Racing with RSL-RL PPO")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if args.max_iterations <= 0:
        parser.error("--max_iterations must be positive")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from q250_uzh.agents.rsl_rl_ppo_cfg import Q250GateRacingPPORunnerCfg
    from q250_uzh.tasks.gate_racing_env import GateRacingEnv, GateRacingEnvCfg

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    project_root = Path(__file__).resolve().parents[1]
    env_cfg = GateRacingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device

    agent_cfg = Q250GateRacingPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed
    if args.device is not None:
        agent_cfg.device = args.device
    agent_cfg.run_name = args.run_name

    log_root = project_root / "logs" / "rsl_rl" / agent_cfg.experiment_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = log_root / (f"{stamp}_{args.run_name}" if args.run_name else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(run_dir)

    print("\n=== Q250 Gate Racing PPO v0.4.0 ===")
    print(f"environments      : {env_cfg.scene.num_envs}")
    print(f"physics rate      : {1.0 / env_cfg.sim.dt:.1f} Hz")
    print(f"policy rate       : {1.0 / (env_cfg.sim.dt * env_cfg.decimation):.1f} Hz")
    print(f"iterations        : {agent_cfg.max_iterations}")
    print(f"device            : {env_cfg.sim.device}")
    print(f"log directory     : {run_dir}")
    print("curriculum        : 3x3 single -> 1.5x1.5 single -> 3 gates")
    print("action            : [collective, p_cmd, q_cmd, r_cmd]")
    print("observation       : current_gate_b + lin_vel_b + gravity_b + ang_vel_b (12D)\n")

    env = GateRacingEnv(env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped_env,
        agent_cfg.to_dict(),
        log_dir=str(run_dir),
        device=agent_cfg.device,
    )

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training finished. Checkpoints are in: {run_dir}")

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
