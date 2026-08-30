from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Q250 v0.5.0 Look-Ahead Racing with PPO")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--transfer_checkpoint",
        type=str,
        default="checkpoints/gate_racing/model_399.pt",
        help="v0.4.0 12-D checkpoint used to initialize the 21-D actor/critic",
    )
    parser.add_argument("--no_transfer", action="store_true", help="debug only: start v0.5.0 from random weights")
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

    from q250_uzh.agents.rsl_rl_ppo_cfg import Q250LookAheadRacingPPORunnerCfg
    from q250_uzh.checkpoint_transfer import save_transfer_snapshot, transfer_checkpoint_to_runner
    from q250_uzh.tasks.lookahead_racing_env import LookAheadRacingEnv, LookAheadRacingEnvCfg

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    project_root = Path(__file__).resolve().parents[1]
    env_cfg = LookAheadRacingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device

    agent_cfg = Q250LookAheadRacingPPORunnerCfg()
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

    print("\n=== Q250 Look-Ahead Racing PPO v0.5.0 ===")
    print(f"environments      : {env_cfg.scene.num_envs}")
    print(f"physics rate      : {1.0 / env_cfg.sim.dt:.1f} Hz")
    print(f"policy rate       : {1.0 / (env_cfg.sim.dt * env_cfg.decimation):.1f} Hz")
    print(f"iterations        : {agent_cfg.max_iterations}")
    print(f"device            : {env_cfg.sim.device}")
    print(f"log directory     : {run_dir}")
    print("curriculum        : 3 vertical -> 3 oriented -> 5 oriented gates")
    print("action            : [collective, p_cmd, q_cmd, r_cmd]")
    print("observation       : old 12D prefix + next_gate_b + current_normal_b + next_normal_b = 21D")

    env = LookAheadRacingEnv(env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)

    if not args.no_transfer:
        source = Path(args.transfer_checkpoint).expanduser()
        if not source.is_absolute():
            source = project_root / source
        if not source.exists():
            raise FileNotFoundError(
                f"v0.4 checkpoint not found: {source}\n"
                "Run .\\import_v04_checkpoint.ps1 first, or pass --transfer_checkpoint explicitly."
            )
        report = transfer_checkpoint_to_runner(runner, source, old_obs_dim=12, new_obs_dim=21)
        print("=== v0.4 -> v0.5 transfer ===")
        print(f"source checkpoint     : {source.resolve()}")
        print(f"source iteration      : {report.source_iteration}")
        print(f"exact tensors copied  : {report.copied_exact}")
        print(f"input layers expanded : {report.expanded_input_layers}")
        print(f"noise tensors reset   : {report.skipped_noise}")
        print(f"kept new tensors      : {report.kept_new}")
        if report.expanded_input_layers < 2:
            raise RuntimeError("Expected actor and critic 12D->21D input layers to be expanded")
        # Snapshot the transferred initialization before PPO modifies it.
        save_transfer_snapshot(runner, run_dir / "model_transfer_init.pt", source=source)
        print("optimizer           : fresh v0.5 optimizer (not transferred)\n")
    else:
        print("[WARNING] --no_transfer selected: v0.5 starts from random weights.\n")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training finished. Checkpoints are in: {run_dir}")

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
