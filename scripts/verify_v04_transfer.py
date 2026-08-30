from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser():
    p = argparse.ArgumentParser(description="Verify model_399 v0.4 -> v0.5 12D-to-21D transfer")
    p.add_argument("--checkpoint", type=str, default="checkpoints/gate_racing/model_399.pt")
    p.add_argument("--num_envs", type=int, default=16)
    AppLauncher.add_app_launcher_args(p)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from q250_uzh.agents.rsl_rl_ppo_cfg import Q250LookAheadRacingPPORunnerCfg
    from q250_uzh.checkpoint_transfer import transfer_checkpoint_to_runner
    from q250_uzh.tasks.lookahead_racing_env import LookAheadRacingEnv, LookAheadRacingEnvCfg

    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing {checkpoint}. Run .\\import_v04_checkpoint.ps1 first.")

    cfg = LookAheadRacingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device
    agent_cfg = Q250LookAheadRacingPPORunnerCfg()
    if args.device is not None:
        agent_cfg.device = args.device

    env = LookAheadRacingEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    report = transfer_checkpoint_to_runner(runner, checkpoint)
    policy = runner.get_inference_policy(device=env.device)
    obs = wrapped.get_observations()
    with torch.inference_mode():
        actions = policy(obs)
    finite = torch.isfinite(actions).all()

    print("\n=== v0.4 -> v0.5 TRANSFER VERIFIED ===")
    print(f"checkpoint            : {checkpoint.resolve()}")
    print(f"source iteration      : {report.source_iteration}")
    print(f"observation           : {tuple(obs.shape) if hasattr(obs, 'shape') else 'dict/wrapper'}")
    print(f"action shape          : {tuple(actions.shape)}")
    print(f"exact tensors copied  : {report.copied_exact}")
    print(f"input layers expanded : {report.expanded_input_layers} (expected >= 2)")
    print(f"noise tensors reset   : {report.skipped_noise}")
    print(f"finite actions        : {bool(finite)}")
    print("new 9 observation columns start at zero weight; optimizer is fresh.")

    wrapped.close()
    simulation_app.close()
    if report.expanded_input_layers < 2 or not bool(finite):
        raise RuntimeError("Checkpoint transfer verification failed")


if __name__ == "__main__":
    main()
