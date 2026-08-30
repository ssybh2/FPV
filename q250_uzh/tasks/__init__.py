"""Gym registration for Q250 Isaac Lab tasks.

Import this module only after Isaac Sim/AppLauncher has been started.
"""

import gymnasium as gym


gym.register(
    id="Isaac-Q250-FlyToPoint-Direct-v0",
    entry_point="q250_uzh.tasks.fly_to_point_env:FlyToPointEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "q250_uzh.tasks.fly_to_point_env:FlyToPointEnvCfg",
        "rsl_rl_cfg_entry_point": "q250_uzh.agents.rsl_rl_ppo_cfg:Q250FlyToPointPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Q250-GateRacing-Direct-v0",
    entry_point="q250_uzh.tasks.gate_racing_env:GateRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "q250_uzh.tasks.gate_racing_env:GateRacingEnvCfg",
        "rsl_rl_cfg_entry_point": "q250_uzh.agents.rsl_rl_ppo_cfg:Q250GateRacingPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Q250-LookAheadRacing-Direct-v0",
    entry_point="q250_uzh.tasks.lookahead_racing_env:LookAheadRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "q250_uzh.tasks.lookahead_racing_env:LookAheadRacingEnvCfg",
        "rsl_rl_cfg_entry_point": "q250_uzh.agents.rsl_rl_ppo_cfg:Q250LookAheadRacingPPORunnerCfg",
    },
)
