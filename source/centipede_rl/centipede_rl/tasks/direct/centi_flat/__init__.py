# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym
from . import agents


# Register Gym environments for the centi_flat task
gym.register(
    id="Template-Centi-Flat-Direct-v0",
    entry_point=f"{__name__}.centi_flat_env:CentiFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.centi_flat_env_cfg:CentiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
