# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

# source/centipede_rl/centipede_rl/__init__.py

# ==== Isaac Lab task registry (for list_envs.py, etc.) ====
from .tasks.direct.centi_flat.centi_flat_env import CentipedeVelTrackEnv as _Env
from .tasks.direct.centi_flat.centi_flat_env_cfg import CentipedeVelTrackEnvCfg as _Cfg

TASK_REGISTRY = {
    # This name shows up in scripts/list_envs.py output
    "Centipede-Flat-Direct": (_Env, _Cfg)
}

# ==== Gymnasium registration (for RL trainers' Hydra lookup) ====
import gymnasium as gym
from gymnasium.error import NameNotFound

GYM_ID = "Centipede-Flat-Direct"  # <- use this in --task

try:
    gym.spec(GYM_ID)
except NameNotFound:
    gym.register(
        id=GYM_ID,
        entry_point="centipede_rl.tasks.direct.centi_flat.centi_flat_env:CentipedeVelTrackEnv",
        disable_env_checker=True,
        kwargs={
            # The trainer reads these to load your config via Hydra:
            "env_cfg_entry_point": "centipede_rl.tasks.direct.centi_flat.centi_flat_env_cfg:CentipedeVelTrackEnvCfg",
            # Optional: set an agent cfg; otherwise trainer uses its default PPO
            "rsl_rl_cfg_entry_point": "centipede_rl.tasks.direct.centi_flat.agents.rsl_rl_ppo_cfg:CentipedePPORunnerCfg",
        },
    )
