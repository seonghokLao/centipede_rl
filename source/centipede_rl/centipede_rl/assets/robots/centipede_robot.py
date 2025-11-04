# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""
Articulation configuration for your centi_flat robot loaded from USD.
Update the usd_path below to where you store `new_robot_full_v3.usd` in your repo.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Tip: place the USD somewhere under your repo (e.g., source/centipede_rl/assets/usd/)
# and set this path accordingly. During training, absolute paths are simplest.
USD_PATH = os.getenv(
    "CENTI_FLAT_USD",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "usd", "new_robot_full_v3.usd")),
)

CENTI_FLAT_ROBOT_CFG = ArticulationCfg(
    # Spawn from your USD file
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        # (Optional) override solver/rigid props if you want; otherwise USD values are used.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=5e-3,
            stabilization_threshold=1e-3,
        ),
    ),

    # Default initial state (world pose is set per-env in your RL env; this is the local state)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),          # lift a bit to avoid ground penetration; tune as needed
        rot=(1.0, 0.0, 0.0, 0.0),     # quaternion (w, x, y, z)
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        # You can pre-bias joint positions by name here if desired:
        # joint_pos={"joint_a": 0.0, "joint_b": 0.0}
        # joint_vel={"joint_a": 0.0, "joint_b": 0.0}
    ),

    # One actuator covering ALL actuated joints by regex. Start simple; split later if needed.
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],      # apply to every joint defined as actuated in the USD
            effort_limit_sim=400.0,       # simulation cap; tune for your hardware model
            stiffness=0.0,                # set >0.0 for (P) position/velocity tracking in implicit model
            damping=5.0,                  # viscous damping; tune per robot
            # You can also pass dicts to override per-joint gains if needed:
            # stiffness={"hip_1": 20.0, "knee_1": 10.0, ...},
            # damping={"hip_1": 1.0, "knee_1": 0.5, ...},
        ),
    },
)

