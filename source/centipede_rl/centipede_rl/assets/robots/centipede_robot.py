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
CENTI_FLAT_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= "./source/centipede_rl/centipede_rl/assets/ten_leg/new_robot_full_v3.usd",
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
    articulation_root_prim_path="/swing_base1/swing_base1",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),

    # Split actuators per group name. Tune gains as needed.
    actuators={
        # Spine (alpha)
        "h1": ImplicitActuatorCfg(
            joint_names_expr=["body_h1"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "h2": ImplicitActuatorCfg(
            joint_names_expr=["body_h2"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "h3": ImplicitActuatorCfg(
            joint_names_expr=["body_h3"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "h4": ImplicitActuatorCfg(
            joint_names_expr=["body_h4"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        # "h5": ImplicitActuatorCfg(
        #     joint_names_expr=["body_h5"],
        #     effort_limit_sim=1.5,
        #     velocity_limit_sim=100.0,
        #     stiffness=10,
        #     damping=0.2,
        # ),
        # Vertical (alpha_v)
        "v1": ImplicitActuatorCfg(
            joint_names_expr=["body_v1"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "v2": ImplicitActuatorCfg(
            joint_names_expr=["body_v2"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "v3": ImplicitActuatorCfg(
            joint_names_expr=["body_v3"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "v4": ImplicitActuatorCfg(
            joint_names_expr=["body_v4"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        # "v5": ImplicitActuatorCfg(
        #     joint_names_expr=["body_v5"],
        #     effort_limit_sim=1.5,
        #     velocity_limit_sim=100.0,
        #     stiffness=10,
        #     damping=0.2,
        # ),
        # Leg pitch L/R (beta)
        "l1": ImplicitActuatorCfg(
            joint_names_expr=["l_leg_joint1"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "l2": ImplicitActuatorCfg(
            joint_names_expr=["l_leg_joint2"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "l3": ImplicitActuatorCfg(
            joint_names_expr=["l_leg_joint3"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "l4": ImplicitActuatorCfg(
            joint_names_expr=["l_leg_joint4"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "l5": ImplicitActuatorCfg(
            joint_names_expr=["l_leg_joint5"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r1": ImplicitActuatorCfg(
            joint_names_expr=["r_leg_joint1"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r2": ImplicitActuatorCfg(
            joint_names_expr=["r_leg_joint2"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r3": ImplicitActuatorCfg(
            joint_names_expr=["r_leg_joint3"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r4": ImplicitActuatorCfg(
            joint_names_expr=["r_leg_joint4"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r5": ImplicitActuatorCfg(
            joint_names_expr=["r_leg_joint5"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        # Leg swing “act” (stance/swing)
        "s1": ImplicitActuatorCfg(
            joint_names_expr=["leg_swing_joint1"],
            effort_limit_sim=150.0,
            stiffness=15.0,
            damping=2.0,
        ),
        "s2": ImplicitActuatorCfg(
            joint_names_expr=["leg_swing_joint2"],
            effort_limit_sim=150.0,
            stiffness=15.0,
            damping=2.0,
        ),
        "s3": ImplicitActuatorCfg(
            joint_names_expr=["leg_swing_joint3"],
            effort_limit_sim=150.0,
            stiffness=15.0,
            damping=2.0,
        ),
        "s4": ImplicitActuatorCfg(
            joint_names_expr=["leg_swing_joint4"],
            effort_limit_sim=150.0,
            stiffness=15.0,
            damping=2.0,
        ),
        "s5": ImplicitActuatorCfg(
            joint_names_expr=["leg_swing_joint5"],
            effort_limit_sim=150.0,
            stiffness=15.0,
            damping=2.0,
        ),
    },
)
