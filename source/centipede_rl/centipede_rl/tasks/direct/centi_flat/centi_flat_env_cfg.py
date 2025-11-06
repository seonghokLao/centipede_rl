# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


# TODO: Replace this import with your robot's config, or build an ArticulationCfg below.
# Example if you already defined one somewhere:
# from my_project.assets.robots.centi_flat import CENTI_FLAT_ROBOT_CFG
# For now we keep CARTPOLE as a placeholder; swap it out before training.
from centipede_rl.assets.robots.centipede_robot import CENTI_FLAT_ROBOT_CFG


@configclass
class CentiFlatEnvCfg(DirectRLEnvCfg):
    # === Env timing ===
    decimation = 2
    episode_length_s = 5.0

    # === Spaces ===
    # Update these to match your robot’s action/obs sizes.
    action_space = 1
    observation_space = 4
    state_space = 0

    # === Simulation ===
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # === Robot ===
    # Point to your robot config and prim path where each env’s robot will spawn.
    robot_cfg: ArticulationCfg = CENTI_FLAT_ROBOT_CFG.replace(  # <-- replace with your robot cfg
        prim_path="/World/envs/env_.*/new_robot_full_v3",
    )

    print(robot_cfg.prim_path)

    # === Scene ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=4.0,
        replicate_physics=True,
    )

    leg_act_joint_names   = [
        "leg_swing_joint1", "leg_swing_joint2",
        "leg_swing_joint3", "leg_swing_joint4",
        "leg_swing_joint5"
    ]
    leg_pitch_joint_names = [
        "l_leg_joint1", "r_leg_joint1",
        "l_leg_joint2", "r_leg_joint2",
        "l_leg_joint3", "r_leg_joint3",
        "l_leg_joint4", "r_leg_joint4",
        "l_leg_joint5", "r_leg_joint5",
    ]
    spine_joint_names     = [
        "body_h1", "body_h2",
        "body_h3", "body_h4",
        "body_h5"
    ]
    vertical_joint_names  = [
        "body_v1", "body_v2",
        "body_v3", "body_v4",
        "body_v5"
    ]

    # Parameter limits for clamping
    wave_num_range = (0.8, 1.5)    # number of spatial waves along the body
    body_amp_range = (0.0, 0.9)    # [rad] spine yaw/pitch target amplitude
    leg_amp_range  = (0.0, 0.8)    # [rad] leg swing amplitude
    v_amp_range    = (0.0, 0.7)    # [Hz] temporal frequency (~ wave speed)

    # === Task params (edit for your robot) ===
    # Names of controllable joints (example placeholders)
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    action_scales = (0.05, 0.05, 0.05, 0.1)

    # Desired-velocity sampling
    target_vel_range = (0.0, 1.0)  # [m/s] sample per env at reset

    # Target position-control stiffness/damping for joints
    kp_body = 20.0
    kd_body = 2.0
    kp_leg  = 30.0
    kd_leg  = 3.0

    # Reward weights
    rew_w_alive = 1.0
    rew_w_vel   = 5.0   # –|vx - v_des|
    rew_w_smooth= -0.001 # –||Δaction||
    rew_w_eff   = -0.0005 # –||qdot||
    rew_w_flat  = 0.5   # encourage keeping base level (optional)

    # Reset/termination
    max_roll_pitch = 0.9

