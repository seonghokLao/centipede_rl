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
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # <-- replace with your robot


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
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(  # <-- replace with your robot cfg
        prim_path="/World/envs/env_.*/Robot"
    )

    # === Scene ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # === Task params (edit for your robot) ===
    # Names of controllable joints (example placeholders)
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # Action scaling (e.g., Nm or N depending on actuator)
    action_scale = 100.0

    # Reward scales (tune per task)
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

    # Reset/termination
    initial_pole_angle_range = [-0.25, 0.25]   # radians
    max_cart_pos = 3.0

