# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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
        "h5": ImplicitActuatorCfg(
            joint_names_expr=["body_h5"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
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
        "v5": ImplicitActuatorCfg(
            joint_names_expr=["body_v5"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
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



class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    centi = CENTI_FLAT_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/new_robot_full_v3")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_centi_state = scene["centi"].data.default_root_state.clone()
            root_centi_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the centi's orientation and velocity
            scene["centi"].write_root_pose_to_sim(root_centi_state[:, :7])
            scene["centi"].write_root_velocity_to_sim(root_centi_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["centi"].data.default_joint_pos.clone(),
                scene["centi"].data.default_joint_vel.clone(),
            )
            scene["centi"].write_joint_state_to_sim(joint_pos, joint_vel)
            joint_pos, joint_vel = (
                scene["Dofbot"].data.default_joint_pos.clone(),
                scene["Dofbot"].data.default_joint_vel.clone(),
            )
            scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting centi state...")

        # drive around
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])

        scene["centi"].set_joint_velocity_target(action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()