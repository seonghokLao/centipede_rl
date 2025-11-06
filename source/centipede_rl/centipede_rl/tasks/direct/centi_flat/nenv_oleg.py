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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
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


# --- OLEG_CONFIG: Custom robot configuration ---
OLEG_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/centipede_rl/centipede_rl/assets/usd/O_leg.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    articulation_root_prim_path="/body_module1/body_module1",
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_knee_joint1": 0.0,
            "left_knee_joint2": 0.0,
            "left_knee_joint3": 0.0,
            "left_knee_joint4": 0.0,
            "left_knee_joint5": 0.0,
            "left_knee_joint6": 0.0,

            "right_knee_joint1": 0.0,
            "right_knee_joint2": 0.0,
            "right_knee_joint3": 0.0,
            "right_knee_joint4": 0.0,
            "right_knee_joint5": 0.0,
            "right_knee_joint6": 0.0,
            "gear_motor_joint1": 0.0,
            "gear_motor_joint2": 0.0,
            "gear_motor_joint3": 0.0,
            "gear_motor_joint4": 0.0,   
            "gear_motor_joint5": 0.0,
            "between_bodies1": 0.0,
            "between_bodies2": 0.0, 
            "between_bodies3": 0.0,
            "between_bodies4": 0.0,
            "between_bodies5": 0.0,

        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "l1": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint1"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "l2": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint2"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "l3": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint3"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "l4": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint4"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "l5": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint5"],
            effort_limit_sim=1.5,               
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,        
        ),
        "l6": ImplicitActuatorCfg(
            joint_names_expr=["left_knee_joint6"],
            effort_limit_sim=1.5,
            velocity_limit_sim=100.0, 
            stiffness=10,
            damping=0.2,
        ),
        "r1": ImplicitActuatorCfg(
            joint_names_expr=["right_knee_joint1"],
            effort_limit_sim=1.5,       
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r2": ImplicitActuatorCfg(
            joint_names_expr=["right_knee_joint2"],
            effort_limit_sim=1.5,       
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r3": ImplicitActuatorCfg(
            joint_names_expr=["right_knee_joint3"],
            effort_limit_sim=1.5,       
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r4": ImplicitActuatorCfg(
            joint_names_expr=["right_knee_joint4"],     
            effort_limit_sim=1.5,   
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r5": ImplicitActuatorCfg(
            joint_names_expr=["right_knee_joint5"],     
            effort_limit_sim=1.5,   
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "r6": ImplicitActuatorCfg(
            joint_names_expr=["right_knee_joint6"],     
            effort_limit_sim=1.5,   
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "g1": ImplicitActuatorCfg(
            joint_names_expr=["gear_motor_joint1"],     
            effort_limit_sim=3.0,   
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),      
        "g2": ImplicitActuatorCfg(
            joint_names_expr=["gear_motor_joint2"],     
            effort_limit_sim=3.0,   
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "g3": ImplicitActuatorCfg(
            joint_names_expr=["gear_motor_joint3"],     
            effort_limit_sim=3.0,   
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "g4": ImplicitActuatorCfg(
            joint_names_expr=["gear_motor_joint4"],    
            effort_limit_sim=3.0,       
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "g5": ImplicitActuatorCfg(
            joint_names_expr=["gear_motor_joint5"],
            effort_limit_sim=3.0,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "b1": ImplicitActuatorCfg(
            joint_names_expr=["between_bodies1"],
            effort_limit_sim=3.0,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "b2": ImplicitActuatorCfg(
            joint_names_expr=["between_bodies2"],
            effort_limit_sim=3.0,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "b3": ImplicitActuatorCfg(
            joint_names_expr=["between_bodies3"],
            effort_limit_sim=3.0,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "b4": ImplicitActuatorCfg(
            joint_names_expr=["between_bodies4"],
            effort_limit_sim=3.0,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
        ),
        "b5": ImplicitActuatorCfg(
            joint_names_expr=["between_bodies5"],
            effort_limit_sim=3.0,
            velocity_limit_sim=100.0,
            stiffness=10,
            damping=0.2,
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
    Oleg = OLEG_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Oleg")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():

        # indices for legs on the left and right side from head to tail
        left_legs = np.array([2, 6, 10, 14, 18, 21])
        right_legs = np.array([1, 5, 9, 13, 17, 20])
        # indices for vertical and horizontal body joints from head to tail
        body_vertical = np.array([0, 4, 8, 12, 16])
        body_horizontal = np.array([3, 7, 11, 15, 19])

        wave_action = scene["Oleg"].data.default_joint_pos

        # wave_action[:, left_legs] = 3.14 * np.sin(2 * np.pi * 0.5 * sim_time)
        # wave_action[:, right_legs] = 3.14 * np.sin(2 * np.pi * 0.5 * sim_time)
        temporal_freq = 1.0 * 2 * np.pi  # 1 Hz
        # set sine wave to body joints
        wave_action[:, body_vertical] = 0.7 * np.sin(temporal_freq* sim_time)
        wave_action[:, body_horizontal] = 0.7 * np.sin(temporal_freq* sim_time + 1.57)
        scene["Oleg"].set_joint_position_target(wave_action)    

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
