# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# Interactive Robot Controller - Enhanced with keyboard control

import argparse
import random as rdom
from click import command
from isaaclab.app import AppLauncher
import math
import threading
import time
from utils.helper_funcs import nearest_equivalent_angle, alpha, get_F_tw1, get_F_tw2, F_Leg, F_leg_act, alpha_v, biased_angle_from_time, get_wheel_angle

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Interactive robot controller for Isaac Lab environment with keyboard control."
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

from centipede_rl.assets.robots.centipede_robot import CENTI_FLAT_ROBOT_CFG

# Global variables for control
current_command = "stop"
command_lock = threading.Lock()

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # robot
    Oleg = CENTI_FLAT_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Oleg")

def keyboard_listener():
    """Listen for keyboard input to control the robot."""
    global current_command
    
    print("\n" + "="*60)
    print("ROBOT CONTROL INTERFACE")
    print("="*60)
    print("Controls:")
    print("  w - Move Forward")
    print("  a - Turn Left") 
    print("  d - Turn Right")
    print("  s - Move Backward")
    print("  f - Self-Rolling Motion")
    print("  x - Stop/Rest")
    print("  q - Quit Simulation")
    print("  z - Side-Winding Left")
    print("  c - Side-Winding Right")
    print("  e - Lift Front Body")
    print("="*60)
    print("Enter command and press Enter:")
    
    command_mapping = {
        'w': 'forward',
        'a': 'left', 
        'd': 'right',
        's': 'backward',
        'f': 'self_right',
        'x': 'stop',
        'q': 'quit',
        'z': 'side_winding_left',
        'c': 'side_winding_right',
        'e': 'lift_front',
    }
    
    while True:
        try:
            user_input = input().strip().lower()
            if user_input in command_mapping:
                with command_lock:
                    current_command = command_mapping[user_input]
                
                if user_input == 'q':
                    print("Quitting simulation...")
                    break
                else:
                    print(f"Command: {command_mapping[user_input]}")
            else:
                print(f"Invalid input '{user_input}'. Use w/a/s/d/f/x/z/c/e/q")
        except (EOFError, KeyboardInterrupt):
            with command_lock:
                current_command = 'quit'
            break

def apply_motion_pattern(robot, command, sim_time, temporal_freq, wave_action, joint_pos_all, env_ind):
    """Apply the motion pattern based on the current command."""
    
    # Wave parameters for 5-leg robot
    wave_num = 1.3
    N = 5  # 5 legs
    xis = 1 - wave_num / 5
    lfs = xis
    num_leg = 5
    body_amp = np.radians(15)
    leg_amp = np.radians(35)
    v_amp = -np.radians(30)
    dutyf = 0.5
    
    # Shape basis functions (N-1 = 4 body segments)
    shape_basis1 = np.sin(np.arange(0, xis * (N - 1), xis) * 2 * np.pi)
    shape_basis2 = np.cos(np.arange(0, xis * (N - 1), xis) * 2 * np.pi)
    phase_max = lfs * np.pi + np.pi / 2
    
    # Get joint indices by name (correct way for your robot)
    leg_swing_names = ["leg_swing_joint1", "leg_swing_joint2", "leg_swing_joint3", 
                       "leg_swing_joint4", "leg_swing_joint5"]
    left_leg_names = ["l_leg_joint1", "l_leg_joint2", "l_leg_joint3", 
                      "l_leg_joint4", "l_leg_joint5"]
    right_leg_names = ["r_leg_joint1", "r_leg_joint2", "r_leg_joint3", 
                       "r_leg_joint4", "r_leg_joint5"]
    spine_names = ["body_h1", "body_h2", "body_h3", "body_h4"]
    vertical_names = ["body_v1", "body_v2", "body_v3", "body_v4"]
    
    leg_swing_ids = robot.find_joints(leg_swing_names)[0]
    left_leg_ids = robot.find_joints(left_leg_names)[0]
    right_leg_ids = robot.find_joints(right_leg_names)[0]
    spine_ids = robot.find_joints(spine_names)[0]
    vertical_ids = robot.find_joints(vertical_names)[0]
    
    if command == "stop":
        # Return to neutral position
        return robot.data.default_joint_pos

    elif command in ["backward", "forward", "left", "right", "lift_front"]:
        # Calculate wave motion
        F_tw1 = get_F_tw1(temporal_freq * sim_time, body_amp)
        F_tw2 = get_F_tw2(temporal_freq * sim_time, body_amp)
        phi = math.atan2(F_tw2, F_tw1) - phase_max
        alpha_array = alpha(F_tw1, shape_basis1, F_tw2, shape_basis2)
        vertical_alpha = alpha_v(temporal_freq * sim_time, v_amp, xis, N, 0)
        
        # Calculate leg pitch angles (beta)
        beta_array = np.zeros(num_leg)
        for ind in range(num_leg):
            time_offset = -temporal_freq * sim_time - phi + ind * lfs * 2 * np.pi
            beta_array[ind] = -F_Leg(leg_amp, time_offset, dutyf)
        
        # Calculate leg swing activations
        swing_array = np.zeros(num_leg)
        for ind in range(num_leg):
            time_offset = -temporal_freq * sim_time - phi + ind * lfs * 2 * np.pi
            swing_val = F_leg_act(time_offset, dutyf)
            swing_array[ind] = 0.4 if swing_val > 0 else -0.4
        
        # Apply leg swing activations
        for ind in range(num_leg):
            wave_action[env_ind, leg_swing_ids[ind]] = swing_array[ind]
        
        # Apply leg pitch (beta) to left and right legs
        for ind in range(num_leg):
            wave_action[env_ind, left_leg_ids[ind]] = beta_array[ind]
            wave_action[env_ind, right_leg_ids[ind]] = beta_array[ind]
        
        # Apply body horizontal (spine) motion
        for i in range(len(alpha_array)):
            wave_action[env_ind, spine_ids[i]] = alpha_array[i]
        
        # Apply body vertical motion
        for i in range(len(vertical_alpha)):
            wave_action[env_ind, vertical_ids[i]] = vertical_alpha[i]
        
        # Command-specific modifications
        if command == "backward":
            # Reverse horizontal body motion
            for i in range(len(alpha_array)):
                wave_action[env_ind, spine_ids[i]] = -alpha_array[i]
            
            # Recalculate leg motions with reversed phase
            for ind in range(num_leg):
                time_offset = temporal_freq * sim_time + phi + ind * lfs * 2 * np.pi
                beta_array[ind] = -F_Leg(leg_amp, time_offset, dutyf)
                swing_val = F_leg_act(time_offset, dutyf)
                swing_array[ind] = 0.4 if swing_val > 0 else -0.4
            
            # Reapply reversed leg motions
            for ind in range(num_leg):
                wave_action[env_ind, leg_swing_ids[ind]] = swing_array[ind]
                wave_action[env_ind, left_leg_ids[ind]] = beta_array[ind]
                wave_action[env_ind, right_leg_ids[ind]] = beta_array[ind]
        
        elif command == "left":
            # Turn left by biasing front body segments
            for i in range(min(3, len(spine_ids))):
                wave_action[env_ind, spine_ids[i]] = -0.7
        
        elif command == "right":
            # Turn right by biasing front body segments
            for i in range(min(3, len(spine_ids))):
                wave_action[env_ind, spine_ids[i]] = 0.7
        
        elif command == "lift_front":
            # Lift front body segment
            wave_action[env_ind, vertical_ids[0]] = -0.8
            
    elif command == "self_right":
        # Self-rolling motion
        for i in range(len(vertical_ids)):
            wave_action[env_ind, vertical_ids[i]] = 0.7 * np.sin(0.5 * temporal_freq * sim_time)
        for i in range(len(spine_ids)):
            wave_action[env_ind, spine_ids[i]] = 0.7 * np.cos(0.5 * temporal_freq * sim_time)

    elif command == "side_winding_left":
        for i in range(len(vertical_ids)):
            wave_action[env_ind, vertical_ids[i]] = 0.3 * np.cos(temporal_freq * sim_time)
        for i in range(len(spine_ids)):
            wave_action[env_ind, spine_ids[i]] = 0.3 * np.sin(temporal_freq * sim_time)

    elif command == "side_winding_right":
        for i in range(len(vertical_ids)):
            wave_action[env_ind, vertical_ids[i]] = 0.3 * np.cos(temporal_freq * sim_time)
        for i in range(len(spine_ids)):
            wave_action[env_ind, spine_ids[i]] = -0.3 * np.sin(temporal_freq * sim_time)
    
    return wave_action

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulation with interactive control."""
    global current_command
    
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot = scene["Oleg"]
    
    temporal_freq = 0.5 * 2 * np.pi  # 1 Hz
    env_num = args_cli.num_envs
    
    # Start keyboard listener in separate thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    print("[INFO]: Robot controller started. Use keyboard commands to control the robot.")
    
    while simulation_app.is_running():
        # Get current command safely
        with command_lock:
            command = current_command
        
        # Check for quit command
        if command == 'quit':
            break
        
        # Initialize wave action with default positions
        wave_action = robot.data.default_joint_pos.clone()
        joint_pos_all = robot.data.joint_pos
        
        # Apply motion for each environment
        for env_ind in range(env_num):
            wave_action = apply_motion_pattern(
                robot, command, sim_time, temporal_freq, 
                wave_action, joint_pos_all, env_ind
            )
        
        # Apply the action to the robot
        robot.set_joint_position_target(wave_action)
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
        
        # Optional: Print status every few seconds
        if count % 240 == 0:  # Every ~5 seconds at 48Hz
            print(f"Time: {sim_time:.1f}s, Current command: {command}")

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
    print("[INFO]: Interactive setup complete...")
    
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()