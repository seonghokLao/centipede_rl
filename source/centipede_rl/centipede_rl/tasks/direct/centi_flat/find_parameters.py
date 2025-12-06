# Copyright (c) 2022-2025
# Simplified parameter sweep focusing on actuator parameters only
# Fixed CPG parameters: wave_num=1.3, body_amp=20°, leg_amp=35°, v_amp=-20°

import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

parser = argparse.ArgumentParser(description="Sweep stiffness and damping parameters")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--test_duration", type=float, default=8.0, help="Test duration per parameter set (seconds)")
parser.add_argument("--output_dir", type=str, default="./param_sweep_results", help="Output directory for results")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import math

# Create output directory
os.makedirs(args_cli.output_dir, exist_ok=True)

# FIXED CPG PARAMETERS (from your specification)
WAVE_NUM = 1.3
BODY_AMP = np.radians(20)
LEG_AMP = np.radians(35)
V_AMP = -np.radians(20)

# Parameter ranges to sweep (actuator parameters only)
# Format: (body_stiffness, body_damping, leg_stiffness, leg_damping, effort_limit)
PARAM_GRID = [
    # Baseline variants with different stiffness
    (4, 1.0, 4, 1.0, 1.5),    # Current baseline
    (6, 1.0, 6, 1.0, 2.0),    # Medium stiffness
    (8, 1.0, 8, 1.0, 2.5),    # Higher stiffness
    (10, 1.0, 10, 1.0, 3.0),  # High stiffness
    (12, 1.0, 12, 1.0, 3.0),  # Very high stiffness
    
    # Different damping ratios
    (8, 0.5, 8, 0.5, 2.5),    # Low damping
    (8, 1.5, 8, 1.5, 2.5),    # High damping
    (8, 2.0, 8, 2.0, 2.5),    # Very high damping
    
    # Asymmetric: stiffer body, softer legs
    (12, 1.0, 6, 1.0, 2.5),   # Stiff body, soft legs
    (10, 1.5, 6, 0.8, 2.5),   # Stiff damped body, soft legs
    
    # Asymmetric: softer body, stiffer legs
    (6, 1.0, 12, 1.0, 3.0),   # Soft body, stiff legs
    (6, 0.8, 10, 1.2, 3.0),   # Soft body, stiff damped legs
    
    # Higher effort variants
    (8, 1.0, 8, 1.0, 3.5),    # High effort
    (10, 1.0, 12, 1.0, 4.0),  # Very high effort
    (12, 1.5, 12, 1.5, 4.0),  # Max stiffness + high effort
    
    # Fine-tuned promising configs
    (8, 0.8, 10, 1.0, 3.0),   # Balanced
    (10, 1.2, 10, 1.0, 3.0),  # Slightly damped body
    (6, 1.0, 10, 0.8, 2.5),   # Flexible body, responsive legs
]


def create_robot_config(body_stiffness, body_damping, leg_stiffness, leg_damping, effort_limit):
    """Create robot configuration with specific actuator parameters"""
    
    ROBOT_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="./source/centipede_rl/centipede_rl/assets/ten_leg/new_robot_full_v3.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,  # Increased for stability
                solver_velocity_iteration_count=1,  # Added for better contact
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
        actuators={
            # Spine (horizontal) - body parameters
            "h1": ImplicitActuatorCfg(
                joint_names_expr=["body_h1"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=body_stiffness,
                damping=body_damping,
            ),
            "h2": ImplicitActuatorCfg(
                joint_names_expr=["body_h2"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=body_stiffness,
                damping=body_damping,
            ),
            "h3": ImplicitActuatorCfg(
                joint_names_expr=["body_h3"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=body_stiffness,
                damping=body_damping,
            ),
            "h4": ImplicitActuatorCfg(
                joint_names_expr=["body_h4"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=body_stiffness,
                damping=body_damping,
            ),
            # Vertical - body parameters
            "v1": ImplicitActuatorCfg(
                joint_names_expr=["body_v1"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=body_stiffness,
                damping=body_damping,
            ),
            "v2": ImplicitActuatorCfg(
                joint_names_expr=["body_v2"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=body_stiffness,
                damping=body_damping,
            ),
            "v3": ImplicitActuatorCfg(
                joint_names_expr=["body_v3"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=body_stiffness,
                damping=body_damping,
            ),
            "v4": ImplicitActuatorCfg(
                joint_names_expr=["body_v4"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=body_stiffness,
                damping=body_damping,
            ),
            # Leg pitch L/R - leg parameters
            "l1": ImplicitActuatorCfg(
                joint_names_expr=["l_leg_joint1"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "l2": ImplicitActuatorCfg(
                joint_names_expr=["l_leg_joint2"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "l3": ImplicitActuatorCfg(
                joint_names_expr=["l_leg_joint3"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "l4": ImplicitActuatorCfg(
                joint_names_expr=["l_leg_joint4"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "l5": ImplicitActuatorCfg(
                joint_names_expr=["l_leg_joint5"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "r1": ImplicitActuatorCfg(
                joint_names_expr=["r_leg_joint1"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "r2": ImplicitActuatorCfg(
                joint_names_expr=["r_leg_joint2"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "r3": ImplicitActuatorCfg(
                joint_names_expr=["r_leg_joint3"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "r4": ImplicitActuatorCfg(
                joint_names_expr=["r_leg_joint4"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            "r5": ImplicitActuatorCfg(
                joint_names_expr=["r_leg_joint5"],
                effort_limit_sim=effort_limit,
                velocity_limit_sim=50,
                stiffness=leg_stiffness,
                damping=leg_damping,
            ),
            # Leg swing - higher stiffness for ground contact
            "s1": ImplicitActuatorCfg(
                joint_names_expr=["leg_swing_joint1"],
                effort_limit_sim=effort_limit,
                stiffness=leg_stiffness * 1.5,  # 50% higher for swing
                damping=leg_damping * 0.5,
            ),
            "s2": ImplicitActuatorCfg(
                joint_names_expr=["leg_swing_joint2"],
                effort_limit_sim=effort_limit,
                stiffness=leg_stiffness * 1.5,
                damping=leg_damping * 0.5,
            ),
            "s3": ImplicitActuatorCfg(
                joint_names_expr=["leg_swing_joint3"],
                effort_limit_sim=effort_limit,
                stiffness=leg_stiffness * 1.5,
                damping=leg_damping * 0.5,
            ),
            "s4": ImplicitActuatorCfg(
                joint_names_expr=["leg_swing_joint4"],
                effort_limit_sim=effort_limit,
                stiffness=leg_stiffness * 1.5,
                damping=leg_damping * 0.5,
            ),
            "s5": ImplicitActuatorCfg(
                joint_names_expr=["leg_swing_joint5"],
                effort_limit_sim=effort_limit,
                stiffness=leg_stiffness * 1.5,
                damping=leg_damping * 0.5,
            ),
        },
    )
    
    return ROBOT_CFG


class ParamSweepSceneCfg(InteractiveSceneCfg):
    """Scene configuration for parameter sweep"""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


def compute_cpg_motion(t, num_envs, device):
    """CPG motion computation with FIXED parameters"""
    N = 5
    num_legs = 5
    
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t] * num_envs, device=device)
    
    wave_num_t = torch.tensor([WAVE_NUM] * num_envs, device=device)
    body_amp_t = torch.tensor([BODY_AMP] * num_envs, device=device)
    leg_amp_t = torch.tensor([LEG_AMP] * num_envs, device=device)
    v_amp_t = torch.tensor([V_AMP] * num_envs, device=device)
    
    xis = 1.0 - wave_num_t / 5.0
    lfs = xis
    dutyf = 0.5
    
    idx = torch.arange(N-1, device=device, dtype=torch.float32)
    phase_idx = idx.unsqueeze(0) * xis.unsqueeze(1)
    sb1 = torch.sin(phase_idx * 2.0 * math.pi)
    sb2 = torch.cos(phase_idx * 2.0 * math.pi)
    
    t_mod = torch.zeros_like(wave_num_t)
    ramp_time = 1.0
    over = t > ramp_time
    t_mod[over] = 2.0 * (t[over] - ramp_time + (math.pi / 2.0))
    
    ramp_mult = torch.clamp(t / ramp_time, 0.0, 1.0)
    
    F1 = body_amp_t * torch.cos(t_mod)
    F2 = body_amp_t * torch.sin(t_mod)
    
    alpha_array = (F1.unsqueeze(1) * sb1 + F2.unsqueeze(1) * sb2) * ramp_mult.unsqueeze(1)
    
    idx_v = torch.arange(N-1, device=device, dtype=torch.float32)
    x_vals = idx_v.unsqueeze(0) * xis.unsqueeze(1) * 2.0 * math.pi
    x_vals = x_vals + t_mod.unsqueeze(1)
    vertical_alpha = v_amp_t.unsqueeze(1) * torch.cos(2.0 * x_vals + math.pi / 2.0) * ramp_mult.unsqueeze(1)
    
    phase_max = lfs * math.pi + math.pi / 2.0
    phi = torch.atan2(F2, F1) - phase_max
    
    idx_leg = torch.arange(num_legs, device=device, dtype=torch.float32)
    times = phi.unsqueeze(1) + idx_leg.unsqueeze(0) * lfs.unsqueeze(1) * 2.0 * math.pi
    two_pi = 2.0 * math.pi
    time_mod = torch.remainder(times, two_pi)
    on = time_mod < (two_pi * dutyf)
    act_vals = torch.where(
        on,
        torch.full_like(time_mod, 0.4),
        torch.full_like(time_mod, -0.4),
    ) * ramp_mult.unsqueeze(1)
    
    leg_amp_exp = leg_amp_t.view(-1, *([1] * (time_mod.ndim - 1)))
    cos_on = torch.cos(time_mod / (2.0 * dutyf))
    cos_off = torch.cos((time_mod - two_pi) / (2.0 * (1.0 - dutyf)))
    beta_array = -torch.where(on, leg_amp_exp * cos_on, leg_amp_exp * cos_off)
    
    beta_LR = beta_array.unsqueeze(2).repeat(1, 1, 2).reshape(num_envs, 2 * num_legs) * ramp_mult.unsqueeze(1)
    
    return {
        'leg_act': act_vals,
        'leg_pitch': beta_LR,
        'spine': alpha_array,
        'vertical': vertical_alpha
    }


def test_single_config(scene, robot, body_stiff, body_damp, leg_stiff, leg_damp, effort, duration):
    """Test a single parameter configuration"""
    device = robot.device
    num_envs = robot.num_instances
    
    # Get joint indices
    leg_act_names = [f"leg_swing_joint{i}" for i in range(1, 6)]
    leg_pitch_names = [f"{side}_leg_joint{i}" for i in range(1, 6) for side in ['l', 'r']]
    spine_names = [f"body_h{i}" for i in range(1, 5)]
    vertical_names = [f"body_v{i}" for i in range(1, 5)]
    
    leg_act_ids = torch.tensor(robot.find_joints(leg_act_names)[0], device=device)
    leg_pitch_ids = torch.tensor(robot.find_joints(leg_pitch_names)[0], device=device)
    spine_ids = torch.tensor(robot.find_joints(spine_names)[0], device=device)
    vertical_ids = torch.tensor(robot.find_joints(vertical_names)[0], device=device)
    
    ids_all = torch.cat([leg_act_ids, leg_pitch_ids, spine_ids, vertical_ids])
    
    # Reset robot
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    
    # Run simulation
    sim_dt = scene._sim.get_physics_dt()
    sim_time = 0.0
    
    # Tracking metrics
    velocities = []
    positions = []
    torques = []
    joint_vels = []
    contact_forces = []
    
    steps = int(duration / sim_dt)
    
    for step in range(steps):
        # Compute CPG motion (FIXED parameters)
        cpg_output = compute_cpg_motion(sim_time, num_envs, device)
        
        # Assemble joint targets
        q_targets = torch.cat([
            cpg_output['leg_act'],
            cpg_output['leg_pitch'],
            cpg_output['spine'],
            cpg_output['vertical']
        ], dim=1)
        
        # Apply action
        robot.set_joint_position_target(q_targets, joint_ids=ids_all)
        
        # Step simulation
        scene.write_data_to_sim()
        scene._sim.step()
        scene.update(sim_dt)
        
        sim_time += sim_dt
        
        # Record metrics (after ramp-up period)
        if sim_time > 1.5:
            velocities.append(robot.data.root_lin_vel_w[:, 0].cpu().numpy())
            positions.append(robot.data.root_pos_w[:, 0].cpu().numpy())
            torques.append(torch.norm(robot.data.applied_torque, dim=1).mean().item())
            joint_vels.append(torch.norm(robot.data.joint_vel, dim=1).mean().item())
    
    # Compute statistics
    velocities = np.array(velocities)
    positions = np.array(positions)
    
    results = {
        'body_stiffness': body_stiff,
        'body_damping': body_damp,
        'leg_stiffness': leg_stiff,
        'leg_damping': leg_damp,
        'effort_limit': effort,
        'mean_velocity': float(np.mean(velocities)),
        'std_velocity': float(np.std(velocities)),
        'max_velocity': float(np.max(velocities)),
        'min_velocity': float(np.min(velocities)),
        'final_position': float(np.mean(positions[-1])),
        'distance_traveled': float(np.mean(positions[-1] - positions[0])),
        'mean_torque': float(np.mean(torques)),
        'mean_joint_vel': float(np.mean(joint_vels)),
    }
    
    return results


def sweep_parameters():
    """Main sweep function - tests all configs sequentially"""
    all_results = []
    total_tests = len(PARAM_GRID)
    
    print(f"\n{'='*80}")
    print(f"ACTUATOR PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"Fixed CPG Parameters:")
    print(f"  wave_num = {WAVE_NUM}")
    print(f"  body_amp = {np.degrees(BODY_AMP):.1f}°")
    print(f"  leg_amp = {np.degrees(LEG_AMP):.1f}°")
    print(f"  v_amp = {np.degrees(V_AMP):.1f}°")
    print(f"\nTesting {total_tests} actuator configurations...")
    print(f"{'='*80}\n")
    
    for test_idx, (body_stiff, body_damp, leg_stiff, leg_damp, effort) in enumerate(PARAM_GRID):
        print(f"[{test_idx+1}/{total_tests}] Testing Configuration:")
        print(f"  Body: K={body_stiff}, D={body_damp}")
        print(f"  Legs: K={leg_stiff}, D={leg_damp}")
        print(f"  Effort: {effort} Nm")
        
        # Create robot config with these parameters
        robot_cfg = create_robot_config(body_stiff, body_damp, leg_stiff, leg_damp, effort)
        robot_cfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Create scene
        scene_cfg = ParamSweepSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0)
        scene_cfg.robot = robot_cfg
        
        # Setup simulation
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/120)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        
        # Get robot and test
        robot = scene["robot"]
        
        result = test_single_config(
            scene, robot, body_stiff, body_damp, leg_stiff, leg_damp, effort, args_cli.test_duration
        )
        
        all_results.append(result)
        
        print(f"  → Velocity: {result['mean_velocity']:.4f} ± {result['std_velocity']:.4f} m/s")
        print(f"  → Distance: {result['distance_traveled']:.4f} m")
        print(f"  → Torque: {result['mean_torque']:.4f} Nm\n")
        
        # Clear simulation for next config
        sim.clear_all_callbacks()
        sim.clear_instance()
    
    return all_results


def analyze_and_plot_results(results):
    """Analyze results and create visualizations"""
    
    # Sort by velocity
    sorted_results = sorted(results, key=lambda x: x['mean_velocity'], reverse=True)
    
    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS BY VELOCITY:")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Body K':<8} {'Body D':<8} {'Leg K':<8} {'Leg D':<8} {'Effort':<8} {'Vel (m/s)':<12} {'Dist (m)':<10}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1:<6} {r['body_stiffness']:<8} {r['body_damping']:<8.1f} "
              f"{r['leg_stiffness']:<8} {r['leg_damping']:<8.1f} {r['effort_limit']:<8.1f} "
              f"{r['mean_velocity']:<12.4f} {r['distance_traveled']:<10.4f}")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args_cli.output_dir, f"sweep_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Create visualization
    create_plots(sorted_results, timestamp)
    
    return sorted_results[0]


def create_plots(results, timestamp):
    """Create comprehensive visualization plots"""
    fig = plt.figure(figsize=(20, 12))
    
    # Extract data arrays
    body_stiff = np.array([r['body_stiffness'] for r in results])
    leg_stiff = np.array([r['leg_stiffness'] for r in results])
    effort = np.array([r['effort_limit'] for r in results])
    velocity = np.array([r['mean_velocity'] for r in results])
    distance = np.array([r['distance_traveled'] for r in results])
    torque = np.array([r['mean_torque'] for r in results])
    
    # Plot 1: Body Stiffness vs Velocity
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(body_stiff, velocity, c=effort, s=100, alpha=0.6, cmap='viridis')
    ax1.set_xlabel('Body Stiffness')
    ax1.set_ylabel('Mean Velocity (m/s)')
    ax1.set_title('Body Stiffness vs Velocity')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Effort (Nm)')
    
    # Plot 2: Leg Stiffness vs Velocity
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(leg_stiff, velocity, c=effort, s=100, alpha=0.6, cmap='viridis')
    ax2.set_xlabel('Leg Stiffness')
    ax2.set_ylabel('Mean Velocity (m/s)')
    ax2.set_title('Leg Stiffness vs Velocity')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Effort (Nm)')
    
    # Plot 3: Effort vs Velocity
    ax3 = plt.subplot(2, 3, 3)
    scatter3 = ax3.scatter(effort, velocity, c=body_stiff, s=100, alpha=0.6, cmap='coolwarm')
    ax3.set_xlabel('Effort Limit (Nm)')
    ax3.set_ylabel('Mean Velocity (m/s)')
    ax3.set_title('Effort vs Velocity')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Body Stiffness')
    
    # Plot 4: Distance Traveled
    ax4 = plt.subplot(2, 3, 4)
    top_10_idx = np.argsort(velocity)[-10:]
    labels = [f"B{results[i]['body_stiffness']}/L{results[i]['leg_stiffness']}" for i in top_10_idx]
    ax4.barh(range(10), distance[top_10_idx])
    ax4.set_yticks(range(10))
    ax4.set_yticklabels(labels)
    ax4.set_xlabel('Distance Traveled (m)')
    ax4.set_title('Top 10 Configurations - Distance')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Efficiency (velocity per unit torque)
    ax5 = plt.subplot(2, 3, 5)
    efficiency = velocity / (torque + 1e-6)
    scatter5 = ax5.scatter(torque, velocity, c=efficiency, s=100, alpha=0.6, cmap='plasma')
    ax5.set_xlabel('Mean Torque (Nm)')
    ax5.set_ylabel('Mean Velocity (m/s)')
    ax5.set_title('Efficiency: Velocity vs Torque')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter5, ax=ax5, label='Efficiency (v/τ)')
    
    # Plot 6: Parameter Correlation Heatmap
    ax6 = plt.subplot(2, 3, 6)
    top_5_idx = np.argsort(velocity)[-5:]
    param_matrix = np.array([
        [results[i]['body_stiffness'], results[i]['leg_stiffness'],
        results[i]['effort_limit'], results[i]['mean_velocity']]
        for i in top_5_idx
    ])
    im = ax6.imshow(param_matrix.T, aspect='auto', cmap='YlOrRd')
    ax6.set_xticks(range(5))
    ax6.set_xticklabels([f"Config {i+1}" for i in range(5)])
    ax6.set_yticks(range(4))
    ax6.set_yticklabels(['Body K', 'Leg K', 'Effort', 'Velocity'])
    ax6.set_title('Top 5 Configurations Heatmap')
    plt.colorbar(im, ax=ax6)

    for i in range(5):
        for j in range(4):
            text = ax6.text(i, j, f'{param_matrix[i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(args_cli.output_dir, f"param_sweep_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()

def main():
    """Main function"""
    print("Starting parameter sweep with fixed CPG parameters...")
    # Run sweep
    results = sweep_parameters()

    # Analyze and visualize
    best_config = analyze_and_plot_results(results)

    print(f"\n{'='*80}")
    print("BEST CONFIGURATION:")
    print(f"{'='*80}")
    print(f"Body Stiffness: {best_config['body_stiffness']}")
    print(f"Body Damping: {best_config['body_damping']}")
    print(f"Leg Stiffness: {best_config['leg_stiffness']}")
    print(f"Leg Damping: {best_config['leg_damping']}")
    print(f"Effort Limit: {best_config['effort_limit']} Nm")
    print(f"Mean Velocity: {best_config['mean_velocity']:.4f} m/s")
    print(f"Distance Traveled: {best_config['distance_traveled']:.4f} m")
    print(f"Mean Torque: {best_config['mean_torque']:.4f} Nm")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()