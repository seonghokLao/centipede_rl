# Copyright (c) 2022-2025
# Fast parameter sweep changing gains at runtime without restarting Sim
# Fixed CPG parameters: wave_num=1.3, body_amp=20°, leg_amp=35°, v_amp=-20°

import argparse
import numpy as np
import torch
import json
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher
from isaaclab.utils.math import quat_from_euler_xyz

# --- CLI Args ---
parser = argparse.ArgumentParser(description="Runtime stiffness/damping sweep")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--test_duration", type=float, default=5.0, help="Test duration per config (seconds)")
parser.add_argument("--output_dir", type=str, default="./param_sweep_results", help="Output directory")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg

# --- FIXED CPG PARAMETERS ---
WAVE_NUM = 1.3
BODY_AMP = np.radians(20)
LEG_AMP = np.radians(35)
V_AMP = -np.radians(20)

# --- SWEEP GRID ---
# (body_stiff, body_damp, leg_stiff, leg_damp, effort_limit)
PARAM_GRID = [
    (4.0, 1.0, 4.0, 1.0, 1.5),   # Baseline
    (8.0, 1.0, 8.0, 1.0, 2.5),   # Stiff
    (15.0, 2.0, 15.0, 2.0, 3.5), # Very Stiff
    (8.0, 0.5, 12.0, 1.0, 2.5),  # Hybrid 1
    (4.0, 0.5, 8.0, 1.0, 2.5),   # Soft Body, Stiff Legs
]

def create_base_robot_config():
    """Create a generic robot config. We will tune gains at runtime."""
    ROBOT_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="./source/centipede_rl/centipede_rl/assets/ten_leg/new_robot_full_v3.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12, # High iterations for stability
                solver_velocity_iteration_count=1,
            ),
        ),
        articulation_root_prim_path="/swing_base1/swing_base1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
        ),
        actuators={
            # We define groups, but initial values don't matter much 
            # as we will overwrite them in the loop.
            "body": ImplicitActuatorCfg(
                joint_names_expr=["body_.*"], # Regex for all body joints
                effort_limit_sim=5.0,
                velocity_limit_sim=50.0,
                stiffness=10.0, 
                damping=1.0,
            ),
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_leg_joint.*"], # Regex for all leg pitch joints
                effort_limit_sim=5.0,
                velocity_limit_sim=50.0,
                stiffness=10.0, 
                damping=1.0,
            ),
            "swing": ImplicitActuatorCfg(
                joint_names_expr=["leg_swing_joint.*"],
                effort_limit_sim=5.0,
                velocity_limit_sim=50.0,
                stiffness=10.0, 
                damping=1.0,
            ),
        },
    )
    return ROBOT_CFG

class SweepSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = create_base_robot_config()
    robot = robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

def set_runtime_gains(robot, body_k, body_d, leg_k, leg_d, effort):
    """
    Updates the PhysX drive parameters directly without reloading the scene.
    """
    device = robot.device
    num_envs = robot.num_instances
    num_dof = robot.num_joints
    
    # 1. Find indices for our groups
    # Note: Regex matching is powerful here
    # We cast to LongTensor to use them as indices for the full tensor
    body_ids = torch.tensor(robot.find_joints("body_.*")[0], device=device, dtype=torch.long)
    leg_ids = torch.tensor(robot.find_joints(".*_leg_joint.*")[0], device=device, dtype=torch.long)
    swing_ids = torch.tensor(robot.find_joints("leg_swing_joint.*")[0], device=device, dtype=torch.long)
    
    # 2. Construct Full Tensors (Avoids partial update bugs)
    # Initialize everything to Body params as a baseline default
    k_full = torch.full((num_envs, num_dof), body_k, device=device)
    d_full = torch.full((num_envs, num_dof), body_d, device=device)
    
    # 3. Overwrite specific groups
    # We update the full tensor in-place before sending it to PhysX
    k_full[:, leg_ids] = leg_k
    d_full[:, leg_ids] = leg_d
    
    k_full[:, swing_ids] = leg_k * 1.5
    d_full[:, swing_ids] = leg_d * 0.5
    
    # 4. Apply to PhysX View
    # FIX: Create explicit indices for ALL joints on CPU (Device -1)
    # The API requires the 'indices' argument to be present.
    all_indices = torch.arange(num_dof, dtype=torch.long, device="cpu")
    
    robot.root_physx_view.set_dof_stiffnesses(k_full.to("cpu"), indices=all_indices)
    robot.root_physx_view.set_dof_dampings(d_full.to("cpu"), indices=all_indices)
    
    # 5. Update Effort Limits
    # Note: This often requires setting max forces
    max_forces = torch.full((num_envs, num_dof), effort, device=device)
    robot.root_physx_view.set_dof_max_forces(max_forces.to("cpu"), indices=all_indices) # Sets for ALL joints

def compute_cpg(t, num_envs, device):
    """Simplified CPG for fixed gait"""
    # ... (Same logic as your previous script, condensed for brevity) ...
    # Uses GLOBAL constants WAVE_NUM, BODY_AMP, etc.
    if not isinstance(t, torch.Tensor): t = torch.tensor([t]*num_envs, device=device)
    
    # Constants
    wave = torch.tensor([WAVE_NUM]*num_envs, device=device)
    amp_b = torch.tensor([BODY_AMP]*num_envs, device=device)
    amp_l = torch.tensor([LEG_AMP]*num_envs, device=device)
    amp_v = torch.tensor([V_AMP]*num_envs, device=device)
    
    xis = 1.0 - wave/5.0
    lfs = xis
    dutyf = 0.5
    
    # Time mod
    ramp = 1.0
    t_eff = torch.zeros_like(t)
    mask = t > ramp
    t_eff[mask] = 2.0 * (t[mask] - ramp + math.pi/2.0)
    gain = torch.clamp(t/ramp, 0.0, 1.0).unsqueeze(1)
    
    # Spine
    idx = torch.arange(4, device=device).unsqueeze(0) # 4 spine joints
    phase = idx * xis.unsqueeze(1)
    # Simple traveling wave
    alpha = amp_b.unsqueeze(1) * torch.sin(t_eff.unsqueeze(1) + phase*2*math.pi) * gain
    
    # Vertical
    vert = amp_v.unsqueeze(1) * torch.cos(2.0*(idx*xis.unsqueeze(1)*2*math.pi + t_eff.unsqueeze(1)) + math.pi/2) * gain

    # Legs
    F1 = torch.cos(t_eff); F2 = torch.sin(t_eff)
    phi = torch.atan2(F2, F1) - (lfs*math.pi + math.pi/2)
    
    idx_l = torch.arange(5, device=device).unsqueeze(0)
    leg_t = phi.unsqueeze(1) + idx_l * lfs.unsqueeze(1) * 2 * math.pi
    leg_t_mod = torch.remainder(leg_t, 2*math.pi)
    
    on = leg_t_mod < (2*math.pi*dutyf)
    
    # Lift (Act)
    act = torch.where(on, torch.tensor(0.4, device=device), torch.tensor(-0.4, device=device)) * gain
    
    # Pitch (Beta)
    cos_on = torch.cos(leg_t_mod / (2*dutyf))
    cos_off = torch.cos((leg_t_mod - 2*math.pi)/(2*(1-dutyf)))
    beta = -torch.where(on, amp_l.unsqueeze(1)*cos_on, amp_l.unsqueeze(1)*cos_off)
    beta_lr = beta.unsqueeze(2).repeat(1,1,2).view(num_envs, 10) * gain
    
    return act, beta_lr, alpha, vert

def main():
    # 1. Setup Simulation (Runs ONCE)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/120)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SweepSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0))
    robot = scene["robot"]
    sim.reset()

    print(f"Simulation started. Robot has {robot.num_joints} joints.")
    all_results = []
    
    # 2. Main Sweep Loop
    for idx, (b_k, b_d, l_k, l_d, eff) in enumerate(PARAM_GRID):
        print(f"\n[{idx+1}/{len(PARAM_GRID)}] Config: Body(K={b_k}, D={b_d}) | Legs(K={l_k}, D={l_d})")
        
        # --- A. UPDATE PARAMETERS RUNTIME ---
        set_runtime_gains(robot, b_k, b_d, l_k, l_d, eff)
        
        # --- B. RESET ROBOT STATE ---
        default_root = robot.data.default_root_state
        rot_quat = quat_from_euler_xyz(
            torch.tensor(math.pi/2, device=robot.device),
            torch.tensor(0.0, device=robot.device),
            torch.tensor(0.0, device=robot.device),
        )
        default_root[:, 3:7] = rot_quat
        default_root[:, 2] += -0.2  # Raise slightly above ground

        robot.write_root_pose_to_sim(default_root[:, :7])
        robot.write_root_velocity_to_sim(default_root[:, 7:])

        # robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
        # robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        scene.reset()
        
        # --- C. RUN TEST ---
        sim_time = 0.0
        start_pos = robot.data.root_pos_w[:, 0].clone()
        velocities = []
        
        steps = int(args_cli.test_duration / sim.get_physics_dt())
        
        # Pre-calculate joint indices for action application
        # FIX: Cast lists to Tensors for torch.cat
        act_ids = torch.tensor(robot.find_joints("leg_swing_joint.*")[0], device=robot.device, dtype=torch.long)
        pitch_ids = torch.tensor(robot.find_joints(".*_leg_joint.*")[0], device=robot.device, dtype=torch.long)
        spine_ids = torch.tensor(robot.find_joints("body_h.*")[0], device=robot.device, dtype=torch.long)
        vert_ids = torch.tensor(robot.find_joints("body_v.*")[0], device=robot.device, dtype=torch.long)
        
        all_ids = torch.cat([act_ids, pitch_ids, spine_ids, vert_ids])

        for _ in range(steps):
            # CPG
            act, pitch, spine, vert = compute_cpg(sim_time, robot.num_instances, robot.device)
            
            # Combine targets (Ensure order matches all_ids!)
            # Note: Your `pitch` logic produces [L1, R1, L2, R2...] or [L1...L5, R1...R5]? 
            # Check your USD joint order. Assuming your CPG matches USD.
            targets = torch.cat([act, pitch, spine, vert], dim=1)
            
            robot.set_joint_position_target(targets.float(), joint_ids=all_ids)
            
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            sim_time += sim.get_physics_dt()
            
            if sim_time > 1.0: # Skip warmup for metrics
                velocities.append(robot.data.root_lin_vel_w[:, 0].mean().item())

        # --- D. LOG RESULTS ---
        dist = (robot.data.root_pos_w[:, 0] - start_pos).mean().item()
        avg_vel = np.mean(velocities) if len(velocities) > 0 else 0.0
        
        print(f" -> Distance: {dist:.3f} m | Vel: {avg_vel:.3f} m/s")
        
        all_results.append({
            "config": (b_k, b_d, l_k, l_d, eff),
            "vel": avg_vel,
            "dist": dist
        })

    # 3. Summary
    print("\n" + "="*50)
    print("SWEEP COMPLETE - TOP CONFIGS")
    print("="*50)
    all_results.sort(key=lambda x: x['vel'], reverse=True)
    for r in all_results[:5]:
        c = r['config']
        print(f"Vel: {r['vel']:.3f} | Body: {c[0]}/{c[1]}, Legs: {c[2]}/{c[3]}")

    simulation_app.close()

if __name__ == "__main__":
    main()