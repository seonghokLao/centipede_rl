# Copyright (c) 2022-2025
# Parallel Parameter Sweep
import argparse
import numpy as np
import torch
import math
import itertools
from isaaclab.app import AppLauncher
from isaaclab.utils.math import quat_from_euler_xyz

# --- CLI Args ---
parser = argparse.ArgumentParser(description="Runtime stiffness/damping sweep")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments (Batch Size)")
parser.add_argument("--test_duration", type=float, default=4.0, help="Test duration per batch (seconds)")
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

# --- 1. DEFINE LARGE SEARCH SPACE ---
# Define ranges for your sweep
range_body_k = np.round(np.linspace(4.0, 40.0, num=10), 2).tolist()   # 5 steps from 4 to 20
range_body_d = np.round(np.linspace(0.5, 15.0, num=10), 2).tolist()    # 4 steps from 0.5 to 3
range_leg_k  = np.round(np.linspace(4.0, 40.0, num=10), 2).tolist()   
range_leg_d  = np.round(np.linspace(0.5, 15.0, num=10), 2).tolist()
range_effort = np.round(np.linspace(1.0, 10.0, num=10), 2).tolist()

# Generate all combinations (Cartesian product)
# Each item is (body_k, body_d, leg_k, leg_d, effort)
FULL_PARAM_GRID = list(itertools.product(
    range_body_k, range_body_d, range_leg_k, range_leg_d, range_effort
))

print(f"[INFO] Total configurations to sweep: {len(FULL_PARAM_GRID)}")
print(f"[INFO] Batch size (num_envs): {args_cli.num_envs}")
print(f"[INFO] Estimated Batches: {math.ceil(len(FULL_PARAM_GRID)/args_cli.num_envs)}")

def create_base_robot_config():
    """Create a generic robot config."""
    ROBOT_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="./source/centipede_rl/centipede_rl/assets/ten_leg/new_robot_full_v3.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        articulation_root_prim_path="/swing_base1/swing_base1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
        ),
        actuators={
            # Initial values don't matter, we overwrite them immediately
            "body": ImplicitActuatorCfg(
                joint_names_expr=["body_.*"],
                effort_limit_sim=5.0, velocity_limit_sim=50.0, stiffness=10.0, damping=1.0,
            ),
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_leg_joint.*"],
                effort_limit_sim=5.0, velocity_limit_sim=50.0, stiffness=10.0, damping=1.0,
            ),
            "swing": ImplicitActuatorCfg(
                joint_names_expr=["leg_swing_joint.*"],
                effort_limit_sim=5.0, velocity_limit_sim=50.0, stiffness=10.0, damping=1.0,
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

def set_parallel_gains(robot, b_k, b_d, l_k, l_d, effort, active_envs_indices):
    """
    Apply different gains to different environments in parallel.
    Inputs are 1D Tensors of size (num_envs,).
    """
    device = robot.device
    num_envs = robot.num_instances
    num_dof = robot.num_joints
    
    # Get Joint Indices
    body_ids = torch.tensor(robot.find_joints("body_.*")[0], device=device, dtype=torch.long)
    leg_ids = torch.tensor(robot.find_joints(".*_leg_joint.*")[0], device=device, dtype=torch.long)
    swing_ids = torch.tensor(robot.find_joints("leg_swing_joint.*")[0], device=device, dtype=torch.long)
    
    # Initialize Full Tensors with Default values (prevents garbage data in unused envs)
    # Shape: (num_envs, num_dof)
    k_full = torch.ones((num_envs, num_dof), device=device) * 10.0
    d_full = torch.ones((num_envs, num_dof), device=device) * 1.0
    f_full = torch.ones((num_envs, num_dof), device=device) * 5.0

    # --- VECTORIZED ASSIGNMENT ---
    # We use unsqueeze(1) to turn shape (N) into (N, 1) so it broadcasts across joints
    
    # Body
    k_full[:, body_ids] = b_k.unsqueeze(1)
    d_full[:, body_ids] = b_d.unsqueeze(1)
    
    # Legs
    k_full[:, leg_ids] = l_k.unsqueeze(1)
    d_full[:, leg_ids] = l_d.unsqueeze(1)
    
    # Swing (Derived from Legs)
    k_full[:, swing_ids] = l_k.unsqueeze(1) * 1.5
    d_full[:, swing_ids] = l_d.unsqueeze(1) * 0.5
    
    # Effort
    f_full[:] = effort.unsqueeze(1)

    # Apply to PhysX
    # Note: We must update ALL envs, even if the last batch is partial, 
    # to ensure consistency.
    all_indices = torch.arange(num_dof, dtype=torch.long, device="cpu")
    
    robot.root_physx_view.set_dof_stiffnesses(k_full.to("cpu"), indices=all_indices)
    robot.root_physx_view.set_dof_dampings(d_full.to("cpu"), indices=all_indices)
    robot.root_physx_view.set_dof_max_forces(f_full.to("cpu"), indices=all_indices)

def compute_cpg(t, num_envs, device):
    """Same CPG logic, ensures tensors match num_envs"""
    if not isinstance(t, torch.Tensor): t = torch.tensor([t]*num_envs, device=device)
    
    wave = torch.tensor([WAVE_NUM]*num_envs, device=device)
    amp_b = torch.tensor([BODY_AMP]*num_envs, device=device)
    amp_l = torch.tensor([LEG_AMP]*num_envs, device=device)
    amp_v = torch.tensor([V_AMP]*num_envs, device=device)
    
    xis = 1.0 - wave/5.0
    lfs = xis
    dutyf = 0.5
    ramp = 1.0
    
    t_eff = torch.zeros_like(t)
    mask = t > ramp
    t_eff[mask] = 4.0 * (t[mask] - ramp + math.pi/2.0)
    gain = torch.clamp(t/ramp, 0.0, 1.0).unsqueeze(1)
    
    idx = torch.arange(4, device=device).unsqueeze(0) 
    phase = idx * xis.unsqueeze(1)
    alpha = amp_b.unsqueeze(1) * torch.sin(t_eff.unsqueeze(1) + phase*2*math.pi) * gain
    vert = amp_v.unsqueeze(1) * torch.cos(2.0*(idx*xis.unsqueeze(1)*2*math.pi + t_eff.unsqueeze(1)) + math.pi/2) * gain

    F1 = torch.cos(t_eff); F2 = torch.sin(t_eff)
    phi = torch.atan2(F2, F1) - (lfs*math.pi + math.pi/2)
    
    idx_l = torch.arange(5, device=device).unsqueeze(0)
    leg_t = phi.unsqueeze(1) + idx_l * lfs.unsqueeze(1) * 2 * math.pi
    leg_t_mod = torch.remainder(leg_t, 2*math.pi)
    on = leg_t_mod < (2*math.pi*dutyf)
    
    act = torch.where(on, torch.tensor(0.4, device=device), torch.tensor(-0.4, device=device)) * gain
    
    cos_on = torch.cos(leg_t_mod / (2*dutyf))
    cos_off = torch.cos((leg_t_mod - 2*math.pi)/(2*(1-dutyf)))
    beta = -torch.where(on, amp_l.unsqueeze(1)*cos_on, amp_l.unsqueeze(1)*cos_off)
    beta_lr = beta.unsqueeze(2).repeat(1,1,2).view(num_envs, 10) * gain
    
    return act, beta_lr, alpha, vert

def main():
    # 1. Setup Simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/120)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set num_envs to batch size
    scene = InteractiveScene(SweepSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0))
    robot = scene["robot"]
    sim.reset()

    print(f"Simulation started. Robot has {robot.num_joints} joints.")
    all_results = []
    
    # Chunk the sweep grid into batches of size num_envs
    total_configs = len(FULL_PARAM_GRID)
    batch_size = args_cli.num_envs
    
    # 2. BATCH LOOP
    for batch_start in range(0, total_configs, batch_size):
        batch_end = min(batch_start + batch_size, total_configs)
        current_batch_configs = FULL_PARAM_GRID[batch_start:batch_end]
        current_batch_len = len(current_batch_configs)
        
        print(f"\nProcessing batch {batch_start} to {batch_end} ({current_batch_len} configs)...")

        # --- PREPARE PARALLEL TENSORS ---
        # We create tensors of size (num_envs,). 
        # If the last batch is smaller than num_envs, we pad with default values 
        # (but we only record results for the valid indices).
        
        # 1. Unpack params into lists
        bk_list, bd_list, lk_list, ld_list, eff_list = zip(*current_batch_configs)
        
        # 2. Convert to Tensors and Pad if necessary
        def to_padded_tensor(data_list):
            t = torch.tensor(data_list, device=robot.device, dtype=torch.float)
            if len(t) < batch_size:
                padding = torch.zeros(batch_size - len(t), device=robot.device)
                t = torch.cat([t, padding])
            return t

        b_k_t = to_padded_tensor(bk_list)
        b_d_t = to_padded_tensor(bd_list)
        l_k_t = to_padded_tensor(lk_list)
        l_d_t = to_padded_tensor(ld_list)
        eff_t = to_padded_tensor(eff_list)
        
        # --- A. UPDATE PARAMETERS PARALLEL ---
        # We pass indices, though currently we just update all rows 
        # and ignore the padded ones later.
        set_parallel_gains(robot, b_k_t, b_d_t, l_k_t, l_d_t, eff_t, None)
        
        # --- B. RESET ROBOT ---
        # default_root = robot.data.default_root_state.clone()
        # rot_quat = quat_from_euler_xyz(
        #     torch.tensor(math.pi/2, device=robot.device),
        #     torch.tensor(0.0, device=robot.device),
        #     torch.tensor(0.0, device=robot.device),
        # )
        # default_root[:, 3:7] = rot_quat
        # default_root[:, 2] += -0.2

        # robot.write_root_pose_to_sim(default_root[:, :7])
        # robot.write_root_velocity_to_sim(default_root[:, 7:])
        # robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        # scene.reset()
        env_origins = scene.env_origins  # Shape: (num_envs, 3)
        
        # 2. Define Local Offset (Where the robot is relative to its own square)
        # e.g., (0, 0, 0.6)
        local_pos = torch.zeros_like(env_origins)
        local_pos[:, 2] = 0.6  # Z-height
        
        # 3. Add them together: Global Pos = Grid Origin + Local Offset
        global_pos = env_origins + local_pos
        
        # 4. Define Rotation
        # Create one quaternion and repeat it for all envs
        q_single = quat_from_euler_xyz(
            torch.tensor(math.pi/2, device=robot.device),
            torch.tensor(0.0, device=robot.device),
            torch.tensor(0.0, device=robot.device)
        )
        global_rot = q_single.unsqueeze(0).repeat(args_cli.num_envs, 1)

        # 5. Apply
        robot.write_root_pose_to_sim(torch.cat([global_pos, global_rot], dim=1))
        robot.write_root_velocity_to_sim(torch.zeros((args_cli.num_envs, 6), device=robot.device))
        
        # Reset joint states
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        scene.reset()
        
        # --- C. RUN TEST ---
        sim_time = 0.0
        # Capture start pos for ALL envs
        start_pos = robot.data.root_pos_w[:, 0].clone()
        
        # Accumulate velocities per environment
        # Shape: (steps, num_envs)
        vel_history = []
        
        steps = int(args_cli.test_duration / sim.get_physics_dt())
        
        # Joint Indices
        act_ids = torch.tensor(robot.find_joints("leg_swing_joint.*")[0], device=robot.device, dtype=torch.long)
        pitch_ids = torch.tensor(robot.find_joints(".*_leg_joint.*")[0], device=robot.device, dtype=torch.long)
        spine_ids = torch.tensor(robot.find_joints("body_h.*")[0], device=robot.device, dtype=torch.long)
        vert_ids = torch.tensor(robot.find_joints("body_v.*")[0], device=robot.device, dtype=torch.long)
        all_ids = torch.cat([act_ids, pitch_ids, spine_ids, vert_ids])

        for _ in range(steps):
            act, pitch, spine, vert = compute_cpg(sim_time, robot.num_instances, robot.device)
            targets = torch.cat([act, pitch, spine, vert], dim=1)
            robot.set_joint_position_target(targets.float(), joint_ids=all_ids)
            
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            sim_time += sim.get_physics_dt()
            
            if sim_time > 1.0: 
                # Store current velocity for all envs
                vel_history.append(robot.data.root_lin_vel_w[:, 0].clone())

        # --- D. LOG RESULTS PER ENVIRONMENT ---
        # 1. Calculate metrics for all envs in parallel
        # Final Distances: (num_envs,)
        final_dists = (robot.data.root_pos_w[:, 0] - start_pos)
        
        # Average Velocities: (num_envs,)
        if len(vel_history) > 0:
            avg_vels = torch.stack(vel_history).mean(dim=0)
        else:
            avg_vels = torch.zeros(batch_size, device=robot.device)

        # 2. Map results back to configurations
        # Only iterate up to current_batch_len to ignore padding
        for local_i in range(current_batch_len):
            config = current_batch_configs[local_i]
            vel = avg_vels[local_i].item()
            dist = final_dists[local_i].item()
            
            all_results.append({
                "config": config,
                "vel": vel,
                "dist": dist
            })

    # 3. Summary
    print("\n" + "="*50)
    print("MASSIVE SWEEP COMPLETE - TOP 10 CONFIGS")
    print("="*50)
    all_results.sort(key=lambda x: x['vel'], reverse=True)
    
    for r in all_results[:10]:
        c = r['config']
        # (body_k, body_d, leg_k, leg_d, effort)
        print(f"Vel: {r['vel']:.3f} | B_K:{c[0]} B_D:{c[1]} L_K:{c[2]} L_D:{c[3]} Eff:{c[4]}")

    # Optional: Save to JSON
    import json
    with open(f"{args_cli.output_dir}/sweep_results.json", "w") as f:
        # Convert tuples to lists for JSON serialization
        json_ready = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in res.items()} for res in all_results]
        json.dump(json_ready, f, indent=4)

    simulation_app.close()

if __name__ == "__main__":
    main()