# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .centi_flat_env_cfg import CentiFlatEnvCfg

def _get_F_tw1(t, body_amp):
    return body_amp * torch.cos(t)


def _get_F_tw2(t, body_amp):
    return body_amp * torch.sin(t)


def _alpha(F_tw1, shape_basis1, F_tw2, shape_basis2):
    return F_tw1.unsqueeze(1) * shape_basis1 + F_tw2.unsqueeze(1) * shape_basis2


def _F_Leg(Aleg, time, dutyf):
    two_pi = 2.0 * math.pi
    time_mod = torch.remainder(time, two_pi)
    on = time_mod < (two_pi * dutyf)
    
    Aleg_exp = Aleg.view(-1, *([1] * (time_mod.ndim - 1)))
    
    cos_on = torch.cos(time_mod / (2.0 * dutyf))
    cos_off = torch.cos((time_mod - two_pi) / (2.0 * (1.0 - dutyf)))
    
    out_on = Aleg_exp * cos_on
    out_off = Aleg_exp * cos_off
    
    return torch.where(on, out_on, out_off)


def _F_leg_act(time, dutyf):
    two_pi = 2.0 * math.pi
    time_mod = torch.remainder(time, two_pi)
    on = time_mod < (two_pi * dutyf)
    return on.to(time.dtype) * 2.0


def _get_beta(phi, num_leg, Aleg, dutyf, lfs):
    idx = torch.arange(num_leg, device=phi.device, dtype=phi.dtype)
    times = phi.unsqueeze(1) + idx.unsqueeze(0) * lfs.unsqueeze(1) * 2.0 * math.pi
    return -_F_Leg(Aleg, times, dutyf)


def _get_act(phi, lfs, num_leg, dutyf):
    idx = torch.arange(num_leg, device=phi.device, dtype=phi.dtype)
    times = phi.unsqueeze(1) + idx.unsqueeze(0) * lfs.unsqueeze(1) * 2.0 * math.pi
    return _F_leg_act(times, dutyf)


def _alpha_v(t, v_amp, xis, N, shape_tile=0.0):
    E = t.shape[0]
    idx = torch.arange(N-1, device=t.device, dtype=t.dtype)
    x_vals = idx.unsqueeze(0) * xis.unsqueeze(1) * 2.0 * math.pi
    x_vals = x_vals + shape_tile + t.unsqueeze(1)
    return v_amp.unsqueeze(1) * torch.cos(2.0 * x_vals + math.pi / 2.0)


class CentiFlatEnv(DirectRLEnv):
    cfg: CentiFlatEnvCfg

    def __init__(self, cfg: CentiFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Resolve joint indices you want to control/observe (edit names in cfg)
        # self._body_joint_ids = self.robot.find_joints(self.cfg.body_joint_names)[0]
        # self._leg_joint_ids  = self.robot.find_joints(self.cfg.leg_joint_names)[0]

        # -- Robot handles
        # self.leg_act_joint_ids   = self.robot.find_joints(self.cfg.leg_act_joint_names)[0]
        # self.leg_pitch_joint_ids = self.robot.find_joints(self.cfg.leg_pitch_joint_names)[0]
        # self.spine_joint_ids     = self.robot.find_joints(self.cfg.spine_joint_names)[0]
        # self.vertical_joint_ids  = self.robot.find_joints(self.cfg.vertical_joint_names)[0]
        self.leg_act_joint_ids = torch.tensor(
            self.robot.find_joints(self.cfg.leg_act_joint_names)[0], 
            device=self.device, dtype=torch.long
        )
        self.leg_pitch_joint_ids = torch.tensor(
            self.robot.find_joints(self.cfg.leg_pitch_joint_names)[0],
            device=self.device, dtype=torch.long
        )
        self.spine_joint_ids = torch.tensor(
            self.robot.find_joints(self.cfg.spine_joint_names)[0],
            device=self.device, dtype=torch.long
        )
        self.vertical_joint_ids = torch.tensor(
            self.robot.find_joints(self.cfg.vertical_joint_names)[0],
            device=self.device, dtype=torch.long
        )


    # ----- Scene -----
    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)
        # Ground
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Clone envs
        self.scene.clone_environments(copy_from_source=False)

        # CPU sim needs explicit filtering
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register in scene
        self.scene.articulations["robot"] = self.robot

        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Resolve joint indices once
        # self._body_joint_ids = self.robot.find_joints(self.cfg.body_joint_names)[0]
        # self._leg_joint_ids  = self.robot.find_joints(self.cfg.leg_joint_names)[0]

        # Task buffers
        n_env = self.scene.num_envs
        self.params = torch.zeros((n_env, 4), device=self.device)
        self.target_vel = torch.zeros(n_env, device=self.device)
        self.prev_actions = torch.zeros((n_env, 4), device=self.device)
        self.time_s = torch.zeros(n_env, device=self.device)

    # ----- RL plumbing -----
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    

    def _apply_action(self) -> None:
        # Scale and send efforts to a chosen DOF set (edit for your actuators)
        # self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)
        self.time_s += self.physics_dt * self.cfg.decimation
        scales = torch.tensor(self.cfg.action_scales, device=self.device)
        # print(self.params.shape)
        self.params = self.params + self.actions * scales
        lo = torch.tensor([self.cfg.wave_num_range[0], self.cfg.body_amp_range[0],
                        self.cfg.leg_amp_range[0],  self.cfg.v_amp_range[0]], device=self.device)
        hi = torch.tensor([self.cfg.wave_num_range[1], self.cfg.body_amp_range[1],
                        self.cfg.leg_amp_range[1],  self.cfg.v_amp_range[1]], device=self.device)
        self.params = torch.max(torch.min(self.params, hi), lo)

        # unpack
        wave_num = self.params[:, 0]
        body_amp = self.params[:, 1]
        leg_amp  = self.params[:, 2]
        v_amp    = self.params[:, 3]

        # MuJoCo constants mirrored
        N = 5
        num_legs = 5  # Number of legs
        E = self.scene.num_envs
        xis = 1.0 - wave_num / 5.0
        lfs = xis
        dutyf = 0.5


        # Precompute bases per env: shape_basis* are [E, N]
        idx = torch.arange(N-1, device=self.device, dtype=self.params.dtype)  # [N]
        phase_idx = idx.unsqueeze(0) * xis.unsqueeze(1)  # [E, N]
        sb1 = torch.sin(phase_idx * 2.0 * math.pi)  # [E, N]
        sb2 = torch.cos(phase_idx * 2.0 * math.pi)  # [E, N]

        t = torch.zeros_like(wave_num)
        over = self.time_s > 1.0
        t[over] = 2.0 * (self.time_s[over] - 1.0 + (math.pi / 2.0))

        # CPG / shape fields
        F1 = _get_F_tw1(t, body_amp)  # [E]
        F2 = _get_F_tw2(t, body_amp)  # [E]
        alpha_array = _alpha(F1, sb1, F2, sb2)  # [E, N=4]
        vertical_alpha = _alpha_v(t, v_amp, xis=xis, N=N, shape_tile=0.0)  # [E, N=4]

        phase_max = lfs * math.pi + math.pi / 2.0  # [E]
        phi = torch.atan2(F2, F1) - phase_max  # [E]

        # Leg contact activations: [E, num_legs]
        act_array = _get_act(phi, lfs, num_legs, dutyf)  # [E, 5]
        act_vals = torch.where(
            act_array > 0,
            torch.full_like(act_array, 0.4),
            torch.full_like(act_array, -0.4),
        )  # [E, 5]
        # Duplicate for L/R: [E, 5] -> [E, 5, 2] -> [E, 10]
        act_vals_LR = act_vals.unsqueeze(2).repeat(1, 1, 2).reshape(E, 2 * num_legs)

        # Leg pitch offsets: [E, num_legs]
        beta_array = _get_beta(phi, num_legs, leg_amp, dutyf, lfs)  # [E, 5]
        # Duplicate for L/R: [E, 5] -> [E, 5, 2] -> [E, 10]
        beta_LR = beta_array.unsqueeze(2).repeat(1, 1, 2).reshape(E, 2 * num_legs)

        # Zero during initialization
        under = self.time_s < 1.0
        if under.any():
            alpha_array[under] *= 0.0
            vertical_alpha[under] *= 0.0
            beta_LR[under] *= 0.0
            act_vals_LR[under] *= 0.0

        # Concatenate joint targets in correct order
        # Expected: [leg_swing(5)] + [leg_pitch_L/R(10)] + [spine_h(4)] + [spine_v(4)]
        # Total: 5 + 10 + 4 + 4 = 23 joints
        q_targets = torch.cat(
            [
                act_vals,          # [E, 5] - leg swing joints (NOT duplicated)
                beta_LR,           # [E, 10] - leg pitch L/R
                alpha_array,       # [E, 4] - spine horizontal
                vertical_alpha     # [E, 4] - spine vertical
            ],
            dim=1,
        )  # [E, 23]

        ids_all = torch.cat([
            self.leg_act_joint_ids,    # 5 joints
            self.leg_pitch_joint_ids,  # 10 joints
            self.spine_joint_ids,      # 4 joints
            self.vertical_joint_ids,   # 4 joints
        ])  # Total: 23 joints

        self.robot.set_joint_position_target(q_targets, joint_ids=ids_all)


    def _get_observations(self) -> dict:
        # Build observation vector (edit to match your task)
        root_lin_vel_w = self.robot.data.root_lin_vel_w[:, 0]  # [env], x component
        root_ang_w = self.robot.data.root_ang_vel_w           # [env, 3]
        base_quat = self.robot.data.root_quat_w               # [env, 4]

        obs = torch.cat(
            [
                self.params,                                  # 4
                root_lin_vel_w.unsqueeze(1),                  # 1
                self.target_vel.unsqueeze(1),                 # 1
                self.prev_actions,                            # 4
                base_quat,                                    # 4
                root_ang_w,                                   # 3
            ],
            dim=1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        vx = self.robot.data.root_lin_vel_w[:, 0]
        vel_err = torch.abs(vx - self.target_vel)

        smooth = torch.norm(self.actions - self.prev_actions, dim=1)
        qdot = torch.norm(self.robot.data.joint_vel, dim=1)

        rew = (
            self.cfg.rew_w_alive
            - self.cfg.rew_w_vel * vel_err
            - self.cfg.rew_w_smooth * smooth
            - self.cfg.rew_w_eff * qdot
        )
        
        self.prev_actions = self.actions.clone()
        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # crude tilt detect from gravity-aligned linear accel or quat; here we approximate from base quaternion:
        # |roll| or |pitch| > threshold → done
        # (If you have a quaternion->euler helper, use it; else skip or add torso frame tilt check.)
        done_fall = torch.zeros_like(time_out, dtype=torch.bool)

        return done_fall, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset robot to defaults at each env origin
        default_root = self.robot.data.default_root_state[env_ids]
        default_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)

        # Reset joints
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )
        print(env_ids)

        # Sample new target velocity per env
        self.target_vel[env_ids] = sample_uniform(
            self.cfg.target_vel_range[0],
            self.cfg.target_vel_range[1],
            (len(env_ids),),
            device=self.device,
        )

        # Reset parameters near mid-range for faster start
        mid = torch.tensor([
            sum(self.cfg.wave_num_range)/2.0,
            sum(self.cfg.body_amp_range)/2.0,
            sum(self.cfg.leg_amp_range)/2.0,
            sum(self.cfg.v_amp_range)/2.0,
        ], device=self.device)
        self.params[env_ids] = mid
        self.prev_actions[env_ids] = 0.0
        self.time_s[env_ids] = 0.0


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    return rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel

