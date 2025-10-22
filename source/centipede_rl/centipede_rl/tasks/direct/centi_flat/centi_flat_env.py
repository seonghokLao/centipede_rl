# source/centipede_rl/centipede_rl/tasks/direct/centi_flat/centi_flat_env.py
from __future__ import annotations
import math, torch
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
import isaaclab.sim as sim_utils

from .centi_flat_env_cfg import CentipedeVelTrackEnvCfg

# Optional: set this to your floor USD path if you have one.
# Otherwise the GroundPlane is enough for training.
FLOOR_USD = None
# Example: FLOOR_USD = r"J:/path/to/your/project/source/centipede_rl/centipede_rl/assets/usd/floor.usd"

class CentipedeVelTrackEnv(DirectRLEnv):
    cfg: CentipedeVelTrackEnvCfg

    def __init__(self, cfg: CentipedeVelTrackEnvCfg, **kw):
        super().__init__(cfg, **kw)

        # ---- Add ground / floor here (NOT via InteractiveSceneCfg) ----
        if FLOOR_USD:
            sim_utils.spawn_from_usd(
                prim_path="/World/Floor",
                cfg=sim_utils.UsdFileCfg(usd_path=FLOOR_USD),
            )
        else:
            sim_utils.spawn_ground_plane(
                prim_path="/World/Ground",
                cfg=sim_utils.GroundPlaneCfg(),
            )

        # Robot handle (spawned by base class using cfg.robot)
        self.robot: Articulation = self.scene["robot"]

        # Pick joint groups to drive (adjust substrings to your USD joint names)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._last_action = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, 4, device=self.device)
        self._cmd_vx = torch.zeros(self.num_envs, device=self.device)
        self._resample_timer = torch.zeros(self.num_envs, device=self.device)
        self._time_left = torch.zeros(self.num_envs, device=self.device)
        self._act_lo = torch.tensor(self.cfg.action_low, device=self.device)
        self._act_hi = torch.tensor(self.cfg.action_high, device=self.device)

        # joint grouping (now that self.robot exists)
        jn = self.robot.joint_names
        self._body_jids = [i for i, n in enumerate(jn) if "body" in n]
        self._leg_jids  = [i for i, n in enumerate(jn) if "leg"  in n]

    # ---------------- DirectRLEnv API ----------------

    @property
    def action_spec(self): return (4,)
    @property
    def observation_spec(self): return (7,)

    def _setup_scene(self):
        # 1) create the articulation from your cfg
        self.robot = Articulation(self.cfg.robot)

        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        print("[DEBUG] Robot at /World/envs/env_0/Robot exists:", robot_prim.IsValid())

        # list children so you can spot the articulation root name
        if robot_prim:
            print("[DEBUG] Children under /World/envs/env_0/Robot:")
            for ch in robot_prim.GetChildren():
                print("   ", ch.GetPath().pathString)

        # 2) add ground or custom floor
        if FLOOR_USD:
            sim_utils.spawn_from_usd(
                prim_path="/World/Floor",
                cfg=sim_utils.UsdFileCfg(usd_path=FLOOR_USD),
            )
        else:
            sim_utils.spawn_ground_plane(
                prim_path="/World/Ground",
                cfg=sim_utils.GroundPlaneCfg(),
            )

        # 3) clone vectorized environments and (CPU only) filter global collisions
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 4) register entities in the scene with keys (this enables self.scene["robot"])
        self.scene.articulations["robot"] = self.robot

        # 5) optional light
        dome = sim_utils.DomeLightCfg(intensity=2000.0)
        dome.func("/World/Light", dome)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._phase[env_ids] = 0.0
        self._last_action[env_ids].zero_()
        self._prev_action[env_ids].zero_()
        self._time_left[env_ids] = self.cfg.episode_length_s
        self._sample_cmd(env_ids)

    def _pre_physics_step(self, actions):
        a = torch.clamp(actions, self._act_lo, self._act_hi)
        self._prev_action = self._last_action.clone()
        self._last_action = a

        wave_num, body_amp, leg_amp, v_amp = a[:,0], a[:,1], a[:,2], a[:,3]
        dt = self.step_dt
        self._phase = (self._phase + v_amp * dt * 2.0 * math.pi).fmod(2.0 * math.pi)

        if self._body_jids:
            B = len(self._body_jids)
            idx = torch.linspace(0, 1, B, device=self.device).unsqueeze(0)
            offs = wave_num.unsqueeze(1) * (2.0 * math.pi) * idx
            tgt  = body_amp.unsqueeze(1) * torch.sin(self._phase.unsqueeze(1) + offs)
            self.robot.set_joint_position_target(tgt, joint_ids=self._body_jids)

        if self._leg_jids:
            tgt = leg_amp.unsqueeze(1) * torch.sin(self._phase.unsqueeze(1))
            self.robot.set_joint_position_target(tgt, joint_ids=self._leg_jids)

        # resample commands periodically
        self._resample_timer -= dt
        mask = self._resample_timer <= 0
        if torch.any(mask):
            self._sample_cmd(torch.nonzero(mask).squeeze(-1))

        # episode time
        self._time_left -= dt

    def _get_observations(self):
        R = self.robot.root_x_w[:, :3, :3]
        v_w = self.robot.root_lin_vel_w
        v_l = torch.einsum("nij,nj->ni", R.transpose(1,2), v_w)
        obs = torch.cat([self._cmd_vx.unsqueeze(1), v_l[:, :2], self._last_action], dim=1)
        return {"policy": obs}  # <-- dict, key = "policy"

    def _get_rewards(self):
        R = self.robot.root_x_w[:, :3, :3]
        v_w = self.robot.root_lin_vel_w
        vx  = torch.einsum("nij,nj->ni", R.transpose(1,2), v_w)[:, 0]

        r_track  = self.cfg.w_track_speed * torch.exp(-2.0 * (vx - self._cmd_vx) ** 2)
        r_smooth = self.cfg.w_action_rate * torch.sum((self._last_action - self._prev_action) ** 2, dim=1)
        z_up     = R[:, 2, 2].clamp(min=0)          # upright proxy
        r_up     = self.cfg.w_upright * z_up
        return r_track + r_smooth + r_up

    def _get_dones(self):
        R = self.robot.root_x_w[:, :3, :3]
        z = R[:, 2, 2].clamp(-1, 1)
        tilt = torch.acos(z)
        timeout = (self._time_left <= 0)
        return torch.logical_or(timeout, tilt > self.cfg.max_base_tilt_rad)

    def _get_dones_observations_rewards(self):
        return self._get_dones(), self._get_observations(), self._get_rewards()

    # ---------------- helpers ----------------

    def _sample_cmd(self, env_ids):
        lo, hi = self.cfg.cmd_vx_range
        self._cmd_vx[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
        tlo, thi = self.cfg.cmd_resample_min_s, self.cfg.cmd_resample_max_s
        self._resample_timer[env_ids] = tlo + (thi - tlo) * torch.rand(len(env_ids), device=self.device)
