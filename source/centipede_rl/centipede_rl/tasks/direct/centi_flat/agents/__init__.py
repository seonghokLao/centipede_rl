from dataclasses import dataclass, field
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

PKG_ROOT = Path(__file__).resolve().parents[4]
USD_DIR   = PKG_ROOT / "assets" / "usd"
CENTIPEDE_USD = str(USD_DIR / "new_robot_full_v3.usd")

@dataclass
class CentipedeCfg(ArticulationCfg):
    prim_path: str = "/World/envs/env.*/new_robot_full_v3"

    spawn: sim_utils.UsdFileCfg = field(
        default_factory=lambda: sim_utils.UsdFileCfg(
            usd_path=CENTIPEDE_USD,
            copy_from_source=False,             # re-root under /World/envs/env_0/Robot
            # source_prim_path="/new_robot_full_v3",  # this is your defaultPrim path inside the USD
        )
    )

    articulation_root_prim_path: str | None = "/swing_base1/swing_base1"
    actuators: tuple = field(default_factory=tuple)
