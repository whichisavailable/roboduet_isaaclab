from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from robot_lab.tasks.manager_based.locomotion.velocity.cus_velocity_env_cfg import (
    GO2ARM_BASE_BODY_NAME,
    GO2ARM_FOOT_BODY_NAMES,
)

from .rough_env_cfg import UnitreeGo2ArmRoughEnvCfg


@configclass
class UnitreeGo2ArmFlatEnvCfg(UnitreeGo2ArmRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        terrain_contact_filter = self._terrain_contact_filter_prim_paths()
        self.scene.FL_foot_contact.filter_prim_paths_expr = terrain_contact_filter
        self.scene.FR_foot_contact.filter_prim_paths_expr = terrain_contact_filter
        self.scene.RL_foot_contact.filter_prim_paths_expr = terrain_contact_filter
        self.scene.RR_foot_contact.filter_prim_paths_expr = terrain_contact_filter

        self.observations.arm_privileged.lpy.params = {"asset_cfg": SceneEntityCfg("robot", body_names=["link6"])}
        self.rewards.total_reward.params["foot_asset_cfg"] = SceneEntityCfg(
            "robot", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True
        )
        self.rewards.total_reward.params["base_body_cfg"] = SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME])
        self.terminations.base_height_termination.params["sensor_cfg"] = None

        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.FL_foot_scanner = None
        self.scene.FR_foot_scanner = None
        self.scene.RL_foot_scanner = None
        self.scene.RR_foot_scanner = None
