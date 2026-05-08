import math
from collections.abc import Callable
from typing import cast

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.assets.unitree import UNITREE_Go2Arm_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.cus_velocity_env_cfg import (
    GO2ARM_ALL_JOINT_NAMES,
    GO2ARM_ARM_JOINT_NAMES,
    GO2ARM_BASE_BODY_NAME,
    GO2ARM_FOOT_BODY_NAMES,
    GO2ARM_FOOT_SCANNER_NAMES,
    GO2ARM_LEG_JOINT_NAMES,
    GO2ARM_NON_FOOT_BODY_REGEX,
    GO2ARM_SIMPLIFIED_ILLEGAL_CONTACT_BODY_NAMES,
    Go2ArmDefaultDeltaJointPositionActionCfg,
    LocomotionVelocityRoughEnvCfg,
)

GO2ARM_STAGE_SWITCH_ITERATION = 10000


@configclass
class RoboDuetDogPolicyObsCfg(ObsGroup):
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(
        func=mdp.roboduet_leg_joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_LEG_JOINT_NAMES, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.roboduet_joint_vel_loco,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_LEG_JOINT_NAMES, preserve_order=True)},
    )
    actions = ObsTerm(func=mdp.roboduet_leg_action)
    commands_dog = ObsTerm(func=mdp.roboduet_commands_dog_policy, params={"command_name": "roboduet"})
    commands_arm = ObsTerm(func=mdp.roboduet_commands_arm_obs, params={"command_name": "roboduet"})
    roll_pitch = ObsTerm(
        func=mdp.roboduet_roll_pitch,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME])},
    )
    clock_inputs = ObsTerm(func=mdp.roboduet_clock_inputs, params={"command_name": "roboduet"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class RoboDuetDogPrivilegedObsCfg(ObsGroup):
    friction = ObsTerm(
        func=mdp.roboduet_privileged_friction,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True)},
    )
    restitution = ObsTerm(
        func=mdp.roboduet_privileged_restitution,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True)},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class RoboDuetArmPolicyObsCfg(ObsGroup):
    joint_pos = ObsTerm(
        func=mdp.roboduet_arm_joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ARM_JOINT_NAMES, preserve_order=True)},
    )
    actions = ObsTerm(func=mdp.roboduet_arm_action)
    commands_arm = ObsTerm(func=mdp.roboduet_commands_arm_obs, params={"command_name": "roboduet"})
    roll_pitch = ObsTerm(
        func=mdp.roboduet_roll_pitch,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME])},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class RoboDuetArmPrivilegedObsCfg(ObsGroup):
    friction = ObsTerm(
        func=mdp.roboduet_privileged_friction,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True)},
    )
    restitution = ObsTerm(
        func=mdp.roboduet_privileged_restitution,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True)},
    )
    lpy = ObsTerm(
        func=mdp.roboduet_current_lpy,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"])},
    )
    ee_quat_in_base = ObsTerm(
        func=mdp.roboduet_current_ee_quat_in_base,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"])},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


PRETRAINED_REWARD_SCALES = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -0.02,
    "ang_vel_xy": -0.001,
    "orientation_control": -5.0,
    "loco_energy": -0.00004,
    "feet_slip": -0.04,
    "feet_clearance_cmd_linear": -30.0,
    "tracking_contacts_shaped_force": 4.0,
    "tracking_contacts_shaped_vel": 4.0,
    "collision": -5.0,
    "dof_vel": -1.0e-4,
    "dof_acc": -2.5e-7,
    "action_rate": -0.01,
    "action_smoothness_1": -0.1,
    "action_smoothness_2": -0.1,
    "torques": -1.0e-5,
    "hip_action_l2": -0.05,
}

HYBRID_REWARD_SCALES = {
    **PRETRAINED_REWARD_SCALES,
    "tracking_lin_vel": 0.7,
    "tracking_ang_vel": 0.25,
    "orientation_control": -10.0,
    "arm_manip_commands_tracking_combine": 1.0,
    "arm_energy": -4.0e-5,
    "arm_dof_vel": -0.001,
    "arm_dof_acc": -2.5e-6,
    "arm_action_rate": -0.1,
    "arm_action_smoothness_1": -0.5,
    "arm_action_smoothness_2": -0.5,
    "arm_control_smoothness_1": -0.1,
    "arm_control_limits": -5.0,
}


@configclass
class UnitreeGo2ArmRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rsl_rl_init_noise_std: float = 1.0
    reward_log_interval_iterations: int = 1
    reward_log_steps_per_iteration: int = 24
    enable_debug_reward_logging: bool = False
    enable_collision_group_logging: bool = False
    enable_contact_verification_logging: bool = False
    enable_termination_debug_logging: bool = False
    enable_play_termination_reason_logging: bool = False
    enable_base_frame_validation_logging: bool = True
    base_frame_validation_log_steps: int = 5
    base_frame_validation_done_logs: int = 5
    episode_log_key_prefixes: tuple[str, ...] = (
        "rew_",
        "Len/",
        "Term/",
        "BaseFrame/",
        "Curriculum/",
    )

    def _terrain_contact_filter_prim_paths(self) -> list[str]:
        if self.scene.terrain.terrain_type == "plane":
            return ["/World/ground/terrain/GroundPlane/CollisionPlane"]
        return ["/World/ground/terrain/mesh"]

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2048
        self.scene.robot = UNITREE_Go2Arm_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.merge_fixed_joints = False
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # Upstream scripts/auto_train.py forces Cfg.terrain.mesh_type = "plane".
        # Keep the rough task ID, but make its effective terrain semantics match auto_train.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + GO2ARM_BASE_BODY_NAME
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + GO2ARM_BASE_BODY_NAME

        self.scene.FL_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/FL_foot"
        self.scene.FR_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/FR_foot"
        self.scene.RL_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/RL_foot"
        self.scene.RR_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/RR_foot"
        self.scene.FL_foot_contact.prim_path = "{ENV_REGEX_NS}/Robot/FL_foot"
        self.scene.FR_foot_contact.prim_path = "{ENV_REGEX_NS}/Robot/FR_foot"
        self.scene.RL_foot_contact.prim_path = "{ENV_REGEX_NS}/Robot/RL_foot"
        self.scene.RR_foot_contact.prim_path = "{ENV_REGEX_NS}/Robot/RR_foot"
        terrain_contact_filter = self._terrain_contact_filter_prim_paths()
        self.scene.FL_foot_contact.filter_prim_paths_expr = terrain_contact_filter
        self.scene.FR_foot_contact.filter_prim_paths_expr = terrain_contact_filter
        self.scene.RL_foot_contact.filter_prim_paths_expr = terrain_contact_filter
        self.scene.RR_foot_contact.filter_prim_paths_expr = terrain_contact_filter

        self.observations.policy = None
        self.observations.dog_policy = RoboDuetDogPolicyObsCfg()
        self.observations.dog_privileged = RoboDuetDogPrivilegedObsCfg()
        self.observations.arm_policy = RoboDuetArmPolicyObsCfg()
        self.observations.arm_privileged = RoboDuetArmPrivilegedObsCfg()
        self.observations.critic = None
        self.observations.privileged = None

        self.actions.joint_pos = Go2ArmDefaultDeltaJointPositionActionCfg(
            asset_name="robot",
            joint_names=GO2ARM_ALL_JOINT_NAMES,
            preserve_order=True,
            use_default_offset=True,
            scale=1.0,
            clip=None,
            delta_clip=None,
            action_scale=0.25,
            hip_joint_names=["^(FL|FR|RL|RR)_hip_joint$"],
            hip_scale_reduction=0.5,
            fixed_delta_action_joint_names=["^joint[1-6]$"],
            fixed_delta_action_until_iteration=GO2ARM_STAGE_SWITCH_ITERATION,
            fixed_delta_action_steps_per_iteration=24,
            fixed_delta_action_value=0.0,
        )

        self.commands.base_velocity = None
        self.commands.roboduet = mdp.RoboDuetCommandCfg(
            asset_name="robot",
            base_body_name=GO2ARM_BASE_BODY_NAME,
            ee_body_name="link6",
            switch_iteration=GO2ARM_STAGE_SWITCH_ITERATION,
            steps_per_iteration=24,
            command_curriculum_seed=100,
            resampling_time_s=10.0,
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_yaw=(-1.0, 1.0),
            body_pitch_range=(-0.4, 0.4),
            body_roll_range=(-0.4, 0.4),
            limit_vel_x=(-5.0, 5.0),
            limit_vel_y=(-0.6, 0.6),
            limit_vel_yaw=(-5.0, 5.0),
            limit_body_pitch=(-0.4, 0.4),
            limit_body_roll=(-0.4, 0.4),
            num_bins_vel_x=21,
            num_bins_vel_y=1,
            num_bins_vel_yaw=21,
            num_bins_body_pitch=1,
            num_bins_body_roll=1,
            pretrained_reward_scales={
                key: PRETRAINED_REWARD_SCALES[key]
                for key in ("tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force", "tracking_contacts_shaped_vel")
            },
        )

        roboduet_reward_params = {
            "command_name": "roboduet",
            "pretrained_scales": PRETRAINED_REWARD_SCALES,
            "hybrid_scales": HYBRID_REWARD_SCALES,
            "only_positive_rewards": False,
            "only_positive_rewards_ji22_style": True,
            "sigma_rew_neg": 0.02,
            "tracking_sigma": 0.25,
            "tracking_sigma_yaw": 0.25,
            "gait_force_sigma": 100.0,
            "gait_vel_sigma": 10.0,
            "illegal_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GO2ARM_SIMPLIFIED_ILLEGAL_CONTACT_BODY_NAMES
            ),
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True),
            "leg_joint_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_LEG_JOINT_NAMES, preserve_order=True),
            "arm_joint_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ARM_JOINT_NAMES, preserve_order=True),
            "base_body_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
            "ee_body_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            "manip_weight_lpy": 3.0,
            "manip_weight_rpy": 1.0,
        }
        for reward_term_name in HYBRID_REWARD_SCALES:
            setattr(
                self.rewards,
                reward_term_name,
                RewTerm(
                    func=cast(Callable[..., object], mdp.roboduet_weighted_reward_term),
                    weight=1.0,
                    params={"reward_term_name": reward_term_name, **roboduet_reward_params},
                ),
            )
        self.rewards.total_reward = RewTerm(
            func=cast(Callable[..., object], mdp.roboduet_total_reward_adjustment),
            weight=1.0,
            params=roboduet_reward_params,
        )
        self.rewards.ee_tracking_potential = None

        self.events.randomize_rigid_body_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.1, 3.0),
                "dynamic_friction_range": (0.1, 3.0),
                "restitution_range": (0.0, 0.4),
                "num_buckets": 64,
            },
        )
        self.events.randomize_rigid_body_mass_base = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
                "mass_distribution_params": (-2.0, 2.0),
                "operation": "add",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_rigid_body_mass_ee = None
        self.events.randomize_apply_external_force_torque_base = None
        self.events.randomize_apply_external_force_torque_ee = None
        self.events.randomize_push_robot = None
        # Upstream auto_train samples reset DOF positions as default_joint_pos * U(0.5, 1.5).
        # The shared go2arm base config uses offset reset, so switch the function here as well.
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_scale
        self.events.randomize_reset_joints.params["position_range"] = (0.5, 1.5)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_reset_base.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)}
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }

        self.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
        self.terminations.terrain_out_of_bounds = None
        self.terminations.non_foot_contact_termination = None
        self.terminations.base_orientation_termination = None
        self.terminations.base_height_termination = DoneTerm(
            func=mdp.roboduet_body_height_termination,
            params={
                "minimum_height": 0.28,
                "asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
                "sensor_cfg": None,
            },
        )
        self.terminations.joint_position_termination = None
        self.terminations.joint_velocity_termination = None
        self.terminations.joint_torque_termination = None
        self.terminations.task_success = None
        self.terminations.reverse_termination = DoneTerm(
            func=mdp.roboduet_reverse_termination,
            params={
                "command_name": "roboduet",
                "roll_limit": 0.10,
                "pitch_limit": 0.20,
                "headupdown_thres": 0.10,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.curriculum.go2arm_reaching_stages = None
        self.curriculum.roboduet_stage_switch = CurrTerm(
            func=mdp.roboduet_stage_switch,
            params={"command_name": "roboduet"},
        )

        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.FL_foot_scanner = None
        self.scene.FR_foot_scanner = None
        self.scene.RL_foot_scanner = None
        self.scene.RR_foot_scanner = None
