# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg

from robot_lab.assets.unitree import UNITREE_Go2Arm_ROBODUET_GO2PIPER_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.rough_env_cfg import (
    HYBRID_REWARD_SCALES,
)
from robot_lab.tasks.manager_based.locomotion.velocity.cus_velocity_env_cfg import GO2ARM_LEG_JOINT_NAMES

ROBODUET_GO2PIPER_ARM_JOINT_NAMES = (
    "piper_joint1",
    "piper_joint2",
    "piper_joint3",
    "piper_joint4",
    "piper_joint5",
    "piper_joint6",
)
ROBODUET_GO2PIPER_GRIPPER_JOINT_REGEX = [r"^piper_joint[7-8]$"]
ROBODUET_GO2PIPER_EE_BODY_NAME = "piper_link6"
ROBODUET_GO2PIPER_COLLISION_BODY_REGEX = [
    r"^base$",
    r".*thigh.*",
    r".*calf.*",
    r"^piper_link[1-8]$",
    r"^gripper_base$",
]
ROBODUET_GO2PIPER_EE_ROT_OFFSET_WXYZ = (0.0, 0.7071, 0.0, -0.7071)
ROBODUET_GO2PIPER_EE_POS_OFFSET_LOCAL = (0.12, 0.0, 0.0)


def _set_reward_term_overrides(term_cfg) -> None:
    if term_cfg is None or not hasattr(term_cfg, "params"):
        return
    params = term_cfg.params
    if "illegal_contact_sensor_cfg" in params:
        params["illegal_contact_sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=ROBODUET_GO2PIPER_COLLISION_BODY_REGEX
        )
    if "arm_joint_cfg" in params:
        params["arm_joint_cfg"] = SceneEntityCfg(
            "robot", joint_names=ROBODUET_GO2PIPER_ARM_JOINT_NAMES, preserve_order=True
        )
    if "ee_body_cfg" in params:
        params["ee_body_cfg"] = SceneEntityCfg("robot", body_names=[ROBODUET_GO2PIPER_EE_BODY_NAME])


def _set_term_params(term_cfg, params: dict) -> None:
    if term_cfg is None or not hasattr(term_cfg, "params"):
        return
    term_cfg.params = params


def _set_term_param_item(term_cfg, key: str, value) -> None:
    if term_cfg is None or not hasattr(term_cfg, "params"):
        return
    term_cfg.params[key] = value


def apply_roboduet_go2piper_overrides(env_cfg) -> None:
    env_cfg.roboduet_urdf_mode = "roboduet_go2piper"
    env_cfg.roboduet_arm_joint_names = ROBODUET_GO2PIPER_ARM_JOINT_NAMES
    env_cfg.roboduet_ee_rot_offset_wxyz = ROBODUET_GO2PIPER_EE_ROT_OFFSET_WXYZ
    env_cfg.roboduet_ee_pos_offset_local = ROBODUET_GO2PIPER_EE_POS_OFFSET_LOCAL

    env_cfg.scene.robot = UNITREE_Go2Arm_ROBODUET_GO2PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.robot.spawn.merge_fixed_joints = True
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 4
    env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1

    env_cfg.actions.joint_pos.joint_names = list(GO2ARM_LEG_JOINT_NAMES) + list(ROBODUET_GO2PIPER_ARM_JOINT_NAMES)
    env_cfg.actions.joint_pos.fixed_delta_action_joint_names = [r"^piper_joint[1-6]$"]
    env_cfg.actions.joint_pos.hold_fixed_joint_names = list(ROBODUET_GO2PIPER_GRIPPER_JOINT_REGEX)

    if hasattr(env_cfg.commands, "roboduet") and env_cfg.commands.roboduet is not None:
        env_cfg.commands.roboduet.ee_body_name = ROBODUET_GO2PIPER_EE_BODY_NAME
    if hasattr(env_cfg.commands, "ee_pose") and env_cfg.commands.ee_pose is not None:
        env_cfg.commands.ee_pose.ee_body_name = ROBODUET_GO2PIPER_EE_BODY_NAME

    if hasattr(env_cfg.observations, "arm_policy") and env_cfg.observations.arm_policy is not None:
        _set_term_params(
            getattr(env_cfg.observations.arm_policy, "joint_pos", None),
            {"asset_cfg": SceneEntityCfg("robot", joint_names=ROBODUET_GO2PIPER_ARM_JOINT_NAMES, preserve_order=True)},
        )
    if hasattr(env_cfg.observations, "arm_privileged") and env_cfg.observations.arm_privileged is not None:
        _set_term_params(
            getattr(env_cfg.observations.arm_privileged, "lpy", None),
            {"asset_cfg": SceneEntityCfg("robot", body_names=[ROBODUET_GO2PIPER_EE_BODY_NAME])},
        )
        _set_term_params(
            getattr(env_cfg.observations.arm_privileged, "ee_quat_in_base", None),
            {"asset_cfg": SceneEntityCfg("robot", body_names=[ROBODUET_GO2PIPER_EE_BODY_NAME])},
        )

    _set_term_param_item(
        getattr(env_cfg.events, "randomize_apply_external_force_torque_ee", None),
        "asset_cfg",
        SceneEntityCfg("robot", body_names=[ROBODUET_GO2PIPER_EE_BODY_NAME]),
    )
    _set_term_param_item(
        getattr(env_cfg.events, "randomize_reset_joints", None),
        "asset_cfg",
        SceneEntityCfg(
            "robot",
            joint_names=list(GO2ARM_LEG_JOINT_NAMES) + list(ROBODUET_GO2PIPER_ARM_JOINT_NAMES),
        ),
    )

    for reward_term_name in list(HYBRID_REWARD_SCALES.keys()) + ["total_reward"]:
        reward_term_cfg = getattr(env_cfg.rewards, reward_term_name, None)
        _set_reward_term_overrides(reward_term_cfg)
