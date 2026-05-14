# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import (
    GO2ARM_COMMAND_CURRICULUM_KEYS,
    GO2ARM_FOOT_BODY_NAMES,
    GO2ARM_FOOT_SCANNER_NAMES,
    GO2ARM_FOOT_SPHERE_RADIUS,
    _get_go2arm_foot_kinematics,
    _get_go2arm_ground_height_data,
    _quat_to_abg,
    _quat_to_roll_pitch,
    _go2arm_phase_offsets,
    get_go2arm_precise_foot_contact_forces,
    get_go2arm_precise_foot_contact_timers,
    get_go2arm_precise_foot_normal_forces,
    roboduet_base_pose_w,
    roboduet_base_velocity_b,
    roboduet_current_action,
    roboduet_current_ee_quat_in_base,
    roboduet_current_lpy,
    roboduet_projected_gravity_b,
)

_ROBODUET_COMMAND_LOG_EXTRA_KEYS = (
    "lin_vel_raw",
    "ang_vel_raw",
    "lin_vel_residual",
    "ang_vel_residual",
    "ep_timesteps",
)
_ROBODUET_DOG_ONLY_TERMS = ("tracking_lin_vel", "tracking_ang_vel")
_ROBODUET_CONTACT_OFFSET_TERMS = ("tracking_contacts_shaped_force", "tracking_contacts_shaped_vel")
_ROBODUET_LOG_ONLY_TERMS = ("vis_manip_commands_tracking_lpy", "vis_manip_commands_tracking_rpy")


def _scale_is_nonzero(scale: float | int | None) -> bool:
    return float(scale or 0.0) != 0.0


def _cached_row_tensor(
    env: ManagerBasedRLEnv,
    cache_name: str,
    values: Sequence[float],
    dtype: torch.dtype,
) -> torch.Tensor:
    cache_key = (tuple(float(value) for value in values), str(env.device), str(dtype))
    cached = getattr(env, cache_name, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["value"]
    tensor = torch.tensor(cache_key[0], device=env.device, dtype=dtype).unsqueeze(0)
    setattr(env, cache_name, {"key": cache_key, "value": tensor})
    return tensor


def _ee_pose_command_term(env: ManagerBasedRLEnv, command_name: str):
    """Return the ee-pose command term used by go2arm."""
    # 从 command manager 里取出当前 ee_pose 命令项，后面直接复用它缓存的目标和误差量。
    return env.command_manager.get_term(command_name)


def _get_go2arm_total_reward_params(env: ManagerBasedRLEnv, total_reward_term_name: str) -> dict:
    """Cache total_reward params lookup since the config is static during training."""
    cache_key = ("go2arm_total_reward_params", total_reward_term_name)
    cached = getattr(env, "_go2arm_term_cfg_cache", None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["value"]
    params = env.reward_manager.get_term_cfg(total_reward_term_name).params
    env._go2arm_term_cfg_cache = {"key": cache_key, "value": params}
    return params


def _get_go2arm_potential_term_cfg(env: ManagerBasedRLEnv):
    """Cache ee_tracking_potential term cfg lookup."""
    cached = getattr(env, "_go2arm_potential_term_cfg", None)
    if cached is None:
        cached = env.reward_manager.get_term_cfg("ee_tracking_potential")
        env._go2arm_potential_term_cfg = cached
    return cached


def _ensure_roboduet_logging_buffers(
    env: ManagerBasedRLEnv, hybrid_scales: dict[str, float]
) -> dict[str, torch.Tensor]:
    reward_names = tuple(hybrid_scales.keys())
    expected_command_names = reward_names + _ROBODUET_COMMAND_LOG_EXTRA_KEYS
    if getattr(env, "_roboduet_reward_term_names", None) != reward_names:
        env._roboduet_reward_term_names = reward_names
        env._roboduet_log_episode_sums = {
            name: torch.zeros(env.num_envs, dtype=torch.float32, device=env.device) for name in reward_names
        }
        env._roboduet_log_episode_sums["total"] = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if getattr(env, "_roboduet_command_term_names", None) != expected_command_names:
        env._roboduet_command_term_names = expected_command_names
        env._roboduet_command_sums = {
            name: torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
            for name in expected_command_names
        }
    return env._roboduet_command_sums


def _get_go2arm_support_contact_stats(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float,
) -> dict[str, torch.Tensor]:
    """Cache per-step support-contact statistics reused by reward logging."""
    cache_key = (
        getattr(env, "common_step_counter", -1),
        sensor_cfg.name,
        tuple(int(body_id) for body_id in sensor_cfg.body_ids),
        float(contact_force_threshold),
    )
    cached = getattr(env, "_go2arm_support_contact_cache", None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["value"]

    precise_normal_forces = get_go2arm_precise_foot_normal_forces(env, sensor_cfg=sensor_cfg)
    if precise_normal_forces is None:
        raise RuntimeError("Go2Arm support-contact stats require the dedicated foot sensors to be available.")
    in_contact = precise_normal_forces > contact_force_threshold
    support_count = torch.sum(in_contact.float(), dim=1)
    value = {
        "in_contact": in_contact,
        "support_count": support_count,
    }
    env._go2arm_support_contact_cache = {"key": cache_key, "value": value}
    return value


def _phi_quadratic(value: torch.Tensor, std: float) -> torch.Tensor:
    """Compute phi(v, std) = exp(-v^T v / std)."""
    if value.ndim > 1:
        quad = torch.sum(torch.square(value), dim=-1)
    else:
        quad = torch.square(value)
    return torch.exp(-quad / std)


def _sigmoid_gate(x: torch.Tensor, mu: float = 1.5, l: float = 1.0) -> torch.Tensor:  # noqa: E741
    """Compute the gating sigmoid D(x) with center mu and scale l."""
    if l <= 0.0:
        raise ValueError("l must be positive for the reward gating sigmoid.")
    k = 5.0 / l
    # D(x) = 1 / (1 + exp(-k * (x - mu))).
    return torch.sigmoid(k * (x - mu))


def _quat_roll_abs(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return absolute roll angle from quaternions in wxyz order."""
    # 按 wxyz 顺序拆四元数。
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    # 标准 roll 恢复公式。
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    # 这里取绝对值，只关心横滚偏离大小，不区分左右方向。
    return torch.abs(torch.atan2(sinr_cosp, cosr_cosp))


def ee_position_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward ee position tracking with exp(-d / sigma)."""
    # 直接复用 ee_pose 命令项里每步缓存好的位置误差。
    pos_error = _ee_pose_command_term(env, command_name).position_tracking_error
    # 原始位置跟踪奖励 r = exp(-d / sigma)。
    return torch.exp(-pos_error / std)


def ee_orientation_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward ee orientation tracking with exp(-d / sigma)."""
    # 直接复用 ee_pose 命令项里每步缓存好的姿态误差。
    rot_error = _ee_pose_command_term(env, command_name).orientation_tracking_error
    # 原始姿态跟踪奖励 r = exp(-d / sigma)。
    return torch.exp(-rot_error / std)


def enhance_exponential_tracking_reward(reward: torch.Tensor, power: float = 5.0) -> torch.Tensor:
    """Enhance an already-computed exponential tracking reward with r + r^M."""
    # 这里不改变原始奖励定义，只在原始奖励 r 已经算出来之后，再构造增强后的奖励值。
    return reward + torch.pow(reward, power)


def ee_position_tracking_enhanced(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    power: float = 5.0,
) -> torch.Tensor:
    """Enhanced ee position tracking reward built on top of the original reward."""
    # 先算原始位置跟踪奖励，再做 r + r^M 增强，便于后面同时保留原始值和增强值。
    reward = ee_position_tracking_exp(env, command_name=command_name, std=std)
    return enhance_exponential_tracking_reward(reward, power=power)


def ee_orientation_tracking_enhanced(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    power: float = 5.0,
) -> torch.Tensor:
    """Enhanced ee orientation tracking reward built on top of the original reward."""
    # 先算原始姿态跟踪奖励，再做 r + r^M 增强，便于后面同时保留原始值和增强值。
    reward = ee_orientation_tracking_exp(env, command_name=command_name, std=std)
    return enhance_exponential_tracking_reward(reward, power=power)


def ee_cumulative_tracking_error_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    clip_max: float,
) -> torch.Tensor:
    """Penalty term based on clipped cumulative tracking error."""
    # 直接复用命令项里维护的累计总跟踪误差。
    command_term = _ee_pose_command_term(env, command_name)
    # 按你的要求做上界截断，避免惩罚无界增长。
    penalty = torch.clamp(command_term.cumulative_tracking_error, min=0.0, max=clip_max)

    return penalty


class EETrackingPotentialReward(ManagerTermBase):
    """Potential-based shaping on the total ee tracking error."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 命令名固定指向 ee_pose。
        self.command_name: str = cfg.params["command_name"]
        # 势函数当前定义为 phi = -tracking_error。
        # 仍然保留 current / prev / last_reward 三组缓存，方便沿用现有调试日志。
        self.current_potential = torch.zeros(env.num_envs, device=env.device)
        self.prev_potential = torch.zeros(env.num_envs, device=env.device)
        self.last_reward = torch.zeros(env.num_envs, device=env.device)
        # 标记哪些 env 已经完成过至少一步有效初始化。
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset cached potential values."""
        if env_ids is None:
            env_ids = slice(None)
        self.current_potential[env_ids] = 0.0
        self.prev_potential[env_ids] = 0.0
        self.last_reward[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        gain: float,
        clip_min: float,
        clip_max: float,
        eps: float,
    ) -> torch.Tensor:
        # 这里的 d 明确就是命令项里的总跟踪误差 tracking_error。
        command_term = _ee_pose_command_term(env, self.command_name)
        # 当前势函数定义为 phi_t = -tracking_error_t。
        tracking_error = command_term.tracking_error
        current_potential = -tracking_error
        self.current_potential = current_potential
        # 势奖励仍然保持差分形式：phi_t - phi_{t-1}。
        reward = current_potential - self.prev_potential

        # episode 第一步没有上一时刻，直接置零更稳妥。
        first_step_mask = env.episode_length_buf == 0
        reward = torch.where(first_step_mask | ~self._initialized, torch.zeros_like(reward), reward)

        # 更新缓存，供下一步使用。
        self.prev_potential = current_potential.clone()
        self.last_reward = reward
        self._initialized[:] = True

        return reward


def support_roll_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on the absolute base roll angle."""
    # 读取 base 对应的刚体数据。
    # 用 roll 的平方作为最常见的横滚惩罚形式。
    asset: RigidObject = env.scene[asset_cfg.name]
    penalty = torch.square(_quat_roll_abs(asset.data.root_quat_w))

    return penalty


def support_non_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    count_weight: float = 1.0,
    force_weight: float = 1.0,
    force_scale: float = 1.0,
) -> torch.Tensor:
    """Penalty on non-foot contacts."""
    # 这个项约束“只有脚可以接触地面”，任何非足端接触都记惩罚。
    # 采用“超阈值 body 计数 + 超阈值力幅值”组合形式，阈值用于抑制接触传感器噪声。
    return collision_force_count_penalty(
        env=env,
        threshold=threshold,
        sensor_cfg=sensor_cfg,
        count_weight=count_weight,
        force_weight=force_weight,
        force_scale=force_scale,
    )


def support_feet_slide_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty on feet sliding without extra gating."""
    # 这个项约束“脚在接触地面时不要在世界系地面上横向打滑”。
    precise_normal_forces = get_go2arm_precise_foot_normal_forces(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    if precise_normal_forces is not None:
        # go2arm 下只把“合法足端 patch”视为支撑接触，避免其它 body 上的非法接触误触发滑移项。
        contacts = precise_normal_forces > 1.0
    else:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
    # 用 foot 刚体和球心局部偏移恢复足端碰撞球心线速度。
    feet_lin_vel_w = _get_go2arm_foot_kinematics(env, asset_cfg)["foot_center_lin_vel_w"]
    # 只取世界系 xy 平面速度，作为地面切向滑动速度。
    foot_lateral_vel = torch.sqrt(torch.sum(torch.square(feet_lin_vel_w[:, :, :2]), dim=2)).view(env.num_envs, -1)
    # 接触中的脚滑得越快，惩罚越大。
    return torch.sum(foot_lateral_vel * contacts, dim=1)


def support_foot_air_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty on feet that are not stably in contact with the ground."""
    precise_normal_forces = get_go2arm_precise_foot_normal_forces(env, sensor_cfg=sensor_cfg)
    if precise_normal_forces is not None:
        in_contact = precise_normal_forces > threshold
    else:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        in_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # 对每个 env 统计当前不在接触中的足端数量，作为“不触地”惩罚。
    penalty = torch.sum(~in_contact, dim=1).float()

    return penalty


def posture_deviation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Penalty on deviation from the default joint posture."""
    # 这里 asset_cfg 会在配置里限制为机械臂 6 个关节。
    asset: Articulation = env.scene[asset_cfg.name]
    # 用当前关节位置与默认构型的欧氏距离度量姿态扭曲。
    joint_delta = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if joint_weights is None:
        penalty = torch.linalg.norm(joint_delta, dim=1)
    else:
        weights = torch.as_tensor(joint_weights, dtype=joint_delta.dtype, device=joint_delta.device)
        if weights.ndim != 1 or weights.numel() != joint_delta.shape[1]:
            raise ValueError(
                "posture_deviation_penalty joint_weights must be a 1D sequence with length matching asset_cfg joints."
            )
        penalty = torch.sqrt(torch.sum(weights.unsqueeze(0) * torch.square(joint_delta), dim=1))

    return penalty


def joint_limit_safety_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on approaching joint position limits."""
    # 直接复用 isaaclab 的 joint_pos_limits，配置里限制为机械臂 6 个关节。
    penalty = mdp.joint_pos_limits(env, asset_cfg)

    return penalty


def locomotion_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    threshold: float,
) -> torch.Tensor:
    """Reward locomotion quality from the gap between reference and actual total tracking errors."""
    # 直接复用 ee_pose 命令项里缓存的参考误差和当前总跟踪误差。
    command_term = _ee_pose_command_term(env, command_name)
    # 先取参考误差与当前误差之差的绝对值。
    error_gap = torch.abs(command_term.reference_tracking_error - command_term.tracking_error)
    # 再减去超参数 threshold，并与 0 取最大值。
    d = torch.clamp(error_gap - threshold, min=0.0)
    # 最终按你指定的指数形式 exp(-d / std) 给奖励。
    return torch.exp(-d / std)


def moving_arm_default_deviation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Penalty on arm joint deviation from default."""
    # 这里只约束机械臂 6 个关节不要偏离默认构型，不再额外加“移动时才生效”的门控。
    asset: Articulation = env.scene[asset_cfg.name]
    # 用关节位置相对默认位姿的平方偏差作为惩罚，偏得越远惩罚越大。
    return torch.sum(
        torch.square(
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def moving_arm_joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on arm joint velocity magnitude."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize arm motion during locomotion to avoid rapid arm shaking.
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def _asset_body_or_root_height_w(asset: RigidObject, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return selected body height when body_ids are resolved, otherwise articulation root height."""
    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None:
        return asset.data.root_pos_w[:, 2]
    try:
        if len(body_ids) == 0:
            return asset.data.root_pos_w[:, 2]
    except TypeError:
        pass
    body_height = asset.data.body_pos_w[:, body_ids, 2]
    if body_height.ndim > 1:
        body_height = torch.mean(body_height, dim=1)
    return body_height


def base_height_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalty on base-height deviation normalized by std."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_height = _asset_body_or_root_height_w(asset, asset_cfg)
    if sensor_cfg is not None:
        # 有地形扫描器时，用机身高度减去局部地面高度来构造相对高度误差。
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            # 扫描结果异常时退化成直接使用世界系机身高度，避免 reward 数值异常。
            ground_z = torch.zeros_like(body_height)
        else:
            ground_z = torch.mean(ray_hits, dim=1)
        height_error = body_height - (target_height + ground_z)
    else:
        # 没有地形扫描器时，直接对世界系目标高度做约束。
        height_error = body_height - target_height
    # 奖励形式是 phi(h - h_ref, std) = exp(-(h-h_ref)^2 / std)。
    return torch.abs(height_error) / std


def min_base_height_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """One-sided penalty when base height falls below the minimum height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_height = _asset_body_or_root_height_w(asset, asset_cfg)
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            ground_z = torch.zeros_like(body_height)
        else:
            ground_z = torch.mean(ray_hits, dim=1)
        base_height = body_height - ground_z
    else:
        base_height = body_height
    return torch.clamp(minimum_height - base_height, min=0.0)


def base_roll_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on base roll normalized by std."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # 这里的 roll 用四元数恢复，再套 phi(roll, std)。
    return _quat_roll_abs(asset.data.root_quat_w) / std


def base_pitch_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on base pitch normalized by std."""
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_wxyz = asset.data.root_quat_w
    # 从四元数恢复 pitch 角，再按 phi(pitch, std) 计算。
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    return torch.abs(pitch) / std


def base_pitch_signed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return signed base pitch in radians."""
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_wxyz = asset.data.root_quat_w
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    sinp = 2.0 * (w * y - z * x)
    return torch.asin(torch.clamp(sinp, -1.0, 1.0))


def target_height_conditioned_pitch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    low_height: float,
    high_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pitch sign that contradicts target height.

    For go2arm, positive signed pitch corresponds to pitching the front of the base downward.
    High world-z targets penalize downward pitch; low world-z targets penalize upward pitch.
    Mid-height targets are left unconstrained.
    """
    command_term = _ee_pose_command_term(env, command_name)
    target_z_w = command_term.target_pos_w[:, 2]
    pitch = base_pitch_signed(env, asset_cfg=asset_cfg)
    high_penalty = (target_z_w > high_height).float() * torch.relu(pitch)
    low_penalty = (target_z_w < low_height).float() * torch.relu(-pitch)
    return high_penalty + low_penalty


def target_workspace_position_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    x_min: float,
    x_max: float,
    y_weight: float,
    std: float,
) -> torch.Tensor:
    """Reward target placement inside the comfortable base-frame manipulation workspace."""
    command_term = _ee_pose_command_term(env, command_name)
    target_pos_b = command_term.target_pos_b
    x = target_pos_b[:, 0]
    y = target_pos_b[:, 1]
    x_violation = torch.relu(float(x_min) - x) + torch.relu(x - float(x_max))
    y_penalty = float(y_weight) * torch.abs(y)
    distance = x_violation + y_penalty
    return torch.exp(-distance / max(float(std), 1.0e-6))


def base_roll_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on base roll angular velocity normalized by std."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # 直接取 base frame 下的 roll 角速度 wx。
    return torch.abs(asset.data.root_ang_vel_b[:, 0]) / std


def base_pitch_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on base pitch angular velocity normalized by std."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # 直接取 base frame 下的 pitch 角速度 wy。
    return torch.abs(asset.data.root_ang_vel_b[:, 1]) / std


def base_z_vel_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on base z velocity normalized by std."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # 直接约束 base frame 下的竖直速度 vz。
    return torch.abs(asset.data.root_lin_vel_b[:, 2]) / std


def base_lateral_vel_penalty(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty on base lateral velocity in the body frame normalized by std."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # 鐩存帴绾︽潫 base frame 涓嬬殑渚у悜閫熷害 vy锛岀敤浜庢姂鍒垛€滄枩鐫€璧/妯潃婕傗€濈殑绉诲姩鏂瑰紡銆?
    return torch.abs(asset.data.root_lin_vel_b[:, 1]) / std


def feet_contact_soft_trot_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_std: float,
    height_std: float,
    vel_std: float,
    cycle_time: float,
    phase_offsets: Sequence[float],
    swing_height: float,
    soft_contact_k: float,
    contact_force_threshold: float = 1.0,
    ground_sensor_names: Sequence[str] | None = None,
) -> torch.Tensor:
    # 整体结构参考feet-contact reward，但期望接触 c_des 改成软权重。
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    del asset

    # 用 episode 时间构造步态相位，并给每只脚加各自的相位偏移。
    phase_t = env.episode_length_buf.to(torch.float32) * env.step_dt / cycle_time
    offsets = _go2arm_phase_offsets(env.device, torch.float32, tuple(phase_offsets))
    phase_signal = torch.sin(2.0 * torch.pi * (phase_t.unsqueeze(-1) + offsets))
    # 软接触权重 c_des in (0, 1)，替代原先硬切换的期望接触标签。
    c_des = torch.sigmoid(-soft_contact_k * phase_signal)
    # 摆动期的期望足高，接触权重越低，期望抬脚越高。
    h_hat = (1.0 - c_des) * swing_height

    # 读取每只脚的接触力。
    # go2arm 下优先使用逐 patch 聚合后的精确合法足端接触力与法向力；否则退化到 body 级接触力。
    precise_forces = get_go2arm_precise_foot_contact_forces(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    precise_normal_forces = get_go2arm_precise_foot_normal_forces(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    if precise_forces is not None and precise_normal_forces is not None:
        net_forces = precise_forces
        force_mag = torch.linalg.norm(net_forces, dim=-1)
        force_z = precise_normal_forces
    else:
        net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
        force_mag = torch.linalg.norm(net_forces, dim=-1)
        force_z = torch.abs(net_forces[..., 2])

    # 用 foot 刚体和球心局部偏移恢复世界系足端碰撞球心位置。
    foot_kinematics = _get_go2arm_foot_kinematics(env, asset_cfg)
    foot_sphere_centers_w = foot_kinematics["foot_sphere_centers_w"]
    if ground_sensor_names is not None:
        if len(ground_sensor_names) != len(asset_cfg.body_ids):
            raise ValueError("ground_sensor_names must match the number of feet.")
        ground_z = torch.zeros(env.num_envs, len(asset_cfg.body_ids), device=env.device)
        for i, sensor_name in enumerate(ground_sensor_names):
            sensor: RayCaster = env.scene.sensors[sensor_name]
            ray_hits = sensor.data.ray_hits_w[..., 2]
            if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
                # 地面高度异常时退化成本脚当前高度，避免数值污染。
                ground_z[:, i] = foot_sphere_centers_w[:, i, 2] - GO2ARM_FOOT_SPHERE_RADIUS
            else:
                ground_z[:, i] = torch.mean(ray_hits, dim=1)
        h_z = foot_sphere_centers_w[..., 2] - GO2ARM_FOOT_SPHERE_RADIUS - ground_z
    else:
        h_z = foot_sphere_centers_w[..., 2] - GO2ARM_FOOT_SPHERE_RADIUS

    # 摆动期希望“脚抬起来且不要有大接触力”。
    height_err = h_hat - h_z
    swing_term = (1.0 - c_des) * torch.exp(-(force_mag**2) / force_std) * torch.exp(-(height_err**2) / height_std)

    # 站立期希望“脚落地且世界系水平速度小”，避免支撑脚打滑。
    contact_mask = force_z > contact_force_threshold
    in_contact = contact_mask.float()
    vel_xy = torch.linalg.norm(foot_kinematics["foot_center_lin_vel_w"][..., :2], dim=-1)
    stance_term = c_des * in_contact * torch.exp(-(vel_xy**2) / vel_std)

    # 四只脚的摆动项和支撑项加总，作为整体步态奖励。
    return torch.sum(swing_term + stance_term, dim=1)


def precise_feet_contact_count(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Return support-foot count using the dedicated foot sensors."""
    return _get_go2arm_support_contact_stats(
        env,
        sensor_cfg=sensor_cfg,
        contact_force_threshold=contact_force_threshold,
    )["support_count"]


def diagonal_foot_symmetry_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
    timer_std: float = 0.2,
) -> torch.Tensor:
    """Penalty on diagonal-foot contact, force, and timer mismatch."""
    precise_timer_data = get_go2arm_precise_foot_contact_timers(
        env, sensor_cfg=sensor_cfg, threshold=contact_force_threshold
    )
    if precise_timer_data is not None:
        air_time = precise_timer_data["current_air_time"]
        contact_time = precise_timer_data["current_contact_time"]
    else:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    precise_normal_forces = get_go2arm_precise_foot_normal_forces(env, sensor_cfg=sensor_cfg)
    if precise_normal_forces is not None:
        normal_forces = precise_normal_forces
    else:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        normal_forces = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])

    contact = (normal_forces > contact_force_threshold).float()
    force_state = torch.clamp(normal_forces / max(float(contact_force_threshold), 1.0e-6), min=0.0, max=1.0)
    timer_scale = max(float(timer_std) ** 2, 1.0e-6)

    def _pair_penalty(first: int, second: int) -> torch.Tensor:
        contact_mismatch = torch.square(contact[:, first] - contact[:, second])
        force_mismatch = torch.square(force_state[:, first] - force_state[:, second])
        timer_mismatch = (
            torch.square(air_time[:, first] - air_time[:, second])
            + torch.square(contact_time[:, first] - contact_time[:, second])
        ) / timer_scale
        return 0.5 * contact_mismatch + 0.3 * force_mismatch + 0.2 * timer_mismatch

    return 0.5 * (_pair_penalty(0, 3) + _pair_penalty(1, 2))


def _get_go2arm_foot_centers_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return current foot sphere centers in the root-link frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_centers_w = _get_go2arm_foot_kinematics(env, asset_cfg)["foot_sphere_centers_w"]
    foot_centers_rel_w = foot_centers_w - asset.data.root_link_pos_w.unsqueeze(1)
    foot_centers_b = torch.zeros_like(foot_centers_rel_w)
    for i in range(foot_centers_rel_w.shape[1]):
        foot_centers_b[:, i, :] = quat_apply_inverse(asset.data.root_link_quat_w, foot_centers_rel_w[:, i, :])
    return foot_centers_b


def _get_go2arm_default_foot_sensor_cfg(env: ManagerBasedRLEnv) -> SceneEntityCfg:
    """Return the default four-foot contact sensor config used by go2arm symmetry terms."""
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_cfg.body_ids, _ = contact_sensor.find_bodies(sensor_cfg.body_names, preserve_order=True)
    return sensor_cfg


def _get_go2arm_default_foot_asset_cfg(env: ManagerBasedRLEnv) -> SceneEntityCfg:
    """Return the default four-foot asset config used by go2arm symmetry terms."""
    asset_cfg = SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES, preserve_order=True)
    asset: Articulation = env.scene[asset_cfg.name]
    asset_cfg.body_ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
    return asset_cfg


def _get_go2arm_foot_contact_mask(env: ManagerBasedRLEnv, contact_force_threshold: float = 1.0) -> torch.Tensor:
    """Return a per-foot boolean contact mask using the default go2arm feet."""
    sensor_cfg = _get_go2arm_default_foot_sensor_cfg(env)
    precise_normal_forces = get_go2arm_precise_foot_normal_forces(env, sensor_cfg=sensor_cfg)
    if precise_normal_forces is not None:
        return precise_normal_forces > contact_force_threshold
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > contact_force_threshold
    )


def _compute_go2arm_left_right_symmetry_components(foot_positions_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute front/rear left-right symmetry penalties on x and y separately."""
    x_symmetry = 0.5 * (
        torch.square(foot_positions_b[:, 0, 0] - foot_positions_b[:, 1, 0])
        + torch.square(foot_positions_b[:, 2, 0] - foot_positions_b[:, 3, 0])
    )
    y_symmetry = 0.5 * (
        torch.square(foot_positions_b[:, 0, 1] + foot_positions_b[:, 1, 1])
        + torch.square(foot_positions_b[:, 2, 1] + foot_positions_b[:, 3, 1])
    )
    return x_symmetry, y_symmetry


def _get_go2arm_touchdown_symmetry_components(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Cache left-right touchdown positions and return mirrored x/y mismatch penalties."""
    sensor_cfg = _get_go2arm_default_foot_sensor_cfg(env)
    asset_cfg = _get_go2arm_default_foot_asset_cfg(env)
    contact_force_threshold = 1.0
    state = getattr(env, "_go2arm_touchdown_symmetry_state", None)
    state_key = (
        sensor_cfg.name,
        tuple(int(body_id) for body_id in sensor_cfg.body_ids),
        asset_cfg.name,
        tuple(int(body_id) for body_id in asset_cfg.body_ids),
        float(contact_force_threshold),
    )
    if state is None or state.get("state_key") != state_key:
        last_touchdown_positions_b = torch.zeros(env.num_envs, 4, 3, device=env.device)
        valid_touchdown = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        prev_contact = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        state = {
            "state_key": state_key,
            "last_touchdown_positions_b": last_touchdown_positions_b,
            "valid_touchdown": valid_touchdown,
            "prev_contact": prev_contact,
            "cached_step": None,
            "cached_value": None,
        }
        env._go2arm_touchdown_symmetry_state = state

    step_index = int(getattr(env, "common_step_counter", -1))
    if state["cached_step"] == step_index and state["cached_value"] is not None:
        return state["cached_value"]

    foot_positions_b = _get_go2arm_foot_centers_b(env, asset_cfg)
    in_contact = _get_go2arm_foot_contact_mask(env, contact_force_threshold=contact_force_threshold)

    # episode 重置后清空逐脚触地缓存，避免上一条轨迹的落脚点泄漏到新 episode。
    reset_mask = env.episode_length_buf == 0
    if torch.any(reset_mask):
        state["last_touchdown_positions_b"][reset_mask] = 0.0
        state["valid_touchdown"][reset_mask] = False
        state["prev_contact"][reset_mask] = False

    new_touchdown = in_contact & ~state["prev_contact"]
    if torch.any(new_touchdown):
        state["last_touchdown_positions_b"] = torch.where(
            new_touchdown.unsqueeze(-1),
            foot_positions_b,
            state["last_touchdown_positions_b"],
        )
        state["valid_touchdown"] = state["valid_touchdown"] | new_touchdown

    state["prev_contact"] = in_contact.clone()

    front_pair_valid = state["valid_touchdown"][:, 0] & state["valid_touchdown"][:, 1]
    rear_pair_valid = state["valid_touchdown"][:, 2] & state["valid_touchdown"][:, 3]
    pair_valid = torch.stack((front_pair_valid, rear_pair_valid), dim=1)

    front_x = torch.square(state["last_touchdown_positions_b"][:, 0, 0] - state["last_touchdown_positions_b"][:, 1, 0])
    rear_x = torch.square(state["last_touchdown_positions_b"][:, 2, 0] - state["last_touchdown_positions_b"][:, 3, 0])
    front_y = torch.square(state["last_touchdown_positions_b"][:, 0, 1] + state["last_touchdown_positions_b"][:, 1, 1])
    rear_y = torch.square(state["last_touchdown_positions_b"][:, 2, 1] + state["last_touchdown_positions_b"][:, 3, 1])

    valid_pair_count = torch.clamp(pair_valid.float().sum(dim=1), min=1.0)
    x_penalty = (front_x * front_pair_valid.float() + rear_x * rear_pair_valid.float()) / valid_pair_count
    y_penalty = (front_y * front_pair_valid.float() + rear_y * rear_pair_valid.float()) / valid_pair_count
    no_valid_pair = ~(front_pair_valid | rear_pair_valid)
    x_penalty = torch.where(no_valid_pair, torch.zeros_like(x_penalty), x_penalty)
    y_penalty = torch.where(no_valid_pair, torch.zeros_like(y_penalty), y_penalty)

    cached_value = {
        "touchdown_left_right_x_symmetry": x_penalty,
        "touchdown_left_right_y_symmetry": y_penalty,
    }
    state["cached_step"] = step_index
    state["cached_value"] = cached_value
    return cached_value


def support_left_right_x_symmetry_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on left-right x symmetry while all four feet are supporting."""
    asset_cfg = _get_go2arm_default_foot_asset_cfg(env)
    foot_positions_b = _get_go2arm_foot_centers_b(env, asset_cfg)
    x_symmetry, _ = _compute_go2arm_left_right_symmetry_components(foot_positions_b)
    in_contact = _get_go2arm_foot_contact_mask(env, contact_force_threshold=1.0)
    support_mask = torch.all(in_contact, dim=1)
    return x_symmetry * support_mask.float()


def support_left_right_y_symmetry_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on left-right y mirror symmetry while all four feet are supporting."""
    asset_cfg = _get_go2arm_default_foot_asset_cfg(env)
    foot_positions_b = _get_go2arm_foot_centers_b(env, asset_cfg)
    _, y_symmetry = _compute_go2arm_left_right_symmetry_components(foot_positions_b)
    in_contact = _get_go2arm_foot_contact_mask(env, contact_force_threshold=1.0)
    support_mask = torch.all(in_contact, dim=1)
    return y_symmetry * support_mask.float()


def support_foot_xy_range_penalty(
    env: ManagerBasedRLEnv,
    x_abs_min: float,
    x_abs_max: float | None,
    y_abs_min: float,
    y_abs_max: float | None,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty on supporting foot centers outside a body-frame abs-x/abs-y range."""
    asset_cfg = _get_go2arm_default_foot_asset_cfg(env)
    foot_positions_b = _get_go2arm_foot_centers_b(env, asset_cfg)
    abs_x = torch.abs(foot_positions_b[..., 0])
    abs_y = torch.abs(foot_positions_b[..., 1])

    x_violation = torch.square(torch.clamp(float(x_abs_min) - abs_x, min=0.0))
    y_violation = torch.square(torch.clamp(float(y_abs_min) - abs_y, min=0.0))
    if x_abs_max is not None:
        x_violation = x_violation + torch.square(torch.clamp(abs_x - float(x_abs_max), min=0.0))
    if y_abs_max is not None:
        y_violation = y_violation + torch.square(torch.clamp(abs_y - float(y_abs_max), min=0.0))

    in_contact = _get_go2arm_foot_contact_mask(env, contact_force_threshold=contact_force_threshold)
    violation = (x_violation + y_violation) * in_contact.float()
    contact_count = torch.clamp(in_contact.float().sum(dim=1), min=1.0)
    penalty = torch.sum(violation, dim=1) / contact_count
    no_contact = ~torch.any(in_contact, dim=1)
    return torch.where(no_contact, torch.zeros_like(penalty), penalty)


def touchdown_left_right_x_symmetry_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on left-right touchdown x mismatch using the most recent touchdown cache per foot."""
    return _get_go2arm_touchdown_symmetry_components(env)["touchdown_left_right_x_symmetry"]


def touchdown_left_right_y_symmetry_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on left-right touchdown y mirror mismatch using the most recent touchdown cache per foot."""
    return _get_go2arm_touchdown_symmetry_components(env)["touchdown_left_right_y_symmetry"]


def touchdown_foot_y_distance_penalty(
    env: ManagerBasedRLEnv,
    min_distance: float,
    max_distance: float | None = None,
) -> torch.Tensor:
    """Penalty on recent touchdown points whose lateral distance leaves the target width band."""
    _get_go2arm_touchdown_symmetry_components(env)
    state = env._go2arm_touchdown_symmetry_state
    touchdown_y_distance = torch.abs(state["last_touchdown_positions_b"][..., 1])
    lower_violation = torch.clamp(min_distance - touchdown_y_distance, min=0.0)
    upper_violation = torch.zeros_like(lower_violation)
    if max_distance is not None:
        upper_violation = torch.clamp(touchdown_y_distance - max_distance, min=0.0)
    foot_y_distance = torch.square(lower_violation) + torch.square(upper_violation)
    valid_foot_count = torch.clamp(state["valid_touchdown"].float().sum(dim=1), min=1.0)
    foot_y_distance = torch.sum(foot_y_distance * state["valid_touchdown"].float(), dim=1) / valid_foot_count
    no_valid_foot = ~torch.any(state["valid_touchdown"], dim=1)
    return torch.where(no_valid_foot, torch.zeros_like(foot_y_distance), foot_y_distance)


def mani_regularization_reward(
    env: ManagerBasedRLEnv,
    support_roll_weight: float,
    support_roll_asset_cfg: SceneEntityCfg,
    support_roll_std: float,
    support_feet_slide_weight: float,
    support_feet_slide_sensor_cfg: SceneEntityCfg,
    support_feet_slide_asset_cfg: SceneEntityCfg,
    support_feet_slide_std: float,
    support_foot_air_weight: float,
    support_foot_air_threshold: float,
    support_foot_air_sensor_cfg: SceneEntityCfg,
    support_foot_air_clip_max: float,
    support_non_foot_contact_weight: float,
    support_non_foot_contact_threshold: float,
    support_non_foot_contact_sensor_cfg: SceneEntityCfg,
    support_non_foot_contact_count_weight: float,
    support_non_foot_contact_force_weight: float,
    support_non_foot_contact_force_scale: float,
    support_non_foot_contact_clip_max: float,
    posture_deviation_weight: float,
    posture_deviation_asset_cfg: SceneEntityCfg,
    posture_deviation_std: float,
    posture_deviation_joint_weights: Sequence[float] | None,
    joint_limit_safety_weight: float,
    joint_limit_safety_asset_cfg: SceneEntityCfg,
    joint_limit_safety_std: float,
    support_left_right_x_symmetry_weight: float,
    support_left_right_x_symmetry_std: float,
    support_left_right_y_symmetry_weight: float,
    support_left_right_y_symmetry_std: float,
) -> torch.Tensor:
    """Manipulation regularization as exp(-weighted normalized penalties)."""
    # 这里只汇总 mani 内部已经确定的正则项，不包含 tracking / cumulative / potential 主项。
    support_roll = support_roll_penalty(env, asset_cfg=support_roll_asset_cfg) / support_roll_std
    support_feet_slide = (
        support_feet_slide_penalty(
            env, sensor_cfg=support_feet_slide_sensor_cfg, asset_cfg=support_feet_slide_asset_cfg
        )
        / support_feet_slide_std
    )
    support_foot_air = torch.clamp(
        support_foot_air_penalty(env, threshold=support_foot_air_threshold, sensor_cfg=support_foot_air_sensor_cfg),
        min=0.0,
        max=support_foot_air_clip_max,
    )
    support_non_foot_contact = torch.clamp(
        support_non_foot_contact_penalty(
            env,
            threshold=support_non_foot_contact_threshold,
            sensor_cfg=support_non_foot_contact_sensor_cfg,
            count_weight=support_non_foot_contact_count_weight,
            force_weight=support_non_foot_contact_force_weight,
            force_scale=support_non_foot_contact_force_scale,
        ),
        min=0.0,
        max=support_non_foot_contact_clip_max,
    )
    posture_deviation = (
        posture_deviation_penalty(
            env,
            asset_cfg=posture_deviation_asset_cfg,
            joint_weights=posture_deviation_joint_weights,
        )
        / posture_deviation_std
    )
    joint_limit_safety = (
        joint_limit_safety_penalty(env, asset_cfg=joint_limit_safety_asset_cfg) / joint_limit_safety_std
    )
    support_left_right_x_symmetry = support_left_right_x_symmetry_penalty(env) / support_left_right_x_symmetry_std
    support_left_right_y_symmetry = support_left_right_y_symmetry_penalty(env) / support_left_right_y_symmetry_std
    reward = (
        abs(support_roll_weight) * support_roll
        + abs(support_feet_slide_weight) * support_feet_slide
        + abs(support_foot_air_weight) * support_foot_air
        + abs(support_non_foot_contact_weight) * support_non_foot_contact
        + abs(posture_deviation_weight) * posture_deviation
        + abs(joint_limit_safety_weight) * joint_limit_safety
        + abs(support_left_right_x_symmetry_weight) * support_left_right_x_symmetry
        + abs(support_left_right_y_symmetry_weight) * support_left_right_y_symmetry
    )
    return torch.exp(-reward)


def mani_reward(
    env: ManagerBasedRLEnv,
    position_command_name: str,
    position_std: float,
    position_power: float,
    orientation_command_name: str,
    orientation_std: float,
    orientation_power: float,
    regularization_support_roll_weight: float,
    regularization_support_roll_asset_cfg: SceneEntityCfg,
    regularization_support_roll_std: float,
    regularization_support_feet_slide_weight: float,
    regularization_support_feet_slide_sensor_cfg: SceneEntityCfg,
    regularization_support_feet_slide_asset_cfg: SceneEntityCfg,
    regularization_support_feet_slide_std: float,
    regularization_support_foot_air_weight: float,
    regularization_support_foot_air_threshold: float,
    regularization_support_foot_air_sensor_cfg: SceneEntityCfg,
    regularization_support_foot_air_clip_max: float,
    regularization_support_non_foot_contact_weight: float,
    regularization_support_non_foot_contact_threshold: float,
    regularization_support_non_foot_contact_sensor_cfg: SceneEntityCfg,
    regularization_support_non_foot_contact_count_weight: float,
    regularization_support_non_foot_contact_force_weight: float,
    regularization_support_non_foot_contact_force_scale: float,
    regularization_support_non_foot_contact_clip_max: float,
    regularization_posture_deviation_weight: float,
    regularization_posture_deviation_asset_cfg: SceneEntityCfg,
    regularization_posture_deviation_std: float,
    regularization_posture_deviation_joint_weights: Sequence[float] | None,
    regularization_joint_limit_safety_weight: float,
    regularization_joint_limit_safety_asset_cfg: SceneEntityCfg,
    regularization_joint_limit_safety_std: float,
    regularization_support_left_right_x_symmetry_weight: float,
    regularization_support_left_right_x_symmetry_std: float,
    regularization_support_left_right_y_symmetry_weight: float,
    regularization_support_left_right_y_symmetry_std: float,
    potential_command_name: str,
    potential_std: float,
    cumulative_error_command_name: str,
    cumulative_error_clip_max: float,
) -> torch.Tensor:
    """Compute manipulation reward from regularization, enhanced tracking, potential, and cumulative error."""
    # 原始 EE 位置跟踪奖励。
    ee_position_reward = ee_position_tracking_exp(
        env,
        command_name=position_command_name,
        std=position_std,
    )
    # 增强后的 EE 位置跟踪奖励。
    ee_position_reward_enhanced = enhance_exponential_tracking_reward(
        ee_position_reward,
        power=position_power,
    )
    # 原始 EE 姿态跟踪奖励。
    ee_orientation_reward = ee_orientation_tracking_exp(
        env,
        command_name=orientation_command_name,
        std=orientation_std,
    )
    # 增强后的 EE 姿态跟踪奖励。
    ee_orientation_reward_enhanced = enhance_exponential_tracking_reward(
        ee_orientation_reward,
        power=orientation_power,
    )
    # mani 正则项先内部做加权求和。
    regularization_reward = mani_regularization_reward(
        env,
        support_roll_weight=regularization_support_roll_weight,
        support_roll_asset_cfg=regularization_support_roll_asset_cfg,
        support_roll_std=regularization_support_roll_std,
        support_feet_slide_weight=regularization_support_feet_slide_weight,
        support_feet_slide_sensor_cfg=regularization_support_feet_slide_sensor_cfg,
        support_feet_slide_asset_cfg=regularization_support_feet_slide_asset_cfg,
        support_feet_slide_std=regularization_support_feet_slide_std,
        support_foot_air_weight=regularization_support_foot_air_weight,
        support_foot_air_threshold=regularization_support_foot_air_threshold,
        support_foot_air_sensor_cfg=regularization_support_foot_air_sensor_cfg,
        support_foot_air_clip_max=regularization_support_foot_air_clip_max,
        support_non_foot_contact_weight=regularization_support_non_foot_contact_weight,
        support_non_foot_contact_threshold=regularization_support_non_foot_contact_threshold,
        support_non_foot_contact_sensor_cfg=regularization_support_non_foot_contact_sensor_cfg,
        support_non_foot_contact_count_weight=regularization_support_non_foot_contact_count_weight,
        support_non_foot_contact_force_weight=regularization_support_non_foot_contact_force_weight,
        support_non_foot_contact_force_scale=regularization_support_non_foot_contact_force_scale,
        support_non_foot_contact_clip_max=regularization_support_non_foot_contact_clip_max,
        posture_deviation_weight=regularization_posture_deviation_weight,
        posture_deviation_asset_cfg=regularization_posture_deviation_asset_cfg,
        posture_deviation_std=regularization_posture_deviation_std,
        posture_deviation_joint_weights=regularization_posture_deviation_joint_weights,
        joint_limit_safety_weight=regularization_joint_limit_safety_weight,
        joint_limit_safety_asset_cfg=regularization_joint_limit_safety_asset_cfg,
        joint_limit_safety_std=regularization_joint_limit_safety_std,
        support_left_right_x_symmetry_weight=regularization_support_left_right_x_symmetry_weight,
        support_left_right_x_symmetry_std=regularization_support_left_right_x_symmetry_std,
        support_left_right_y_symmetry_weight=regularization_support_left_right_y_symmetry_weight,
        support_left_right_y_symmetry_std=regularization_support_left_right_y_symmetry_std,
    )
    # 势奖励需要复用 reward manager 里已经初始化好的状态化 term，不能在这里每步临时新建。
    potential_term_cfg = _get_go2arm_potential_term_cfg(env)
    potential_reward = potential_term_cfg.func(
        env,
        command_name=potential_command_name,
        std=potential_std,
    )
    # 累积误差惩罚。
    cumulative_error_penalty = ee_cumulative_tracking_error_penalty(
        env,
        command_name=cumulative_error_command_name,
        clip_max=cumulative_error_clip_max,
    )
    # 按你给定的公式：
    # regularization * (1 + ee位置增强 + ee位置原始 * ee姿态增强) + potential - cumulative_penalty
    return (
        regularization_reward
        * (1.0 + ee_position_reward_enhanced + ee_position_reward * ee_orientation_reward_enhanced)
        + potential_reward
        - cumulative_error_penalty
    )


def loco_regularization_reward(
    env: ManagerBasedRLEnv,
    base_height_weight: float,
    base_height_std: float,
    base_height_target_height: float,
    base_height_asset_cfg: SceneEntityCfg,
    base_height_sensor_cfg: SceneEntityCfg | None,
    base_roll_weight: float,
    base_roll_std: float,
    base_roll_asset_cfg: SceneEntityCfg,
    base_pitch_weight: float,
    base_pitch_std: float,
    base_pitch_asset_cfg: SceneEntityCfg,
    base_roll_ang_vel_weight: float,
    base_roll_ang_vel_std: float,
    base_roll_ang_vel_asset_cfg: SceneEntityCfg,
    base_pitch_ang_vel_weight: float,
    base_pitch_ang_vel_std: float,
    base_pitch_ang_vel_asset_cfg: SceneEntityCfg,
    base_z_vel_weight: float,
    base_z_vel_std: float,
    base_z_vel_asset_cfg: SceneEntityCfg,
    base_lateral_vel_weight: float,
    base_lateral_vel_std: float,
    base_lateral_vel_asset_cfg: SceneEntityCfg,
    leg_posture_deviation_weight: float,
    leg_posture_deviation_std: float,
    leg_posture_deviation_asset_cfg: SceneEntityCfg,
    leg_posture_deviation_joint_weights: Sequence[float] | None,
    touchdown_left_right_x_symmetry_weight: float,
    touchdown_left_right_x_symmetry_std: float,
    touchdown_left_right_y_symmetry_weight: float,
    touchdown_left_right_y_symmetry_std: float,
    touchdown_foot_y_distance_weight: float,
    touchdown_foot_y_distance_std: float,
    touchdown_foot_y_distance_min_distance: float,
    feet_contact_soft_trot_weight: float,
    feet_contact_soft_trot_sensor_cfg: SceneEntityCfg,
    feet_contact_soft_trot_asset_cfg: SceneEntityCfg,
    feet_contact_soft_trot_force_std: float,
    feet_contact_soft_trot_height_std: float,
    feet_contact_soft_trot_vel_std: float,
    feet_contact_soft_trot_cycle_time: float,
    feet_contact_soft_trot_phase_offsets: Sequence[float],
    feet_contact_soft_trot_swing_height: float,
    feet_contact_soft_trot_soft_contact_k: float,
    feet_contact_soft_trot_contact_force_threshold: float = 1.0,
    feet_contact_soft_trot_ground_sensor_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Locomotion regularization as base exp-penalties times a gait factor."""
    # 这里只汇总 loco 内部已经确定的机身/步态正则项。
    # 机械臂摆动惩罚按你的定义单独从总的 loco reward 里减去，不放在这里。
    base_penalty = (
        abs(base_height_weight)
        * base_height_penalty(
            env,
            std=base_height_std,
            target_height=base_height_target_height,
            asset_cfg=base_height_asset_cfg,
            sensor_cfg=base_height_sensor_cfg,
        )
        + abs(base_roll_weight) * base_roll_penalty(env, std=base_roll_std, asset_cfg=base_roll_asset_cfg)
        + abs(base_pitch_weight) * base_pitch_penalty(env, std=base_pitch_std, asset_cfg=base_pitch_asset_cfg)
        + abs(base_roll_ang_vel_weight)
        * base_roll_ang_vel_penalty(env, std=base_roll_ang_vel_std, asset_cfg=base_roll_ang_vel_asset_cfg)
        + abs(base_pitch_ang_vel_weight)
        * base_pitch_ang_vel_penalty(env, std=base_pitch_ang_vel_std, asset_cfg=base_pitch_ang_vel_asset_cfg)
        + abs(base_z_vel_weight) * base_z_vel_penalty(env, std=base_z_vel_std, asset_cfg=base_z_vel_asset_cfg)
        + abs(base_lateral_vel_weight)
        * base_lateral_vel_penalty(env, std=base_lateral_vel_std, asset_cfg=base_lateral_vel_asset_cfg)
        + abs(leg_posture_deviation_weight)
        * (
            posture_deviation_penalty(
                env,
                asset_cfg=leg_posture_deviation_asset_cfg,
                joint_weights=leg_posture_deviation_joint_weights,
            )
            / leg_posture_deviation_std
        )
        + abs(touchdown_left_right_x_symmetry_weight)
        * (touchdown_left_right_x_symmetry_penalty(env) / touchdown_left_right_x_symmetry_std)
        + abs(touchdown_left_right_y_symmetry_weight)
        * (touchdown_left_right_y_symmetry_penalty(env) / touchdown_left_right_y_symmetry_std)
        + abs(touchdown_foot_y_distance_weight)
        * (
            touchdown_foot_y_distance_penalty(env, min_distance=touchdown_foot_y_distance_min_distance)
            / touchdown_foot_y_distance_std
        )
    )
    trot_reward = feet_contact_soft_trot_reward(
        env,
        sensor_cfg=feet_contact_soft_trot_sensor_cfg,
        asset_cfg=feet_contact_soft_trot_asset_cfg,
        force_std=feet_contact_soft_trot_force_std,
        height_std=feet_contact_soft_trot_height_std,
        vel_std=feet_contact_soft_trot_vel_std,
        cycle_time=feet_contact_soft_trot_cycle_time,
        phase_offsets=feet_contact_soft_trot_phase_offsets,
        swing_height=feet_contact_soft_trot_swing_height,
        soft_contact_k=feet_contact_soft_trot_soft_contact_k,
        contact_force_threshold=feet_contact_soft_trot_contact_force_threshold,
        ground_sensor_names=feet_contact_soft_trot_ground_sensor_names,
    )
    normalized_trot_reward = torch.clamp(
        trot_reward / float(len(feet_contact_soft_trot_asset_cfg.body_ids)),
        min=1.0e-6,
        max=1.0,
    )
    gait_factor = torch.pow(normalized_trot_reward, abs(feet_contact_soft_trot_weight))
    return torch.exp(-base_penalty) * gait_factor


def loco_reward(
    env: ManagerBasedRLEnv,
    tracking_command_name: str,
    tracking_std: float,
    tracking_threshold: float,
    tracking_weight: float,
    regularization_base_height_weight: float,
    regularization_base_height_std: float,
    regularization_base_height_target_height: float,
    regularization_base_height_asset_cfg: SceneEntityCfg,
    regularization_base_height_sensor_cfg: SceneEntityCfg | None,
    regularization_base_roll_weight: float,
    regularization_base_roll_std: float,
    regularization_base_roll_asset_cfg: SceneEntityCfg,
    regularization_base_pitch_weight: float,
    regularization_base_pitch_std: float,
    regularization_base_pitch_asset_cfg: SceneEntityCfg,
    regularization_base_roll_ang_vel_weight: float,
    regularization_base_roll_ang_vel_std: float,
    regularization_base_roll_ang_vel_asset_cfg: SceneEntityCfg,
    regularization_base_pitch_ang_vel_weight: float,
    regularization_base_pitch_ang_vel_std: float,
    regularization_base_pitch_ang_vel_asset_cfg: SceneEntityCfg,
    regularization_base_z_vel_weight: float,
    regularization_base_z_vel_std: float,
    regularization_base_z_vel_asset_cfg: SceneEntityCfg,
    regularization_base_lateral_vel_weight: float,
    regularization_base_lateral_vel_std: float,
    regularization_base_lateral_vel_asset_cfg: SceneEntityCfg,
    regularization_leg_posture_deviation_weight: float,
    regularization_leg_posture_deviation_std: float,
    regularization_leg_posture_deviation_asset_cfg: SceneEntityCfg,
    regularization_leg_posture_deviation_joint_weights: Sequence[float] | None,
    regularization_touchdown_left_right_x_symmetry_weight: float,
    regularization_touchdown_left_right_x_symmetry_std: float,
    regularization_touchdown_left_right_y_symmetry_weight: float,
    regularization_touchdown_left_right_y_symmetry_std: float,
    regularization_touchdown_foot_y_distance_weight: float,
    regularization_touchdown_foot_y_distance_std: float,
    regularization_touchdown_foot_y_distance_min_distance: float,
    regularization_feet_contact_soft_trot_weight: float,
    regularization_feet_contact_soft_trot_sensor_cfg: SceneEntityCfg,
    regularization_feet_contact_soft_trot_asset_cfg: SceneEntityCfg,
    regularization_feet_contact_soft_trot_force_std: float,
    regularization_feet_contact_soft_trot_height_std: float,
    regularization_feet_contact_soft_trot_vel_std: float,
    regularization_feet_contact_soft_trot_cycle_time: float,
    regularization_feet_contact_soft_trot_phase_offsets: Sequence[float],
    regularization_feet_contact_soft_trot_swing_height: float,
    regularization_feet_contact_soft_trot_soft_contact_k: float,
    arm_swing_weight: float,
    arm_swing_asset_cfg: SceneEntityCfg,
    arm_dynamic_weight: float,
    arm_dynamic_asset_cfg: SceneEntityCfg,
    regularization_feet_contact_soft_trot_contact_force_threshold: float = 1.0,
    regularization_feet_contact_soft_trot_ground_sensor_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Compute locomotion reward as regularization * (1 + tracking) - arm-swing penalty."""
    # loco 跟踪主项。
    tracking_reward = locomotion_tracking_exp(
        env,
        command_name=tracking_command_name,
        std=tracking_std,
        threshold=tracking_threshold,
    )
    tracking_reward = tracking_weight * tracking_reward
    # loco 正则项内部先做加权和。
    regularization_reward = loco_regularization_reward(
        env,
        base_height_weight=regularization_base_height_weight,
        base_height_std=regularization_base_height_std,
        base_height_target_height=regularization_base_height_target_height,
        base_height_asset_cfg=regularization_base_height_asset_cfg,
        base_height_sensor_cfg=regularization_base_height_sensor_cfg,
        base_roll_weight=regularization_base_roll_weight,
        base_roll_std=regularization_base_roll_std,
        base_roll_asset_cfg=regularization_base_roll_asset_cfg,
        base_pitch_weight=regularization_base_pitch_weight,
        base_pitch_std=regularization_base_pitch_std,
        base_pitch_asset_cfg=regularization_base_pitch_asset_cfg,
        base_roll_ang_vel_weight=regularization_base_roll_ang_vel_weight,
        base_roll_ang_vel_std=regularization_base_roll_ang_vel_std,
        base_roll_ang_vel_asset_cfg=regularization_base_roll_ang_vel_asset_cfg,
        base_pitch_ang_vel_weight=regularization_base_pitch_ang_vel_weight,
        base_pitch_ang_vel_std=regularization_base_pitch_ang_vel_std,
        base_pitch_ang_vel_asset_cfg=regularization_base_pitch_ang_vel_asset_cfg,
        base_z_vel_weight=regularization_base_z_vel_weight,
        base_z_vel_std=regularization_base_z_vel_std,
        base_z_vel_asset_cfg=regularization_base_z_vel_asset_cfg,
        base_lateral_vel_weight=regularization_base_lateral_vel_weight,
        base_lateral_vel_std=regularization_base_lateral_vel_std,
        base_lateral_vel_asset_cfg=regularization_base_lateral_vel_asset_cfg,
        leg_posture_deviation_weight=regularization_leg_posture_deviation_weight,
        leg_posture_deviation_std=regularization_leg_posture_deviation_std,
        leg_posture_deviation_asset_cfg=regularization_leg_posture_deviation_asset_cfg,
        leg_posture_deviation_joint_weights=regularization_leg_posture_deviation_joint_weights,
        touchdown_left_right_x_symmetry_weight=regularization_touchdown_left_right_x_symmetry_weight,
        touchdown_left_right_x_symmetry_std=regularization_touchdown_left_right_x_symmetry_std,
        touchdown_left_right_y_symmetry_weight=regularization_touchdown_left_right_y_symmetry_weight,
        touchdown_left_right_y_symmetry_std=regularization_touchdown_left_right_y_symmetry_std,
        touchdown_foot_y_distance_weight=regularization_touchdown_foot_y_distance_weight,
        touchdown_foot_y_distance_std=regularization_touchdown_foot_y_distance_std,
        touchdown_foot_y_distance_min_distance=regularization_touchdown_foot_y_distance_min_distance,
        feet_contact_soft_trot_weight=regularization_feet_contact_soft_trot_weight,
        feet_contact_soft_trot_sensor_cfg=regularization_feet_contact_soft_trot_sensor_cfg,
        feet_contact_soft_trot_asset_cfg=regularization_feet_contact_soft_trot_asset_cfg,
        feet_contact_soft_trot_force_std=regularization_feet_contact_soft_trot_force_std,
        feet_contact_soft_trot_height_std=regularization_feet_contact_soft_trot_height_std,
        feet_contact_soft_trot_vel_std=regularization_feet_contact_soft_trot_vel_std,
        feet_contact_soft_trot_cycle_time=regularization_feet_contact_soft_trot_cycle_time,
        feet_contact_soft_trot_phase_offsets=regularization_feet_contact_soft_trot_phase_offsets,
        feet_contact_soft_trot_swing_height=regularization_feet_contact_soft_trot_swing_height,
        feet_contact_soft_trot_soft_contact_k=regularization_feet_contact_soft_trot_soft_contact_k,
        feet_contact_soft_trot_contact_force_threshold=regularization_feet_contact_soft_trot_contact_force_threshold,
        feet_contact_soft_trot_ground_sensor_names=regularization_feet_contact_soft_trot_ground_sensor_names,
    )
    # 机械臂摆动惩罚单独减去，并保留独立权重。
    arm_swing_penalty = arm_swing_weight * moving_arm_default_deviation_penalty(env, asset_cfg=arm_swing_asset_cfg)
    arm_dynamic_penalty = arm_dynamic_weight * moving_arm_joint_velocity_penalty(env, asset_cfg=arm_dynamic_asset_cfg)
    # 按你指定的公式：regularization * (1 + tracking) - arm_swing_penalty。
    return regularization_reward * (1.0 + tracking_reward) - arm_swing_penalty - arm_dynamic_penalty


def _actuator_effort_limits(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    joint_ids: list[int] | slice,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return actuator effort limits for selected joints."""

    effort_limits = torch.full_like(asset.data.computed_torque, torch.inf)
    for actuator in asset.actuators.values():
        actuator_joint_ids = torch.as_tensor(actuator.joint_indices, device=env.device, dtype=torch.long)
        actuator_effort_limit = actuator.effort_limit
        if not isinstance(actuator_effort_limit, torch.Tensor):
            actuator_effort_limit = torch.full(
                (env.num_envs, actuator_joint_ids.numel()),
                float(actuator_effort_limit),
                device=env.device,
                dtype=dtype,
            )
        else:
            actuator_effort_limit = actuator_effort_limit.to(device=env.device, dtype=dtype)
            if actuator_effort_limit.dim() == 0:
                actuator_effort_limit = actuator_effort_limit.reshape(1, 1).expand(
                    env.num_envs, actuator_joint_ids.numel()
                )
            elif actuator_effort_limit.dim() == 1:
                actuator_effort_limit = actuator_effort_limit.unsqueeze(0).expand(env.num_envs, -1)
        effort_limits[:, actuator_joint_ids] = actuator_effort_limit

    return torch.clamp(effort_limits[:, joint_ids], min=1.0e-6)


def joint_torque_sq_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_effort_limit: bool = False,
) -> torch.Tensor:
    """Penalty on the sum of squared joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    if normalize_by_effort_limit:
        torque = asset.data.computed_torque[:, asset_cfg.joint_ids]
        effort_limits = _actuator_effort_limits(env, asset, asset_cfg.joint_ids, dtype=torque.dtype)
        torque = torque / effort_limits
    else:
        torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_sq = torch.sum(torch.square(torque), dim=1)
    return torque_sq


def joint_power_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize_by_effort_limit: bool = False,
) -> torch.Tensor:
    """Penalty on the sum of absolute joint power."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    if normalize_by_effort_limit:
        effort_limits = _actuator_effort_limits(env, asset, asset_cfg.joint_ids, dtype=torque.dtype)
        torque = torque / effort_limits
    joint_power = torque * asset.data.joint_vel[:, asset_cfg.joint_ids]
    power_norm = torch.sum(torch.abs(joint_power), dim=1)
    return power_norm


def basic_reward(
    env: ManagerBasedRLEnv,
    is_alive_weight: float,
    collision_weight: float,
    collision_threshold: float,
    collision_sensor_cfg: SceneEntityCfg,
    action_smoothness_first_weight: float,
    action_smoothness_second_weight: float,
    joint_torque_sq_weight: float,
    joint_torque_sq_asset_cfg: SceneEntityCfg,
    joint_power_weight: float,
    joint_power_asset_cfg: SceneEntityCfg,
    joint_torque_sq_normalize_by_effort_limit: bool = False,
    joint_power_normalize_by_effort_limit: bool = False,
) -> torch.Tensor:
    """Weighted sum of basic reward terms."""
    # basic 奖励目前按你说的方式，直接做各子项加权求和。
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    reward += is_alive_weight * mdp.is_alive(env)
    reward += collision_weight * undesired_contacts(
        env,
        threshold=collision_threshold,
        sensor_cfg=collision_sensor_cfg,
    )
    reward += action_smoothness_first_weight * action_smoothness_first_penalty(env)
    reward += action_smoothness_second_weight * action_smoothness_second_penalty(env)
    reward += joint_torque_sq_weight * joint_torque_sq_penalty(
        env,
        asset_cfg=joint_torque_sq_asset_cfg,
        normalize_by_effort_limit=joint_torque_sq_normalize_by_effort_limit,
    )
    reward += joint_power_weight * joint_power_penalty(
        env,
        asset_cfg=joint_power_asset_cfg,
        normalize_by_effort_limit=joint_power_normalize_by_effort_limit,
    )
    return reward


def non_success_termination_penalty(
    env: ManagerBasedRLEnv,
    excluded_terms: Sequence[str] | str = ("task_success",),
) -> torch.Tensor:
    """Return 1 for non-timeout terminations except explicitly excluded terms."""
    if isinstance(excluded_terms, str):
        excluded_terms = (excluded_terms,)
    excluded_term_set = set(excluded_terms)

    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for term_name in env.termination_manager.active_terms:
        if term_name in excluded_term_set:
            continue
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        if term_cfg.time_out:
            continue
        terminated |= env.termination_manager.get_term(term_name)

    return terminated.float()


def _compute_go2arm_reward_terms(
    env: ManagerBasedRLEnv,
    total_reward_term_name: str = "total_reward",
) -> dict[str, torch.Tensor]:
    """Compute go2arm total reward for the current step."""
    params = _get_go2arm_total_reward_params(env, total_reward_term_name)
    command_term = _ee_pose_command_term(env, params["gating_command_name"])

    # -----------------------------
    # 1) 单步快照：同一步里 gate / mani / loco / debug 统一用这一份
    # -----------------------------
    tracking_error = command_term.tracking_error
    position_tracking_error = command_term.position_tracking_error
    orientation_tracking_error = command_term.orientation_tracking_error
    reference_tracking_error = command_term.reference_tracking_error
    cumulative_tracking_error = command_term.cumulative_tracking_error
    ee_pos_b = command_term.ee_pos_b
    target_pos_b = command_term.target_pos_b
    del ee_pos_b, target_pos_b
    # -----------------------------
    # 2) gate：直接用冻结后的 reference_tracking_error
    # -----------------------------
    gate_input = (5.0 / params["gating_l"]) * (reference_tracking_error - params["gating_mu"])
    gate = torch.sigmoid(gate_input)
    if params.get("gating_fixed_d") is not None:
        gate = torch.full_like(gate, float(params["gating_fixed_d"]))

    # -----------------------------
    # 3) manipulation tracking：直接用冻结后的 position/orientation error
    # -----------------------------
    ee_position_raw = torch.exp(-position_tracking_error / params["mani_position_std"])
    ee_position_enhanced = ee_position_raw + torch.pow(ee_position_raw, params["mani_position_power"])

    ee_orientation_raw = torch.exp(-orientation_tracking_error / params["mani_orientation_std"])
    ee_orientation_enhanced = ee_orientation_raw + torch.pow(ee_orientation_raw, params["mani_orientation_power"])

    # -----------------------------
    # 4) mani regularization：先按各惩罚子项线性组合，再映射成 (0, 1] 的正则因子
    # -----------------------------
    support_roll = support_roll_penalty(env, asset_cfg=params["mani_regularization_support_roll_asset_cfg"])
    support_feet_slide = support_feet_slide_penalty(
        env,
        sensor_cfg=params["mani_regularization_support_feet_slide_sensor_cfg"],
        asset_cfg=params["mani_regularization_support_feet_slide_asset_cfg"],
    )
    support_foot_air = support_foot_air_penalty(
        env,
        threshold=params["mani_regularization_support_foot_air_threshold"],
        sensor_cfg=params["mani_regularization_support_foot_air_sensor_cfg"],
    )
    support_non_foot_contact = support_non_foot_contact_penalty(
        env,
        threshold=params["mani_regularization_support_non_foot_contact_threshold"],
        sensor_cfg=params["mani_regularization_support_non_foot_contact_sensor_cfg"],
        count_weight=params["mani_regularization_support_non_foot_contact_count_weight"],
        force_weight=params["mani_regularization_support_non_foot_contact_force_weight"],
        force_scale=params["mani_regularization_support_non_foot_contact_force_scale"],
    )
    target_height_pitch = target_height_conditioned_pitch_penalty(
        env,
        command_name=params["mani_regularization_target_height_pitch_command_name"],
        low_height=params["mani_regularization_target_height_pitch_low_height"],
        high_height=params["mani_regularization_target_height_pitch_high_height"],
        asset_cfg=params["mani_regularization_target_height_pitch_asset_cfg"],
    )
    min_base_height = min_base_height_penalty(
        env,
        minimum_height=params["mani_regularization_min_base_height_minimum_height"],
        asset_cfg=params["mani_regularization_min_base_height_asset_cfg"],
        sensor_cfg=params["mani_regularization_min_base_height_sensor_cfg"],
    )
    posture_deviation = posture_deviation_penalty(
        env,
        asset_cfg=params["mani_regularization_posture_deviation_asset_cfg"],
        joint_weights=params["mani_regularization_posture_deviation_joint_weights"],
    )
    joint_limit_safety = joint_limit_safety_penalty(
        env, asset_cfg=params["mani_regularization_joint_limit_safety_asset_cfg"]
    )
    support_left_right_x_symmetry = support_left_right_x_symmetry_penalty(env)
    support_left_right_y_symmetry = support_left_right_y_symmetry_penalty(env)
    support_foot_xy_range = support_foot_xy_range_penalty(
        env,
        x_abs_min=params["mani_regularization_support_foot_xy_range_x_abs_min"],
        x_abs_max=params["mani_regularization_support_foot_xy_range_x_abs_max"],
        y_abs_min=params["mani_regularization_support_foot_xy_range_y_abs_min"],
        y_abs_max=params["mani_regularization_support_foot_xy_range_y_abs_max"],
    )
    mani_regularization_raw = (
        abs(params["mani_regularization_support_roll_weight"])
        * (support_roll / params["mani_regularization_support_roll_std"])
        + abs(params["mani_regularization_support_feet_slide_weight"])
        * (support_feet_slide / params["mani_regularization_support_feet_slide_std"])
        + abs(params["mani_regularization_support_foot_air_weight"])
        * torch.clamp(support_foot_air, min=0.0, max=params["mani_regularization_support_foot_air_clip_max"])
        + abs(params["mani_regularization_support_non_foot_contact_weight"])
        * torch.clamp(
            support_non_foot_contact,
            min=0.0,
            max=params["mani_regularization_support_non_foot_contact_clip_max"],
        )
        + abs(params["mani_regularization_target_height_pitch_weight"])
        * (target_height_pitch / params["mani_regularization_target_height_pitch_std"])
        + abs(params["mani_regularization_min_base_height_weight"])
        * (min_base_height / params["mani_regularization_min_base_height_std"])
        + abs(params["mani_regularization_posture_deviation_weight"])
        * (posture_deviation / params["mani_regularization_posture_deviation_std"])
        + abs(params["mani_regularization_joint_limit_safety_weight"])
        * (joint_limit_safety / params["mani_regularization_joint_limit_safety_std"])
        + abs(params["mani_regularization_support_left_right_x_symmetry_weight"])
        * (support_left_right_x_symmetry / params["mani_regularization_support_left_right_x_symmetry_std"])
        + abs(params["mani_regularization_support_left_right_y_symmetry_weight"])
        * (support_left_right_y_symmetry / params["mani_regularization_support_left_right_y_symmetry_std"])
        + abs(params["mani_regularization_support_foot_xy_range_weight"])
        * (support_foot_xy_range / params["mani_regularization_support_foot_xy_range_std"])
    )
    # 当前配置里这些 weight 都是负值，表示“惩罚越大，regularization 越小”。
    # 这里把线性组合后的 raw 值通过 exp 映射到 (0, 1]，从而让整个 manipulation
    # regularization 成为一个真正的 0-1 调制因子，而不是直接变成负奖励。

    mani_regularization = torch.exp(-mani_regularization_raw)

    # manipulation 各惩罚项对应到 mani_regularization_raw 的实际加权贡献。
    support_roll_weighted = abs(params["mani_regularization_support_roll_weight"]) * (
        support_roll / params["mani_regularization_support_roll_std"]
    )
    support_feet_slide_weighted = abs(params["mani_regularization_support_feet_slide_weight"]) * (
        support_feet_slide / params["mani_regularization_support_feet_slide_std"]
    )
    support_foot_air_weighted = abs(params["mani_regularization_support_foot_air_weight"]) * torch.clamp(
        support_foot_air,
        min=0.0,
        max=params["mani_regularization_support_foot_air_clip_max"],
    )
    support_non_foot_contact_weighted = abs(
        params["mani_regularization_support_non_foot_contact_weight"]
    ) * torch.clamp(
        support_non_foot_contact,
        min=0.0,
        max=params["mani_regularization_support_non_foot_contact_clip_max"],
    )
    target_height_pitch_weighted = abs(params["mani_regularization_target_height_pitch_weight"]) * (
        target_height_pitch / params["mani_regularization_target_height_pitch_std"]
    )
    min_base_height_weighted = abs(params["mani_regularization_min_base_height_weight"]) * (
        min_base_height / params["mani_regularization_min_base_height_std"]
    )
    posture_deviation_weighted = abs(params["mani_regularization_posture_deviation_weight"]) * (
        posture_deviation / params["mani_regularization_posture_deviation_std"]
    )
    joint_limit_safety_weighted = abs(params["mani_regularization_joint_limit_safety_weight"]) * (
        joint_limit_safety / params["mani_regularization_joint_limit_safety_std"]
    )
    support_left_right_x_symmetry_weighted = abs(params["mani_regularization_support_left_right_x_symmetry_weight"]) * (
        support_left_right_x_symmetry / params["mani_regularization_support_left_right_x_symmetry_std"]
    )
    support_left_right_y_symmetry_weighted = abs(params["mani_regularization_support_left_right_y_symmetry_weight"]) * (
        support_left_right_y_symmetry / params["mani_regularization_support_left_right_y_symmetry_std"]
    )
    support_foot_xy_range_weighted = abs(params["mani_regularization_support_foot_xy_range_weight"]) * (
        support_foot_xy_range / params["mani_regularization_support_foot_xy_range_std"]
    )

    # -----------------------------
    # 5) potential：保持你现在的状态项接法不变
    # -----------------------------
    potential_term_cfg = _get_go2arm_potential_term_cfg(env)
    if isinstance(potential_term_cfg.func, EETrackingPotentialReward):
        ee_tracking_potential_value = potential_term_cfg.func.last_reward
    else:
        ee_tracking_potential_value = torch.zeros(env.num_envs, device=env.device)

    # -----------------------------
    # 6) cumulative penalty：直接用冻结后的 cumulative_tracking_error
    # -----------------------------
    ee_cumulative_penalty = torch.clamp(
        cumulative_tracking_error,
        min=0.0,
        max=params["mani_cumulative_error_clip_max"],
    )
    ee_tracking_potential_weighted = params["mani_potential_weight"] * ee_tracking_potential_value
    # 当前 cumulative 在 total_reward 里是直接减项，日志里改成“已乘系数后的实际贡献”口径。
    ee_cumulative_penalty_weighted = -ee_cumulative_penalty
    workspace_position_reward = target_workspace_position_penalty(
        env,
        command_name=params["workspace_position_command_name"],
        x_min=params["workspace_position_x_min"],
        x_max=params["workspace_position_x_max"],
        y_weight=params["workspace_position_y_weight"],
        std=params["workspace_position_std"],
    )
    workspace_position_reward_weighted = abs(params["workspace_position_weight"]) * workspace_position_reward

    mani_total = (
        mani_regularization * (1.0 + ee_position_enhanced + ee_position_raw * ee_orientation_enhanced)
        + ee_tracking_potential_weighted
        + ee_cumulative_penalty_weighted
    )

    # -----------------------------
    # 7) locomotion tracking：直接用冻结后的 reference / tracking error
    # -----------------------------
    error_gap = torch.abs(reference_tracking_error - tracking_error)
    loco_d = torch.clamp(error_gap - params["loco_tracking_threshold"], min=0.0)
    locomotion_tracking = params["loco_tracking_weight"] * torch.exp(-loco_d / params["loco_tracking_std"])

    # -----------------------------
    # 8) loco regularization：保持你原有写法
    # -----------------------------
    base_height = base_height_penalty(
        env,
        std=params["loco_regularization_base_height_std"],
        target_height=params["loco_regularization_base_height_target_height"],
        asset_cfg=params["loco_regularization_base_height_asset_cfg"],
        sensor_cfg=params["loco_regularization_base_height_sensor_cfg"],
    )
    base_roll = base_roll_penalty(
        env,
        std=params["loco_regularization_base_roll_std"],
        asset_cfg=params["loco_regularization_base_roll_asset_cfg"],
    )
    base_pitch = base_pitch_penalty(
        env,
        std=params["loco_regularization_base_pitch_std"],
        asset_cfg=params["loco_regularization_base_pitch_asset_cfg"],
    )
    base_roll_ang_vel = base_roll_ang_vel_penalty(
        env,
        std=params["loco_regularization_base_roll_ang_vel_std"],
        asset_cfg=params["loco_regularization_base_roll_ang_vel_asset_cfg"],
    )
    base_pitch_ang_vel = base_pitch_ang_vel_penalty(
        env,
        std=params["loco_regularization_base_pitch_ang_vel_std"],
        asset_cfg=params["loco_regularization_base_pitch_ang_vel_asset_cfg"],
    )
    base_z_vel = base_z_vel_penalty(
        env,
        std=params["loco_regularization_base_z_vel_std"],
        asset_cfg=params["loco_regularization_base_z_vel_asset_cfg"],
    )
    base_lateral_vel = base_lateral_vel_penalty(
        env,
        std=params["loco_regularization_base_lateral_vel_std"],
        asset_cfg=params["loco_regularization_base_lateral_vel_asset_cfg"],
    )
    leg_posture_deviation = posture_deviation_penalty(
        env,
        asset_cfg=params["loco_regularization_leg_posture_deviation_asset_cfg"],
        joint_weights=params["loco_regularization_leg_posture_deviation_joint_weights"],
    )
    touchdown_left_right_x_symmetry = touchdown_left_right_x_symmetry_penalty(env)
    touchdown_left_right_y_symmetry = touchdown_left_right_y_symmetry_penalty(env)
    touchdown_foot_y_distance = touchdown_foot_y_distance_penalty(
        env,
        min_distance=params["loco_regularization_touchdown_foot_y_distance_min_distance"],
        max_distance=params.get("loco_regularization_touchdown_foot_y_distance_max_distance"),
    )
    feet_contact_soft_trot_reward_value = feet_contact_soft_trot_reward(
        env,
        sensor_cfg=params["loco_regularization_feet_contact_soft_trot_sensor_cfg"],
        asset_cfg=params["loco_regularization_feet_contact_soft_trot_asset_cfg"],
        force_std=params["loco_regularization_feet_contact_soft_trot_force_std"],
        height_std=params["loco_regularization_feet_contact_soft_trot_height_std"],
        vel_std=params["loco_regularization_feet_contact_soft_trot_vel_std"],
        cycle_time=params["loco_regularization_feet_contact_soft_trot_cycle_time"],
        phase_offsets=params["loco_regularization_feet_contact_soft_trot_phase_offsets"],
        swing_height=params["loco_regularization_feet_contact_soft_trot_swing_height"],
        soft_contact_k=params["loco_regularization_feet_contact_soft_trot_soft_contact_k"],
        contact_force_threshold=params["loco_regularization_feet_contact_soft_trot_contact_force_threshold"],
        ground_sensor_names=params["loco_regularization_feet_contact_soft_trot_ground_sensor_names"],
    )
    feet_contact_soft_trot_normalized = torch.clamp(
        feet_contact_soft_trot_reward_value
        / float(len(params["loco_regularization_feet_contact_soft_trot_asset_cfg"].body_ids)),
        min=1.0e-6,
        max=1.0,
    )
    feet_contact_soft_trot_factor = torch.pow(
        feet_contact_soft_trot_normalized,
        abs(params["loco_regularization_feet_contact_soft_trot_weight"]),
    )
    diagonal_foot_symmetry = diagonal_foot_symmetry_penalty(
        env,
        sensor_cfg=params["loco_regularization_diagonal_foot_symmetry_sensor_cfg"],
        contact_force_threshold=params["loco_regularization_feet_contact_soft_trot_contact_force_threshold"],
    )
    moving_arm_deviation = moving_arm_default_deviation_penalty(env, asset_cfg=params["loco_arm_swing_asset_cfg"])
    moving_arm_dynamic = moving_arm_joint_velocity_penalty(env, asset_cfg=params["loco_arm_dynamic_asset_cfg"])
    base_height_weighted = abs(params["loco_regularization_base_height_weight"]) * base_height
    base_roll_weighted = abs(params["loco_regularization_base_roll_weight"]) * base_roll
    base_pitch_weighted = abs(params["loco_regularization_base_pitch_weight"]) * base_pitch
    base_roll_ang_vel_weighted = abs(params["loco_regularization_base_roll_ang_vel_weight"]) * base_roll_ang_vel
    base_pitch_ang_vel_weighted = abs(params["loco_regularization_base_pitch_ang_vel_weight"]) * base_pitch_ang_vel
    base_z_vel_weighted = abs(params["loco_regularization_base_z_vel_weight"]) * base_z_vel
    base_lateral_vel_weighted = abs(params["loco_regularization_base_lateral_vel_weight"]) * base_lateral_vel
    leg_posture_deviation_weighted = abs(params["loco_regularization_leg_posture_deviation_weight"]) * (
        leg_posture_deviation / params["loco_regularization_leg_posture_deviation_std"]
    )
    touchdown_left_right_x_symmetry_weighted = abs(
        params["loco_regularization_touchdown_left_right_x_symmetry_weight"]
    ) * (touchdown_left_right_x_symmetry / params["loco_regularization_touchdown_left_right_x_symmetry_std"])
    touchdown_left_right_y_symmetry_weighted = abs(
        params["loco_regularization_touchdown_left_right_y_symmetry_weight"]
    ) * (touchdown_left_right_y_symmetry / params["loco_regularization_touchdown_left_right_y_symmetry_std"])
    touchdown_foot_y_distance_weighted = abs(params["loco_regularization_touchdown_foot_y_distance_weight"]) * (
        touchdown_foot_y_distance / params["loco_regularization_touchdown_foot_y_distance_std"]
    )
    diagonal_foot_symmetry_weighted = abs(params["loco_regularization_diagonal_foot_symmetry_weight"]) * (
        diagonal_foot_symmetry / params["loco_regularization_diagonal_foot_symmetry_std"]
    )
    moving_arm_deviation_weighted = params["loco_arm_swing_weight"] * moving_arm_deviation
    moving_arm_dynamic_weighted = params["loco_arm_dynamic_weight"] * moving_arm_dynamic
    loco_regularization_base_raw = (
        base_height_weighted
        + base_roll_weighted
        + base_pitch_weighted
        + base_roll_ang_vel_weighted
        + base_pitch_ang_vel_weighted
        + base_z_vel_weighted
        + base_lateral_vel_weighted
        + leg_posture_deviation_weighted
        + touchdown_left_right_x_symmetry_weighted
        + touchdown_left_right_y_symmetry_weighted
        + touchdown_foot_y_distance_weighted
        + diagonal_foot_symmetry_weighted
    )
    loco_regularization = torch.exp(-loco_regularization_base_raw) * feet_contact_soft_trot_factor
    loco_total = (
        loco_regularization * (1.0 + locomotion_tracking) - moving_arm_deviation_weighted - moving_arm_dynamic_weighted
    )

    # -----------------------------
    # 9) basic：保持你原有写法
    # -----------------------------
    basic_is_alive = mdp.is_alive(env)
    basic_collision = collision_force_count_penalty(
        env,
        threshold=params["basic_collision_threshold"],
        sensor_cfg=params["basic_collision_sensor_cfg"],
        count_weight=params["basic_collision_count_weight"],
        force_weight=params["basic_collision_force_weight"],
        force_scale=params["basic_collision_force_scale"],
    )
    basic_termination_penalty = non_success_termination_penalty(
        env,
        excluded_terms=params.get("basic_termination_penalty_excluded_terms", ("task_success",)),
    )
    basic_action_smoothness_first = action_smoothness_first_penalty(env)
    basic_action_smoothness_second = action_smoothness_second_penalty(env)
    basic_joint_torque_sq = joint_torque_sq_penalty(
        env,
        asset_cfg=params["basic_joint_torque_sq_asset_cfg"],
        normalize_by_effort_limit=params.get("basic_joint_torque_sq_normalize_by_effort_limit", False),
    )
    basic_joint_power = joint_power_penalty(
        env,
        asset_cfg=params["basic_joint_power_asset_cfg"],
        normalize_by_effort_limit=params.get("basic_joint_power_normalize_by_effort_limit", False),
    )
    basic_is_alive_weighted = params["basic_is_alive_weight"] * basic_is_alive
    basic_collision_weighted = params["basic_collision_weight"] * basic_collision
    basic_termination_penalty_weighted = params.get("basic_termination_penalty_weight", 0.0) * basic_termination_penalty
    basic_action_smoothness_first_weighted = (
        params["basic_action_smoothness_first_weight"] * basic_action_smoothness_first
    )
    basic_action_smoothness_second_weighted = (
        params["basic_action_smoothness_second_weight"] * basic_action_smoothness_second
    )
    basic_joint_torque_sq_weighted = params["basic_joint_torque_sq_weight"] * basic_joint_torque_sq
    basic_joint_power_weighted = params["basic_joint_power_weight"] * basic_joint_power
    basic_total = (
        basic_is_alive_weighted
        + basic_collision_weighted
        + basic_termination_penalty_weighted
        + basic_action_smoothness_first_weighted
        + basic_action_smoothness_second_weighted
        + basic_joint_torque_sq_weighted
        + basic_joint_power_weighted
    )

    # -----------------------------
    # 10) 总奖励
    # -----------------------------
    total = (1.0 - gate) * mani_total + gate * loco_total + basic_total + workspace_position_reward_weighted

    # -----------------------------
    # 11) 单次计算后统一缓存，供日志直接复用
    # -----------------------------
    return {"total_reward_debug": total}


def total_reward(
    env: ManagerBasedRLEnv,
    gating_command_name: object = None,
    gating_mu: object = None,
    gating_l: object = None,
    gating_fixed_d: object = None,
    mani_position_command_name: object = None,
    mani_position_std: object = None,
    mani_position_power: object = None,
    mani_orientation_command_name: object = None,
    mani_orientation_std: object = None,
    mani_orientation_power: object = None,
    mani_regularization_support_roll_weight: object = None,
    mani_regularization_support_roll_asset_cfg: object = None,
    mani_regularization_support_roll_std: object = None,
    mani_regularization_support_feet_slide_weight: object = None,
    mani_regularization_support_feet_slide_sensor_cfg: object = None,
    mani_regularization_support_feet_slide_asset_cfg: object = None,
    mani_regularization_support_feet_slide_std: object = None,
    mani_regularization_support_foot_air_weight: object = None,
    mani_regularization_support_foot_air_threshold: object = None,
    mani_regularization_support_foot_air_sensor_cfg: object = None,
    mani_regularization_support_foot_air_clip_max: object = None,
    mani_regularization_support_non_foot_contact_weight: object = None,
    mani_regularization_support_non_foot_contact_threshold: object = None,
    mani_regularization_support_non_foot_contact_sensor_cfg: object = None,
    mani_regularization_support_non_foot_contact_count_weight: object = None,
    mani_regularization_support_non_foot_contact_force_weight: object = None,
    mani_regularization_support_non_foot_contact_force_scale: object = None,
    mani_regularization_support_non_foot_contact_clip_max: object = None,
    mani_regularization_target_height_pitch_weight: object = None,
    mani_regularization_target_height_pitch_command_name: object = None,
    mani_regularization_target_height_pitch_asset_cfg: object = None,
    mani_regularization_target_height_pitch_low_height: object = None,
    mani_regularization_target_height_pitch_high_height: object = None,
    mani_regularization_target_height_pitch_std: object = None,
    mani_regularization_min_base_height_weight: object = None,
    mani_regularization_min_base_height_asset_cfg: object = None,
    mani_regularization_min_base_height_sensor_cfg: object = None,
    mani_regularization_min_base_height_minimum_height: object = None,
    mani_regularization_min_base_height_std: object = None,
    mani_regularization_posture_deviation_weight: object = None,
    mani_regularization_posture_deviation_asset_cfg: object = None,
    mani_regularization_posture_deviation_std: object = None,
    mani_regularization_posture_deviation_joint_weights: object = None,
    mani_regularization_joint_limit_safety_weight: object = None,
    mani_regularization_joint_limit_safety_asset_cfg: object = None,
    mani_regularization_joint_limit_safety_std: object = None,
    mani_regularization_support_left_right_x_symmetry_weight: object = None,
    mani_regularization_support_left_right_x_symmetry_std: object = None,
    mani_regularization_support_left_right_y_symmetry_weight: object = None,
    mani_regularization_support_left_right_y_symmetry_std: object = None,
    mani_regularization_support_foot_xy_range_weight: object = None,
    mani_regularization_support_foot_xy_range_std: object = None,
    mani_regularization_support_foot_xy_range_x_abs_min: object = None,
    mani_regularization_support_foot_xy_range_x_abs_max: object = None,
    mani_regularization_support_foot_xy_range_y_abs_min: object = None,
    mani_regularization_support_foot_xy_range_y_abs_max: object = None,
    mani_potential_command_name: object = None,
    mani_potential_std: object = None,
    mani_potential_weight: object = None,
    mani_cumulative_error_command_name: object = None,
    mani_cumulative_error_clip_max: object = None,
    workspace_position_weight: object = None,
    workspace_position_command_name: object = None,
    workspace_position_x_min: object = None,
    workspace_position_x_max: object = None,
    workspace_position_y_weight: object = None,
    workspace_position_std: object = None,
    workspace_position_clip_max: object = None,
    loco_tracking_command_name: object = None,
    loco_tracking_std: object = None,
    loco_tracking_threshold: object = None,
    loco_tracking_weight: object = None,
    loco_regularization_base_height_weight: object = None,
    loco_regularization_base_height_std: object = None,
    loco_regularization_base_height_target_height: object = None,
    loco_regularization_base_height_asset_cfg: object = None,
    loco_regularization_base_height_sensor_cfg: object = None,
    loco_regularization_base_roll_weight: object = None,
    loco_regularization_base_roll_std: object = None,
    loco_regularization_base_roll_asset_cfg: object = None,
    loco_regularization_base_pitch_weight: object = None,
    loco_regularization_base_pitch_std: object = None,
    loco_regularization_base_pitch_asset_cfg: object = None,
    loco_regularization_base_roll_ang_vel_weight: object = None,
    loco_regularization_base_roll_ang_vel_std: object = None,
    loco_regularization_base_roll_ang_vel_asset_cfg: object = None,
    loco_regularization_base_pitch_ang_vel_weight: object = None,
    loco_regularization_base_pitch_ang_vel_std: object = None,
    loco_regularization_base_pitch_ang_vel_asset_cfg: object = None,
    loco_regularization_base_z_vel_weight: object = None,
    loco_regularization_base_z_vel_std: object = None,
    loco_regularization_base_z_vel_asset_cfg: object = None,
    loco_regularization_base_lateral_vel_weight: object = None,
    loco_regularization_base_lateral_vel_std: object = None,
    loco_regularization_base_lateral_vel_asset_cfg: object = None,
    loco_regularization_leg_posture_deviation_weight: object = None,
    loco_regularization_leg_posture_deviation_std: object = None,
    loco_regularization_leg_posture_deviation_asset_cfg: object = None,
    loco_regularization_leg_posture_deviation_joint_weights: object = None,
    loco_regularization_touchdown_left_right_x_symmetry_weight: object = None,
    loco_regularization_touchdown_left_right_x_symmetry_std: object = None,
    loco_regularization_touchdown_left_right_y_symmetry_weight: object = None,
    loco_regularization_touchdown_left_right_y_symmetry_std: object = None,
    loco_regularization_touchdown_foot_y_distance_weight: object = None,
    loco_regularization_touchdown_foot_y_distance_std: object = None,
    loco_regularization_touchdown_foot_y_distance_min_distance: object = None,
    loco_regularization_touchdown_foot_y_distance_max_distance: object = None,
    loco_regularization_diagonal_foot_symmetry_weight: object = None,
    loco_regularization_diagonal_foot_symmetry_std: object = None,
    loco_regularization_diagonal_foot_symmetry_sensor_cfg: object = None,
    loco_regularization_feet_contact_soft_trot_weight: object = None,
    loco_regularization_feet_contact_soft_trot_sensor_cfg: object = None,
    loco_regularization_feet_contact_soft_trot_asset_cfg: object = None,
    loco_regularization_feet_contact_soft_trot_force_std: object = None,
    loco_regularization_feet_contact_soft_trot_height_std: object = None,
    loco_regularization_feet_contact_soft_trot_vel_std: object = None,
    loco_regularization_feet_contact_soft_trot_cycle_time: object = None,
    loco_regularization_feet_contact_soft_trot_phase_offsets: object = None,
    loco_regularization_feet_contact_soft_trot_swing_height: object = None,
    loco_regularization_feet_contact_soft_trot_soft_contact_k: object = None,
    loco_regularization_feet_contact_soft_trot_contact_force_threshold: object = None,
    loco_regularization_feet_contact_soft_trot_ground_sensor_names: object = None,
    loco_arm_swing_weight: object = None,
    loco_arm_swing_asset_cfg: object = None,
    loco_arm_dynamic_weight: object = None,
    loco_arm_dynamic_asset_cfg: object = None,
    basic_is_alive_weight: object = None,
    basic_collision_weight: object = None,
    basic_collision_threshold: object = None,
    basic_collision_sensor_cfg: object = None,
    basic_collision_count_weight: object = None,
    basic_collision_force_weight: object = None,
    basic_collision_force_scale: object = None,
    basic_termination_penalty_weight: object = None,
    basic_termination_penalty_excluded_terms: object = None,
    basic_action_smoothness_first_weight: object = None,
    basic_action_smoothness_second_weight: object = None,
    basic_joint_torque_sq_weight: object = None,
    basic_joint_torque_sq_asset_cfg: object = None,
    basic_joint_torque_sq_normalize_by_effort_limit: object = None,
    basic_joint_power_weight: object = None,
    basic_joint_power_asset_cfg: object = None,
    basic_joint_power_normalize_by_effort_limit: object = None,
) -> torch.Tensor:
    """Compute total reward as (1-D)*mani + D*loco + basic."""
    _ = (
        gating_command_name,
        gating_mu,
        gating_l,
        gating_fixed_d,
        mani_position_command_name,
        mani_position_std,
        mani_position_power,
        mani_orientation_command_name,
        mani_orientation_std,
        mani_orientation_power,
        mani_regularization_support_roll_weight,
        mani_regularization_support_roll_asset_cfg,
        mani_regularization_support_roll_std,
        mani_regularization_support_feet_slide_weight,
        mani_regularization_support_feet_slide_sensor_cfg,
        mani_regularization_support_feet_slide_asset_cfg,
        mani_regularization_support_feet_slide_std,
        mani_regularization_support_foot_air_weight,
        mani_regularization_support_foot_air_threshold,
        mani_regularization_support_foot_air_sensor_cfg,
        mani_regularization_support_foot_air_clip_max,
        mani_regularization_support_non_foot_contact_weight,
        mani_regularization_support_non_foot_contact_threshold,
        mani_regularization_support_non_foot_contact_sensor_cfg,
        mani_regularization_support_non_foot_contact_count_weight,
        mani_regularization_support_non_foot_contact_force_weight,
        mani_regularization_support_non_foot_contact_force_scale,
        mani_regularization_support_non_foot_contact_clip_max,
        mani_regularization_target_height_pitch_weight,
        mani_regularization_target_height_pitch_command_name,
        mani_regularization_target_height_pitch_asset_cfg,
        mani_regularization_target_height_pitch_low_height,
        mani_regularization_target_height_pitch_high_height,
        mani_regularization_target_height_pitch_std,
        mani_regularization_min_base_height_weight,
        mani_regularization_min_base_height_asset_cfg,
        mani_regularization_min_base_height_sensor_cfg,
        mani_regularization_min_base_height_minimum_height,
        mani_regularization_min_base_height_std,
        mani_regularization_posture_deviation_weight,
        mani_regularization_posture_deviation_asset_cfg,
        mani_regularization_posture_deviation_std,
        mani_regularization_posture_deviation_joint_weights,
        mani_regularization_joint_limit_safety_weight,
        mani_regularization_joint_limit_safety_asset_cfg,
        mani_regularization_joint_limit_safety_std,
        mani_regularization_support_left_right_x_symmetry_weight,
        mani_regularization_support_left_right_x_symmetry_std,
        mani_regularization_support_left_right_y_symmetry_weight,
        mani_regularization_support_left_right_y_symmetry_std,
        mani_regularization_support_foot_xy_range_weight,
        mani_regularization_support_foot_xy_range_std,
        mani_regularization_support_foot_xy_range_x_abs_min,
        mani_regularization_support_foot_xy_range_x_abs_max,
        mani_regularization_support_foot_xy_range_y_abs_min,
        mani_regularization_support_foot_xy_range_y_abs_max,
        mani_potential_command_name,
        mani_potential_std,
        mani_potential_weight,
        mani_cumulative_error_command_name,
        mani_cumulative_error_clip_max,
        workspace_position_weight,
        workspace_position_command_name,
        workspace_position_x_min,
        workspace_position_x_max,
        workspace_position_y_weight,
        workspace_position_std,
        workspace_position_clip_max,
        loco_tracking_command_name,
        loco_tracking_std,
        loco_tracking_threshold,
        loco_tracking_weight,
        loco_regularization_base_height_weight,
        loco_regularization_base_height_std,
        loco_regularization_base_height_target_height,
        loco_regularization_base_height_asset_cfg,
        loco_regularization_base_height_sensor_cfg,
        loco_regularization_base_roll_weight,
        loco_regularization_base_roll_std,
        loco_regularization_base_roll_asset_cfg,
        loco_regularization_base_pitch_weight,
        loco_regularization_base_pitch_std,
        loco_regularization_base_pitch_asset_cfg,
        loco_regularization_base_roll_ang_vel_weight,
        loco_regularization_base_roll_ang_vel_std,
        loco_regularization_base_roll_ang_vel_asset_cfg,
        loco_regularization_base_pitch_ang_vel_weight,
        loco_regularization_base_pitch_ang_vel_std,
        loco_regularization_base_pitch_ang_vel_asset_cfg,
        loco_regularization_base_z_vel_weight,
        loco_regularization_base_z_vel_std,
        loco_regularization_base_z_vel_asset_cfg,
        loco_regularization_base_lateral_vel_weight,
        loco_regularization_base_lateral_vel_std,
        loco_regularization_base_lateral_vel_asset_cfg,
        loco_regularization_leg_posture_deviation_weight,
        loco_regularization_leg_posture_deviation_std,
        loco_regularization_leg_posture_deviation_asset_cfg,
        loco_regularization_leg_posture_deviation_joint_weights,
        loco_regularization_touchdown_left_right_x_symmetry_weight,
        loco_regularization_touchdown_left_right_x_symmetry_std,
        loco_regularization_touchdown_left_right_y_symmetry_weight,
        loco_regularization_touchdown_left_right_y_symmetry_std,
        loco_regularization_touchdown_foot_y_distance_weight,
        loco_regularization_touchdown_foot_y_distance_std,
        loco_regularization_touchdown_foot_y_distance_min_distance,
        loco_regularization_touchdown_foot_y_distance_max_distance,
        loco_regularization_diagonal_foot_symmetry_weight,
        loco_regularization_diagonal_foot_symmetry_std,
        loco_regularization_diagonal_foot_symmetry_sensor_cfg,
        loco_regularization_feet_contact_soft_trot_weight,
        loco_regularization_feet_contact_soft_trot_sensor_cfg,
        loco_regularization_feet_contact_soft_trot_asset_cfg,
        loco_regularization_feet_contact_soft_trot_force_std,
        loco_regularization_feet_contact_soft_trot_height_std,
        loco_regularization_feet_contact_soft_trot_vel_std,
        loco_regularization_feet_contact_soft_trot_cycle_time,
        loco_regularization_feet_contact_soft_trot_phase_offsets,
        loco_regularization_feet_contact_soft_trot_swing_height,
        loco_regularization_feet_contact_soft_trot_soft_contact_k,
        loco_regularization_feet_contact_soft_trot_contact_force_threshold,
        loco_regularization_feet_contact_soft_trot_ground_sensor_names,
        loco_arm_swing_weight,
        loco_arm_swing_asset_cfg,
        loco_arm_dynamic_weight,
        loco_arm_dynamic_asset_cfg,
        basic_is_alive_weight,
        basic_collision_weight,
        basic_collision_threshold,
        basic_collision_sensor_cfg,
        basic_collision_count_weight,
        basic_collision_force_weight,
        basic_collision_force_scale,
        basic_termination_penalty_weight,
        basic_termination_penalty_excluded_terms,
        basic_action_smoothness_first_weight,
        basic_action_smoothness_second_weight,
        basic_joint_torque_sq_weight,
        basic_joint_torque_sq_asset_cfg,
        basic_joint_torque_sq_normalize_by_effort_limit,
        basic_joint_power_weight,
        basic_joint_power_asset_cfg,
        basic_joint_power_normalize_by_effort_limit,
    )
    cache = _compute_go2arm_reward_terms(env)
    return cache["total_reward_debug"]

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame.

    Uses exponential kernel for reward computation.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    defined in :attr:`synced_feet_pair_names` to bias the policy towards a desired gait,
    i.e trotting, bounding, or pacing. Note that this reward is only for quadrupedal gaits
    with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _effective_action_history(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return action history after environment-side action masks when available."""
    current = getattr(env, "_go2arm_effective_action", None)
    previous = getattr(env, "_go2arm_prev_effective_action", None)
    previous_previous = getattr(env, "_go2arm_prev_prev_effective_action", None)
    if current is not None and previous is not None:
        return current, previous, previous_previous
    return (
        env.action_manager.action,
        env.action_manager.prev_action,
        getattr(env.action_manager, "prev_prev_action", None),
    )


def _roboduet_action_ids_for_joint_cfg(
    env: ManagerBasedRLEnv, robot: Articulation, joint_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Map asset joint ids to the corresponding columns in the RoboDuet action tensor."""
    action_term = env.action_manager.get_term("joint_pos")
    action_joint_names = tuple(getattr(action_term, "_joint_names", ()))
    if not action_joint_names:
        raise RuntimeError("RoboDuet joint_pos action term does not expose resolved joint names.")

    joint_ids = joint_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(*joint_ids.indices(robot.num_joints)))
    joint_names = tuple(robot.joint_names[int(joint_id)] for joint_id in joint_ids)
    missing = tuple(joint_name for joint_name in joint_names if joint_name not in action_joint_names)
    if missing:
        raise RuntimeError(f"RoboDuet action tensor is missing joints required by reward cfg: {missing}")
    return torch.tensor(
        [action_joint_names.index(joint_name) for joint_name in joint_names],
        dtype=torch.long,
        device=env.device,
    )


def action_smoothness_first_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on first-order action changes."""
    # 一阶平滑项：惩罚当前动作和上一步动作之间的跳变。
    action, prev_action, _ = _effective_action_history(env)
    diff = torch.square(action - prev_action)
    # 第一步没有有效上一时刻动作时，不对这一项计惩罚。
    diff = diff * (prev_action[:, :] != 0)
    return torch.sum(diff, dim=1)


def action_smoothness_second_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on second-order action changes."""
    # 二阶平滑项：惩罚动作离散二阶差分，抑制动作抖动和“折线感”。
    action, prev_action, prev_prev_action = _effective_action_history(env)
    if prev_prev_action is None:
        # 兼容旧版 IsaacLab：ActionManager 只维护一阶历史时，无法可靠计算二阶差分，
        # 因此这里返回零惩罚而不是伪造历史，避免改变奖励语义。
        return torch.zeros(env.num_envs, device=env.device)
    diff = torch.square(action - 2 * prev_action + prev_prev_action)
    # 前两步历史不足时，不对二阶项计惩罚。
    diff = diff * (prev_action[:, :] != 0)
    diff = diff * (prev_prev_action[:, :] != 0)
    return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def collision_force_count_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    count_weight: float = 1.0,
    force_weight: float = 1.0,
    force_scale: float = 1.0,
) -> torch.Tensor:
    """Penalty on undesired contacts using both count and above-threshold force magnitude.

    The threshold suppresses sensor noise. Contacts below the threshold do not contribute.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    force_excess = torch.clamp(contact_force_norm - threshold, min=0.0)
    contact_count = torch.sum(force_excess > 0.0, dim=1).float()
    scaled_force_excess = torch.sum(force_excess / max(force_scale, 1e-6), dim=1)
    return count_weight * contact_count + force_weight * scaled_force_excess


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _compute_roboduet_reward_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    pretrained_scales: dict[str, float],
    hybrid_scales: dict[str, float],
    only_positive_rewards: bool,
    only_positive_rewards_ji22_style: bool,
    sigma_rew_neg: float,
    tracking_sigma: float,
    tracking_sigma_yaw: float,
    gait_force_sigma: float,
    gait_vel_sigma: float,
    illegal_contact_sensor_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    foot_asset_cfg: SceneEntityCfg,
    leg_joint_cfg: SceneEntityCfg,
    arm_joint_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    ee_body_cfg: SceneEntityCfg,
    manip_weight_lpy_start: float = 4.0,
    manip_weight_lpy_end: float = 3.0,
    manip_weight_rpy_start: float = 0.0,
    manip_weight_rpy_end: float = 1.0,
    manip_weight_keep_sum_constant: bool = True,
    manip_weight_transition_iters: int = 5000,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    del base_body_cfg
    command_sums = _ensure_roboduet_logging_buffers(env, hybrid_scales)
    log_episode_sums = getattr(env, "_roboduet_log_episode_sums")
    term = env.command_manager.get_term(command_name)
    cache_key = (getattr(env, "common_step_counter", -1), bool(term.switch_open))
    cached_state = getattr(env, "_roboduet_reward_step_cache", None)
    if cached_state is not None and cached_state.get("key") == cache_key:
        return cached_state["value"]

    scales = hybrid_scales if term.switch_open else pretrained_scales
    reward_names = tuple(hybrid_scales.keys())
    active_terms = {name for name in reward_names if _scale_is_nonzero(scales.get(name, 0.0))}
    ever_active_terms = {
        name
        for name in reward_names
        if _scale_is_nonzero(pretrained_scales.get(name, 0.0)) or _scale_is_nonzero(hybrid_scales.get(name, 0.0))
    }

    def has_active(*names: str) -> bool:
        return any(name in active_terms for name in names)

    def has_ever_active(*names: str) -> bool:
        return any(name in ever_active_terms for name in names)

    robot: Articulation = env.scene["robot"]
    leg_action_ids = _roboduet_action_ids_for_joint_cfg(env, robot, leg_joint_cfg)
    arm_action_ids = _roboduet_action_ids_for_joint_cfg(env, robot, arm_joint_cfg)
    reward_dt = float(env.step_dt)
    reward_dtype = robot.data.joint_vel.dtype
    reward_dog_linear = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)
    reward_arm_linear = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)
    reward_dog_linear_scaled = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)
    reward_arm_linear_scaled = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)
    reward_pos_dog_scaled = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)
    reward_neg_dog_scaled = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)
    reward_pos_arm_scaled = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)
    reward_neg_arm_scaled = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)
    zero_reward = torch.zeros(env.num_envs, device=env.device, dtype=reward_dtype)

    foot_body_positions_w = None
    foot_body_lin_vel_w = None

    def get_foot_body_positions_w() -> torch.Tensor:
        nonlocal foot_body_positions_w
        if foot_body_positions_w is None:
            foot_body_positions_w = robot.data.body_pos_w[:, foot_asset_cfg.body_ids, :]
        return foot_body_positions_w

    def get_foot_body_lin_vel_w() -> torch.Tensor:
        nonlocal foot_body_lin_vel_w
        if foot_body_lin_vel_w is None:
            foot_body_lin_vel_w = robot.data.body_lin_vel_w[:, foot_asset_cfg.body_ids, :]
        return foot_body_lin_vel_w

    foot_force_vectors = None
    foot_force_norms = None
    foot_force_z = None
    if has_active(
        "feet_slip",
        "feet_clearance_cmd_linear",
        "tracking_contacts_shaped_force",
        "tracking_contacts_shaped_vel",
    ) or has_ever_active("feet_slip"):
        foot_force_vectors = get_go2arm_precise_foot_contact_forces(env, foot_sensor_cfg)
        if foot_force_vectors is None:
            raise RuntimeError("RoboDuet reward requires the shared `contact_forces` sensor to expose all four feet.")
        foot_force_norms = torch.linalg.norm(foot_force_vectors, dim=-1)
        foot_force_z = foot_force_vectors[..., 2]

    root_lin_vel_b, root_ang_vel_b = roboduet_base_velocity_b(robot)
    desired_contact = term.desired_contact_states

    metrics: dict[str, torch.Tensor] = {}
    if has_active("tracking_lin_vel"):
        metrics["tracking_lin_vel"] = torch.exp(
            -torch.sum(torch.square(term.commands_dog[:, :2] - root_lin_vel_b[:, :2]), dim=1) / tracking_sigma
        )
    if has_active("tracking_ang_vel"):
        metrics["tracking_ang_vel"] = torch.exp(
            -torch.square(term.commands_dog[:, 2] - root_ang_vel_b[:, 2]) / tracking_sigma_yaw
        )
    if has_active("lin_vel_z"):
        metrics["lin_vel_z"] = torch.square(root_lin_vel_b[:, 2])
    if has_active("ang_vel_xy"):
        metrics["ang_vel_xy"] = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)

    if has_active("orientation_control"):
        projected_gravity_b = roboduet_projected_gravity_b(robot)
        quat_roll = quat_from_euler_xyz(
            -term.commands_dog[:, 4], torch.zeros_like(term.commands_dog[:, 4]), torch.zeros_like(term.commands_dog[:, 4])
        )
        quat_pitch = quat_from_euler_xyz(
            torch.zeros_like(term.commands_dog[:, 3]), -term.commands_dog[:, 3], torch.zeros_like(term.commands_dog[:, 3])
        )
        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_apply_inverse(
            desired_base_quat,
            _cached_row_tensor(env, "_roboduet_down_vector_cache", (0.0, 0.0, -1.0), projected_gravity_b.dtype).expand(
                env.num_envs, -1
            ),
        )
        metrics["orientation_control"] = torch.sum(
            torch.square(projected_gravity_b[:, :2] - desired_projected_gravity[:, :2]),
            dim=1,
        )

    if has_active("loco_energy"):
        metrics["loco_energy"] = torch.sum(
            torch.square(
                robot.data.applied_torque[:, leg_joint_cfg.joint_ids] * robot.data.joint_vel[:, leg_joint_cfg.joint_ids]
            ),
            dim=1,
        )

    if foot_force_vectors is not None and foot_force_norms is not None and foot_force_z is not None:
        if has_ever_active("feet_slip"):
            foot_contacts = foot_force_z > 1.0
            last_foot_contacts = getattr(env, "_roboduet_last_foot_contacts", None)
            if last_foot_contacts is None or last_foot_contacts.shape != foot_contacts.shape:
                last_foot_contacts = torch.zeros_like(foot_contacts)
                env._roboduet_last_foot_contacts = last_foot_contacts
            if has_active("feet_slip"):
                foot_velocities_xy = get_foot_body_lin_vel_w()[..., :2]
                contact_filter = torch.logical_or(foot_contacts, last_foot_contacts)
                metrics["feet_slip"] = torch.sum(
                    contact_filter.float() * torch.sum(torch.square(foot_velocities_xy), dim=2),
                    dim=1,
                )
            last_foot_contacts.copy_(foot_contacts.detach())

        if has_active("feet_clearance_cmd_linear"):
            phases = 1.0 - torch.abs(1.0 - torch.clamp((term.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
            ground_height_data = _get_go2arm_ground_height_data(env, GO2ARM_FOOT_SCANNER_NAMES)
            foot_body_height = get_foot_body_positions_w()[..., 2]
            ground_height = torch.where(
                ground_height_data["is_valid"],
                ground_height_data["ground_height_w"],
                torch.zeros_like(foot_body_height),
            )
            foot_height = foot_body_height - ground_height
            target_height = 0.04 * phases + 0.02
            metrics["feet_clearance_cmd_linear"] = torch.sum(
                torch.square(target_height - foot_height) * (1.0 - desired_contact), dim=1
            )

        if has_active("tracking_contacts_shaped_force"):
            metrics["tracking_contacts_shaped_force"] = -torch.sum(
                (1.0 - desired_contact) * (1.0 - torch.exp(-torch.square(foot_force_norms) / gait_force_sigma)),
                dim=1,
            ) / 4.0

        if has_active("tracking_contacts_shaped_vel"):
            foot_velocities = get_foot_body_lin_vel_w()
            metrics["tracking_contacts_shaped_vel"] = -torch.sum(
                desired_contact * (1.0 - torch.exp(-torch.sum(torch.square(foot_velocities), dim=2) / gait_vel_sigma)),
                dim=1,
            ) / 4.0

    if has_active("collision"):
        illegal_sensor = env.scene.sensors[illegal_contact_sensor_cfg.name]
        illegal_forces = illegal_sensor.data.net_forces_w[:, illegal_contact_sensor_cfg.body_ids, :]
        metrics["collision"] = torch.sum((torch.norm(illegal_forces, dim=-1) > 0.1).float(), dim=1)

    if has_active("orientation_heuristic"):
        base_pos_w, base_quat_w = roboduet_base_pose_w(robot)
        _, base_pitch = _quat_to_roll_pitch(base_quat_w)
        delta_z = term.commands_arm[:, 0] * torch.sin(term.commands_arm[:, 1]) + 0.38 - base_pos_w[:, 2]
        orientation_guide = torch.zeros_like(base_pitch)
        down_flag = delta_z < -0.10
        up_flag = delta_z > 0.40
        orientation_guide[down_flag] = torch.square(base_pitch - 0.4)[down_flag]
        orientation_guide[up_flag] = torch.square(base_pitch + 0.3)[up_flag]
        metrics["orientation_heuristic"] = orientation_guide

    if has_active("arm_manip_commands_tracking_combine", "vis_manip_commands_tracking_lpy", "vis_manip_commands_tracking_rpy"):
        current_lpy = roboduet_current_lpy(env, ee_body_cfg)
        current_abg = _quat_to_abg(roboduet_current_ee_quat_in_base(env, ee_body_cfg))
        lpy_range = _cached_row_tensor(
            env,
            "_roboduet_lpy_range_cache",
            (
                term.cfg.l_range[1] - term.cfg.l_range[0],
                term.cfg.p_range[1] - term.cfg.p_range[0],
                term.cfg.y_range[1] - term.cfg.y_range[0],
            ),
            current_lpy.dtype,
        )
        rpy_range = _cached_row_tensor(
            env,
            "_roboduet_rpy_range_cache",
            (
                term.cfg.roll_ee_range[1] - term.cfg.roll_ee_range[0],
                term.cfg.pitch_ee_range[1] - term.cfg.pitch_ee_range[0],
                term.cfg.yaw_ee_range[1] - term.cfg.yaw_ee_range[0],
            ),
            current_abg.dtype,
        )
        lpy_error = torch.sum(torch.abs(current_lpy - term.commands_arm_obs[:, :3]) / lpy_range, dim=1)
        rpy_error = torch.sum(torch.abs(current_abg - term.target_abg) / rpy_range, dim=1)
        if has_active("arm_manip_commands_tracking_combine"):
            _step = int(getattr(env, "common_step_counter", 0))
            _cur_iter = float(_step) / float(max(term.cfg.steps_per_iteration, 1))
            _arm_stage_iter = max(0.0, _cur_iter - float(term.cfg.switch_iteration))
            if manip_weight_transition_iters <= 0:
                _alpha = 1.0
            else:
                _alpha = max(0.0, min(1.0, _arm_stage_iter / float(manip_weight_transition_iters)))
            _manip_weight_lpy = manip_weight_lpy_start + (manip_weight_lpy_end - manip_weight_lpy_start) * _alpha
            if manip_weight_keep_sum_constant:
                _manip_weight_rpy = (manip_weight_lpy_start + manip_weight_rpy_start) - _manip_weight_lpy
            else:
                _manip_weight_rpy = manip_weight_rpy_start + (manip_weight_rpy_end - manip_weight_rpy_start) * _alpha
            metrics["arm_manip_commands_tracking_combine"] = torch.exp(
                -(_manip_weight_lpy * lpy_error + _manip_weight_rpy * rpy_error)
            )
        if has_active("vis_manip_commands_tracking_lpy"):
            metrics["vis_manip_commands_tracking_lpy"] = torch.exp(-lpy_error)
        if has_active("vis_manip_commands_tracking_rpy"):
            metrics["vis_manip_commands_tracking_rpy"] = torch.exp(-rpy_error)

    prev_joint_vel_buffer = getattr(env, "_roboduet_prev_joint_vel", None)
    if has_ever_active("dof_acc", "arm_dof_acc"):
        if prev_joint_vel_buffer is None or prev_joint_vel_buffer.shape != robot.data.joint_vel.shape:
            prev_joint_vel_buffer = torch.zeros_like(robot.data.joint_vel)
            env._roboduet_prev_joint_vel = prev_joint_vel_buffer
    prev_joint_vel = prev_joint_vel_buffer if prev_joint_vel_buffer is not None else torch.zeros_like(robot.data.joint_vel)

    if has_active("arm_dof_vel"):
        metrics["arm_dof_vel"] = torch.sum(torch.square(robot.data.joint_vel[:, arm_joint_cfg.joint_ids]), dim=1)

    joint_pos_target = None
    if has_active(
        "arm_energy",
        "torques",
        "arm_action_smoothness_1",
        "arm_action_smoothness_2",
        "action_smoothness_1",
        "action_smoothness_2",
    ):
        joint_pos_target = getattr(env, "_go2arm_joint_pos_target", None)
        if joint_pos_target is None:
            joint_pos_target = torch.zeros_like(env.action_manager.action)

    if has_active("arm_energy"):
        metrics["arm_energy"] = torch.sum(
            torch.square(joint_pos_target[:, arm_action_ids] * robot.data.joint_vel[:, arm_joint_cfg.joint_ids]),
            dim=1,
        )

    if has_active("arm_dof_acc"):
        metrics["arm_dof_acc"] = torch.sum(
            torch.square(
                (prev_joint_vel[:, arm_joint_cfg.joint_ids] - robot.data.joint_vel[:, arm_joint_cfg.joint_ids]) / env.step_dt
            ),
            dim=1,
        )

    action = prev_action = prev_prev_action = None
    if has_active(
        "arm_action_rate",
        "action_rate",
        "action_smoothness_1",
        "action_smoothness_2",
        "arm_action_smoothness_1",
        "arm_action_smoothness_2",
        "hip_action_l2",
    ):
        action, prev_action, prev_prev_action = _effective_action_history(env)

    if has_active("arm_action_rate"):
        metrics["arm_action_rate"] = torch.sum(torch.square(prev_action[:, arm_action_ids] - action[:, arm_action_ids]), dim=1)

    plan_actions = last_plan_actions = None
    if has_active("arm_control_limits", "arm_control_smoothness_1"):
        plan_actions = getattr(env, "plan_actions", None)
        if plan_actions is None:
            plan_actions = torch.zeros(env.num_envs, 2, device=env.device, dtype=reward_dtype)
        last_plan_actions = getattr(env, "last_plan_actions", None)
        if last_plan_actions is None:
            last_plan_actions = torch.zeros_like(plan_actions)

    if has_active("arm_control_limits"):
        metrics["arm_control_limits"] = (
            torch.clamp(term.cfg.limit_body_pitch[0] - plan_actions[:, 0], min=0.0)
            + torch.clamp(plan_actions[:, 0] - term.cfg.limit_body_pitch[1], min=0.0)
            + torch.clamp(term.cfg.limit_body_roll[0] - plan_actions[:, 1], min=0.0)
            + torch.clamp(plan_actions[:, 1] - term.cfg.limit_body_roll[1], min=0.0)
        )

    if has_active("arm_control_smoothness_1"):
        plan_action_valid = (last_plan_actions != 0.0).float()
        metrics["arm_control_smoothness_1"] = torch.sum(
            torch.square(plan_actions - last_plan_actions) * plan_action_valid,
            dim=1,
        )

    if has_active("dof_vel"):
        metrics["dof_vel"] = torch.sum(torch.square(robot.data.joint_vel[:, leg_joint_cfg.joint_ids]), dim=1)

    if has_active("dof_acc"):
        metrics["dof_acc"] = torch.sum(
            torch.square(
                (prev_joint_vel[:, leg_joint_cfg.joint_ids] - robot.data.joint_vel[:, leg_joint_cfg.joint_ids]) / env.step_dt
            ),
            dim=1,
        )

    if has_active("action_rate"):
        metrics["action_rate"] = torch.sum(torch.square(prev_action[:, leg_action_ids] - action[:, leg_action_ids]), dim=1)

    if has_active("action_smoothness_1", "action_smoothness_2", "arm_action_smoothness_1", "arm_action_smoothness_2"):
        last_joint_pos_target = getattr(env, "_go2arm_last_joint_pos_target", None)
        if last_joint_pos_target is None:
            last_joint_pos_target = torch.zeros_like(joint_pos_target)
        last_last_joint_pos_target = getattr(env, "_go2arm_last_last_joint_pos_target", None)
        if last_last_joint_pos_target is None:
            last_last_joint_pos_target = torch.zeros_like(joint_pos_target)
        if prev_prev_action is None:
            prev_prev_action = torch.zeros_like(prev_action)
        leg_valid = (prev_action[:, leg_action_ids] != 0.0).float()
        leg_valid_2 = (prev_prev_action[:, leg_action_ids] != 0.0).float()
        arm_valid = (prev_action[:, arm_action_ids] != 0.0).float()
        arm_valid_2 = (prev_prev_action[:, arm_action_ids] != 0.0).float()
        if has_active("action_smoothness_1"):
            metrics["action_smoothness_1"] = torch.sum(
                torch.square(joint_pos_target[:, leg_action_ids] - last_joint_pos_target[:, leg_action_ids]) * leg_valid,
                dim=1,
            )
        if has_active("action_smoothness_2"):
            metrics["action_smoothness_2"] = torch.sum(
                torch.square(
                    joint_pos_target[:, leg_action_ids]
                    - 2.0 * last_joint_pos_target[:, leg_action_ids]
                    + last_last_joint_pos_target[:, leg_action_ids]
                )
                * leg_valid
                * leg_valid_2,
                dim=1,
            )
        if has_active("arm_action_smoothness_1"):
            metrics["arm_action_smoothness_1"] = torch.sum(
                torch.square(joint_pos_target[:, arm_action_ids] - last_joint_pos_target[:, arm_action_ids]) * arm_valid,
                dim=1,
            )
        if has_active("arm_action_smoothness_2"):
            metrics["arm_action_smoothness_2"] = torch.sum(
                torch.square(
                    joint_pos_target[:, arm_action_ids]
                    - 2.0 * last_joint_pos_target[:, arm_action_ids]
                    + last_last_joint_pos_target[:, arm_action_ids]
                )
                * arm_valid
                * arm_valid_2,
                dim=1,
            )
    if has_active("torques"):
        # Match upstream RoboDuet control_type="M" semantics: _compute_torques() returns a
        # concatenated tensor whose loco slice is the clipped leg torque target and whose arm
        # slice is the arm position target. The legacy local port only penalized Isaac's
        # applied leg torque, which omits the arm position-target slice that upstream
        # _reward_torques() includes via torch.sum(torch.square(self.env.torques), dim=1).
        leg_torque_target_global = getattr(env, "_go2arm_leg_torque_target", None)
        if torch.is_tensor(leg_torque_target_global):
            leg_torque_for_reward = leg_torque_target_global[:, leg_joint_cfg.joint_ids]
        else:
            leg_torque_for_reward = robot.data.applied_torque[:, leg_joint_cfg.joint_ids]
        metrics["torques"] = torch.sum(torch.square(leg_torque_for_reward), dim=1) + torch.sum(
            torch.square(joint_pos_target[:, arm_action_ids]),
            dim=1,
        )

    if has_active("hip_action_l2"):
        metrics["hip_action_l2"] = torch.sum(torch.square(roboduet_current_action(env)[:, [0, 3, 6, 9]]), dim=1)

    if has_active("dof_pos_limits"):
        all_joint_ids = list(leg_joint_cfg.joint_ids) + list(arm_joint_cfg.joint_ids)
        joint_pos_all = robot.data.joint_pos[:, all_joint_ids]
        joint_lower = robot.data.soft_joint_pos_limits[:, all_joint_ids, 0]
        joint_upper = robot.data.soft_joint_pos_limits[:, all_joint_ids, 1]
        out_of_limits = -(joint_pos_all - joint_lower).clamp(max=0.0)
        out_of_limits += (joint_pos_all - joint_upper).clamp(min=0.0)
        metrics["dof_pos_limits"] = torch.sum(out_of_limits, dim=1)

    if has_active("raibert_heuristic"):
        foot_pos_w = get_foot_body_positions_w()
        cur_footsteps_translated = foot_pos_w - robot.data.root_link_pos_w.unsqueeze(1)
        root_yaw_inv = yaw_quat(math_utils.quat_conjugate(robot.data.root_link_quat_w))
        footsteps_in_body_frame = math_utils.quat_apply(
            root_yaw_inv.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
            cur_footsteps_translated.reshape(-1, 3),
        ).reshape(env.num_envs, 4, 3)
        desired_stance_width = 0.3
        desired_stance_length = 0.45
        desired_ys_nom = _cached_row_tensor(
            env,
            "_roboduet_raibert_y_nom_cache",
            (
                desired_stance_width / 2,
                -desired_stance_width / 2,
                desired_stance_width / 2,
                -desired_stance_width / 2,
            ),
            footsteps_in_body_frame.dtype,
        )
        desired_xs_nom = _cached_row_tensor(
            env,
            "_roboduet_raibert_x_nom_cache",
            (
                desired_stance_length / 2,
                desired_stance_length / 2,
                -desired_stance_length / 2,
                -desired_stance_length / 2,
            ),
            footsteps_in_body_frame.dtype,
        )
        phases_raibert = torch.abs(1.0 - (term.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies_raibert = 3.0
        x_vel_des = term.commands_dog[:, 0:1]
        yaw_vel_des = term.commands_dog[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases_raibert * y_vel_des * (0.5 / frequencies_raibert)
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases_raibert * x_vel_des * (0.5 / frequencies_raibert)
        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset
        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        metrics["raibert_heuristic"] = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

    weighted_terms: dict[str, torch.Tensor] = {}
    for name in reward_names:
        scale = float(scales.get(name, 0.0))
        metric_value = metrics.get(name)
        weighted_value = zero_reward
        if metric_value is not None and scale != 0.0:
            weighted_value = metric_value * scale
        weighted_value_scaled = weighted_value * reward_dt
        weighted_terms[name] = weighted_value
        if name in _ROBODUET_LOG_ONLY_TERMS:
            log_episode_sums[name] += weighted_value_scaled
            continue
        reward_dog_linear += weighted_value
        reward_dog_linear_scaled += weighted_value_scaled
        term_sum = torch.sum(weighted_value_scaled)
        is_nonnegative = term_sum >= 0.0
        pos_mask = is_nonnegative.to(dtype=weighted_value.dtype)
        neg_mask = ((term_sum <= 0.0) & ~is_nonnegative).to(dtype=weighted_value.dtype)
        reward_pos_dog_scaled += weighted_value_scaled * pos_mask
        reward_neg_dog_scaled += weighted_value_scaled * neg_mask
        if name not in _ROBODUET_DOG_ONLY_TERMS:
            reward_arm_linear += weighted_value
            reward_arm_linear_scaled += weighted_value_scaled
            reward_pos_arm_scaled += weighted_value_scaled * pos_mask
            reward_neg_arm_scaled += weighted_value_scaled * neg_mask
        if name in scales:
            command_value = (
                float(scales[name]) * reward_dt + weighted_value_scaled
                if name in _ROBODUET_CONTACT_OFFSET_TERMS
                else weighted_value_scaled
            )
            command_sums[name] += command_value
            if name in GO2ARM_COMMAND_CURRICULUM_KEYS:
                term.accumulate_metric(name, command_value)
        log_episode_sums[name] += weighted_value_scaled

    if only_positive_rewards:
        reward_dog_scaled = torch.clamp(reward_dog_linear_scaled, min=0.0)
        reward_arm_scaled = torch.clamp(reward_arm_linear_scaled, min=0.0)
    elif only_positive_rewards_ji22_style:
        reward_dog_scaled = reward_pos_dog_scaled * torch.exp(reward_neg_dog_scaled / float(sigma_rew_neg))
        reward_arm_scaled = reward_pos_arm_scaled * torch.exp(reward_neg_arm_scaled / float(sigma_rew_neg))
    else:
        reward_dog_scaled = reward_dog_linear_scaled
        reward_arm_scaled = reward_arm_linear_scaled
    reward_dog_for_manager = reward_dog_scaled / reward_dt
    reward_arm_for_manager = reward_arm_scaled / reward_dt
    log_episode_sums["total"] += reward_dog_scaled

    command_sums["lin_vel_raw"] += root_lin_vel_b[:, 0]
    command_sums["ang_vel_raw"] += root_ang_vel_b[:, 2]
    command_sums["lin_vel_residual"] += torch.square(root_lin_vel_b[:, 0] - term.commands_dog[:, 0])
    command_sums["ang_vel_residual"] += torch.square(root_ang_vel_b[:, 2] - term.commands_dog[:, 2])
    command_sums["ep_timesteps"] += 1.0
    env._roboduet_reward_dog.copy_(reward_dog_scaled.detach())
    env._roboduet_reward_arm.copy_(reward_arm_scaled.detach())
    if prev_joint_vel_buffer is not None:
        prev_joint_vel_buffer.copy_(robot.data.joint_vel.detach())

    reward_state = {
        "weighted_terms": weighted_terms,
        "reward_dog_linear": reward_dog_linear,
        "reward_dog_linear_scaled": reward_dog_linear_scaled,
        "reward_dog": reward_dog_for_manager,
        "reward_arm": reward_arm_for_manager,
        "reward_dog_scaled": reward_dog_scaled,
        "reward_arm_scaled": reward_arm_scaled,
        "total_adjustment": reward_dog_for_manager - reward_dog_linear,
    }
    env._roboduet_reward_step_cache = {"key": cache_key, "value": reward_state}
    return reward_state


def roboduet_weighted_reward_term(
    env: ManagerBasedRLEnv,
    reward_term_name: str,
    command_name: str,
    pretrained_scales: dict[str, float],
    hybrid_scales: dict[str, float],
    only_positive_rewards: bool,
    only_positive_rewards_ji22_style: bool,
    sigma_rew_neg: float,
    tracking_sigma: float,
    tracking_sigma_yaw: float,
    gait_force_sigma: float,
    gait_vel_sigma: float,
    illegal_contact_sensor_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    foot_asset_cfg: SceneEntityCfg,
    leg_joint_cfg: SceneEntityCfg,
    arm_joint_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    ee_body_cfg: SceneEntityCfg,
    manip_weight_lpy_start: float = 4.0,
    manip_weight_lpy_end: float = 3.0,
    manip_weight_rpy_start: float = 0.0,
    manip_weight_rpy_end: float = 1.0,
    manip_weight_keep_sum_constant: bool = True,
    manip_weight_transition_iters: int = 5000,
) -> torch.Tensor:
    reward_state = _compute_roboduet_reward_state(
        env=env,
        command_name=command_name,
        pretrained_scales=pretrained_scales,
        hybrid_scales=hybrid_scales,
        only_positive_rewards=only_positive_rewards,
        only_positive_rewards_ji22_style=only_positive_rewards_ji22_style,
        sigma_rew_neg=sigma_rew_neg,
        tracking_sigma=tracking_sigma,
        tracking_sigma_yaw=tracking_sigma_yaw,
        gait_force_sigma=gait_force_sigma,
        gait_vel_sigma=gait_vel_sigma,
        illegal_contact_sensor_cfg=illegal_contact_sensor_cfg,
        foot_sensor_cfg=foot_sensor_cfg,
        foot_asset_cfg=foot_asset_cfg,
        leg_joint_cfg=leg_joint_cfg,
        arm_joint_cfg=arm_joint_cfg,
        base_body_cfg=base_body_cfg,
        ee_body_cfg=ee_body_cfg,
        manip_weight_lpy_start=manip_weight_lpy_start,
        manip_weight_lpy_end=manip_weight_lpy_end,
        manip_weight_rpy_start=manip_weight_rpy_start,
        manip_weight_rpy_end=manip_weight_rpy_end,
        manip_weight_keep_sum_constant=manip_weight_keep_sum_constant,
        manip_weight_transition_iters=manip_weight_transition_iters,
    )
    weighted_term = reward_state["weighted_terms"][reward_term_name]
    if reward_term_name in _ROBODUET_LOG_ONLY_TERMS:
        return torch.zeros_like(weighted_term)
    return weighted_term


def roboduet_total_reward_adjustment(
    env: ManagerBasedRLEnv,
    command_name: str,
    pretrained_scales: dict[str, float],
    hybrid_scales: dict[str, float],
    only_positive_rewards: bool,
    only_positive_rewards_ji22_style: bool,
    sigma_rew_neg: float,
    tracking_sigma: float,
    tracking_sigma_yaw: float,
    gait_force_sigma: float,
    gait_vel_sigma: float,
    illegal_contact_sensor_cfg: SceneEntityCfg,
    foot_sensor_cfg: SceneEntityCfg,
    foot_asset_cfg: SceneEntityCfg,
    leg_joint_cfg: SceneEntityCfg,
    arm_joint_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    ee_body_cfg: SceneEntityCfg,
    manip_weight_lpy_start: float = 4.0,
    manip_weight_lpy_end: float = 3.0,
    manip_weight_rpy_start: float = 0.0,
    manip_weight_rpy_end: float = 1.0,
    manip_weight_keep_sum_constant: bool = True,
    manip_weight_transition_iters: int = 5000,
) -> torch.Tensor:
    reward_state = _compute_roboduet_reward_state(
        env=env,
        command_name=command_name,
        pretrained_scales=pretrained_scales,
        hybrid_scales=hybrid_scales,
        only_positive_rewards=only_positive_rewards,
        only_positive_rewards_ji22_style=only_positive_rewards_ji22_style,
        sigma_rew_neg=sigma_rew_neg,
        tracking_sigma=tracking_sigma,
        tracking_sigma_yaw=tracking_sigma_yaw,
        gait_force_sigma=gait_force_sigma,
        gait_vel_sigma=gait_vel_sigma,
        illegal_contact_sensor_cfg=illegal_contact_sensor_cfg,
        foot_sensor_cfg=foot_sensor_cfg,
        foot_asset_cfg=foot_asset_cfg,
        leg_joint_cfg=leg_joint_cfg,
        arm_joint_cfg=arm_joint_cfg,
        base_body_cfg=base_body_cfg,
        ee_body_cfg=ee_body_cfg,
        manip_weight_lpy_start=manip_weight_lpy_start,
        manip_weight_lpy_end=manip_weight_lpy_end,
        manip_weight_rpy_start=manip_weight_rpy_start,
        manip_weight_rpy_end=manip_weight_rpy_end,
        manip_weight_keep_sum_constant=manip_weight_keep_sum_constant,
        manip_weight_transition_iters=manip_weight_transition_iters,
    )
    return reward_state["total_adjustment"]


class RoboDuetReward(ManagerTermBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        pretrained_scales: dict[str, float],
        hybrid_scales: dict[str, float],
        only_positive_rewards: bool,
        only_positive_rewards_ji22_style: bool,
        sigma_rew_neg: float,
        tracking_sigma: float,
        tracking_sigma_yaw: float,
        gait_force_sigma: float,
        gait_vel_sigma: float,
        illegal_contact_sensor_cfg: SceneEntityCfg,
        foot_sensor_cfg: SceneEntityCfg,
        foot_asset_cfg: SceneEntityCfg,
        leg_joint_cfg: SceneEntityCfg,
        arm_joint_cfg: SceneEntityCfg,
        base_body_cfg: SceneEntityCfg,
        ee_body_cfg: SceneEntityCfg,
        manip_weight_lpy_start: float = 4.0,
        manip_weight_lpy_end: float = 3.0,
        manip_weight_rpy_start: float = 0.0,
        manip_weight_rpy_end: float = 1.0,
        manip_weight_keep_sum_constant: bool = True,
        manip_weight_transition_iters: int = 5000,
    ) -> torch.Tensor:
        reward_state = _compute_roboduet_reward_state(
            env=env,
            command_name=command_name,
            pretrained_scales=pretrained_scales,
            hybrid_scales=hybrid_scales,
            only_positive_rewards=only_positive_rewards,
            only_positive_rewards_ji22_style=only_positive_rewards_ji22_style,
            sigma_rew_neg=sigma_rew_neg,
            tracking_sigma=tracking_sigma,
            tracking_sigma_yaw=tracking_sigma_yaw,
            gait_force_sigma=gait_force_sigma,
            gait_vel_sigma=gait_vel_sigma,
            illegal_contact_sensor_cfg=illegal_contact_sensor_cfg,
            foot_sensor_cfg=foot_sensor_cfg,
            foot_asset_cfg=foot_asset_cfg,
            leg_joint_cfg=leg_joint_cfg,
            arm_joint_cfg=arm_joint_cfg,
            base_body_cfg=base_body_cfg,
            ee_body_cfg=ee_body_cfg,
            manip_weight_lpy_start=manip_weight_lpy_start,
            manip_weight_lpy_end=manip_weight_lpy_end,
            manip_weight_rpy_start=manip_weight_rpy_start,
            manip_weight_rpy_end=manip_weight_rpy_end,
            manip_weight_keep_sum_constant=manip_weight_keep_sum_constant,
            manip_weight_transition_iters=manip_weight_transition_iters,
        )
        return reward_state["reward_dog"]
