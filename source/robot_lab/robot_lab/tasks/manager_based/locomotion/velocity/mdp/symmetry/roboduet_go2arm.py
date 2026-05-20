# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch.distributions import Normal

_DOG_POLICY_OBS_DIM = 56
_DOG_PRIVILEGED_OBS_DIM = 2
_ARM_POLICY_OBS_DIM = 20
_ARM_PRIVILEGED_OBS_DIM = 9

_LEG_SIGN = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=torch.float32)
_ARM_SIGN = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
_PLAN_SIGN = torch.tensor([1.0, -1.0], dtype=torch.float32)
_FOOT_LEFT_RIGHT_INDEX = (1, 0, 3, 2)


def augment_dog_ppo_batch(
    *,
    privileged_obs_batch: torch.Tensor,
    obs_history_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    old_mu_batch: torch.Tensor | None = None,
    old_sigma_batch: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    mirrored_history = _mirror_dog_obs_history(obs_history_batch)
    mirrored_privileged = _mirror_dog_privileged_obs(privileged_obs_batch)
    mirrored_actions = _mirror_leg_joint_data(actions_batch)
    output = {
        "mirrored_privileged_obs": mirrored_privileged,
        "mirrored_obs_history": mirrored_history,
        "mirrored_actions": mirrored_actions,
    }
    if old_mu_batch is not None:
        output["mirrored_old_mu"] = _mirror_leg_joint_data(old_mu_batch)
    if old_sigma_batch is not None:
        output["mirrored_old_sigma"] = _mirror_std_like_leg(old_sigma_batch)
        output["mirrored_old_actions_log_prob"] = gaussian_log_prob(
            mirrored_actions,
            output["mirrored_old_mu"],
            output["mirrored_old_sigma"],
        )
    return output


def augment_arm_ppo_batch(
    *,
    privileged_obs_batch: torch.Tensor,
    obs_history_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    old_mu_batch: torch.Tensor | None = None,
    old_sigma_batch: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    mirrored_history = _mirror_arm_obs_history(obs_history_batch)
    mirrored_privileged = _mirror_arm_privileged_obs(privileged_obs_batch)
    mirrored_actions = _mirror_arm_action(actions_batch)
    output = {
        "mirrored_privileged_obs": mirrored_privileged,
        "mirrored_obs_history": mirrored_history,
        "mirrored_actions": mirrored_actions,
    }
    if old_mu_batch is not None:
        output["mirrored_old_mu"] = _mirror_arm_action(old_mu_batch)
    if old_sigma_batch is not None:
        output["mirrored_old_sigma"] = _mirror_std_like_arm(old_sigma_batch)
        output["mirrored_old_actions_log_prob"] = gaussian_log_prob(
            mirrored_actions,
            output["mirrored_old_mu"],
            output["mirrored_old_sigma"],
        )
    return output


def mirror_dog_action_mean(actions: torch.Tensor) -> torch.Tensor:
    return _mirror_leg_joint_data(actions)


def mirror_arm_action_mean(actions: torch.Tensor) -> torch.Tensor:
    return _mirror_arm_action(actions)


def gaussian_log_prob(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    distribution = Normal(mean, std)
    return distribution.log_prob(actions).sum(dim=-1, keepdim=True)


def _mirror_dog_obs_history(obs_history: torch.Tensor) -> torch.Tensor:
    return _mirror_obs_history(obs_history, obs_dim=_DOG_POLICY_OBS_DIM, mirror_step_fn=_mirror_dog_policy_obs)


def _mirror_arm_obs_history(obs_history: torch.Tensor) -> torch.Tensor:
    return _mirror_obs_history(obs_history, obs_dim=_ARM_POLICY_OBS_DIM, mirror_step_fn=_mirror_arm_policy_obs)


def _mirror_obs_history(
    obs_history: torch.Tensor,
    *,
    obs_dim: int,
    mirror_step_fn,
) -> torch.Tensor:
    if obs_history.shape[-1] % obs_dim != 0:
        raise ValueError(f"Observation history dim {obs_history.shape[-1]} is not divisible by obs dim {obs_dim}.")
    history_length = obs_history.shape[-1] // obs_dim
    mirrored = obs_history.reshape(obs_history.shape[0], history_length, obs_dim).clone()
    mirrored = mirror_step_fn(mirrored.reshape(-1, obs_dim)).reshape(obs_history.shape[0], history_length, obs_dim)
    return mirrored.reshape(obs_history.shape[0], history_length * obs_dim)


def _mirror_dog_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != _DOG_POLICY_OBS_DIM:
        raise ValueError(f"Dog policy obs symmetry expects dim {_DOG_POLICY_OBS_DIM}, got {obs.shape[-1]}.")
    mirrored = obs.clone()
    mirrored[:, 0:3] = _apply_sign(obs[:, 0:3], (1.0, -1.0, 1.0))
    mirrored[:, 3:15] = _mirror_leg_joint_data(obs[:, 3:15])
    mirrored[:, 15:27] = _mirror_leg_joint_data(obs[:, 15:27])
    mirrored[:, 27:39] = _mirror_leg_joint_data(obs[:, 27:39])
    mirrored[:, 39:44] = _apply_sign(obs[:, 39:44], (1.0, -1.0, -1.0, 1.0, -1.0))
    mirrored[:, 44:50] = _apply_sign(obs[:, 44:50], (1.0, 1.0, -1.0, -1.0, 1.0, -1.0))
    mirrored[:, 50:52] = _apply_sign(obs[:, 50:52], (-1.0, 1.0))
    mirrored[:, 52:56] = _switch_feet_scalar(obs[:, 52:56])
    return mirrored


def _mirror_arm_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != _ARM_POLICY_OBS_DIM:
        raise ValueError(f"Arm policy obs symmetry expects dim {_ARM_POLICY_OBS_DIM}, got {obs.shape[-1]}.")
    mirrored = obs.clone()
    mirrored[:, 0:6] = _apply_sign(obs[:, 0:6], _ARM_SIGN)
    mirrored[:, 6:12] = _apply_sign(obs[:, 6:12], _ARM_SIGN)
    mirrored[:, 12:18] = _apply_sign(obs[:, 12:18], (1.0, 1.0, -1.0, -1.0, 1.0, -1.0))
    mirrored[:, 18:20] = _apply_sign(obs[:, 18:20], (-1.0, 1.0))
    return mirrored


def _mirror_dog_privileged_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != _DOG_PRIVILEGED_OBS_DIM:
        raise ValueError(f"Dog privileged obs symmetry expects dim {_DOG_PRIVILEGED_OBS_DIM}, got {obs.shape[-1]}.")
    return obs.clone()


def _mirror_arm_privileged_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != _ARM_PRIVILEGED_OBS_DIM:
        raise ValueError(f"Arm privileged obs symmetry expects dim {_ARM_PRIVILEGED_OBS_DIM}, got {obs.shape[-1]}.")
    mirrored = obs.clone()
    mirrored[:, 2:5] = _apply_sign(obs[:, 2:5], (1.0, 1.0, -1.0))
    mirrored[:, 5:9] = _apply_sign(obs[:, 5:9], (1.0, -1.0, 1.0, -1.0))
    return mirrored


def _mirror_leg_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    mirrored = torch.empty_like(joint_data)
    mirrored[..., 0:3] = joint_data[..., 3:6]
    mirrored[..., 3:6] = joint_data[..., 0:3]
    mirrored[..., 6:9] = joint_data[..., 9:12]
    mirrored[..., 9:12] = joint_data[..., 6:9]
    sign = _LEG_SIGN.to(device=joint_data.device, dtype=joint_data.dtype)
    return mirrored * sign


def _mirror_std_like_leg(joint_data: torch.Tensor) -> torch.Tensor:
    return _mirror_leg_joint_data(joint_data).abs()


def _mirror_arm_action(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-1] != 8:
        raise ValueError(f"Arm action symmetry expects dim 8, got {actions.shape[-1]}.")
    mirrored = actions.clone()
    mirrored[:, 0:6] = _apply_sign(actions[:, 0:6], _ARM_SIGN)
    mirrored[:, 6:8] = _apply_sign(actions[:, 6:8], _PLAN_SIGN)
    return mirrored


def _mirror_std_like_arm(actions: torch.Tensor) -> torch.Tensor:
    return _mirror_arm_action(actions).abs()


def _switch_feet_scalar(values: torch.Tensor) -> torch.Tensor:
    return values[..., list(_FOOT_LEFT_RIGHT_INDEX)]


def _apply_sign(values: torch.Tensor, sign_values: tuple[float, ...] | torch.Tensor) -> torch.Tensor:
    if isinstance(sign_values, torch.Tensor):
        sign = sign_values.to(device=values.device, dtype=values.dtype)
    else:
        sign = torch.tensor(sign_values, device=values.device, dtype=values.dtype)
    return values * sign
