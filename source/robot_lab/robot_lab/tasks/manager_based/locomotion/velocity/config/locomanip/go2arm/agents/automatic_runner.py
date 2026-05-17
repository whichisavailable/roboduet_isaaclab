# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
import time
import statistics
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .automatic_models import ArmActorCritic, DogActorCritic
from .automatic_ppo import AutomaticPPO
from .callable_resolver import resolve_callable
from .logger_compat import Logger


def _append_obs_history(
    history: torch.Tensor, history_scratch: torch.Tensor, obs: torch.Tensor, obs_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the latest observation into a flattened history buffer without overlapping copies."""
    if history.shape[-1] > obs_dim:
        history_scratch[:, :-obs_dim].copy_(history[:, obs_dim:])
    history_scratch[:, -obs_dim:].copy_(obs)
    return history_scratch, history


class RoboDuetAutomaticInferencePolicy(nn.Module):
    """复现 Roboduet `auto_train` dog/arm 联合推理逻辑的确定性策略。"""

    skip_generic_export = True

    def __init__(
        self,
        env,
        dog_model: DogActorCritic,
        arm_model: ArmActorCritic,
        dog_history_length: int,
        arm_history_length: int,
        dog_action_dim: int,
        arm_action_dim: int,
        num_plan_actions: int,
    ):
        super().__init__()
        self.env = env
        self.dog_model = dog_model
        self.arm_model = arm_model
        self.dog_history_length = dog_history_length
        self.arm_history_length = arm_history_length
        self.dog_action_dim = dog_action_dim
        self.arm_action_dim = arm_action_dim
        self.num_plan_actions = num_plan_actions
        self.dog_obs_dim = None
        self.arm_obs_dim = None
        self.dog_obs_history = None
        self.arm_obs_history = None
        self.dog_obs_history_scratch = None
        self.arm_obs_history_scratch = None

    def _ensure_histories(self, obs):
        if self.dog_obs_dim is None:
            self.dog_obs_dim = obs["dog_policy"].shape[-1]
            self.arm_obs_dim = obs["arm_policy"].shape[-1]
            self.dog_obs_history = torch.zeros(
                obs["dog_policy"].shape[0], self.dog_obs_dim * self.dog_history_length, device=obs["dog_policy"].device
            )
            self.dog_obs_history_scratch = torch.zeros_like(self.dog_obs_history)
            self.arm_obs_history = torch.zeros(
                obs["arm_policy"].shape[0], self.arm_obs_dim * self.arm_history_length, device=obs["arm_policy"].device
            )
            self.arm_obs_history_scratch = torch.zeros_like(self.arm_obs_history)

    def reset(self, dones: torch.Tensor | None = None):
        if dones is None:
            if self.dog_obs_history is not None:
                self.dog_obs_history.zero_()
                self.dog_obs_history_scratch.zero_()
            if self.arm_obs_history is not None:
                self.arm_obs_history.zero_()
                self.arm_obs_history_scratch.zero_()
            return
        env_ids = dones.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        if self.dog_obs_history is not None:
            self.dog_obs_history[env_ids] = 0.0
            self.dog_obs_history_scratch[env_ids] = 0.0
        if self.arm_obs_history is not None:
            self.arm_obs_history[env_ids] = 0.0
            self.arm_obs_history_scratch[env_ids] = 0.0

    def act_inference(self, obs):
        self._ensure_histories(obs)
        dog_obs = obs["dog_policy"]
        arm_obs = obs["arm_policy"]
        self.dog_obs_history, self.dog_obs_history_scratch = _append_obs_history(
            self.dog_obs_history, self.dog_obs_history_scratch, dog_obs, self.dog_obs_dim
        )
        self.arm_obs_history, self.arm_obs_history_scratch = _append_obs_history(
            self.arm_obs_history, self.arm_obs_history_scratch, arm_obs, self.arm_obs_dim
        )

        command_term = self.env.unwrapped.command_manager.get_term("roboduet")
        with torch.inference_mode():
            dog_action = self.dog_model.act_student(self.dog_obs_history)
            if command_term.switch_open:
                arm_action_cd = self.arm_model.act_student(self.arm_obs_history)
                self.env.unwrapped.set_plan_actions(arm_action_cd[:, self.arm_action_dim :])
                arm_action = arm_action_cd[:, : self.arm_action_dim]
            else:
                arm_action = torch.zeros(
                    dog_action.shape[0], self.arm_action_dim, device=dog_action.device, dtype=dog_action.dtype
                )
            return torch.cat((dog_action[:, : self.dog_action_dim], arm_action), dim=-1)

    def forward(self, obs):
        return self.act_inference(obs)


class RoboDuetAutomaticRunner:
    """Custom runner that mirrors Roboduet upstream `auto_train`."""

    _DOG_SYMMETRY_CALLABLE = (
        "robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry.roboduet_go2arm:augment_dog_ppo_batch"
    )
    _DOG_SYMMETRY_MIRROR_CALLABLE = (
        "robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry.roboduet_go2arm:mirror_dog_action_mean"
    )
    _ARM_SYMMETRY_CALLABLE = (
        "robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry.roboduet_go2arm:augment_arm_ppo_batch"
    )
    _ARM_SYMMETRY_MIRROR_CALLABLE = (
        "robot_lab.tasks.manager_based.locomotion.velocity.mdp.symmetry.roboduet_go2arm:mirror_arm_action_mean"
    )

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.log_dir = log_dir
        self.device = device
        self.current_learning_iteration = 0
        clip_actions_cfg = self.cfg.get("clip_actions", None)
        self.clip_actions = None if clip_actions_cfg is None else float(clip_actions_cfg)
        self.export_deploy_models = bool(self.cfg.get("roboduet_export_deploy_models", True))
        self.gpu_world_size = 1
        self.gpu_global_rank = 0
        self.is_distributed = False

        obs = self.env.get_observations()
        self.dog_obs_dim = int(obs["dog_policy"].shape[-1])
        self.dog_privileged_dim = int(obs["dog_privileged"].shape[-1])
        self.arm_obs_dim = int(obs["arm_policy"].shape[-1])
        self.arm_privileged_dim = int(obs["arm_privileged"].shape[-1])

        dog_cfg = dict(self.cfg["dog_model"])
        arm_cfg = dict(self.cfg["arm_model"])
        algorithm_cfg = dict(self.cfg["algorithm"])
        logger_algorithm_cfg = dict(algorithm_cfg)
        logger_algorithm_cfg.setdefault("rnd_cfg", None)
        self.cfg["algorithm"] = logger_algorithm_cfg
        algorithm_cfg.pop("rnd_cfg", None)
        dog_algorithm_cfg = dict(algorithm_cfg)
        arm_algorithm_cfg = dict(algorithm_cfg)
        if bool(self.cfg.get("symmetry", False)):
            symmetry_loss_coef = float(self.cfg.get("symmetry_loss_coef", 1.0))
            dog_algorithm_cfg["symmetry_callable"] = self._DOG_SYMMETRY_CALLABLE
            dog_algorithm_cfg["symmetry_mirror_callable"] = self._DOG_SYMMETRY_MIRROR_CALLABLE
            dog_algorithm_cfg["symmetry_loss_coef"] = symmetry_loss_coef
            arm_algorithm_cfg["symmetry_callable"] = self._ARM_SYMMETRY_CALLABLE
            arm_algorithm_cfg["symmetry_mirror_callable"] = self._ARM_SYMMETRY_MIRROR_CALLABLE
            arm_algorithm_cfg["symmetry_loss_coef"] = symmetry_loss_coef

        dog_model_class = resolve_callable(dog_cfg.pop("class_name"))
        arm_model_class = resolve_callable(arm_cfg.pop("class_name"))
        algorithm_class = resolve_callable(algorithm_cfg.pop("class_name"))
        dog_algorithm_cfg.pop("class_name", None)
        arm_algorithm_cfg.pop("class_name", None)

        self.dog_history_length = int(dog_cfg.pop("history_length"))
        self.arm_history_length = int(arm_cfg.pop("history_length"))
        self.dog_action_dim = int(dog_cfg.pop("num_actions"))
        self.arm_action_dim = int(arm_cfg.pop("num_actions"))
        self.num_plan_actions = int(arm_cfg.pop("num_plan_actions"))
        self.arm_action_total_dim = self.arm_action_dim + self.num_plan_actions
        self._dog_policy_command_slice = self._resolve_dog_policy_command_slice()

        self.dog_model: DogActorCritic = dog_model_class(
            num_obs=self.dog_obs_dim,
            num_privileged_obs=self.dog_privileged_dim,
            num_obs_history=self.dog_obs_dim * self.dog_history_length,
            num_actions=self.dog_action_dim,
            **dog_cfg,
        ).to(self.device)
        self.arm_model: ArmActorCritic = arm_model_class(
            num_obs=self.arm_obs_dim,
            num_privileged_obs=self.arm_privileged_dim,
            num_obs_history=self.arm_obs_dim * self.arm_history_length,
            num_actions=self.arm_action_total_dim,
            **arm_cfg,
        ).to(self.device)
        self._configure_stage_switch()
        self._load_pretrained_components()

        self.alg_dog: AutomaticPPO = algorithm_class(self.dog_model, device=self.device, **dog_algorithm_cfg)
        self.alg_dog.init_storage(
            self.env.num_envs,
            int(self.cfg["num_steps_per_env"]),
            [self.dog_obs_dim],
            [self.dog_privileged_dim],
            [self.dog_obs_dim * self.dog_history_length],
            [self.dog_action_dim],
            [self.dog_action_dim],
        )
        self.alg_arm: AutomaticPPO = algorithm_class(self.arm_model, device=self.device, **arm_algorithm_cfg)
        self.alg_arm.init_storage(
            self.env.num_envs,
            int(self.cfg["num_steps_per_env"]),
            [self.arm_obs_dim],
            [self.arm_privileged_dim],
            [self.arm_obs_dim * self.arm_history_length],
            [self.arm_action_total_dim],
            [self.arm_action_total_dim],
        )

        self.dog_obs_history = torch.zeros(
            self.env.num_envs, self.dog_obs_dim * self.dog_history_length, device=self.device
        )
        self.dog_obs_history_scratch = torch.zeros_like(self.dog_obs_history)
        self.arm_obs_history = torch.zeros(
            self.env.num_envs, self.arm_obs_dim * self.arm_history_length, device=self.device
        )
        self.arm_obs_history_scratch = torch.zeros_like(self.arm_obs_history)
        self.fake_arm_actions = torch.zeros(self.env.num_envs, self.arm_action_dim, device=self.device)
        self.full_action = torch.zeros(
            self.env.num_envs, self.dog_action_dim + self.arm_action_dim, device=self.device
        )
        self.arm_rewbuffer = deque(maxlen=100)
        self.cur_arm_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.inference_policy = RoboDuetAutomaticInferencePolicy(
            self.env,
            self.dog_model,
            self.arm_model,
            self.dog_history_length,
            self.arm_history_length,
            self.dog_action_dim,
            self.arm_action_dim,
            self.num_plan_actions,
        )
        # keep a minimal compatibility surface for play/export code that inspects runner.alg.policy
        self.alg = SimpleNamespace(policy=self.inference_policy)
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.env.reset()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _resolve_stage_switch_iteration(self) -> int:
        if bool(self.cfg.get("roboduet_disable_two_stage", False)):
            return 0

        explicit_iteration = self.cfg.get("roboduet_stage_switch_iteration")
        if explicit_iteration is not None:
            return int(explicit_iteration)

        has_dog_ckpt = bool(self.cfg.get("roboduet_pretrained_dog_checkpoint"))
        has_arm_ckpt = bool(self.cfg.get("roboduet_pretrained_arm_checkpoint"))
        if has_dog_ckpt and has_arm_ckpt:
            return 2000
        return 10000

    def _configure_stage_switch(self) -> None:
        # 统一把命令项和动作冻结掩码的切换迭代绑到同一个值，保持与上游两阶段训练一致。
        switch_iteration = self._resolve_stage_switch_iteration()
        command_term = self._command_term()
        command_term.cfg.switch_iteration = switch_iteration

        raw_env = self.env.unwrapped
        if hasattr(raw_env, "cfg") and hasattr(raw_env.cfg, "commands") and hasattr(raw_env.cfg.commands, "roboduet"):
            raw_env.cfg.commands.roboduet.switch_iteration = switch_iteration

        action_term = None
        if hasattr(raw_env, "action_manager"):
            try:
                action_term = raw_env.action_manager.get_term("joint_pos")
            except KeyError:
                action_term = None
        if action_term is not None and hasattr(action_term, "cfg") and hasattr(
            action_term.cfg, "fixed_delta_action_until_iteration"
        ):
            action_term.cfg.fixed_delta_action_until_iteration = switch_iteration
        if hasattr(raw_env, "cfg") and hasattr(raw_env.cfg, "actions") and hasattr(raw_env.cfg.actions, "joint_pos"):
            raw_env.cfg.actions.joint_pos.fixed_delta_action_until_iteration = switch_iteration

        command_term._update_switch_state()

    @staticmethod
    def _extract_state_dict(loaded_obj, state_key: str):
        if isinstance(loaded_obj, dict) and state_key in loaded_obj:
            return loaded_obj[state_key]
        return loaded_obj

    @staticmethod
    def _state_dict_precheck(state_dict, model: nn.Module | None = None) -> tuple[int, int, int]:
        if not isinstance(state_dict, dict) or model is None:
            return 0, 0, 0
        model_state = model.state_dict()
        model_keys = list(model_state.keys())
        state_keys = set(state_dict.keys())
        missing = [key for key in model_keys if key not in state_keys]
        unexpected = [key for key in state_dict.keys() if key not in model_state]
        shape_mismatch = []
        for key in model_keys:
            if key not in state_dict:
                continue
            if torch.is_tensor(state_dict[key]) and tuple(state_dict[key].shape) != tuple(model_state[key].shape):
                shape_mismatch.append(key)
        return len(missing), len(unexpected), len(shape_mismatch)

    @staticmethod
    def _load_result_counts(load_result) -> tuple[int, int]:
        missing = list(getattr(load_result, "missing_keys", []) or [])
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        return len(missing), len(unexpected)

    @staticmethod
    def _derive_companion_arm_checkpoint_path(dog_checkpoint_path: str) -> str | None:
        dog_dir = os.path.basename(os.path.dirname(dog_checkpoint_path))
        if dog_dir != "checkpoints_dog":
            return None
        run_dir = os.path.dirname(os.path.dirname(dog_checkpoint_path))
        arm_dir = os.path.join(run_dir, "checkpoints_arm")
        dog_name = os.path.basename(dog_checkpoint_path)
        if dog_name == "ac_weights_last_dog.pt":
            arm_name = "ac_weights_last_arm.pt"
        elif dog_name.startswith("ac_weights_") and dog_name.endswith(".pt"):
            arm_name = dog_name
        else:
            return None
        return os.path.join(arm_dir, arm_name)

    def _load_companion_arm_checkpoint(
        self, dog_checkpoint_path: str, strict: bool = True, map_location: str | None = None
    ) -> str | None:
        arm_checkpoint_path = self._derive_companion_arm_checkpoint_path(dog_checkpoint_path)
        if arm_checkpoint_path is None:
            print("[ROBODUET LOAD] arm: no companion checkpoint inferred for this dog checkpoint path.")
            return None
        if not os.path.exists(arm_checkpoint_path):
            print(
                f"[ROBODUET LOAD] arm: companion checkpoint not found at {arm_checkpoint_path}. "
                "This is expected for pure stage1 checkpoints before arm saving starts."
            )
            return None

        arm_loaded = torch.load(arm_checkpoint_path, weights_only=False, map_location=map_location)
        arm_state_dict = self._extract_state_dict(arm_loaded, "arm_model_state_dict")
        pre_missing, pre_unexpected, pre_shape = self._state_dict_precheck(arm_state_dict, self.arm_model)
        arm_load_result = self.arm_model.load_state_dict(arm_state_dict, strict=strict)
        missing, unexpected = self._load_result_counts(arm_load_result)
        print(
            f"[ROBODUET LOAD] arm path={arm_checkpoint_path} "
            f"precheck(missing={pre_missing}, unexpected={pre_unexpected}, shape={pre_shape}) "
            f"load(missing={missing}, unexpected={unexpected})"
        )
        return arm_checkpoint_path

    def _load_pretrained_components(self) -> None:
        dog_checkpoint = self.cfg.get("roboduet_pretrained_dog_checkpoint")
        arm_checkpoint = self.cfg.get("roboduet_pretrained_arm_checkpoint")
        if not dog_checkpoint and not arm_checkpoint:
            return
        if not dog_checkpoint or not arm_checkpoint:
            raise ValueError(
                "RoboDuet auto-train separate pretrained loading requires both "
                "`roboduet_pretrained_dog_checkpoint` and `roboduet_pretrained_arm_checkpoint`."
            )

        dog_loaded = torch.load(dog_checkpoint, weights_only=False, map_location=self.device)
        arm_loaded = torch.load(arm_checkpoint, weights_only=False, map_location=self.device)
        # 兼容两种来源：上游 `ac_weights_*.pt` 的裸 state_dict，或本地组合 checkpoint。
        self.dog_model.load_state_dict(self._extract_state_dict(dog_loaded, "dog_model_state_dict"), strict=True)
        self.arm_model.load_state_dict(self._extract_state_dict(arm_loaded, "arm_model_state_dict"), strict=True)

    def _command_term(self):
        return self.env.unwrapped.command_manager.get_term("roboduet")

    def _resolve_dog_policy_command_slice(self) -> slice | None:
        fixed_dims = 3 + 5 + 6 + 2 + 4
        variable_dims = self.dog_obs_dim - fixed_dims
        if variable_dims != 3 * self.dog_action_dim:
            return None
        command_start = 3 + variable_dims
        return slice(command_start, command_start + 5)

    def _clear_cached(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.dog_obs_history[env_ids] = 0.0
        self.dog_obs_history_scratch[env_ids] = 0.0
        self.arm_obs_history[env_ids] = 0.0
        self.arm_obs_history_scratch[env_ids] = 0.0

    def _get_dog_observations(self, obs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if obs is None:
            obs_manager = self.env.unwrapped.observation_manager
            dog_obs = obs_manager.compute_group("dog_policy").to(self.device)
            dog_privileged = obs_manager.compute_group("dog_privileged").to(self.device)
        else:
            dog_obs = obs["dog_policy"].to(self.device)
            dog_privileged = obs["dog_privileged"].to(self.device)
        self.dog_obs_history, self.dog_obs_history_scratch = _append_obs_history(
            self.dog_obs_history, self.dog_obs_history_scratch, dog_obs, self.dog_obs_dim
        )
        return {"obs": dog_obs, "privileged_obs": dog_privileged, "obs_history": self.dog_obs_history}

    def _get_dog_observations_from_cached_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._dog_policy_command_slice is None:
            return self._get_dog_observations()
        dog_obs = obs["dog_policy"].to(self.device)
        term = self._command_term()
        dog_obs[:, self._dog_policy_command_slice].copy_(term.commands_dog * term.commands_scale_dog)
        dog_privileged = obs["dog_privileged"].to(self.device)
        self.dog_obs_history, self.dog_obs_history_scratch = _append_obs_history(
            self.dog_obs_history, self.dog_obs_history_scratch, dog_obs, self.dog_obs_dim
        )
        return {"obs": dog_obs, "privileged_obs": dog_privileged, "obs_history": self.dog_obs_history}

    def _get_arm_observations(self, obs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if obs is None:
            obs_manager = self.env.unwrapped.observation_manager
            arm_obs = obs_manager.compute_group("arm_policy").to(self.device)
            arm_privileged = obs_manager.compute_group("arm_privileged").to(self.device)
        else:
            arm_obs = obs["arm_policy"].to(self.device)
            arm_privileged = obs["arm_privileged"].to(self.device)
        self.arm_obs_history, self.arm_obs_history_scratch = _append_obs_history(
            self.arm_obs_history, self.arm_obs_history_scratch, arm_obs, self.arm_obs_dim
        )
        return {"obs": arm_obs, "privileged_obs": arm_privileged, "obs_history": self.arm_obs_history}

    def _step_env(self, dog_actions: torch.Tensor, arm_actions: torch.Tensor):
        if not self._command_term().switch_open:
            arm_actions = self.fake_arm_actions
        self.full_action[:, : self.dog_action_dim].copy_(dog_actions[:, : self.dog_action_dim])
        self.full_action[:, self.dog_action_dim :].copy_(arm_actions[:, : self.arm_action_dim])
        full_action = self._clip_full_action(self.full_action)
        obs, _rew, dones, extras = self.env.step(full_action)
        raw_env = self.env.unwrapped
        rewards_dog = getattr(raw_env, "_roboduet_reward_dog").to(self.device)
        rewards_arm = getattr(raw_env, "_roboduet_reward_arm").to(self.device)
        return obs, rewards_dog, rewards_arm, dones.to(self.device), extras

    def _clip_full_action(self, full_action: torch.Tensor) -> torch.Tensor:
        if self.clip_actions is not None:
            full_action.clamp_(min=-self.clip_actions, max=self.clip_actions)
        return full_action

    @staticmethod
    def _make_loss_dict(prefix: str, loss_tuple) -> dict[str, float]:
        return {
            f"{prefix}/value_function": float(loss_tuple[0]),
            f"{prefix}/surrogate": float(loss_tuple[1]),
            f"{prefix}/adaptation_module": float(loss_tuple[2]),
            f"{prefix}/adaptation_module_test": float(loss_tuple[5]),
        }

    @staticmethod
    def _make_zero_loss_dict(prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/value_function": 0.0,
            f"{prefix}/surrogate": 0.0,
            f"{prefix}/adaptation_module": 0.0,
            f"{prefix}/adaptation_module_test": 0.0,
        }

    def _process_arm_reward_step(self, rewards_arm: torch.Tensor, dones: torch.Tensor) -> None:
        self.cur_arm_reward_sum += rewards_arm
        new_ids = (dones > 0).nonzero(as_tuple=False)
        self.arm_rewbuffer.extend(self.cur_arm_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        self.cur_arm_reward_sum[new_ids] = 0

    def _log_roboduet_scalars(self, it: int) -> None:
        writer = self.logger.writer
        if writer is not None:
            writer.add_scalar("Policy/dog_mean_std", self.dog_model.std.mean().item(), it)
            writer.add_scalar("Policy/arm_mean_std", self.arm_model.std.mean().item(), it)
            if len(self.arm_rewbuffer) > 0:
                writer.add_scalar("Train/mean_arm_reward", statistics.mean(self.arm_rewbuffer), it)
                if getattr(self.logger, "logger_type", "tensorboard") != "wandb":
                    writer.add_scalar(
                        "Train/mean_arm_reward/time", statistics.mean(self.arm_rewbuffer), int(self.logger.tot_time)
                    )
            term = self._command_term()
            curriculum = term._curriculum
            weights = curriculum.weights
            total_weight = weights.sum()
            if total_weight > 0:
                probs = weights / total_weight
                x_vel_grid = curriculum.grid[0]
                yaw_vel_grid = curriculum.grid[2]
                writer.add_scalar("Curriculum/x_vel_mean", float(np.dot(probs, x_vel_grid)), it)
                writer.add_scalar("Curriculum/yaw_vel_mean", float(np.dot(probs, yaw_vel_grid)), it)
                writer.add_scalar("Curriculum/active_bins", int(np.count_nonzero(weights)), it)
                writer.add_scalar("Curriculum/min_weight", float(weights.min()), it)
                x_var = float(np.dot(probs, x_vel_grid**2) - np.dot(probs, x_vel_grid) ** 2)
                writer.add_scalar("Curriculum/x_vel_std", float(np.sqrt(max(x_var, 0.0))), it)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.alg_dog.train_mode()
        self.alg_arm.train_mode()
        self.logger.init_logging_writer()

        arm_obs_dict = self._get_arm_observations()
        dog_obs_dict = None
        if not self._command_term().switch_open:
            dog_obs_dict = self._get_dog_observations()
        num_steps_per_env = int(self.cfg["num_steps_per_env"])

        start_it = self.current_learning_iteration
        total_it = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, total_it):
            arm_rollout_active = bool(self._command_term().switch_open)
            rollout_obs = None
            if arm_rollout_active:
                rollout_obs = self.env.get_observations()
                arm_obs_dict = self._get_arm_observations(rollout_obs)
            elif dog_obs_dict is None:
                dog_obs_dict = self._get_dog_observations()

            rollout_start_time = time.perf_counter()
            with torch.inference_mode():
                for rollout_step in range(num_steps_per_env):
                    if arm_rollout_active:
                        actions_arm_cd = self.alg_arm.act(
                            arm_obs_dict["obs"],
                            arm_obs_dict["privileged_obs"],
                            arm_obs_dict["obs_history"],
                        )
                        self.env.unwrapped.set_plan_actions(actions_arm_cd[:, self.arm_action_dim :])
                        arm_actions = actions_arm_cd[:, : self.arm_action_dim]
                        dog_obs_dict = self._get_dog_observations_from_cached_obs(rollout_obs)
                    else:
                        arm_actions = self.fake_arm_actions

                    actions_dog = self.alg_dog.act(
                        dog_obs_dict["obs"], dog_obs_dict["privileged_obs"], dog_obs_dict["obs_history"]
                    )
                    obs, rewards_dog, rewards_arm, dones, extras = self._step_env(actions_dog, arm_actions)
                    self.logger.process_env_step(rewards_dog, dones, extras)
                    self._process_arm_reward_step(rewards_arm, dones)
                    self.alg_dog.process_env_step(rewards_dog, dones, extras)

                    if arm_rollout_active:
                        rollout_obs = obs
                        self.alg_arm.process_env_step(rewards_arm, dones, extras)

                    done_env_ids = dones.nonzero(as_tuple=False).flatten()
                    self._clear_cached(done_env_ids)
                    if arm_rollout_active:
                        arm_obs_dict = self._get_arm_observations(rollout_obs)
                    else:
                        dog_obs_dict = self._get_dog_observations(obs)

                collect_time = time.perf_counter() - rollout_start_time
                if arm_rollout_active:
                    actions_arm_cd = self.alg_arm.act(
                        arm_obs_dict["obs"],
                        arm_obs_dict["privileged_obs"],
                        arm_obs_dict["obs_history"],
                    )
                    self.env.unwrapped.set_plan_actions(actions_arm_cd[:, self.arm_action_dim :])
                    dog_obs_dict = self._get_dog_observations_from_cached_obs(rollout_obs)
                    self.alg_arm.compute_returns(arm_obs_dict["obs_history"], arm_obs_dict["privileged_obs"])
                self.alg_dog.compute_returns(dog_obs_dict["obs_history"], dog_obs_dict["privileged_obs"])

            update_start_time = time.perf_counter()
            if arm_rollout_active:
                arm_loss_tuple = self.alg_arm.update(un_adapt=False)
            else:
                arm_loss_tuple = None
            dog_loss_tuple = self.alg_dog.update()
            learn_time = time.perf_counter() - update_start_time
            self.current_learning_iteration = it

            loss_dict = self._make_loss_dict("dog", dog_loss_tuple)
            loss_dict["dog/symmetry"] = float(getattr(self.alg_dog, "last_symmetry_loss", 0.0))
            if arm_loss_tuple is not None:
                loss_dict.update(self._make_loss_dict("arm", arm_loss_tuple))
                loss_dict["arm/symmetry"] = float(getattr(self.alg_arm, "last_symmetry_loss", 0.0))
            else:
                loss_dict.update(self._make_zero_loss_dict("arm"))
                loss_dict["arm/symmetry"] = 0.0
            action_std = (
                torch.cat((self.dog_model.std.detach(), self.arm_model.std.detach()))
                if self._command_term().switch_open
                else self.dog_model.std.detach()
            )
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg_dog.learning_rate,
                action_std=action_std,
                policy_std_dict={
                    "dog": self.dog_model.std.detach(),
                    "arm": self.arm_model.std.detach(),
                },
                rnd_weight=None,
            )
            self._log_roboduet_scalars(it)

            if self.log_dir is not None and it % int(self.cfg["save_interval"]) == 0:
                self.save(it=it, save_arm=bool(self._command_term().switch_open))

        if self.log_dir is not None:
            self.save(it=self.current_learning_iteration, save_arm=bool(self._command_term().switch_open))
        self.logger.stop_logging_writer()

    def save(self, it: int | None = None, save_arm: bool | None = None) -> None:
        """Save RoboDuet checkpoints with upstream `auto_train` directory semantics."""
        if self.log_dir is None:
            return
        save_iteration = self.current_learning_iteration if it is None else int(it)
        should_save_arm = bool(self._command_term().switch_open) if save_arm is None else bool(save_arm)
        self._save_dog_checkpoint(save_iteration)
        if should_save_arm:
            self._save_arm_checkpoint(save_iteration)

    def _save_dog_checkpoint(self, it: int) -> None:
        dog_dir = os.path.join(self.log_dir, "checkpoints_dog")
        os.makedirs(dog_dir, exist_ok=True)
        torch.save(self.dog_model.state_dict(), os.path.join(dog_dir, f"ac_weights_{it:06d}.pt"))
        torch.save(self.dog_model.state_dict(), os.path.join(dog_dir, "ac_weights_last_dog.pt"))
        if self.export_deploy_models:
            self._save_dog_deploy_model()

    def _save_arm_checkpoint(self, it: int) -> None:
        arm_dir = os.path.join(self.log_dir, "checkpoints_arm")
        os.makedirs(arm_dir, exist_ok=True)
        torch.save(self.arm_model.state_dict(), os.path.join(arm_dir, f"ac_weights_{it:06d}.pt"))
        torch.save(self.arm_model.state_dict(), os.path.join(arm_dir, "ac_weights_last_arm.pt"))
        if self.export_deploy_models:
            self._save_arm_deploy_models()

    def _deploy_model_dir(self) -> str | None:
        if self.log_dir is None:
            return None
        deploy_dir = os.path.join(self.log_dir, "deploy_model")
        os.makedirs(deploy_dir, exist_ok=True)
        return deploy_dir

    def _save_dog_deploy_model(self) -> None:
        deploy_dir = self._deploy_model_dir()
        if deploy_dir is None:
            return
        # 保持与上游 `auto_train` 相同的 deploy_model 文件名，方便直接复用后处理脚本。
        dog_adaptation = copy.deepcopy(self.dog_model.adaptation_module).to("cpu")
        torch.jit.script(dog_adaptation).save(os.path.join(deploy_dir, "adaptation_module_latest_dog.jit"))
        dog_body = copy.deepcopy(self.dog_model.actor_body).to("cpu")
        torch.jit.script(dog_body).save(os.path.join(deploy_dir, "body_latest_dog.jit"))

    def _save_arm_deploy_models(self) -> None:
        deploy_dir = self._deploy_model_dir()
        if deploy_dir is None:
            return
        arm_adaptation = copy.deepcopy(self.arm_model.adaptation_module).to("cpu")
        torch.jit.script(arm_adaptation).save(os.path.join(deploy_dir, "adaptation_module_latest_arm.jit"))
        arm_body = copy.deepcopy(self.arm_model.actor_body).to("cpu")
        torch.jit.script(arm_body).save(os.path.join(deploy_dir, "body_latest_arm.jit"))
        arm_history = copy.deepcopy(self.arm_model.actor_history_encoder).to("cpu")
        torch.jit.script(arm_history).save(os.path.join(deploy_dir, "history_latest_arm.jit"))

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        del load_cfg
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        if isinstance(loaded_dict, dict) and "dog_model_state_dict" in loaded_dict:
            dog_state_dict = loaded_dict["dog_model_state_dict"]
            arm_state_dict = loaded_dict["arm_model_state_dict"]
            dog_pre_missing, dog_pre_unexpected, dog_pre_shape = self._state_dict_precheck(dog_state_dict, self.dog_model)
            arm_pre_missing, arm_pre_unexpected, arm_pre_shape = self._state_dict_precheck(arm_state_dict, self.arm_model)
            dog_load_result = self.dog_model.load_state_dict(dog_state_dict, strict=strict)
            arm_load_result = self.arm_model.load_state_dict(arm_state_dict, strict=strict)
            dog_missing, dog_unexpected = self._load_result_counts(dog_load_result)
            arm_missing, arm_unexpected = self._load_result_counts(arm_load_result)
            print(
                f"[ROBODUET LOAD] path={path} branch=combined "
                f"dog_precheck=({dog_pre_missing},{dog_pre_unexpected},{dog_pre_shape}) "
                f"dog_load=({dog_missing},{dog_unexpected}) "
                f"arm_precheck=({arm_pre_missing},{arm_pre_unexpected},{arm_pre_shape}) "
                f"arm_load=({arm_missing},{arm_unexpected})"
            )
            self.current_learning_iteration = int(loaded_dict.get("iter", 0))
            self.inference_policy.reset()
            self._last_load_debug = {
                "path": path,
                "branch": "combined_robotlab_checkpoint",
                "iter": self.current_learning_iteration,
            }
            return loaded_dict.get("infos")

        if not isinstance(loaded_dict, dict):
            raise TypeError(
                f"Unsupported RoboDuet checkpoint object from {path}: {type(loaded_dict).__name__}. "
                "Expected a RobotLab combined checkpoint dict or a raw dog state_dict."
            )

        # backward compatible fallback: dog raw checkpoint, matching upstream `checkpoints_dog/ac_weights_*.pt`.
        # When the dog checkpoint is in `checkpoints_dog/`, also load the paired arm checkpoint from `checkpoints_arm/`.
        dog_pre_missing, dog_pre_unexpected, dog_pre_shape = self._state_dict_precheck(loaded_dict, self.dog_model)
        dog_load_result = self.dog_model.load_state_dict(loaded_dict, strict=strict)
        dog_missing, dog_unexpected = self._load_result_counts(dog_load_result)
        print(
            f"[ROBODUET LOAD] path={path} branch=raw_dog "
            f"dog_precheck=({dog_pre_missing},{dog_pre_unexpected},{dog_pre_shape}) "
            f"dog_load=({dog_missing},{dog_unexpected})"
        )
        arm_checkpoint_path = self._load_companion_arm_checkpoint(path, strict=strict, map_location=map_location)
        self.inference_policy.reset()
        self._last_load_debug = {
            "path": path,
            "arm_path": arm_checkpoint_path,
            "branch": "raw_dog_state_dict",
            "iter": self.current_learning_iteration,
        }
        return None

    def load_dog_checkpoint(
        self,
        path: str,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        """Load only the dog actor-critic weights. Used by stage2 bootstrap in train.py."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Accept either a raw state_dict (upstream checkpoints_dog/ac_weights_*.pt)
        # or a combined robotlab checkpoint that contains "dog_model_state_dict".
        dog_state_dict = self._extract_state_dict(loaded_dict, "dog_model_state_dict")
        if not isinstance(dog_state_dict, dict):
            raise TypeError(
                f"Unsupported dog checkpoint at {path}: {type(loaded_dict).__name__}. "
                "Expected a raw state_dict or a dict with 'dog_model_state_dict'."
            )
        pre_missing, pre_unexpected, pre_shape = self._state_dict_precheck(dog_state_dict, self.dog_model)
        load_result = self.dog_model.load_state_dict(dog_state_dict, strict=strict)
        missing, unexpected = self._load_result_counts(load_result)
        print(
            f"[ROBODUET LOAD] path={path} branch=dog_only "
            f"precheck=({pre_missing},{pre_unexpected},{pre_shape}) "
            f"load=({missing},{unexpected})"
        )
        self.inference_policy.reset()
        return loaded_dict.get("infos") if isinstance(loaded_dict, dict) else None

    def get_inference_policy(self, device: str | None = None):
        self.alg_dog.eval_mode()
        self.alg_arm.eval_mode()
        if device is not None:
            self.dog_model.to(device)
            self.arm_model.to(device)
        self.inference_policy.reset()
        return self.inference_policy
