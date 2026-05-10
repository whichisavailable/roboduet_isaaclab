# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
import time
import statistics
from collections import deque
from types import SimpleNamespace

import torch
import torch.nn as nn

from .automatic_models import ArmActorCritic, DogActorCritic
from .automatic_ppo import AutomaticPPO
from .callable_resolver import resolve_callable
from .logger_compat import Logger


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

    def _ensure_histories(self, obs):
        if self.dog_obs_dim is None:
            self.dog_obs_dim = obs["dog_policy"].shape[-1]
            self.arm_obs_dim = obs["arm_policy"].shape[-1]
            self.dog_obs_history = torch.zeros(
                obs["dog_policy"].shape[0], self.dog_obs_dim * self.dog_history_length, device=obs["dog_policy"].device
            )
            self.arm_obs_history = torch.zeros(
                obs["arm_policy"].shape[0], self.arm_obs_dim * self.arm_history_length, device=obs["arm_policy"].device
            )

    def reset(self, dones: torch.Tensor | None = None):
        if dones is None:
            if self.dog_obs_history is not None:
                self.dog_obs_history.zero_()
            if self.arm_obs_history is not None:
                self.arm_obs_history.zero_()
            return
        env_ids = dones.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        if self.dog_obs_history is not None:
            self.dog_obs_history[env_ids] = 0.0
        if self.arm_obs_history is not None:
            self.arm_obs_history[env_ids] = 0.0

    def act_inference(self, obs):
        self._ensure_histories(obs)
        dog_obs = obs["dog_policy"]
        arm_obs = obs["arm_policy"]
        self.dog_obs_history = torch.cat((self.dog_obs_history[:, self.dog_obs_dim :], dog_obs), dim=-1)
        self.arm_obs_history = torch.cat((self.arm_obs_history[:, self.arm_obs_dim :], arm_obs), dim=-1)

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

        dog_model_class = resolve_callable(dog_cfg.pop("class_name"))
        arm_model_class = resolve_callable(arm_cfg.pop("class_name"))
        algorithm_class = resolve_callable(algorithm_cfg.pop("class_name"))

        self.dog_history_length = int(dog_cfg.pop("history_length"))
        self.arm_history_length = int(arm_cfg.pop("history_length"))
        self.dog_action_dim = int(dog_cfg.pop("num_actions"))
        self.arm_action_dim = int(arm_cfg.pop("num_actions"))
        self.num_plan_actions = int(arm_cfg.pop("num_plan_actions"))
        self.arm_action_total_dim = self.arm_action_dim + self.num_plan_actions

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

        self.alg_dog: AutomaticPPO = algorithm_class(self.dog_model, device=self.device, **algorithm_cfg)
        self.alg_dog.init_storage(
            self.env.num_envs,
            int(self.cfg["num_steps_per_env"]),
            [self.dog_obs_dim],
            [self.dog_privileged_dim],
            [self.dog_obs_dim * self.dog_history_length],
            [self.dog_action_dim],
            [self.dog_action_dim],
        )
        self.alg_arm: AutomaticPPO = algorithm_class(self.arm_model, device=self.device, **algorithm_cfg)
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
        self.arm_obs_history = torch.zeros(
            self.env.num_envs, self.arm_obs_dim * self.arm_history_length, device=self.device
        )
        self.fake_arm_actions = torch.zeros(self.env.num_envs, self.arm_action_dim, device=self.device)
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

    def _clear_cached(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.dog_obs_history[env_ids] = 0.0
        self.arm_obs_history[env_ids] = 0.0

    def _get_dog_observations(self, obs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if obs is None:
            obs_manager = self.env.unwrapped.observation_manager
            dog_obs = obs_manager.compute_group("dog_policy").to(self.device)
            dog_privileged = obs_manager.compute_group("dog_privileged").to(self.device)
        else:
            dog_obs = obs["dog_policy"].to(self.device)
            dog_privileged = obs["dog_privileged"].to(self.device)
        self.dog_obs_history = torch.cat((self.dog_obs_history[:, self.dog_obs_dim :], dog_obs), dim=-1)
        return {"obs": dog_obs, "privileged_obs": dog_privileged, "obs_history": self.dog_obs_history}

    def _get_arm_observations(self, obs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if obs is None:
            obs_manager = self.env.unwrapped.observation_manager
            arm_obs = obs_manager.compute_group("arm_policy").to(self.device)
            arm_privileged = obs_manager.compute_group("arm_privileged").to(self.device)
        else:
            arm_obs = obs["arm_policy"].to(self.device)
            arm_privileged = obs["arm_privileged"].to(self.device)
        self.arm_obs_history = torch.cat((self.arm_obs_history[:, self.arm_obs_dim :], arm_obs), dim=-1)
        return {"obs": arm_obs, "privileged_obs": arm_privileged, "obs_history": self.arm_obs_history}

    def _step_env(self, dog_actions: torch.Tensor, arm_actions: torch.Tensor):
        if not self._command_term().switch_open:
            arm_actions = self.fake_arm_actions
        full_action_raw = torch.cat((dog_actions, arm_actions), dim=-1)
        full_action = self._clip_full_action(full_action_raw)
        obs, _rew, dones, extras = self.env.step(full_action)
        raw_env = self.env.unwrapped
        self._record_alignment_debug_step(dones)
        rewards_dog = getattr(raw_env, "_roboduet_reward_dog").to(self.device)
        rewards_arm = getattr(raw_env, "_roboduet_reward_arm").to(self.device)
        return obs, rewards_dog, rewards_arm, dones.to(self.device), extras

    def _clip_full_action(self, full_action: torch.Tensor) -> torch.Tensor:
        if self.clip_actions is None:
            clipped_action = full_action
        else:
            clipped_action = torch.clamp(full_action, -self.clip_actions, self.clip_actions)
        self._record_action_clip_debug(full_action, clipped_action)
        return clipped_action

    def _record_action_clip_debug(self, raw_action: torch.Tensor, clipped_action: torch.Tensor) -> None:
        del clipped_action
        if not hasattr(self, "_align_debug_steps"):
            self._reset_alignment_debug_accumulators()
        raw_dog = raw_action[:, : self.dog_action_dim].detach()
        self._align_debug_action_clip_steps += 1
        self._align_debug_dog_raw_action_abs_mean_sum += float(raw_dog.abs().mean().item())
        self._align_debug_dog_raw_action_abs_max = max(
            self._align_debug_dog_raw_action_abs_max, float(raw_dog.abs().max().item())
        )

    def _reset_alignment_debug_accumulators(self) -> None:
        self._align_debug_steps = 0
        self._align_debug_done_count = 0.0
        self._align_debug_terminated_count = 0.0
        self._align_debug_timeout_count = 0.0
        self._align_debug_height_mean_sum = 0.0
        self._align_debug_height_min = float("inf")
        self._align_debug_term_done_sums = {}
        self._align_debug_leg_control_steps = 0
        self._align_debug_leg_torque_target_abs_mean_sum = 0.0
        self._align_debug_leg_computed_torque_abs_mean_sum = 0.0
        self._align_debug_leg_applied_torque_abs_mean_sum = 0.0
        self._align_debug_leg_torque_target_diff_abs_mean_sum = 0.0
        self._align_debug_leg_torque_clip_abs_mean_sum = 0.0
        self._align_debug_action_clip_steps = 0
        self._align_debug_dog_raw_action_abs_mean_sum = 0.0
        self._align_debug_dog_raw_action_abs_max = 0.0

    def _record_alignment_debug_step(self, dones: torch.Tensor) -> None:
        if not hasattr(self, "_align_debug_steps"):
            self._reset_alignment_debug_accumulators()

        raw_env = self.env.unwrapped
        dones = dones.detach().to(device=raw_env.device, dtype=torch.bool)
        done_count = float(dones.sum().item())
        self._align_debug_steps += 1
        self._align_debug_done_count += done_count

        terminated = getattr(raw_env, "reset_terminated", None)
        if terminated is not None:
            terminated = terminated.detach().to(device=raw_env.device, dtype=torch.bool)
            self._align_debug_terminated_count += float((terminated & dones).sum().item())
        timeouts = getattr(raw_env, "reset_time_outs", None)
        if timeouts is not None:
            timeouts = timeouts.detach().to(device=raw_env.device, dtype=torch.bool)
            self._align_debug_timeout_count += float((timeouts & dones).sum().item())

        termination_manager = getattr(raw_env, "termination_manager", None)
        if termination_manager is not None and done_count > 0.0:
            for term_name in termination_manager.active_terms:
                term_value = termination_manager.get_term(term_name).detach().to(device=raw_env.device, dtype=torch.bool)
                self._align_debug_term_done_sums[term_name] = self._align_debug_term_done_sums.get(term_name, 0.0) + float(
                    (term_value & dones).sum().item()
                )

        robot = raw_env.scene["robot"]
        base_height = robot.data.root_pos_w[:, 2]
        gravity_b = robot.data.projected_gravity_b
        self._align_debug_height_mean_sum += float(base_height.mean().item())
        self._align_debug_height_min = min(self._align_debug_height_min, float(base_height.min().item()))

        leg_joint_ids = getattr(raw_env, "_go2arm_leg_joint_ids", None)
        leg_torque_target_global = getattr(raw_env, "_go2arm_leg_torque_target", None)
        active_envs = ~dones
        if leg_joint_ids is not None and leg_torque_target_global is not None and torch.any(active_envs):
            leg_joint_ids_tensor = torch.as_tensor(leg_joint_ids, dtype=torch.long, device=raw_env.device)
            leg_torque_target = leg_torque_target_global[:, leg_joint_ids_tensor][active_envs]
            self._align_debug_leg_control_steps += 1
            self._align_debug_leg_torque_target_abs_mean_sum += float(leg_torque_target.abs().mean().item())
            if hasattr(robot.data, "computed_torque"):
                leg_computed_torque = robot.data.computed_torque[:, leg_joint_ids_tensor][active_envs]
                self._align_debug_leg_computed_torque_abs_mean_sum += float(leg_computed_torque.abs().mean().item())
                self._align_debug_leg_torque_target_diff_abs_mean_sum += float(
                    (leg_torque_target - leg_computed_torque).abs().mean().item()
                )
            if hasattr(robot.data, "applied_torque"):
                leg_applied_torque = robot.data.applied_torque[:, leg_joint_ids_tensor][active_envs]
                self._align_debug_leg_applied_torque_abs_mean_sum += float(leg_applied_torque.abs().mean().item())
                if hasattr(robot.data, "computed_torque"):
                    self._align_debug_leg_torque_clip_abs_mean_sum += float(
                        (robot.data.computed_torque[:, leg_joint_ids_tensor][active_envs] - leg_applied_torque)
                        .abs()
                        .mean()
                        .item()
                    )

    def _print_alignment_debug(self, it: int) -> None:
        if not hasattr(self, "_align_debug_steps") or self._align_debug_steps == 0:
            return

        steps = float(self._align_debug_steps)
        num_envs = float(self.env.num_envs)
        done_rate = self._align_debug_done_count / max(steps * num_envs, 1.0)
        terminal_rate_on_done = self._align_debug_terminated_count / max(self._align_debug_done_count, 1.0)
        timeout_rate_on_done = self._align_debug_timeout_count / max(self._align_debug_done_count, 1.0)
        mean_episode_length = statistics.mean(self.logger.lenbuffer) if len(self.logger.lenbuffer) > 0 else float("nan")
        term_on_done = ",".join(
            f"{name}:{count / max(self._align_debug_done_count, 1.0):.3g}"
            for name, count in sorted(self._align_debug_term_done_sums.items())
        )
        if not term_on_done:
            term_on_done = "none"

        leg_steps = float(max(self._align_debug_leg_control_steps, 1))
        action_clip_steps = float(max(self._align_debug_action_clip_steps, 1))
        print(
            "[roboduet-debug] "
            f"it={it} ep_len={mean_episode_length:.6g} "
            f"term={terminal_rate_on_done:.3g} timeout={timeout_rate_on_done:.3g} "
            f"h_mean={self._align_debug_height_mean_sum / steps:.6g} h_min={self._align_debug_height_min:.6g} "
            f"leg_tau={self._align_debug_leg_torque_target_abs_mean_sum / leg_steps:.6g} "
            f"tau_isaac={self._align_debug_leg_computed_torque_abs_mean_sum / leg_steps:.6g} "
            f"tau_diff={self._align_debug_leg_torque_target_diff_abs_mean_sum / leg_steps:.6g} "
            f"tau_applied={self._align_debug_leg_applied_torque_abs_mean_sum / leg_steps:.6g} "
            f"tau_clip={self._align_debug_leg_torque_clip_abs_mean_sum / leg_steps:.6g} "
            f"raw_act={self._align_debug_dog_raw_action_abs_mean_sum / action_clip_steps:.6g} "
            f"raw_act_max={self._align_debug_dog_raw_action_abs_max:.6g} "
            f"done={term_on_done}",
            flush=True,
        )

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
        command_term = self._command_term()
        raw_env = self.env.unwrapped
        switch_open = bool(command_term.switch_open)
        arm_obs_abs_max = command_term.commands_arm_obs.abs().max().item()
        cmd_pitch_roll_abs_mean = command_term.commands_dog[:, 3:5].abs().mean().item()
        cmd_velocity_abs_mean = command_term.commands_dog[:, :3].abs().mean().item()
        effective_leg_action_abs_mean = float("nan")
        effective_arm_action_abs_mean = float("nan")
        effective_arm_action_abs_max = float("nan")
        if hasattr(raw_env, "_go2arm_effective_action"):
            effective_action = raw_env._go2arm_effective_action
            effective_leg_action_abs_mean = effective_action[:, : self.dog_action_dim].abs().mean().item()
            effective_arm_action_abs_mean = effective_action[:, self.dog_action_dim :].abs().mean().item()
            effective_arm_action_abs_max = effective_action[:, self.dog_action_dim :].abs().max().item()

        writer = self.logger.writer
        if writer is not None:
            writer.add_scalar("RoboDuet/switch_open", float(switch_open), it)
            writer.add_scalar("Policy/dog_mean_std", self.dog_model.std.mean().item(), it)
            writer.add_scalar("Policy/arm_mean_std", self.arm_model.std.mean().item(), it)
            writer.add_scalar("RoboDuet/stage1_arm_effective_action_abs_max", effective_arm_action_abs_max, it)
            writer.add_scalar("RoboDuet/stage1_arm_obs_abs_max", arm_obs_abs_max, it)
            writer.add_scalar("RoboDuet/effective_leg_action_abs_mean", effective_leg_action_abs_mean, it)
            writer.add_scalar("RoboDuet/effective_arm_action_abs_mean", effective_arm_action_abs_mean, it)
            writer.add_scalar("RoboDuet/effective_arm_action_abs_max", effective_arm_action_abs_max, it)
            writer.add_scalar("RoboDuet/stage1_command_pitch_roll_abs_mean", cmd_pitch_roll_abs_mean, it)
            writer.add_scalar("RoboDuet/stage1_command_velocity_abs_mean", cmd_velocity_abs_mean, it)
            if len(self.arm_rewbuffer) > 0:
                writer.add_scalar("Train/mean_arm_reward", statistics.mean(self.arm_rewbuffer), it)
                if getattr(self.logger, "logger_type", "tensorboard") != "wandb":
                    writer.add_scalar(
                        "Train/mean_arm_reward/time", statistics.mean(self.arm_rewbuffer), int(self.logger.tot_time)
                    )

        self._print_alignment_debug(it)

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
            if arm_rollout_active:
                arm_obs_dict = self._get_arm_observations()
            elif dog_obs_dict is None:
                dog_obs_dict = self._get_dog_observations()

            self._reset_alignment_debug_accumulators()
            rollout_start_time = time.perf_counter()
            with torch.inference_mode():
                for rollout_step in range(num_steps_per_env + 1):
                    if arm_rollout_active:
                        actions_arm_cd = self.alg_arm.act(
                            arm_obs_dict["obs"],
                            arm_obs_dict["privileged_obs"],
                            arm_obs_dict["obs_history"],
                        )
                        self.env.unwrapped.set_plan_actions(actions_arm_cd[:, self.arm_action_dim :])
                        arm_actions = actions_arm_cd[:, : self.arm_action_dim]
                        dog_obs_dict = self._get_dog_observations()
                    else:
                        arm_actions = self.fake_arm_actions

                    if rollout_step > 0:
                        self.alg_dog.process_env_step(rewards_dog, dones, extras)
                        if rollout_step == num_steps_per_env:
                            break

                    actions_dog = self.alg_dog.act(
                        dog_obs_dict["obs"], dog_obs_dict["privileged_obs"], dog_obs_dict["obs_history"]
                    )
                    obs, rewards_dog, rewards_arm, dones, extras = self._step_env(actions_dog, arm_actions)
                    self.logger.process_env_step(rewards_dog, dones, extras)
                    self._process_arm_reward_step(rewards_arm, dones)

                    if arm_rollout_active:
                        arm_obs_dict = self._get_arm_observations(obs)
                        self.alg_arm.process_env_step(rewards_arm, dones, extras)

                    done_env_ids = dones.nonzero(as_tuple=False).flatten()
                    self._clear_cached(done_env_ids)
                    if not arm_rollout_active:
                        dog_obs_dict = self._get_dog_observations(obs)

                collect_time = time.perf_counter() - rollout_start_time
                if arm_rollout_active:
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
            if arm_loss_tuple is not None:
                loss_dict.update(self._make_loss_dict("arm", arm_loss_tuple))
            else:
                loss_dict.update(self._make_zero_loss_dict("arm"))
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
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        save_dict = {
            "dog_model_state_dict": self.dog_model.state_dict(),
            "arm_model_state_dict": self.arm_model.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

        if self.log_dir is not None:
            dog_dir = os.path.join(self.log_dir, "checkpoints_dog")
            arm_dir = os.path.join(self.log_dir, "checkpoints_arm")
            os.makedirs(dog_dir, exist_ok=True)
            os.makedirs(arm_dir, exist_ok=True)
            torch.save(
                self.dog_model.state_dict(), os.path.join(dog_dir, f"ac_weights_{self.current_learning_iteration:06d}.pt")
            )
            torch.save(self.dog_model.state_dict(), os.path.join(dog_dir, "ac_weights_last_dog.pt"))
            torch.save(
                self.arm_model.state_dict(), os.path.join(arm_dir, f"ac_weights_{self.current_learning_iteration:06d}.pt")
            )
            torch.save(self.arm_model.state_dict(), os.path.join(arm_dir, "ac_weights_last_arm.pt"))
            if self.export_deploy_models:
                self._save_deploy_models()

    def _save_deploy_models(self) -> None:
        if self.log_dir is None:
            return

        deploy_dir = os.path.join(self.log_dir, "deploy_model")
        os.makedirs(deploy_dir, exist_ok=True)

        # 保持与上游 `auto_train` 相同的 deploy_model 文件名，方便直接复用后处理脚本。
        dog_adaptation = copy.deepcopy(self.dog_model.adaptation_module).to("cpu")
        torch.jit.script(dog_adaptation).save(os.path.join(deploy_dir, "adaptation_module_latest_dog.jit"))
        dog_body = copy.deepcopy(self.dog_model.actor_body).to("cpu")
        torch.jit.script(dog_body).save(os.path.join(deploy_dir, "body_latest_dog.jit"))

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
        if "dog_model_state_dict" in loaded_dict:
            self.dog_model.load_state_dict(loaded_dict["dog_model_state_dict"], strict=strict)
            self.arm_model.load_state_dict(loaded_dict["arm_model_state_dict"], strict=strict)
            self.current_learning_iteration = int(loaded_dict.get("iter", 0))
            self.inference_policy.reset()
            return loaded_dict.get("infos")
        # backward compatible fallback: dog-only or arm-only raw checkpoints
        self.dog_model.load_state_dict(loaded_dict, strict=strict)
        self.inference_policy.reset()
        return None

    def get_inference_policy(self, device: str | None = None):
        self.alg_dog.eval_mode()
        self.alg_arm.eval_mode()
        if device is not None:
            self.dog_model.to(device)
            self.arm_model.to(device)
        self.inference_policy.reset()
        return self.inference_policy
