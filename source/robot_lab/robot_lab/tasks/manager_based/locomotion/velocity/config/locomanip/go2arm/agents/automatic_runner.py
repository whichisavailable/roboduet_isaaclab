# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
import time
from types import SimpleNamespace

import torch
import torch.nn as nn

from .automatic_models import ArmActorCritic, DogActorCritic
from .automatic_ppo import AutomaticPPO
from .callable_resolver import resolve_callable


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
        self.git_status_repos: list[str] = []
        self.export_deploy_models = bool(self.cfg.get("roboduet_export_deploy_models", True))
        self.log_interval = max(1, int(self.cfg.get("log_interval", 1)))

        obs = self.env.get_observations()
        self.dog_obs_dim = int(obs["dog_policy"].shape[-1])
        self.dog_privileged_dim = int(obs["dog_privileged"].shape[-1])
        self.arm_obs_dim = int(obs["arm_policy"].shape[-1])
        self.arm_privileged_dim = int(obs["arm_privileged"].shape[-1])

        dog_cfg = dict(self.cfg["dog_model"])
        arm_cfg = dict(self.cfg["arm_model"])
        algorithm_cfg = dict(self.cfg["algorithm"])

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

        print(
            "[INFO] RoboDuet runner initialized: "
            f"num_envs={self.env.num_envs}, "
            f"num_steps_per_env={int(self.cfg['num_steps_per_env'])}, "
            f"save_interval={int(self.cfg['save_interval'])}, "
            f"log_interval={self.log_interval}"
        )
        reset_start_time = time.perf_counter()
        self.env.reset()
        print(f"[INFO] RoboDuet runner env.reset() finished in {time.perf_counter() - reset_start_time:.2f}s.")

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

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

    def _get_dog_observations(self) -> dict[str, torch.Tensor]:
        obs = self.env.get_observations()
        dog_obs = obs["dog_policy"].to(self.device)
        dog_privileged = obs["dog_privileged"].to(self.device)
        self.dog_obs_history = torch.cat((self.dog_obs_history[:, self.dog_obs_dim :], dog_obs), dim=-1)
        return {"obs": dog_obs, "privileged_obs": dog_privileged, "obs_history": self.dog_obs_history}

    def _get_arm_observations(self) -> dict[str, torch.Tensor]:
        obs = self.env.get_observations()
        arm_obs = obs["arm_policy"].to(self.device)
        arm_privileged = obs["arm_privileged"].to(self.device)
        self.arm_obs_history = torch.cat((self.arm_obs_history[:, self.arm_obs_dim :], arm_obs), dim=-1)
        return {"obs": arm_obs, "privileged_obs": arm_privileged, "obs_history": self.arm_obs_history}

    def _step_env(self, dog_actions: torch.Tensor, arm_actions: torch.Tensor):
        if not self._command_term().switch_open:
            arm_actions = self.fake_arm_actions
        full_action = torch.cat((dog_actions, arm_actions), dim=-1)
        _obs, _rew, dones, extras = self.env.step(full_action)
        raw_env = self.env.unwrapped
        rewards_dog = getattr(raw_env, "_roboduet_reward_dog").to(self.device)
        rewards_arm = getattr(raw_env, "_roboduet_reward_arm").to(self.device)
        return rewards_dog, rewards_arm, dones.to(self.device), extras

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.alg_dog.train_mode()
        self.alg_arm.train_mode()

        arm_obs_dict = self._get_arm_observations()
        num_steps_per_env = int(self.cfg["num_steps_per_env"])

        total_it = self.current_learning_iteration + num_learning_iterations
        print(
            "[INFO] Starting RoboDuet training: "
            f"start_iteration={self.current_learning_iteration}, "
            f"total_iterations={total_it}, "
            f"num_steps_per_env={num_steps_per_env}, "
            f"switch_open={self._command_term().switch_open}"
        )
        for it in range(self.current_learning_iteration, total_it):
            iteration_start_time = time.perf_counter()
            with torch.inference_mode():
                for rollout_step in range(num_steps_per_env + 1):
                    if self._command_term().switch_open:
                        actions_arm_cd = self.alg_arm.act(
                            arm_obs_dict["obs"],
                            arm_obs_dict["privileged_obs"],
                            arm_obs_dict["obs_history"],
                        )
                        self.env.unwrapped.set_plan_actions(actions_arm_cd[:, self.arm_action_dim :])
                        arm_actions = actions_arm_cd[:, : self.arm_action_dim]
                    else:
                        arm_actions = self.fake_arm_actions

                    dog_obs_dict = self._get_dog_observations()
                    if rollout_step > 0:
                        self.alg_dog.process_env_step(rewards_dog, dones, extras)
                        if rollout_step == num_steps_per_env:
                            break

                    actions_dog = self.alg_dog.act(
                        dog_obs_dict["obs"], dog_obs_dict["privileged_obs"], dog_obs_dict["obs_history"]
                    )
                    rewards_dog, rewards_arm, dones, extras = self._step_env(actions_dog, arm_actions)

                    if self._command_term().switch_open:
                        arm_obs_dict = self._get_arm_observations()
                        self.alg_arm.process_env_step(rewards_arm, dones, extras)

                    done_env_ids = dones.nonzero(as_tuple=False).flatten()
                    self._clear_cached(done_env_ids)

                rollout_duration = time.perf_counter() - iteration_start_time
                if self._command_term().switch_open:
                    self.alg_arm.compute_returns(arm_obs_dict["obs_history"], arm_obs_dict["privileged_obs"])
                self.alg_dog.compute_returns(dog_obs_dict["obs_history"], dog_obs_dict["privileged_obs"])

            update_start_time = time.perf_counter()
            if self._command_term().switch_open:
                self.alg_arm.update(un_adapt=False)
            self.alg_dog.update()
            update_duration = time.perf_counter() - update_start_time
            self.current_learning_iteration = it + 1
            iteration_duration = time.perf_counter() - iteration_start_time

            if (
                self.current_learning_iteration == 1
                or self.current_learning_iteration % self.log_interval == 0
                or self.current_learning_iteration == total_it
            ):
                print(
                    "[INFO] RoboDuet iteration "
                    f"{self.current_learning_iteration}/{total_it}: "
                    f"rollout={rollout_duration:.2f}s, "
                    f"update={update_duration:.2f}s, "
                    f"total={iteration_duration:.2f}s, "
                    f"switch_open={self._command_term().switch_open}"
                )

            if self.log_dir is not None and self.current_learning_iteration % int(self.cfg["save_interval"]) == 0:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        save_dict = {
            "dog_model_state_dict": self.dog_model.state_dict(),
            "arm_model_state_dict": self.arm_model.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)

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
