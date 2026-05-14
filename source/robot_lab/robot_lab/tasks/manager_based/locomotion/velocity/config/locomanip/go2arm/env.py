# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
import time

import torch

from isaaclab.envs import ManagerBasedRLEnv

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.cus_velocity_env_cfg import (
    GO2ARM_ARM_JOINT_NAMES,
    GO2ARM_LEG_JOINT_NAMES,
    resolve_go2arm_arm_joint_names,
)


class Go2ArmManagerBasedRLEnv(ManagerBasedRLEnv):
    """go2arm 额外调试日志环境。"""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._debug_zero_action = bool(getattr(cfg, "debug_zero_action", False))
        self._enable_play_termination_reason_logging = bool(
            getattr(cfg, "enable_play_termination_reason_logging", False)
        )
        self._episode_log_key_prefixes = tuple(getattr(cfg, "episode_log_key_prefixes", ()) or ())
        reward_log_interval_iterations = getattr(cfg, "reward_log_interval_iterations", None)
        reward_log_steps_per_iteration = int(getattr(cfg, "reward_log_steps_per_iteration", 24))
        if reward_log_interval_iterations is not None:
            self._reward_log_interval = max(1, int(reward_log_interval_iterations) * reward_log_steps_per_iteration)
        else:
            self._reward_log_interval = max(1, int(getattr(cfg, "reward_log_interval", 100)))
        self._reward_log_counter = 0
        self._reward_log_sums: dict[str, torch.Tensor] = {}
        self._reward_log_counts: dict[str, torch.Tensor] = {}
        self._roboduet_reward_step_cache = None
        self.action_manager.prev_prev_action = torch.zeros_like(self.action_manager.action)
        self.num_plan_actions = 2
        self.plan_actions = torch.zeros(self.num_envs, self.num_plan_actions, device=self.device)
        self.last_plan_actions = torch.zeros_like(self.plan_actions)
        self._go2arm_joint_pos_target = torch.zeros_like(self.action_manager.action)
        self._go2arm_last_joint_pos_target = torch.zeros_like(self.action_manager.action)
        self._go2arm_last_last_joint_pos_target = torch.zeros_like(self.action_manager.action)
        self._roboduet_reward_dog = torch.zeros(self.num_envs, device=self.device)
        self._roboduet_reward_arm = torch.zeros(self.num_envs, device=self.device)
        self._validate_go2arm_shared_contact_feet()
        arm_joint_names = resolve_go2arm_arm_joint_names(cfg)
        self._go2arm_arm_joint_ids, _ = self.scene["robot"].find_joints(arm_joint_names, preserve_order=True)
        self._go2arm_hold_joint_ids: tuple[int, ...] = ()
        hold_joint_patterns = tuple(getattr(getattr(cfg.actions, "joint_pos", None), "hold_fixed_joint_names", ()) or ())
        if hold_joint_patterns:
            hold_joint_ids = []
            for joint_id, joint_name in enumerate(self.scene["robot"].joint_names):
                if any(re.fullmatch(pattern, joint_name) for pattern in hold_joint_patterns):
                    hold_joint_ids.append(int(joint_id))
            self._go2arm_hold_joint_ids = tuple(hold_joint_ids)
        self._go2arm_leg_joint_ids, _ = self.scene["robot"].find_joints(GO2ARM_LEG_JOINT_NAMES, preserve_order=True)
        self._configure_roboduet_motor_randomization(cfg)
        self._configure_roboduet_gravity_randomization(cfg)
        self._roboduet_step_profile_enabled = bool(getattr(cfg, "roboduet_profile_collection", False))
        self._roboduet_step_profile_sync_cuda = bool(getattr(cfg, "roboduet_profile_sync_cuda", True))
        self._roboduet_step_profile_totals: dict[str, float] = {}
        self._roboduet_step_profile_count = 0

    def _roboduet_profile_sync(self) -> None:
        if self._roboduet_step_profile_sync_cuda and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(self.device))

    def _roboduet_profile_stamp(self) -> float:
        self._roboduet_profile_sync()
        return time.perf_counter()

    def _roboduet_profile_add(self, key: str, duration_s: float) -> None:
        self._roboduet_step_profile_totals[key] = self._roboduet_step_profile_totals.get(key, 0.0) + float(duration_s)

    def consume_roboduet_step_profile(self) -> tuple[dict[str, float], int]:
        totals = dict(self._roboduet_step_profile_totals)
        count = int(self._roboduet_step_profile_count)
        self._roboduet_step_profile_totals.clear()
        self._roboduet_step_profile_count = 0
        return totals, count

    def _configure_roboduet_motor_randomization(self, cfg) -> None:
        robot = self.scene["robot"]
        self._roboduet_randomize_motor_strength = bool(getattr(cfg, "roboduet_randomize_motor_strength", False))
        self._roboduet_randomize_motor_offset = bool(getattr(cfg, "roboduet_randomize_motor_offset", False))
        self._roboduet_motor_strength_range = tuple(
            float(value) for value in getattr(cfg, "roboduet_motor_strength_range", (0.9, 1.1))
        )
        self._roboduet_motor_offset_range = tuple(
            float(value) for value in getattr(cfg, "roboduet_motor_offset_range", (-0.02, 0.02))
        )
        interval_s = float(getattr(cfg, "roboduet_motor_randomization_interval_s", 4.0))
        self._roboduet_motor_randomization_interval_steps = max(1, int(math.ceil(interval_s / self.step_dt)))
        self._roboduet_motor_strengths = torch.ones(
            self.num_envs, robot.data.default_joint_pos.shape[1], dtype=torch.float32, device=self.device
        )
        self._roboduet_motor_offsets = torch.zeros_like(self._roboduet_motor_strengths)
        self._roboduet_leg_joint_ids_tensor = torch.as_tensor(
            self._go2arm_leg_joint_ids, dtype=torch.long, device=self.device
        )
        self._randomize_roboduet_motor_props(torch.arange(self.num_envs, device=self.device))

    def _randomize_roboduet_motor_props(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._roboduet_randomize_motor_strength:
            low, high = self._roboduet_motor_strength_range
            values = torch.rand(env_ids.numel(), 1, dtype=torch.float32, device=self.device) * (high - low) + low
            self._roboduet_motor_strengths[env_ids, :] = values
        else:
            self._roboduet_motor_strengths[env_ids, :] = 1.0
        if self._roboduet_randomize_motor_offset:
            low, high = self._roboduet_motor_offset_range
            self._roboduet_motor_offsets[env_ids, :] = (
                torch.rand(
                    env_ids.numel(),
                    self._roboduet_motor_offsets.shape[1],
                    dtype=torch.float32,
                    device=self.device,
                )
                * (high - low)
                + low
            )
        else:
            self._roboduet_motor_offsets[env_ids, :] = 0.0

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if hasattr(self, "_roboduet_motor_strengths"):
            self._randomize_roboduet_motor_props(env_ids)

    def _configure_roboduet_gravity_randomization(self, cfg) -> None:
        self._roboduet_randomize_gravity = bool(getattr(cfg, "roboduet_randomize_gravity", False))
        self._roboduet_nominal_gravity = tuple(float(value) for value in self.sim.cfg.gravity)
        self._roboduet_gravity_range = tuple(float(value) for value in getattr(cfg, "roboduet_gravity_range", (-1.0, 1.0)))
        interval_s = float(getattr(cfg, "roboduet_gravity_interval_s", 8.0))
        duration = float(getattr(cfg, "roboduet_gravity_impulse_duration", 0.99))
        self._roboduet_gravity_interval_steps = max(1, int(math.ceil(interval_s / self.step_dt)))
        self._roboduet_gravity_duration_steps = max(
            1, int(math.ceil(self._roboduet_gravity_interval_steps * duration))
        )
        if self._roboduet_randomize_gravity:
            self._randomize_roboduet_gravity()

    def _apply_roboduet_gravity(self, gravity_w: torch.Tensor) -> None:
        gravity_list = [float(value) for value in gravity_w.detach().cpu().tolist()]
        mdp.randomize_physics_scene_gravity(
            self,
            None,
            gravity_distribution_params=(gravity_list, gravity_list),
            operation="abs",
            distribution="uniform",
        )
        gravity_dir = gravity_w.to(device=self.device, dtype=torch.float32)
        gravity_dir = gravity_dir / torch.clamp(torch.linalg.norm(gravity_dir), min=1.0e-8)
        robot = self.scene["robot"]
        robot.data.GRAVITY_VEC_W[:] = gravity_dir.unsqueeze(0)

    def _randomize_roboduet_gravity(self) -> None:
        low, high = self._roboduet_gravity_range
        nominal_gravity = torch.tensor(self._roboduet_nominal_gravity, dtype=torch.float32)
        gravity_offset = torch.empty(3, dtype=torch.float32).uniform_(low, high)
        self._apply_roboduet_gravity(nominal_gravity + gravity_offset)

    def _reset_roboduet_gravity(self) -> None:
        self._apply_roboduet_gravity(torch.tensor(self._roboduet_nominal_gravity, dtype=torch.float32))

    def _update_roboduet_gravity_randomization(self) -> None:
        if not self._roboduet_randomize_gravity:
            return
        step = int(self.common_step_counter)
        interval = self._roboduet_gravity_interval_steps
        duration = self._roboduet_gravity_duration_steps
        if step > 0 and step % interval == 0:
            self._randomize_roboduet_gravity()
        if step >= duration and (step - duration) % interval == 0:
            self._reset_roboduet_gravity()

    def set_plan_actions(self, plan_actions: torch.Tensor) -> None:
        """镜像 upstream `env.plan(...)`，先缓存，再把 pitch/roll 规划动作写回命令项。"""
        if plan_actions.shape[-1] != self.num_plan_actions:
            raise ValueError(
                f"Go2Arm RoboDuet expects {self.num_plan_actions} plan actions, but got shape {tuple(plan_actions.shape)}."
            )
        self.plan_actions.copy_(plan_actions * 0.4)
        self._command_term("roboduet").apply_plan_actions(plan_actions)

    def _validate_go2arm_shared_contact_feet(self) -> None:
        """Ensure the shared whole-body contact sensor exposes the four Go2Arm feet."""
        try:
            foot_body_ids, _ = self.scene["robot"].find_bodies(mdp.GO2ARM_FOOT_BODY_NAMES, preserve_order=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Go2Arm whole-body contact requires the robot asset to expose "
                "FL_foot/FR_foot/RL_foot/RR_foot as articulation bodies. "
                "The current asset does not provide that layout."
            ) from exc
        if len(foot_body_ids) != 4:
            raise RuntimeError(
                f"Go2Arm whole-body contact expected 4 articulation foot bodies, but found {len(foot_body_ids)}."
            )
        if "contact_forces" not in self.scene.sensors:
            raise RuntimeError(
                "Go2Arm whole-body contact requires the shared `contact_forces` sensor, "
                "but it is missing from the scene."
            )
        contact_sensor = self.scene.sensors["contact_forces"]
        try:
            contact_body_ids, _ = contact_sensor.find_bodies(mdp.GO2ARM_FOOT_BODY_NAMES, preserve_order=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "The shared `contact_forces` sensor does not expose FL_foot/FR_foot/RL_foot/RR_foot."
            ) from exc
        if len(contact_body_ids) != 4:
            raise RuntimeError(
                "The shared `contact_forces` sensor must include all four feet for go2arm contact semantics."
            )
        self._go2arm_has_shared_contact_feet = True
        self._go2arm_shared_contact_sensor = contact_sensor
        self._go2arm_shared_contact_foot_body_ids = tuple(int(body_id) for body_id in contact_body_ids)

    def _as_log_tensor(self, value: float | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.device, dtype=torch.float32)
        return torch.tensor(float(value), device=self.device, dtype=torch.float32)

    def _accumulate_scalar_log(
        self, episode_dict: dict[str, float | torch.Tensor], key: str, value: float | torch.Tensor, count: float = 1.0
    ) -> None:
        episode_dict[key] = self._as_log_tensor(value)
        self._reward_log_sums[key] = self._reward_log_sums.get(key, self._as_log_tensor(0.0)) + self._as_log_tensor(
            value
        )
        self._reward_log_counts[key] = self._reward_log_counts.get(key, self._as_log_tensor(0.0)) + self._as_log_tensor(
            count
        )

    def _accumulate_log_only(self, key: str, value: float, count: float = 1.0) -> None:
        self._reward_log_sums[key] = self._reward_log_sums.get(key, self._as_log_tensor(0.0)) + self._as_log_tensor(
            value
        )
        self._reward_log_counts[key] = self._reward_log_counts.get(key, self._as_log_tensor(0.0)) + self._as_log_tensor(
            count
        )

    def _accumulate_tensor_mean_log(
        self,
        episode_dict: dict[str, float | torch.Tensor],
        key: str,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        write_episode: bool = True,
    ) -> None:
        valid_values = values[mask] if mask is not None else values.reshape(-1)
        valid_count = int(valid_values.numel())
        if valid_count > 0:
            sum_value = valid_values.sum()
            mean_value = sum_value / float(valid_count)
        else:
            sum_value = self._as_log_tensor(0.0)
            mean_value = self._as_log_tensor(0.0)
        if write_episode:
            episode_dict[key] = mean_value
        self._accumulate_log_only(key, sum_value, count=float(valid_count))

    def _log_roboduet_reward_terms(
        self,
        episode_dict: dict[str, float | torch.Tensor],
        done_mask: torch.Tensor,
        write_episode: bool = True,
    ) -> None:
        if not torch.any(done_mask):
            return
        log_episode_sums = getattr(self, "_roboduet_log_episode_sums", None)
        if not isinstance(log_episode_sums, dict):
            return
        done_ids = torch.where(done_mask)[0]
        reward_names = tuple(getattr(self, "_roboduet_reward_term_names", ()))
        for name in reward_names + ("total",):
            if name not in log_episode_sums:
                continue
            values = log_episode_sums[name][done_ids]
            if write_episode:
                self._accumulate_tensor_mean_log(episode_dict, f"rew_{name}", values, write_episode=True)
            log_episode_sums[name][done_ids] = 0.0
        command_sums = getattr(self, "_roboduet_command_sums", None)
        if isinstance(command_sums, dict):
            for values in command_sums.values():
                values[done_ids] = 0.0

    def _command_term(self, command_name: str | None = None):
        if command_name is not None:
            return self.command_manager.get_term(command_name)
        for candidate in ("roboduet", "ee_pose"):
            try:
                return self.command_manager.get_term(candidate)
            except Exception:  # noqa: BLE001
                continue
        raise KeyError("No supported go2arm command term found. Expected 'roboduet' or 'ee_pose'.")

    def _go2arm_fixed_arm_until_iteration(self) -> int | None:
        action_term = None
        if hasattr(self, "action_manager"):
            try:
                action_term = self.action_manager.get_term("joint_pos")
            except KeyError:
                action_term = None
        fixed_until_iteration = getattr(getattr(action_term, "cfg", None), "fixed_delta_action_until_iteration", None)
        if fixed_until_iteration is not None:
            return int(fixed_until_iteration)
        joint_pos_cfg = getattr(getattr(self.cfg, "actions", None), "joint_pos", None)
        if joint_pos_cfg is None:
            return None
        fixed_until_iteration = getattr(joint_pos_cfg, "fixed_delta_action_until_iteration", None)
        if fixed_until_iteration is None:
            return None
        return int(fixed_until_iteration)

    def _go2arm_should_keep_arm_fixed(self) -> bool:
        fixed_until_iteration = self._go2arm_fixed_arm_until_iteration()
        if fixed_until_iteration is None or fixed_until_iteration <= 0:
            return False

        action_term = None
        if hasattr(self, "action_manager"):
            try:
                action_term = self.action_manager.get_term("joint_pos")
            except KeyError:
                action_term = None
        steps_per_iteration = int(
            getattr(
                getattr(action_term, "cfg", None),
                "fixed_delta_action_steps_per_iteration",
                getattr(getattr(self.cfg.actions, "joint_pos", None), "fixed_delta_action_steps_per_iteration", 24),
            )
        )
        current_step = int(getattr(self, "common_step_counter", 0))
        current_iteration = float(current_step) / float(max(steps_per_iteration, 1))
        return current_iteration < float(fixed_until_iteration)

    def _keep_go2arm_arm_fixed(self) -> None:
        fixed_joint_ids = tuple(int(joint_id) for joint_id in self._go2arm_arm_joint_ids) + tuple(self._go2arm_hold_joint_ids)
        if len(fixed_joint_ids) == 0:
            return
        robot = self.scene["robot"]
        joint_ids = torch.as_tensor(fixed_joint_ids, dtype=torch.long, device=self.device)
        default_pos = robot.data.default_joint_pos[:, joint_ids]
        zero_vel = torch.zeros_like(default_pos)
        robot.write_joint_state_to_sim(default_pos, zero_vel, joint_ids=joint_ids)

    def _classify_episode_bucket(
        self, sampled_target_pos_b: torch.Tensor, target_pos_w: torch.Tensor, command_cfg
    ) -> str:
        x_tag = "x_near" if float(sampled_target_pos_b[0]) <= 0.5 else "x_far"
        z_world = float(target_pos_w[2])
        low_range = getattr(command_cfg, "secondary_world_z_range", None)
        high_range = getattr(command_cfg, "tertiary_world_z_range", None)
        # After the final stage, the curriculum disables explicit low/high secondary
        # samplers and uses one full z range. Keep episode length buckets comparable.
        if low_range is None or high_range is None:
            world_z_range = getattr(command_cfg, "world_z_range", None)
            if world_z_range is not None:
                world_z_min, world_z_max = float(world_z_range[0]), float(world_z_range[1])
                if low_range is None and world_z_min < 0.40:
                    low_range = (world_z_min, min(0.40, world_z_max))
                if high_range is None and world_z_max > 0.80:
                    high_range = (max(0.80, world_z_min), world_z_max)
        z_tag = "z_normal"
        if low_range is not None and float(low_range[0]) <= z_world <= float(low_range[1]):
            z_tag = "z_low_hard"
        elif high_range is not None and float(high_range[0]) <= z_world <= float(high_range[1]):
            z_tag = "z_high_hard"
        return f"{x_tag}/{z_tag}"

    def _accumulate_done_episode_stats(
        self,
        episode_dict: dict[str, float | torch.Tensor],
        done_mask: torch.Tensor,
        prev_episode_length_buf: torch.Tensor,
        prev_sampled_target_pos_b: torch.Tensor,
        prev_target_pos_w: torch.Tensor,
        terminal_tracking_errors: dict[str, torch.Tensor] | None,
    ) -> None:
        if not torch.any(done_mask):
            return

        done_ids = torch.where(done_mask)[0]
        done_episode_lengths = prev_episode_length_buf[done_ids].to(torch.float32) + 1.0
        if terminal_tracking_errors is not None:
            for term_name, term_value in terminal_tracking_errors.items():
                self._accumulate_tensor_mean_log(
                    episode_dict,
                    f"R/tracking/terminal_{term_name}",
                    term_value[done_ids],
                    write_episode=True,
                )
        self._accumulate_log_only(
            "R/misc/terminal_episode_length",
            done_episode_lengths.sum().item(),
            count=float(done_episode_lengths.numel()),
        )
        episode_dict["R/misc/terminal_episode_length"] = done_episode_lengths.sum() / float(
            done_episode_lengths.numel()
        )

        if not hasattr(self.cfg.commands, "ee_pose"):
            return
        command_cfg = self.cfg.commands.ee_pose
        bucket_to_lengths: dict[str, list[float]] = {}
        for local_idx, env_id in enumerate(done_ids.tolist()):
            bucket = self._classify_episode_bucket(
                prev_sampled_target_pos_b[env_id],
                prev_target_pos_w[env_id],
                command_cfg,
            )
            bucket_to_lengths.setdefault(bucket, []).append(float(done_episode_lengths[local_idx].item()))

        for bucket, lengths in bucket_to_lengths.items():
            bucket_sum = float(sum(lengths))
            bucket_count = float(len(lengths))
            bucket_key = f"Len/{bucket}"
            episode_dict[bucket_key] = self._as_log_tensor(bucket_sum / bucket_count)
            self._accumulate_log_only(bucket_key, bucket_sum, count=bucket_count)

    def _filter_episode_log_dict(self, log_dict: dict[str, float | torch.Tensor]) -> dict[str, float | torch.Tensor]:
        if not log_dict or not self._episode_log_key_prefixes:
            return log_dict
        return {
            key: value
            for key, value in log_dict.items()
            if any(key.startswith(prefix) for prefix in self._episode_log_key_prefixes)
        }

    def _merge_reset_logs_into_episode(self, extras: dict, episode_dict: dict[str, float | torch.Tensor]) -> None:
        log_dict = extras.get("log")
        if not isinstance(log_dict, dict):
            return
        filtered_log_dict = dict(log_dict)
        roboduet_reward_names = tuple(getattr(self, "_roboduet_reward_term_names", ()))
        if roboduet_reward_names:
            blocked_reward_keys = {f"Episode_Reward/{name}" for name in roboduet_reward_names}
            blocked_reward_keys.add("Episode_Reward/total_reward")
            inv_step_dt = 1.0 / float(self.step_dt)
            for reward_name in roboduet_reward_names:
                reward_key = f"Episode_Reward/{reward_name}"
                if reward_key in log_dict:
                    filtered_log_dict[f"rew_{reward_name}"] = self._as_log_tensor(log_dict[reward_key]) * inv_step_dt
            total_reward_key = "Episode_Reward/total_reward"
            if total_reward_key in log_dict:
                filtered_log_dict["rew_total"] = self._as_log_tensor(log_dict[total_reward_key]) * inv_step_dt
            filtered_log_dict = {
                key: value for key, value in filtered_log_dict.items() if key not in blocked_reward_keys
            }
        filtered_log_dict = self._filter_episode_log_dict(filtered_log_dict)
        if filtered_log_dict:
            episode_dict.update(filtered_log_dict)
        extras["log"] = filtered_log_dict

    def _log_final_termination_terms(
        self,
        episode_dict: dict[str, float | torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Log only the terminal causes on envs that ended this step."""
        done_mask = terminated | truncated
        if not torch.any(done_mask):
            return

        done_count = int(done_mask.sum().item())
        self._accumulate_log_only(
            "Term/done_count",
            float(done_count),
            count=1.0,
        )
        episode_dict["Term/done_count"] = self._as_log_tensor(float(done_count))
        self._accumulate_log_only(
            "Term/terminated",
            terminated[done_mask].float().mean().item(),
            count=1.0,
        )
        episode_dict["Term/terminated"] = terminated[done_mask].float().mean()

        for term_name in self.termination_manager.active_terms:
            term_value = self.termination_manager.get_term(term_name).float()
            self._accumulate_log_only(
                f"Term/{term_name}",
                term_value[done_mask].mean().item(),
                count=1.0,
            )
            episode_dict[f"Term/{term_name}"] = term_value[done_mask].mean()

    def _collect_termination_reasons(self, terminated: torch.Tensor, truncated: torch.Tensor) -> list[dict]:
        """Collect per-env terminal causes for play-time diagnostics."""
        done_mask = terminated | truncated
        if not torch.any(done_mask):
            return []

        term_values = {
            term_name: self.termination_manager.get_term(term_name).detach().to(device="cpu", dtype=torch.bool)
            for term_name in self.termination_manager.active_terms
        }
        terminated_cpu = terminated.detach().to(device="cpu", dtype=torch.bool)
        truncated_cpu = truncated.detach().to(device="cpu", dtype=torch.bool)

        reasons = []
        for env_id in torch.where(done_mask.detach().to(device="cpu"))[0].tolist():
            active_reasons = [term_name for term_name, term_value in term_values.items() if bool(term_value[env_id])]
            if not active_reasons:
                if bool(truncated_cpu[env_id]):
                    active_reasons.append("truncated")
                elif bool(terminated_cpu[env_id]):
                    active_reasons.append("terminated")
            reasons.append(
                {
                    "env_id": int(env_id),
                    "terminated": bool(terminated_cpu[env_id]),
                    "truncated": bool(truncated_cpu[env_id]),
                    "reasons": active_reasons,
                }
            )
        return reasons

    def step(self, action: torch.Tensor):
        profile_enabled = bool(self._roboduet_step_profile_enabled)
        if profile_enabled:
            step_start = self._roboduet_profile_stamp()
            section_start = step_start

        self._roboduet_reward_step_cache = None
        self.action_manager.prev_prev_action = self.action_manager.prev_action.clone()
        prev_episode_length_buf = self.episode_length_buf.clone()
        command_term = self._command_term()
        prev_sampled_target_pos_b = None
        prev_target_pos_w = None
        if hasattr(command_term, "sampled_target_pos_b") and hasattr(command_term, "target_pos_w"):
            prev_sampled_target_pos_b = getattr(command_term, "sampled_target_pos_b").clone()
            prev_target_pos_w = getattr(command_term, "target_pos_w").clone()

        if self._debug_zero_action:
            action = torch.zeros_like(action)

        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("action", section_end - section_start)
            section_start = section_end
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            if self._go2arm_should_keep_arm_fixed():
                self._keep_go2arm_arm_fixed()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("sim", section_end - section_start)
            section_start = section_end

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.command_manager.compute(dt=self.step_dt)
        motor_rand_ids = torch.where(
            self.episode_length_buf % self._roboduet_motor_randomization_interval_steps == 0
        )[0]
        self._randomize_roboduet_motor_props(motor_rand_ids)
        self._update_roboduet_gravity_randomization()
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("command", section_end - section_start)
            section_start = section_end
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("reward_termination", section_end - section_start)
            section_start = section_end

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("observation_recorder", section_end - section_start)
            section_start = section_end

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("reset", section_end - section_start)
            section_start = section_end

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("events", section_end - section_start)
            section_start = section_end
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("observation_final", section_end - section_start)
            section_start = section_end

        obs = self.obs_buf
        rew = self.reward_buf
        terminated = self.reset_terminated
        truncated = self.reset_time_outs
        extras = self.extras
        self._go2arm_last_last_joint_pos_target.copy_(self._go2arm_last_joint_pos_target)
        self._go2arm_last_joint_pos_target.copy_(self._go2arm_joint_pos_target)
        self.last_plan_actions.copy_(self.plan_actions)
        done_mask = terminated | truncated
        if torch.any(done_mask):
            self.action_manager.prev_prev_action[done_mask] = 0.0
            self.plan_actions[done_mask] = 0.0
            self.last_plan_actions[done_mask] = 0.0
            self._go2arm_joint_pos_target[done_mask] = 0.0
            self._go2arm_last_joint_pos_target[done_mask] = 0.0
            self._go2arm_last_last_joint_pos_target[done_mask] = 0.0
        if self._enable_play_termination_reason_logging and torch.any(done_mask):
            extras["go2arm_termination_reasons"] = self._collect_termination_reasons(
                terminated=terminated, truncated=truncated
            )

        episode_dict: dict[str, float | torch.Tensor] = extras.setdefault("episode", {})
        self._merge_reset_logs_into_episode(extras, episode_dict)

        reward_term_name = "total_reward"
        terminal_tracking_errors = None
        if reward_term_name in self.reward_manager.active_terms and hasattr(self, "_roboduet_reward_term_names"):
            self._log_roboduet_reward_terms(episode_dict, done_mask, write_episode=False)

        if prev_sampled_target_pos_b is not None and prev_target_pos_w is not None and hasattr(self.cfg.commands, "ee_pose"):
            self._accumulate_done_episode_stats(
                episode_dict,
                done_mask=done_mask,
                prev_episode_length_buf=prev_episode_length_buf,
                prev_sampled_target_pos_b=prev_sampled_target_pos_b,
                prev_target_pos_w=prev_target_pos_w,
                terminal_tracking_errors=terminal_tracking_errors,
            )
        self._log_final_termination_terms(episode_dict, terminated=terminated, truncated=truncated)

        next_reward_log_counter = self._reward_log_counter + 1
        should_emit_log = next_reward_log_counter >= self._reward_log_interval

        self._reward_log_counter = next_reward_log_counter

        if should_emit_log:
            log_dict = extras.setdefault("log", {})
            for name, value_sum in self._reward_log_sums.items():
                count = self._reward_log_counts.get(name, self._as_log_tensor(0.0))
                log_dict[name] = (value_sum / count).item() if float(count.item()) > 0.0 else 0.0
            extras["log"] = self._filter_episode_log_dict(log_dict)
            self._reward_log_sums.clear()
            self._reward_log_counts.clear()
            self._reward_log_counter = 0

        extras["episode"] = self._filter_episode_log_dict(episode_dict)
        if profile_enabled:
            section_end = self._roboduet_profile_stamp()
            self._roboduet_profile_add("bookkeeping", section_end - section_start)
            self._roboduet_profile_add("total", section_end - step_start)
            self._roboduet_step_profile_count += 1

        return obs, rew, terminated, truncated, extras
