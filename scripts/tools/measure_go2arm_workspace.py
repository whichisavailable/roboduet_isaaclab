# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Measure Go2Arm workspace expansion from body roll/pitch.

This script reports two workspace metrics:

* physical: FK-only voxel occupancy. A voxel is reachable if any sampled arm
  joint configuration reaches it once.
* reliable: policy rollout occupancy. A voxel is reliable if the target reaches
  the requested position-error threshold in more than the configured success-rate
  threshold of trials before episode timeout.

The comparison is always between an independent fixed arm mount and the same arm
mount carried by the Go2 body under a roll/pitch grid.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import torch

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_SCRIPT_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure Go2Arm physical and reliable workspace expansion.")
    parser.add_argument("--mode", choices=("physical", "reliable", "both"), default="physical")
    parser.add_argument("--task", type=str, default="RobotLab-Isaac-Flat-Go2Arm-v0")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs. Mainly speeds up reliable rollout.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fabric", action="store_true", default=False)

    parser.add_argument("--num-arm-samples", type=int, default=1_000_000)
    parser.add_argument("--fk-batch-size", type=int, default=None)
    parser.add_argument("--base-grid", type=int, default=41)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--roll-range", type=float, nargs=2, default=(-0.4, 0.4), metavar=("MIN", "MAX"))
    parser.add_argument("--pitch-range", type=float, nargs=2, default=(-0.4, 0.4), metavar=("MIN", "MAX"))

    parser.add_argument("--checkpoint", type=str, default=None, help="Optional dog/combined checkpoint path.")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--reliable-max-targets", type=int, default=2000)
    parser.add_argument("--trials-per-target", type=int, default=10)
    parser.add_argument("--success-rate-threshold", type=float, default=0.89)
    parser.add_argument("--success-pos-threshold", type=float, default=0.05)

    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli, hydra_args = _build_parser().parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Imports that require Isaac Sim to be launched."""

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402

import cli_args  # noqa: E402
import robot_lab.tasks  # noqa: F401,E402
from robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.callable_resolver import (  # noqa: E402
    resolve_callable,
)


GO2ARM_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GO2ARM_EE_BODY_NAME = "link6"
GO2ARM_MOUNT_OFFSET_B = (-0.01, 0.0, 0.085)
GO2ARM_DEFAULT_BASE_HEIGHT = 0.34
GO2ARM_DEFAULT_TRUNK_REF_HEIGHT = 0.38


def _go2arm_urdf_path() -> Path:
    return (
        REPO_ROOT
        / "source"
        / "robot_lab"
        / "data"
        / "Robots"
        / "unitree"
        / "go2arm_description"
        / "urdf"
        / "go2_piper_description_mjc_NoGripper.urdf"
    )


def _read_arm_joint_limits() -> torch.Tensor:
    root = ET.parse(_go2arm_urdf_path()).getroot()
    limits: list[tuple[float, float]] = []
    joints_by_name = {joint.attrib.get("name"): joint for joint in root.findall("joint")}
    for joint_name in GO2ARM_ARM_JOINT_NAMES:
        joint = joints_by_name.get(joint_name)
        if joint is None:
            raise RuntimeError(f"Could not find {joint_name!r} in {_go2arm_urdf_path()}.")
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise RuntimeError(f"Joint {joint_name!r} has no lower/upper limit in {_go2arm_urdf_path()}.")
        limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
    return torch.tensor(limits, dtype=torch.float32)


def _configure_eval_env(env_cfg, agent_cfg=None, *, num_envs: int) -> None:
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if getattr(env_cfg.observations, "dog_policy", None) is not None:
        env_cfg.observations.dog_policy.enable_corruption = False
    if getattr(env_cfg.observations, "dog_privileged", None) is not None:
        env_cfg.observations.dog_privileged.enable_corruption = False
    if getattr(env_cfg.observations, "arm_policy", None) is not None:
        env_cfg.observations.arm_policy.enable_corruption = False
    if getattr(env_cfg.observations, "arm_privileged", None) is not None:
        env_cfg.observations.arm_privileged.enable_corruption = False

    for event_name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_ee",
        "randomize_apply_external_force_torque_base",
        "randomize_apply_external_force_torque_ee",
        "randomize_push_robot",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    if getattr(env_cfg.events, "randomize_reset_joints", None) is not None:
        env_cfg.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
    if getattr(env_cfg.events, "randomize_reset_base", None) is not None:
        env_cfg.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    roboduet_cfg = getattr(getattr(env_cfg, "commands", None), "roboduet", None)
    if roboduet_cfg is not None:
        roboduet_cfg.switch_iteration = 0
        roboduet_cfg.fixed_play_dog_command = (0.0, 0.0, 0.0)
        roboduet_cfg.disable_play_resampling = True
        roboduet_cfg.disable_play_arm_resampling = True
        roboduet_cfg.resampling_time_s = 1.0e9
        roboduet_cfg.resampling_time_range = (1.0e9, 1.0e9)
    if hasattr(env_cfg.actions, "joint_pos"):
        env_cfg.actions.joint_pos.fixed_delta_action_until_iteration = 0
    if hasattr(env_cfg, "roboduet_randomize_gravity"):
        env_cfg.roboduet_randomize_gravity = False
    if hasattr(env_cfg, "roboduet_randomize_motor_strength"):
        env_cfg.roboduet_randomize_motor_strength = False
    if hasattr(env_cfg, "roboduet_randomize_motor_offset"):
        env_cfg.roboduet_randomize_motor_offset = False

    if agent_cfg is not None:
        agent_cfg.seed = int(args_cli.seed)
        if hasattr(agent_cfg, "roboduet_disable_two_stage"):
            agent_cfg.roboduet_disable_two_stage = True
        if hasattr(agent_cfg, "roboduet_stage_switch_iteration"):
            agent_cfg.roboduet_stage_switch_iteration = 0


def _make_env(num_envs: int):
    print(f"[INFO] Creating physical FK env: task={args_cli.task}, num_envs={num_envs}", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_eval_env(env_cfg, num_envs=num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env.reset()
    print("[INFO] Physical FK env ready.", flush=True)
    return env


def _make_agent_env(num_envs: int):
    print(f"[INFO] Creating reliable rollout env: task={args_cli.task}, num_envs={num_envs}", flush=True)
    rsl_args = argparse.Namespace(
        seed=args_cli.seed,
        resume=False,
        load_run=args_cli.load_run,
        checkpoint=args_cli.checkpoint,
        run_name=None,
        logger=None,
        log_project_name=None,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, rsl_args)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_eval_env(env_cfg, agent_cfg, num_envs=num_envs)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Resolving Go2Arm checkpoint under: {log_root_path}", flush=True)
    checkpoint_path = _resolve_go2arm_checkpoint(log_root_path, agent_cfg)
    log_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(log_dir) in {"checkpoints_dog", "checkpoints_arm"}:
        log_dir = os.path.dirname(log_dir)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner_class = resolve_callable(agent_cfg.class_name)
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading Go2Arm checkpoint: {checkpoint_path}", flush=True)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("[INFO] Reliable rollout env and policy ready.", flush=True)
    return env, policy


def _resolve_go2arm_checkpoint(log_root_path: str, agent_cfg) -> str:
    if args_cli.checkpoint:
        requested = retrieve_file_path(args_cli.checkpoint)
        return _dog_checkpoint_for_runner_load(requested)

    arm_checkpoint = get_checkpoint_path(
        log_root_path,
        agent_cfg.load_run,
        "ac_weights_last_arm.pt",
        other_dirs=["checkpoints_arm"],
    )
    return _dog_checkpoint_for_runner_load(arm_checkpoint)


def _dog_checkpoint_for_runner_load(path: str) -> str:
    path_obj = Path(path)
    if path_obj.parent.name != "checkpoints_arm":
        return str(path_obj)
    dog_dir = path_obj.parent.parent / "checkpoints_dog"
    dog_name = path_obj.name.replace("_arm", "_dog")
    dog_path = dog_dir / dog_name
    if not dog_path.exists():
        raise FileNotFoundError(
            "RoboDuet runner loads a dog checkpoint and then its companion arm checkpoint. "
            f"Resolved arm checkpoint {path_obj}, but companion dog checkpoint is missing: {dog_path}"
        )
    return str(dog_path)


def _get_robot(env):
    return env.unwrapped.scene["robot"]


def _env_origins(env) -> torch.Tensor:
    return env.unwrapped.scene.env_origins


def _arm_joint_ids(robot) -> list[int]:
    return [robot.joint_names.index(name) for name in GO2ARM_ARM_JOINT_NAMES]


def _ee_body_id(robot) -> int:
    return robot.body_names.index(GO2ARM_EE_BODY_NAME)


def _sim_forward(env) -> None:
    raw_env = env.unwrapped
    raw_env.scene.write_data_to_sim()
    raw_env.sim.forward()
    raw_env.scene.update(0.0)


def _sample_arm_joint_positions(num_samples: int, limits: torch.Tensor, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args_cli.seed))
    low = limits[:, 0].to(device)
    high = limits[:, 1].to(device)
    return low + (high - low) * torch.rand((num_samples, limits.shape[0]), generator=generator, device=device)


def _collect_mount_local_points(env, num_samples: int) -> torch.Tensor:
    robot = _get_robot(env)
    device = robot.device
    num_envs = env.unwrapped.num_envs
    batch_size = int(args_cli.fk_batch_size or num_envs)
    if batch_size > num_envs:
        raise ValueError(f"--fk-batch-size ({batch_size}) cannot exceed env num_envs ({num_envs}).")

    limits = _read_arm_joint_limits()
    print(
        f"[INFO] Sampling arm FK points: samples={num_samples}, batch_size={batch_size}, device={device}",
        flush=True,
    )
    sampled_arm_q = _sample_arm_joint_positions(num_samples, limits, device)
    arm_ids = _arm_joint_ids(robot)
    ee_id = _ee_body_id(robot)
    mount_offset = torch.tensor(GO2ARM_MOUNT_OFFSET_B, device=device, dtype=torch.float32).unsqueeze(0)

    all_points = []
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, :3] += _env_origins(env)
    zero_vel = torch.zeros_like(robot.data.joint_vel)
    env_ids_full = torch.arange(num_envs, device=device)

    with torch.inference_mode():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            n = end - start
            env_ids = env_ids_full[:n]
            root_pose = default_root_state[:n, :7].clone()
            joint_pos = robot.data.default_joint_pos[:n].clone()
            joint_pos[:, arm_ids] = sampled_arm_q[start:end]
            robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:n, 7:13] * 0.0, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, zero_vel[:n], env_ids=env_ids)
            _sim_forward(env)

            ee_pos_w = robot.data.body_pos_w[:n, ee_id]
            root_pos_w = robot.data.root_pos_w[:n]
            root_quat_w = robot.data.root_quat_w[:n]
            mount_pos_w = root_pos_w + quat_apply(root_quat_w, mount_offset.expand(n, -1))
            ee_pos_mount = quat_apply_inverse(root_quat_w, ee_pos_w - mount_pos_w)
            all_points.append(ee_pos_mount.detach().clone())
            if start == 0 or end == num_samples or (start // batch_size) % 25 == 0:
                print(f"[INFO] FK samples: {end}/{num_samples}", flush=True)

    return torch.cat(all_points, dim=0)


def _roll_pitch_values(grid_size: int, value_range: Iterable[float], *, device: torch.device) -> torch.Tensor:
    low, high = (float(v) for v in value_range)
    if grid_size <= 1:
        return torch.tensor([(low + high) * 0.5], device=device)
    return torch.linspace(low, high, grid_size, device=device)


def _voxelize_points(points: torch.Tensor, voxel_size: float) -> set[tuple[int, int, int]]:
    vox = torch.floor(points / float(voxel_size)).to(torch.int64)
    unique = torch.unique(vox, dim=0).detach().cpu().tolist()
    return {tuple(int(v) for v in row) for row in unique}


def _update_voxel_set(voxel_set: set[tuple[int, int, int]], points: torch.Tensor, voxel_size: float) -> None:
    voxel_set.update(_voxelize_points(points, voxel_size))


def _voxel_centers(voxels: set[tuple[int, int, int]], voxel_size: float, max_targets: int, seed: int) -> torch.Tensor:
    if not voxels:
        return torch.empty(0, 3, dtype=torch.float32)
    sorted_voxels = sorted(voxels)
    total = len(sorted_voxels)
    if max_targets > 0 and total > max_targets:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        indices = torch.randperm(total, generator=generator)[:max_targets].sort()[0].tolist()
        selected = [sorted_voxels[i] for i in indices]
    else:
        selected = sorted_voxels
    centers = (torch.tensor(selected, dtype=torch.float32) + 0.5) * float(voxel_size)
    return centers


def _transform_mount_points(mount_points: torch.Tensor, roll: float, pitch: float, *, device: torch.device) -> torch.Tensor:
    points = mount_points.to(device)
    n = points.shape[0]
    root_pos = torch.tensor((0.0, 0.0, GO2ARM_DEFAULT_BASE_HEIGHT), device=device).expand(n, -1)
    mount_offset = torch.tensor(GO2ARM_MOUNT_OFFSET_B, device=device).expand(n, -1)
    zero = torch.zeros(n, device=device)
    roll_t = torch.full((n,), float(roll), device=device)
    pitch_t = torch.full((n,), float(pitch), device=device)
    quat = quat_from_euler_xyz(roll_t, pitch_t, zero)
    return root_pos + quat_apply(quat, mount_offset + points)


def compute_physical_workspace(env) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]], torch.Tensor]:
    print(
        "[INFO] Starting physical workspace pass: "
        f"num_arm_samples={args_cli.num_arm_samples}, base_grid={args_cli.base_grid}, "
        f"voxel_size={args_cli.voxel_size}",
        flush=True,
    )
    mount_points = _collect_mount_local_points(env, int(args_cli.num_arm_samples))
    device = env.unwrapped.device
    voxel_size = float(args_cli.voxel_size)

    fixed_points = _transform_mount_points(mount_points, 0.0, 0.0, device=device)
    fixed_voxels = _voxelize_points(fixed_points, voxel_size)
    print(f"[INFO] Fixed mount voxelization complete: voxels={len(fixed_voxels)}", flush=True)
    expanded_voxels: set[tuple[int, int, int]] = set()

    rolls = _roll_pitch_values(args_cli.base_grid, args_cli.roll_range, device=device)
    pitches = _roll_pitch_values(args_cli.base_grid, args_cli.pitch_range, device=device)
    total = int(rolls.numel() * pitches.numel())
    done = 0
    with torch.inference_mode():
        for roll in rolls.tolist():
            for pitch in pitches.tolist():
                transformed = _transform_mount_points(mount_points, roll, pitch, device=device)
                _update_voxel_set(expanded_voxels, transformed, voxel_size)
                done += 1
                if done == 1 or done == total or done % 100 == 0:
                    print(f"[INFO] Roll/pitch grid voxelization: {done}/{total}", flush=True)
    print(f"[INFO] Expanded voxelization complete: voxels={len(expanded_voxels)}", flush=True)
    return fixed_voxels, expanded_voxels, mount_points


def _volume(voxels: set[tuple[int, int, int]]) -> float:
    return float(len(voxels)) * float(args_cli.voxel_size) ** 3


def _gain(expanded: float, fixed: float) -> float:
    return 0.0 if fixed <= 0.0 else (expanded - fixed) / fixed * 100.0


def _world_targets_to_lpy(raw_env, target_w: torch.Tensor) -> torch.Tensor:
    robot = raw_env.scene["robot"]
    base_pos = robot.data.root_pos_w[: target_w.shape[0]]
    yaw_zero_quat = quat_from_euler_xyz(
        torch.zeros(target_w.shape[0], device=target_w.device),
        torch.zeros(target_w.shape[0], device=target_w.device),
        torch.zeros(target_w.shape[0], device=target_w.device),
    )
    # Evaluation resets yaw to zero, so this is equivalent to RoboDuet's yaw-aligned frame.
    delta = quat_apply_inverse(yaw_zero_quat, target_w - base_pos)
    delta[:, 2] = target_w[:, 2] - GO2ARM_DEFAULT_TRUNK_REF_HEIGHT
    length = torch.linalg.norm(delta, dim=1)
    pitch = torch.atan2(delta[:, 2], torch.sqrt(torch.clamp(delta[:, 0] ** 2 + delta[:, 1] ** 2, min=1.0e-8)))
    yaw = torch.atan2(delta[:, 1], delta[:, 0])
    return torch.stack((length, pitch, yaw), dim=-1)


def _set_fixed_targets(raw_env, target_w: torch.Tensor) -> None:
    term = raw_env.command_manager.get_term("roboduet")
    lpy = _world_targets_to_lpy(raw_env, target_w)
    n = target_w.shape[0]
    term.switch_open = True
    term.commands_dog[:n] = 0.0
    term.commands_arm[:n] = lpy
    term.commands_arm_obs[:n] = 0.0
    term.commands_arm_obs[:n, :3] = lpy
    term.target_abg[:n] = 0.0
    term.obj_quats[:n] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=target_w.device).expand(n, -1)
    term.T_trajs[:n] = float("inf")
    term.arm_time[:n] = 0.0
    if hasattr(term, "_refresh_command_buffer"):
        term._refresh_command_buffer()


def _current_ee_pos_w(raw_env, n: int) -> torch.Tensor:
    robot = raw_env.scene["robot"]
    ee_id = robot.body_names.index(GO2ARM_EE_BODY_NAME)
    return robot.data.body_pos_w[:n, ee_id]


def _evaluate_reliable_volume(
    env,
    policy,
    voxels: set[tuple[int, int, int]],
    physical_volume: float,
    *,
    label: str,
) -> float:
    targets = _voxel_centers(voxels, float(args_cli.voxel_size), int(args_cli.reliable_max_targets), int(args_cli.seed))
    if targets.numel() == 0:
        return 0.0

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    trial_count = int(args_cli.trials_per_target)
    successes = torch.zeros(targets.shape[0], dtype=torch.int32)
    evaluated = torch.zeros(targets.shape[0], dtype=torch.int32)
    threshold = float(args_cli.success_pos_threshold)

    jobs = [(target_idx, trial_idx) for target_idx in range(targets.shape[0]) for trial_idx in range(trial_count)]
    max_steps = int(env.unwrapped.max_episode_length)
    print(
        "[INFO] Starting reliable rollout pass: "
        f"label={label}, targets={targets.shape[0]}, trials_per_target={trial_count}, "
        f"total_trials={len(jobs)}, num_envs={num_envs}, max_steps={max_steps}, "
        f"success_pos_threshold={args_cli.success_pos_threshold}, "
        f"success_rate_threshold={args_cli.success_rate_threshold}",
        flush=True,
    )

    for start in range(0, len(jobs), num_envs):
        batch_jobs = jobs[start : start + num_envs]
        n = len(batch_jobs)
        batch_target_indices = torch.tensor([job[0] for job in batch_jobs], dtype=torch.long)
        env_targets_rel = targets[batch_target_indices].to(device)

        obs, _ = env.reset()
        raw_env = env.unwrapped
        env_targets_w = env_targets_rel + raw_env.scene.env_origins[:n]
        _set_fixed_targets(raw_env, env_targets_w)
        obs = env.get_observations()

        reached = torch.zeros(n, dtype=torch.bool, device=device)
        active = torch.ones(n, dtype=torch.bool, device=device)
        if hasattr(policy, "reset"):
            policy.reset()

        with torch.inference_mode():
            for _ in range(max_steps):
                actions = policy.act_inference(obs) if hasattr(policy, "act_inference") else policy(obs)
                obs, _, dones, _ = env.step(actions)
                ee_pos = _current_ee_pos_w(raw_env, n)
                err = torch.linalg.norm(ee_pos - env_targets_w, dim=1)
                reached |= active & (err <= threshold)
                active &= ~dones[:n].to(torch.bool)
                if not active.any():
                    break

        reached_cpu = reached.detach().cpu()
        for local_i, ok in enumerate(reached_cpu.tolist()):
            target_i = int(batch_target_indices[local_i])
            evaluated[target_i] += 1
            successes[target_i] += int(ok)

        completed = min(start + len(batch_jobs), len(jobs))
        if completed == len(jobs) or completed % max(num_envs * 10, 1) == 0:
            print(f"[INFO] Reliable rollout ({label}): {completed}/{len(jobs)} trials", flush=True)

    success_rate = successes.to(torch.float32) / torch.clamp(evaluated.to(torch.float32), min=1.0)
    reliable_fraction = torch.mean((success_rate > float(args_cli.success_rate_threshold)).to(torch.float32)).item()
    return float(reliable_fraction) * float(physical_volume)


def _print_results(results: dict[str, float]) -> None:
    ordered_keys = (
        "physical_fixed_mount_volume_m3",
        "physical_expanded_volume_m3",
        "physical_gain_percent",
        "reliable_fixed_mount_volume_m3",
        "reliable_expanded_volume_m3",
        "reliable_gain_percent",
    )
    print("[GO2ARM WORKSPACE RESULTS]", flush=True)
    for key in ordered_keys:
        value = results.get(key, float("nan"))
        print(f"{key}: {value:.9g}", flush=True)


def main() -> None:
    print(
        f"[INFO] Go2Arm workspace measurement starting: mode={args_cli.mode}, "
        f"headless={getattr(args_cli, 'headless', False)}",
        flush=True,
    )
    # Reliable mode also needs physical candidates, so all modes start with the
    # same FK/voxel pass.
    physical_env = _make_env(args_cli.num_envs)
    fixed_voxels, expanded_voxels, _ = compute_physical_workspace(physical_env)
    physical_env.close()

    physical_fixed = _volume(fixed_voxels)
    physical_expanded = _volume(expanded_voxels)
    results = {
        "physical_fixed_mount_volume_m3": physical_fixed,
        "physical_expanded_volume_m3": physical_expanded,
        "physical_gain_percent": _gain(physical_expanded, physical_fixed),
        "reliable_fixed_mount_volume_m3": float("nan"),
        "reliable_expanded_volume_m3": float("nan"),
        "reliable_gain_percent": float("nan"),
    }

    if args_cli.mode in {"reliable", "both"}:
        reliable_env, policy = _make_agent_env(args_cli.num_envs)
        reliable_fixed = _evaluate_reliable_volume(reliable_env, policy, fixed_voxels, physical_fixed, label="fixed")
        reliable_expanded = _evaluate_reliable_volume(
            reliable_env, policy, expanded_voxels, physical_expanded, label="expanded"
        )
        reliable_env.close()
        results["reliable_fixed_mount_volume_m3"] = reliable_fixed
        results["reliable_expanded_volume_m3"] = reliable_expanded
        results["reliable_gain_percent"] = _gain(reliable_expanded, reliable_fixed)

    _print_results(results)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
