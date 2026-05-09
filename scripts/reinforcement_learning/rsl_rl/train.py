# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import re
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--low_vram_num_envs",
    type=int,
    default=None,
    help=(
        "Auto-cap num_envs to this value when GPU memory is <= 4.5 GiB and --num_envs is not provided. "
        "Disabled by default to keep RoboDuet auto_train semantics explicit."
    ),
)
parser.add_argument(
    "--roboduet_alignment_check",
    action="store_true",
    default=False,
    help="Create the RoboDuet environment and check key effective training semantics, then exit without training.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# reduce CUDA memory fragmentation unless user already set it
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip
from robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.callable_resolver import (
    resolve_callable,
)

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _reset_go2arm_arm_std(policy, arm_action_indices: tuple[int, ...], arm_std: float) -> bool:
    """Reset Go2Arm arm action std for supported RSL-RL policy variants."""
    index = torch.as_tensor(arm_action_indices, dtype=torch.long)
    reset_done = False

    std_param = getattr(policy, "std", None)
    if isinstance(std_param, torch.nn.Parameter) and std_param.ndim == 1:
        index_device = index.to(std_param.device)
        with torch.no_grad():
            std_param.data[index_device] = float(arm_std)
        reset_done = True

    distribution = getattr(policy, "distribution", None)
    dist_std_param = getattr(distribution, "std_param", None)
    if isinstance(dist_std_param, torch.nn.Parameter) and dist_std_param.ndim == 1:
        index_device = index.to(dist_std_param.device)
        with torch.no_grad():
            dist_std_param.data[index_device] = float(arm_std)
        reset_done = True

    dist_log_std_param = getattr(distribution, "log_std_param", None)
    if isinstance(dist_log_std_param, torch.nn.Parameter) and dist_log_std_param.ndim == 1:
        index_device = index.to(dist_log_std_param.device)
        log_arm_std = torch.log(torch.tensor(float(arm_std), device=dist_log_std_param.device))
        with torch.no_grad():
            dist_log_std_param.data[index_device] = log_arm_std
        reset_done = True

    return reset_done


def _get_go2arm_policy(runner):
    """Return the policy module across RSL-RL versions."""
    get_policy = getattr(runner.alg, "get_policy", None)
    if callable(get_policy):
        return get_policy()
    for owner in (runner.alg, runner):
        for attr_name in ("actor_critic", "policy"):
            policy = getattr(owner, attr_name, None)
            if policy is not None:
                return policy
    return None


def _install_go2arm_mani_phase_reset_hook(runner, agent_cfg) -> None:
    """Install a robot_lab-only hook that resets arm std and PPO lr at a configured iteration."""
    reset_iteration = getattr(agent_cfg, "go2arm_mani_phase_reset_iteration", None)
    if reset_iteration is None:
        return

    arm_std = float(getattr(agent_cfg, "go2arm_mani_phase_reset_arm_std", 0.4))
    learning_rate = float(getattr(agent_cfg, "go2arm_mani_phase_reset_learning_rate", 3.0e-4))
    arm_action_indices = tuple(getattr(agent_cfg, "go2arm_mani_phase_reset_arm_action_indices", range(12, 18)))
    num_steps_per_env = int(getattr(agent_cfg, "num_steps_per_env", runner.cfg.get("num_steps_per_env", 1)))
    start_iteration = int(getattr(runner, "current_learning_iteration", 0))

    original_act = runner.alg.act
    hook_state = {"step_calls": 0, "triggered": False}

    def act_with_go2arm_mani_reset(*args, **kwargs):
        rollout_iteration = start_iteration + hook_state["step_calls"] // max(num_steps_per_env, 1)
        step_in_iteration = hook_state["step_calls"] % max(num_steps_per_env, 1)
        if (not hook_state["triggered"]) and rollout_iteration == int(reset_iteration) and step_in_iteration == 0:
            policy = _get_go2arm_policy(runner)
            std_reset = False
            if policy is not None:
                std_reset = _reset_go2arm_arm_std(policy, arm_action_indices=arm_action_indices, arm_std=arm_std)
            runner.alg.learning_rate = learning_rate
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = learning_rate
            hook_state["triggered"] = True
            print(
                "[INFO] Go2Arm mani phase reset at iteration "
                f"{rollout_iteration}: arm_std={arm_std}, lr={learning_rate}, std_reset={std_reset}"
            )
        hook_state["step_calls"] += 1
        return original_act(*args, **kwargs)

    runner.alg.act = act_with_go2arm_mani_reset


def _sync_resume_iteration_to_env(runner, env, agent_cfg) -> None:
    """Sync RSL-RL resume iteration into env counters used by Go2Arm curriculum/action masking."""
    raw_env = getattr(env, "unwrapped", env)
    if not hasattr(raw_env, "common_step_counter"):
        return

    current_iteration = int(getattr(runner, "current_learning_iteration", 0))
    if current_iteration <= 0:
        return

    runner_cfg = getattr(runner, "cfg", {})
    num_steps_per_env = int(getattr(agent_cfg, "num_steps_per_env", runner_cfg.get("num_steps_per_env", 1)))
    resume_step = current_iteration * max(num_steps_per_env, 1)
    previous_step = int(getattr(raw_env, "common_step_counter", 0))
    raw_env.common_step_counter = max(previous_step, resume_step)

    alg = getattr(runner, "alg", None)
    action_mask_until_iteration = getattr(alg, "action_mask_until_iteration", None)
    if action_mask_until_iteration is not None and current_iteration >= int(action_mask_until_iteration):
        alg.action_mask = None
        alg.action_mask_until_iteration = 0
        print(f"[INFO] Disabled completed PPO action mask at resumed iteration {current_iteration}.")

    action_term = None
    if hasattr(raw_env, "action_manager"):
        try:
            action_term = raw_env.action_manager.get_term("joint_pos")
        except KeyError:
            action_term = None
    fixed_until_iteration = getattr(getattr(action_term, "cfg", None), "fixed_delta_action_until_iteration", None)
    if fixed_until_iteration is not None and current_iteration >= int(fixed_until_iteration):
        action_term.cfg.fixed_delta_action_until_iteration = 0
        if hasattr(raw_env.cfg, "actions") and hasattr(raw_env.cfg.actions, "joint_pos"):
            raw_env.cfg.actions.joint_pos.fixed_delta_action_until_iteration = 0
        print(f"[INFO] Disabled completed env fixed arm-action mask at resumed iteration {current_iteration}.")

    # The wrapper reset recomputes curriculum terms before command/event reset, so commands and
    # reset ranges immediately match the resumed training stage instead of the fresh-env stage.
    env.reset()
    if hasattr(raw_env, "command_manager"):
        try:
            ee_pose_cfg = raw_env.command_manager.get_term("ee_pose").cfg
        except KeyError:
            ee_pose_cfg = None
        if ee_pose_cfg is not None:
            print(
                "[INFO] Go2Arm resumed ee_pose curriculum: "
                f"world_z_range={ee_pose_cfg.world_z_range}, "
                f"secondary_world_z_range={ee_pose_cfg.secondary_world_z_range}, "
                f"secondary_sample_prob={ee_pose_cfg.secondary_sample_prob}, "
                f"tertiary_world_z_range={ee_pose_cfg.tertiary_world_z_range}, "
                f"tertiary_sample_prob={ee_pose_cfg.tertiary_sample_prob}"
            )
    print(
        "[INFO] Synced env common_step_counter for resume: "
        f"iteration={current_iteration}, step={raw_env.common_step_counter}"
    )


def _run_roboduet_alignment_check(env, agent_cfg) -> None:
    """Check RoboDuet effective training semantics and exit before training."""
    raw_env = env.unwrapped
    obs = env.get_observations()
    checks: list[tuple[str, object, object, bool]] = []

    def add_check(name: str, actual: object, expected: object, ok: bool | None = None) -> None:
        if ok is None:
            ok = actual == expected
        checks.append((name, actual, expected, bool(ok)))

    def add_close_check(name: str, actual: float, expected: float, tol: float = 1.0e-6) -> None:
        add_check(name, actual, expected, abs(float(actual) - float(expected)) <= tol)

    add_check("dog_policy_obs_dim", int(obs["dog_policy"].shape[-1]), 56)
    add_check("dog_privileged_obs_dim", int(obs["dog_privileged"].shape[-1]), 2)
    add_check("arm_policy_obs_dim", int(obs["arm_policy"].shape[-1]), 20)
    add_check("arm_privileged_obs_dim", int(obs["arm_privileged"].shape[-1]), 9)
    add_check("dog_action_dim", int(agent_cfg.dog_model.num_actions), 12)
    add_check("arm_action_dim", int(agent_cfg.arm_model.num_actions), 6)
    add_check("arm_plan_action_dim", int(agent_cfg.arm_model.num_plan_actions), 2)
    add_check("full_action_dim", int(env.num_actions), 18)
    add_close_check("step_dt", float(raw_env.step_dt), 0.02)
    add_check("max_episode_length_steps", int(raw_env.max_episode_length), 1000)
    add_check("num_steps_per_env", int(agent_cfg.num_steps_per_env), 24)

    command_term = raw_env.command_manager.get_term("roboduet")
    action_term = raw_env.action_manager.get_term("joint_pos")

    robot = raw_env.scene["robot"]
    all_joint_names = tuple(robot.joint_names)
    leg_joint_ids = getattr(raw_env, "_go2arm_leg_joint_ids", None)
    if leg_joint_ids is None:
        leg_joint_ids, _ = robot.find_joints([r"^(FL|FR|RL|RR)_(hip|thigh|calf)_joint$"], preserve_order=True)
    leg_joint_names = tuple(all_joint_names[int(joint_id)] for joint_id in leg_joint_ids)
    leg_joint_name_set = set(leg_joint_names)
    expected_stage1_frozen_joint_names = tuple(
        joint_name for joint_name in all_joint_names if joint_name not in leg_joint_name_set
    )

    action_joint_names = tuple(getattr(action_term, "_joint_names", ()))
    fixed_joint_ids = getattr(action_term, "_fixed_delta_action_joint_ids", None)
    if fixed_joint_ids is not None:
        if isinstance(fixed_joint_ids, torch.Tensor):
            fixed_joint_ids = fixed_joint_ids.detach().cpu().tolist()
        stage1_frozen_joint_names = tuple(action_joint_names[int(joint_id)] for joint_id in fixed_joint_ids)
    else:
        fixed_joint_name_patterns = getattr(action_term.cfg, "fixed_delta_action_joint_names", None)
        if fixed_joint_name_patterns is None:
            stage1_frozen_joint_names = ()
        else:
            stage1_frozen_joint_names = tuple(
                joint_name
                for joint_name in action_joint_names
                if any(re.fullmatch(pattern, joint_name) for pattern in fixed_joint_name_patterns)
            )
    stage1_frozen_joint_name_set = set(stage1_frozen_joint_names)
    actual_stage1_frozen_joint_names = tuple(
        joint_name for joint_name in all_joint_names if joint_name in stage1_frozen_joint_name_set
    )
    missing_stage1_frozen_non_leg_joint_names = tuple(
        joint_name for joint_name in expected_stage1_frozen_joint_names if joint_name not in stage1_frozen_joint_name_set
    )
    unexpected_stage1_frozen_joint_names = tuple(
        joint_name for joint_name in stage1_frozen_joint_names if joint_name not in expected_stage1_frozen_joint_names
    )
    duplicate_stage1_frozen_joint_names = tuple(
        joint_name for joint_name in stage1_frozen_joint_names if stage1_frozen_joint_names.count(joint_name) > 1
    )
    add_check(
        "stage1_frozen_joint_names_equal_non_leg_joint_names",
        actual_stage1_frozen_joint_names,
        expected_stage1_frozen_joint_names,
        stage1_frozen_joint_name_set == set(expected_stage1_frozen_joint_names)
        and len(stage1_frozen_joint_names) == len(stage1_frozen_joint_name_set),
    )
    add_check("stage1_unfrozen_non_leg_joint_names", missing_stage1_frozen_non_leg_joint_names, ())
    add_check("stage1_unexpected_frozen_joint_names", unexpected_stage1_frozen_joint_names, ())
    add_check("stage1_duplicate_frozen_joint_names", duplicate_stage1_frozen_joint_names, ())

    expected_switch_iteration = 0 if bool(getattr(agent_cfg, "roboduet_disable_two_stage", False)) else 10000
    explicit_switch_iteration = getattr(agent_cfg, "roboduet_stage_switch_iteration", None)
    if explicit_switch_iteration is not None:
        expected_switch_iteration = int(explicit_switch_iteration)
    elif getattr(agent_cfg, "roboduet_pretrained_dog_checkpoint", None) and getattr(
        agent_cfg, "roboduet_pretrained_arm_checkpoint", None
    ):
        expected_switch_iteration = 2000
    add_check("command_switch_iteration", int(command_term.cfg.switch_iteration), expected_switch_iteration)
    add_check(
        "action_fixed_until_iteration",
        int(action_term.cfg.fixed_delta_action_until_iteration),
        expected_switch_iteration,
    )
    expected_initial_switch_open = expected_switch_iteration <= 0
    add_check("initial_switch_open", bool(command_term.switch_open), expected_initial_switch_open)

    active_terms = tuple(raw_env.termination_manager.active_terms)
    effective_non_timeout_terms = []
    for term_name in active_terms:
        term_cfg = raw_env.termination_manager.get_term_cfg(term_name)
        if term_cfg.time_out:
            continue
        if term_name == "reverse_termination" and not command_term.switch_open:
            continue
        effective_non_timeout_terms.append(term_name)
    add_check(
        "stage1_effective_non_timeout_terminations",
        tuple(effective_non_timeout_terms),
        ("base_height_termination",),
    )

    print("\n[INFO] RoboDuet alignment check results:")
    failed = []
    for name, actual, expected, ok in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}: actual={actual!r}, expected={expected!r}")
        if not ok:
            failed.append(name)

    if failed:
        failed_list = ", ".join(failed)
        raise RuntimeError(f"RoboDuet alignment check failed: {failed_list}")
    print("[INFO] RoboDuet alignment check passed. Training was not started.\n")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if int(agent_cfg.seed) == -1:
        agent_cfg.seed = int(torch.randint(0, 10000, (1,)).item())
        print(f"[INFO] RoboDuet random seed selected: {agent_cfg.seed}")

    # auto downscale environment count for low-VRAM GPUs when user doesn't override --num_envs
    if args_cli.num_envs is None and args_cli.low_vram_num_envs is not None:
        device = env_cfg.sim.device if env_cfg.sim.device is not None else ""
        if isinstance(device, str) and "cuda" in device and torch.cuda.is_available():
            cuda_index = 0
            if ":" in device:
                try:
                    cuda_index = int(device.split(":")[-1])
                except ValueError:
                    cuda_index = 0
            total_mem_gib = torch.cuda.get_device_properties(cuda_index).total_memory / (1024**3)
            if total_mem_gib <= 4.5:
                original_num_envs = env_cfg.scene.num_envs
                env_cfg.scene.num_envs = min(env_cfg.scene.num_envs, args_cli.low_vram_num_envs)
                if env_cfg.scene.num_envs < original_num_envs:
                    logger.warning(
                        "Detected low-VRAM GPU (%.2f GiB). Auto reducing num_envs from %s to %s. "
                        "You can override with --num_envs.",
                        total_mem_gib,
                        original_num_envs,
                        env_cfg.scene.num_envs,
                    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        runner_class = resolve_callable(agent_cfg.class_name)
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        _sync_resume_iteration_to_env(runner, env, agent_cfg)

    if args_cli.roboduet_alignment_check:
        _run_roboduet_alignment_check(env, agent_cfg)
        env.close()
        return

    if agent_cfg.class_name != (
        "robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.automatic_runner:"
        "RoboDuetAutomaticRunner"
    ):
        _install_go2arm_mani_phase_reset_hook(runner, agent_cfg)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
