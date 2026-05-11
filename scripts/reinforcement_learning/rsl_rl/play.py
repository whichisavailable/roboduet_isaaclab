# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

_GO2ARM_PLAY_FIXED_COMMAND_TIME_S = 1.0e9
_GO2ARM_PLAY_STAGE1_ONLY_SWITCH_ITERATION = 1_000_000_000

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--go2arm_ee_pos",
    type=float,
    nargs=3,
    metavar=("X_B", "Y_B", "Z_W"),
    default=None,
    help="Fixed Go2Arm ee target position for play: base-frame x/y and world-frame z.",
)
parser.add_argument(
    "--go2arm_ee_rpy",
    type=float,
    nargs=3,
    metavar=("ROLL_B", "PITCH_B", "YAW_B"),
    default=None,
    help="Fixed Go2Arm ee target orientation for play in base-frame roll/pitch/yaw radians.",
)
parser.add_argument(
    "--go2arm_trace_actions",
    action="store_true",
    default=False,
    help="Print Go2Arm action and joint state diagnostics during play.",
)
parser.add_argument(
    "--go2arm_dog_cmd",
    type=float,
    nargs=3,
    metavar=("VX", "VY", "WZ"),
    default=None,
    help="Fixed RoboDuet dog command for Go2Arm play: vx, vy, yaw-rate. Randomly sampled once if omitted.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip
from robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.callable_resolver import (
    resolve_callable,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def _print_go2arm_termination_reasons(extras: dict) -> None:
    """Print per-env Go2Arm termination causes when the env exposes them."""
    termination_reasons = extras.get("go2arm_termination_reasons")
    if not termination_reasons:
        return

    for item in termination_reasons:
        env_id = item.get("env_id", "?")
        state = "truncated" if item.get("truncated") else "terminated"
        reasons = ", ".join(item.get("reasons", ())) or "unknown"
        print(f"[TERMINATION env{env_id}] {state}: {reasons}")


def _fmt_tensor(values: torch.Tensor, max_items: int = 12) -> list[float]:
    values = values.detach().flatten().cpu()
    return [round(float(x), 4) for x in values[:max_items].tolist()]


def _print_go2arm_action_state(env, policy_action: torch.Tensor, step: int, obs: dict[str, torch.Tensor] | None = None) -> None:
    """Print only direction-relevant Go2Arm diagnostics for backwards-walking debugging."""
    robot = env.unwrapped.scene["robot"]
    action_manager = env.unwrapped.action_manager
    try:
        action_term = action_manager.get_term("joint_pos")
    except KeyError:
        action_term = None

    current_action = action_manager.action[0].detach().cpu()
    policy_action = policy_action.detach().cpu()
    effective_action = getattr(env.unwrapped, "_go2arm_effective_action", None)
    if torch.is_tensor(effective_action):
        effective_action = effective_action[0].detach().cpu()
    else:
        effective_action = current_action

    root_quat_w = robot.data.root_quat_w[0].detach().cpu()
    root_lin_vel_w = getattr(robot.data, "root_lin_vel_w", None)
    root_lin_vel_b = getattr(robot.data, "root_lin_vel_b", None)
    root_ang_vel_b = getattr(robot.data, "root_ang_vel_b", None)
    if torch.is_tensor(root_lin_vel_w):
        root_lin_vel_w = root_lin_vel_w[0].detach().cpu()
    else:
        root_lin_vel_w = torch.zeros(3)
    if torch.is_tensor(root_lin_vel_b):
        root_lin_vel_b = root_lin_vel_b[0].detach().cpu()
    else:
        root_lin_vel_b = torch.zeros(3)
    if torch.is_tensor(root_ang_vel_b):
        root_ang_vel_b = root_ang_vel_b[0].detach().cpu()
    else:
        root_ang_vel_b = torch.zeros(3)

    forward_w = quat_apply(root_quat_w.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]])).squeeze(0)
    forward_xy = forward_w[:2]
    forward_xy_norm = torch.linalg.norm(forward_xy).clamp_min(1.0e-6)
    forward_xy_unit = forward_xy / forward_xy_norm
    speed_along_forward_w = torch.dot(root_lin_vel_w[:2], forward_xy_unit)
    yaw_w = torch.atan2(forward_w[1], forward_w[0])

    cmd_raw = torch.zeros(5)
    cmd_scaled = torch.zeros(5)
    switch_open = False
    try:
        command_term = env.unwrapped.command_manager.get_term("roboduet")
        switch_open = bool(command_term.switch_open)
        cmd_raw = command_term.commands_dog[0].detach().cpu()
        cmd_scaled = (command_term.commands_dog[0] * command_term.commands_scale_dog[0]).detach().cpu()
    except Exception as exc:  # noqa: BLE001
        print(f"[GO2ARM DIR step={step}] command_error={type(exc).__name__}: {exc}")

    obs_cmd = torch.full((5,), float("nan"))
    obs_pg = torch.full((3,), float("nan"))
    obs_rp = torch.full((2,), float("nan"))
    dog_obs_tensor = None
    if obs is not None:
        try:
            dog_obs_tensor = obs["dog_policy"]
        except (KeyError, TypeError, AttributeError):
            dog_obs_tensor = None
    if torch.is_tensor(dog_obs_tensor):
        dog_obs = dog_obs_tensor[0].detach().cpu()
        obs_pg = dog_obs[0:3]
        obs_cmd = dog_obs[39:44]
        obs_rp = dog_obs[50:52]

    leg_torque_target = getattr(env.unwrapped, "_go2arm_leg_torque_target", None)
    if torch.is_tensor(leg_torque_target):
        tau_norm = float(leg_torque_target[0].detach().cpu().norm().item())
    else:
        tau_norm = 0.0
    leg_position_target = getattr(action_term, "_leg_position_target", None) if action_term is not None else None
    if torch.is_tensor(leg_position_target):
        target_norm = float(leg_position_target[0].detach().cpu().norm().item())
    else:
        target_norm = 0.0

    print(
        f"[GO2ARM DIR step={step}] switch_open={switch_open} "
        f"cmd_raw={_fmt_tensor(cmd_raw, 5)} cmd_obs={_fmt_tensor(obs_cmd, 5)} "
        f"vel_b={_fmt_tensor(root_lin_vel_b, 3)} vel_w={_fmt_tensor(root_lin_vel_w, 3)} "
        f"forward_xy={_fmt_tensor(forward_xy_unit, 2)} yaw_w={float(yaw_w.item()):.4f} "
        f"v_forward_w={float(speed_along_forward_w.item()):.4f} "
        f"cmd_x={float(cmd_raw[0].item()):.4f} vx_b={float(root_lin_vel_b[0].item()):.4f} "
        f"backwards={(float(cmd_raw[0].item()) * float(root_lin_vel_b[0].item())) < -0.05}"
    )
    print(
        f"[GO2ARM DIR step={step}] obs_pg={_fmt_tensor(obs_pg, 3)} obs_rp={_fmt_tensor(obs_rp, 2)} "
        f"act_FL={_fmt_tensor(effective_action[0:3], 3)} act_FR={_fmt_tensor(effective_action[3:6], 3)} "
        f"act_RL={_fmt_tensor(effective_action[6:9], 3)} act_RR={_fmt_tensor(effective_action[9:12], 3)} "
        f"policy_norm={float(policy_action[:12].norm().item()):.4f} target_norm={target_norm:.4f} tau_norm={tau_norm:.4f}"
    )
 
 
def _sample_go2arm_dog_command_once(roboduet_cfg) -> tuple[float, float, float]:
    """Sample one non-zero dog velocity command from the configured RoboDuet ranges."""
    ranges = (roboduet_cfg.lin_vel_x, roboduet_cfg.lin_vel_y, roboduet_cfg.ang_vel_yaw)
    for _ in range(100):
        command = tuple(
            float(torch.empty((), dtype=torch.float32).uniform_(float(cmd_range[0]), float(cmd_range[1])).item())
            for cmd_range in ranges
        )
        if abs(command[0]) > 0.07 or abs(command[1]) > 0.07 or abs(command[2]) > 0.10:
            return command
    return tuple(float((cmd_range[0] + cmd_range[1]) * 0.5) for cmd_range in ranges)


def _configure_go2arm_stage1_dog_play(env_cfg, agent_cfg) -> bool:
    """Keep RoboDuet play in stage1 and use one fixed dog command for playback."""
    roboduet_cfg = getattr(getattr(env_cfg, "commands", None), "roboduet", None)
    if roboduet_cfg is None:
        return False

    if args_cli.go2arm_dog_cmd is None:
        dog_cmd = _sample_go2arm_dog_command_once(roboduet_cfg)
        dog_cmd_source = "sampled"
    else:
        dog_cmd = tuple(float(value) for value in args_cli.go2arm_dog_cmd)
        dog_cmd_source = "cli"
    fixed_time_s = float(_GO2ARM_PLAY_FIXED_COMMAND_TIME_S)
    stage1_switch_iteration = int(_GO2ARM_PLAY_STAGE1_ONLY_SWITCH_ITERATION)

    # Keep play in stage1 for dog-only validation: arm command observations stay
    # zero, the RoboDuet inference policy emits zero arm actions, and the action
    # term keeps joint1-6 deltas fixed.  The custom runner can overwrite env_cfg
    # from agent_cfg during construction, so set both configs here.
    roboduet_cfg.switch_iteration = stage1_switch_iteration
    if hasattr(agent_cfg, "roboduet_disable_two_stage"):
        agent_cfg.roboduet_disable_two_stage = False
    if hasattr(agent_cfg, "roboduet_stage_switch_iteration"):
        agent_cfg.roboduet_stage_switch_iteration = stage1_switch_iteration
    env_cfg.actions.joint_pos.fixed_delta_action_until_iteration = stage1_switch_iteration

    # Only freeze the dog command after one initial sample/CLI command.
    roboduet_cfg.fixed_play_dog_command = dog_cmd
    roboduet_cfg.disable_play_resampling = True
    roboduet_cfg.resampling_time_s = fixed_time_s
    roboduet_cfg.resampling_time_range = (fixed_time_s, fixed_time_s)

    print(
        "[INFO] Go2Arm RoboDuet stage1 dog play command: "
        f"source={dog_cmd_source}, dog(vx,vy,wz)={dog_cmd}, "
        f"resampling_time_s={fixed_time_s:g}, switch_iteration={stage1_switch_iteration}."
    )
    return True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    if "go2arm" in task_name.lower():
        env_cfg.observations.dog_policy.enable_corruption = False
        env_cfg.observations.dog_privileged.enable_corruption = False
        env_cfg.observations.arm_policy.enable_corruption = False
        env_cfg.observations.arm_privileged.enable_corruption = False
        # Match upstream `load_env()` evaluation semantics: keep reset randomization,
        # but disable domain randomization and external disturbances during play.
        env_cfg.events.randomize_rigid_body_material = None
        env_cfg.events.randomize_rigid_body_mass_base = None
        env_cfg.events.randomize_rigid_body_mass_ee = None
        env_cfg.events.randomize_apply_external_force_torque_base = None
        env_cfg.events.randomize_apply_external_force_torque_ee = None
        env_cfg.events.randomize_push_robot = None
        env_cfg.roboduet_randomize_gravity = False
        env_cfg.roboduet_randomize_motor_strength = False
        env_cfg.roboduet_randomize_motor_offset = False
        env_cfg.enable_play_termination_reason_logging = True
        # Keep arm disabled for stage1 dog-only playback.  The action term keeps
        # joint1-6 deltas fixed for the duration of playback.
        fixed_roboduet_play = _configure_go2arm_stage1_dog_play(env_cfg, agent_cfg)
        print("[INFO] Go2Arm play override: disabled RoboDuet eval-time DR/disturbances; reset randomization is kept.")
        print("[INFO] Go2Arm play override: kept stage1 arm-action freeze for dog-only playback.")
        if fixed_roboduet_play:
            print("[INFO] Go2Arm play override: fixed one dog command and disabled play-time command resampling.")
    else:
        if env_cfg.observations.policy is not None:
            env_cfg.observations.policy.enable_corruption = False
        if getattr(env_cfg.observations, "privileged", None) is not None:
            env_cfg.observations.privileged.enable_corruption = False
        env_cfg.events.randomize_apply_external_force_torque = None
        env_cfg.events.randomize_push_robot = None
        env_cfg.curriculum.command_levels_lin_vel = None
        env_cfg.curriculum.command_levels_ang_vel = None
    go2arm_fixed_target = args_cli.go2arm_ee_pos is not None or args_cli.go2arm_ee_rpy is not None
    if "go2arm" in task_name.lower() and go2arm_fixed_target:
        env_cfg.curriculum.go2arm_reaching_stages = None
        if args_cli.go2arm_ee_pos is not None:
            ee_x_b, ee_y_b, ee_z_w = args_cli.go2arm_ee_pos
            env_cfg.commands.ee_pose.position_range_b = (ee_x_b, ee_x_b, ee_y_b, ee_y_b, 0.0, 0.0)
            env_cfg.commands.ee_pose.world_z_range = (ee_z_w, ee_z_w)
        else:
            env_cfg.commands.ee_pose.position_range_b = (0.05, 2.00, -0.35, 0.35, 0.0, 0.0)
            env_cfg.commands.ee_pose.world_z_range = (0.02, 1.20)
        if args_cli.go2arm_ee_rpy is not None:
            ee_roll_b, ee_pitch_b, ee_yaw_b = args_cli.go2arm_ee_rpy
            env_cfg.commands.ee_pose.euler_xyz_range_b = (
                ee_roll_b,
                ee_roll_b,
                ee_pitch_b,
                ee_pitch_b,
                ee_yaw_b,
                ee_yaw_b,
            )
        print(
            f"[INFO] Go2Arm fixed ee command override: pos={args_cli.go2arm_ee_pos}, rpy={args_cli.go2arm_ee_rpy}"
        )
        env_cfg.commands.ee_pose.sample_z_in_world_frame = True
        env_cfg.commands.ee_pose.reject_position_cuboid = None
        env_cfg.commands.ee_pose.max_sampling_tries = 1
        env_cfg.commands.ee_pose.secondary_position_range_b = None
        env_cfg.commands.ee_pose.secondary_euler_xyz_range_b = None
        env_cfg.commands.ee_pose.secondary_world_z_range = None
        env_cfg.commands.ee_pose.secondary_sample_prob = 0.0
        env_cfg.commands.ee_pose.tertiary_position_range_b = None
        env_cfg.commands.ee_pose.tertiary_euler_xyz_range_b = None
        env_cfg.commands.ee_pose.tertiary_world_z_range = None
        env_cfg.commands.ee_pose.tertiary_sample_prob = 0.0
        env_cfg.events.randomize_reset_joints.params["position_range"] = (-0.04, 0.04)
        env_cfg.events.randomize_reset_joints.params["velocity_range"] = (-0.05, 0.05)
        env_cfg.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.06, 0.06),
            "y": (-0.06, 0.06),
            "yaw": (-0.18, 0.18),
        }

    if "go2arm" in task_name.lower() and args_cli.keyboard:
        raise ValueError("Go2Arm/RoboDuet does not support generic `play.py --keyboard`; use the task-specific control path.")

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif "go2arm" in task_name.lower():
        # RoboDuet upstream saves latest policy weights as `ac_weights_last_dog.pt` and
        # `ac_weights_last_arm.pt`.  Do not let the generic IsaacLab resolver pick stale
        # root-level `model_*.pt` files or numbered `ac_weights_006000.pt` by default.
        go2arm_checkpoint_pattern = agent_cfg.load_checkpoint
        if go2arm_checkpoint_pattern is None or str(go2arm_checkpoint_pattern).startswith("model_"):
            go2arm_checkpoint_pattern = "ac_weights_last_dog.pt"
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.load_run,
            go2arm_checkpoint_pattern,
            other_dirs=["checkpoints_dog"],
        )
        print(
            "[INFO] Go2Arm play checkpoint override: using dog checkpoint under "
            f"checkpoints_dog/ matching {go2arm_checkpoint_pattern!r}."
        )
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    if "go2arm" in task_name.lower() and os.path.basename(log_dir) in {"checkpoints_dog", "checkpoints_arm"}:
        log_dir = os.path.dirname(log_dir)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner_class = resolve_callable(agent_cfg.class_name)
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = getattr(runner.alg, "actor_critic", None)
    if policy_nn is None and hasattr(policy, "act_inference"):
        policy_nn = policy

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # 导出策略到 jit / onnx。
    # 只要策略对象自己提供了 as_jit()/as_onnx()，就直接使用策略自带的导出包装。
    # 这样可以确保 go2arm 这类自定义 privileged teacher policy 走正确的导出语义。
    export_model_dir = os.path.join(log_dir, "exported")
    if getattr(policy_nn, "skip_generic_export", False):
        print("[INFO] Skipping generic play-time export for RoboDuet automatic policy. Use training-time deploy_model artifacts instead.")
    elif policy_nn is not None and hasattr(policy_nn, "as_jit") and hasattr(policy_nn, "as_onnx"):
        os.makedirs(export_model_dir, exist_ok=True)

        # 直接导出 TorchScript。
        jit_model = policy_nn.as_jit()
        jit_model.to("cpu")
        torch.jit.script(jit_model).save(os.path.join(export_model_dir, "policy.pt"))

        # 直接导出 ONNX。
        onnx_model = policy_nn.as_onnx(verbose=False)
        onnx_model.to("cpu")
        onnx_model.eval()
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            os.path.join(export_model_dir, "policy.onnx"),
            export_params=True,
            opset_version=18,
            verbose=False,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
        )
    elif policy_nn is not None:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    use_mean_action = "go2arm" in task_name.lower() and hasattr(policy, "act_inference")
    if "go2arm" in task_name.lower() and not use_mean_action:
        print("[INFO] Go2Arm play fallback: policy has no act_inference(); using policy(obs) instead.")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    trace_step = 0
    trace_interval = 20 if args_cli.go2arm_trace_actions else None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            pre_step_obs = obs
            if use_mean_action:
                actions = policy.act_inference(obs)
            else:
                actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            _print_go2arm_termination_reasons(extras)
            if trace_interval is not None and timestep % trace_interval == 0:
                _print_go2arm_action_state(env, actions[0], trace_step, pre_step_obs)
            trace_step += 1
            # reset recurrent states for episodes that have terminated
            if hasattr(policy, "reset"):
                policy.reset(dones)
            elif policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
