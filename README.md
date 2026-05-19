## Overview

**robot_lab_roboduet** 是一个基于 `robot_lab` 的本地变体，用于在 IsaacLab 上复现并扩展 RoboDuet 风格的全身协同移动操作训练流程。

当前仓库主要聚焦于：

- 将 RoboDuet 的训练范式迁移到本地 `Go2 + Piper` 资产链路
- 保留上游 `auto_train` 的两阶段 `dog -> dog+arm` 训练语义
- 与本地 `go2arm` 任务代码和 IsaacLab 工作流兼容
- 提供平地基线任务与本地 rough terrain 扩展任务

## RoboDuet Task

当前任务面向四足底盘搭载 6 自由度机械臂的协同 locomotion + manipulation，主要参考：

*RoboDuet: Learning a Cooperative Policy for Whole-Body Legged Loco-Manipulation*.

和原始上游实现相比，这个仓库不是逐行镜像，而是一个建立在 `robot_lab` 之上的本地 IsaacLab 移植版本。目标是尽可能保留有效训练行为，同时适配本地资产、环境和调试方式。

## Robot Setup

- 机器人主体：Unitree Go2
- 机械臂：Piper 6-DoF
- 安装链路：`arm_mount -> link1 -> link2 -> link3 -> link4 -> link5 -> link6`
- 末端执行器：`link6`

默认训练路径使用本地 `go2arm` 的 URDF 资产链路。若需要直接对齐上游 RoboDuet `auto_train` 的机器人描述，可在训练和回放脚本中额外启用 `--roboduet_urdf`。

## Default Joint State

当前任务使用的本地 Go2Arm 默认关节初始姿态如下：

```text
Leg joints
FL_hip_joint   = 0.0
FL_thigh_joint = 0.8
FL_calf_joint  = -1.5
FR_hip_joint   = 0.0
FR_thigh_joint = 0.8
FR_calf_joint  = -1.5
RL_hip_joint   = 0.0
RL_thigh_joint = 0.8
RL_calf_joint  = -1.5
RR_hip_joint   = 0.0
RR_thigh_joint = 0.8
RR_calf_joint  = -1.5

Arm joints
joint1 = 0.0
joint2 = 0.314
joint3 = -0.2967
joint4 = 0.0
joint5 = 0.0
joint6 = 0.0
```

这组默认姿态同时也是当前本地 joint-delta action 参数化的中心点。

## Local Version

- Isaac Sim 4.5
- Isaac Lab 2.2.1

## Installation

请在 IsaacLab 对应的 Python 环境中安装本扩展：

```bash
python -m pip install -e source/robot_lab
```

如果你本地同时以 editable 方式安装了 `IsaacLab` 或 `rsl_rl`，建议保持它们处于同一个 Python 环境，避免版本错配。

## Registered Environments

当前仓库注册了两个 Go2Arm loco-manipulation 环境：

- `RobotLab-Isaac-Flat-Go2Arm-v0`
- `RobotLab-Isaac-Rough-Go2Arm-v0`

推荐使用顺序：

- 先从 `RobotLab-Isaac-Flat-Go2Arm-v0` 开始，便于验证行为和调试训练流程
- 平地版本稳定后，再切到 `RobotLab-Isaac-Rough-Go2Arm-v0`

## Training

默认 PPO runner 是本地 RoboDuet 风格的 automatic runner，保留了上游的两阶段训练结构：

- stage 1：仅 dog 的 locomotion 阶段
- stage 2：dog-arm 协同阶段

默认情况下：

- 从头训练时，两阶段切换点为 `10000` iterations
- 当 runner config 同时提供 dog 和 arm 预训练权重时，默认切换点变为 `2000`

平地训练示例：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Flat-Go2Arm-v0 \
  --headless \
  --num_envs 2048
```

rough 训练示例：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Rough-Go2Arm-v0 \
  --headless \
  --num_envs 2048
```

常用参数：

- `--resume`：从已有 run 继续训练
- `--load_run <run-folder>`：指定要恢复的 run
- `--checkpoint <path-or-pattern>`：覆盖 checkpoint 选择逻辑
- `--roboduet_stage2_dog_checkpoint <path>`：直接用 dog checkpoint 引导 stage 2
- `--roboduet_alignment_check`：只创建环境并检查关键 RoboDuet 语义，不启动训练
- `--roboduet_urdf`：切换到上游风格的 Go2Piper URDF 覆盖
- `--symmetry`：启用镜像增强和 mirror consistency loss
- `--omni` / `--omni_stage1` / `--omni_stage2`：启用 RoboDuet 风格的 omni reward 聚合

恢复训练示例：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Flat-Go2Arm-v0 \
  --headless \
  --resume \
  --load_run <run-folder> \
  --checkpoint checkpoints_dog/ac_weights_004000.pt \
  --max_iterations 12000
```

说明：

- 对 RoboDuet automatic runner，`--resume` 默认会优先从 `checkpoints_dog/` 解析 dog checkpoint
- 从 `ac_weights_004000.pt` 恢复时，训练迭代数会回到 `4000`
- `--max_iterations` 表示恢复后还要继续跑多少 iteration

## Training Outputs

除了标准日志目录外，RoboDuet 训练还会额外导出与上游兼容的产物：

- `checkpoints_dog/ac_weights_*.pt`
- `checkpoints_arm/ac_weights_*.pt`
- `deploy_model/adaptation_module_latest_dog.jit`
- `deploy_model/body_latest_dog.jit`
- `deploy_model/adaptation_module_latest_arm.jit`
- `deploy_model/body_latest_arm.jit`
- `deploy_model/history_latest_arm.jit`

这样更方便和上游 checkpoint 以及部署模块做对比。

## Play

基础回放命令：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Flat-Go2Arm-v0 \
  --checkpoint <path-to-model-or-ac_weights.pt>
```

当前仓库中的回放行为：

- `play.py` 对 RoboDuet automatic runner 使用确定性 `act_inference()`
- 回放时默认关闭训练阶段的机械臂冻结逻辑，便于直接观察 arm 输出
- 对 automatic runner 训练得到的策略，更推荐直接使用导出的 `deploy_model/` 产物进行部署

常用回放参数：

- `--roboduet_urdf`
- `--go2arm_stage1_play`
- `--go2arm_dog_cmd vx vy wz`
- `--go2arm_arm_cmd l p y`
- `--go2arm_ee_pos X Y Z`
- `--go2arm_ee_rpy R P Y`
- `--go2arm_trace_actions`

## Current Alignment Scope

当前仓库主要对齐的是 RoboDuet 在本地 IsaacLab 栈上的有效训练语义，重点包括：

- 观测分组：`dog_policy`、`dog_privileged`、`arm_policy`、`arm_privileged`
- 两阶段训练调度
- `12` 维 dog action head 与 `6 + 2` 维 arm action head
- 上游风格的 checkpoint 与 deploy 导出布局
- 带阶段感知的 reward 聚合与 command 处理

当前有两点本地差异是有意保留的：

- stage-1 的 arm freeze 是通过在 IsaacLab action 路径中强制 arm delta action 为零实现的
- 足端接触处理比上游更严格，因为本地任务使用了更细的 foot contact sensing 来恢复合法支撑接触

## Key Files

和 RoboDuet Go2Arm 任务最相关的文件包括：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/flat_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/agents/rsl_rl_ppo_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/agents/automatic_runner.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `scripts/reinforcement_learning/rsl_rl/train.py`
- `scripts/reinforcement_learning/rsl_rl/play.py`

## Notes

- 当前迁移目标聚焦本地 `go2arm` 任务，不包含上游 `go1` 分支
- 默认资产链路仍然是本地 `Go2 + Piper`，不会直接把所有流程切到上游 URDF
- `rough` 是建立在 flat 基线之上的本地扩展任务，不属于上游 `auto_train` 的默认配置
- 这份 README 目前是偏“可用性 + 仓库导览”的初版，后续可以再补充更细的任务设计说明

## Citation

This repository is a modified local variant of `robot_lab`.
