## Overview

**roboduet_isaaclab** 是一个基于 `roboduet` (Isaacgym)的IsaacLab版本，目标是严格对齐 RoboDuet 原仓库`go2`的网络结构、观测、命令、奖励、随机化、终止、课程和两阶段训练流程。

虽然代码里仍然保留了 `RobotLab-Isaac-Rough-Go2Arm-v0` 这个任务注册点，但当前 `rough_env_cfg.py` 里也把 terrain 强制设成了 `plane`。从训练上看，当前仓库和roboduet中只有flat任务是一致的。

## Alignment Status

主要对齐点：
- 两阶段训练：`dog-only -> dog+arm`
- 观测分组：`dog_policy`56D / `dog_privileged`2D / `arm_policy`20D / `arm_privileged`9D
- 动作结构：`12` 维 dog 动作 + `6` 维 arm 动作 + `2` 维 plan action
- 历史观测的adaptation model
- locomotion command curriculum
- reward term、scale 和 stage switch 逻辑
- `step_dt = 0.02`
- `max_episode_length = 1000`
- `num_steps_per_env = 24`
- stage1 机械臂冻结关节是否正确
- 命令切换迭代是否正确

## Robot Setup

- 机器人主体：Unitree Go2
- 机械臂：Piper 6-DoF
- 安装链路：`arm_mount -> link1 -> link2 -> link3 -> link4 -> link5 -> link6`
- 末端执行器：`link6`

默认训练使用本地 `Go2 + Piper` 资产链路

## Default Joint State

当前任务的默认关节初始姿态为：

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

这组默认姿态同时也是 joint-delta action 的偏置中心。狗的默认姿态与上游对齐，但是机械臂由于型号不同，并不完全对齐。

## Local Version

当前本地版本：

- Isaac Sim 5.1
- Isaac Lab 2.3.2

## Installation

在 IsaacLab 对应的 Python 环境中安装扩展：

```bash
python -m pip install -e source/robot_lab
```

## Registered Environments

代码中当前注册了两个环境：

- `RobotLab-Isaac-Flat-Go2Arm-v0`
- `RobotLab-Isaac-Rough-Go2Arm-v0`

但需要注意：

- **建议实际使用 `RobotLab-Isaac-Flat-Go2Arm-v0`。**
- `rough` 入口当前不是一个真正独立的 rough-terrain RoboDuet 任务。
- `rough_env_cfg.py` 当前同样将 terrain 强制成 `plane`，保留该 ID 主要是为了兼容本地代码路径和后续扩展。

## Training

当前默认 runner 是本地 RoboDuet automatic runner，对齐原仓库 `auto_train` 的两阶段训练流程：

- stage 1：只训练 dog locomotion policy
- stage 2：打开 arm policy 和 whole-body 协同控制

切换规则：

- 从头训练或--resume：`10000` iterations 切 stage 2
- 如果显式指定狗的权重`--roboduet_stage2_dog_checkpoint <path>`，可以直接从stage2开始训练

训练入口：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Flat-Go2Arm-v0 \
  --headless \
  --num_envs 4096
```

### Train Arguments

与 RoboDuet / Go2Arm 强相关的训练参数如下：

| 参数 | 作用 |
| --- | --- |
| `--task` | 任务名。当前主线请用 `RobotLab-Isaac-Flat-Go2Arm-v0` |
| `--num_envs` | 覆盖并行环境数 |
| `--headless` | 无界面训练 |
| `--seed` | 设置随机种子 |
| `--max_iterations` | 覆盖训练 iteration 总数 |
| `--resume` | 从已有 run 恢复训练 |
| `--load_run <run-folder>` | 指定恢复的 run 目录 |
| `--checkpoint <path-or-pattern>` | 恢复 checkpoint；RoboDuet runner 默认会优先从 `checkpoints_dog/` 解析 |
| `--roboduet_stage2_dog_checkpoint <path>` | 直接从 dog checkpoint 引导进入 stage 2，runner iteration 会被同步到 `10000`，不能和--resume一起使用 |
| `--symmetry` | 启用 RoboDuet Go2Arm 镜像增强和 mirror consistency loss |
| `--omni` | 同时打开 stage1 和 stage2 的 omni reward|
| `--omni1` | 仅打开 stage1 omni reward |
| `--omni2` | 仅打开 stage2 omni reward |
| `--video` | 训练时录视频 |
| `--video_interval` / `--video_length` | 控制训练视频录制频率和长度 |

### Resume Example

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

- `--resume` 对 RoboDuet automatic runner 默认会优先查找 `checkpoints_dog/ac_weights_last_dog.pt`
- 从 `ac_weights_004000.pt` 恢复时，内部训练 iteration 会恢复到 `4000`
- `--max_iterations 12000` 的含义是“恢复后再继续跑 12000 iterations”

### Training Outputs

训练产物除了标准日志，还会额外导出与原仓库兼容的文件：

- `checkpoints_dog/ac_weights_*.pt`
- `checkpoints_dog/ac_weights_last_dog.pt`
- `checkpoints_arm/ac_weights_*.pt`
- `checkpoints_arm/ac_weights_last_arm.pt`
- `deploy_model/adaptation_module_latest_dog.jit`
- `deploy_model/body_latest_dog.jit`
- `deploy_model/adaptation_module_latest_arm.jit`
- `deploy_model/body_latest_arm.jit`
- `deploy_model/history_latest_arm.jit`

## Play

基本回放命令：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Flat-Go2Arm-v0 \
  --checkpoint <path-to-ac_weights_last_dog.pt>
```

对 RoboDuet automatic runner：

- `play.py` 默认走确定性 `act_inference()`
- 默认直接打开 stage 2 回放
- 默认关闭 eval-time DR 和外部扰动，但保留 reset 随机化

### Play Arguments

| 参数 | 作用 |
| --- | --- |
| `--checkpoint` | 指定play checkpoint。默认会优先从 `checkpoints_dog/` 找 `ac_weights_last_dog.pt` |
| `--stage1` | 强制在 stage 1 dog-only 模式下play |
| `--arm_fix` | 仅与 `--stage1` 一起使用；每步把 arm 强制重置到默认姿态，模拟上游 `keep_arm_fixed` |
| `--go2arm_dog_cmd ...` | 固定 dog command。stage1 需要 5 维：`vx vy wz pitch roll`；stage2 需要 3 维：`vx vy wz` |
| `--go2arm_arm_cmd L P Y` | stage2 下固定 arm 的 `l/p/y` 位置命令 |
| `--go2arm_trace_actions` | 定期打印 dog/arm 命令与实际执行状态 |
| `--video` / `--video_length` | 回放并录制视频 |
| `--real-time` | 实时速率回放 |

说明：

- stage1 回放时，arm command 保持为零，arm policy 也不会输出实际机械臂动作，狗的pitch和roll由命令显示指定（没给默认为0）
- stage2 回放默认把 dog command 固定为 `(0, 0, 0)`，除非显式传入 `--go2arm_dog_cmd`
- `--go2arm_arm_cmd` 只固定 arm 的位置命令，姿态命令仍会按 RoboDuet 范围采样一次并保持不变

## Task Design Details

和roboduet仓库对齐，有些地方与原文有区别。

### 1. Network Structure

当前 automatic runner 使用两套独立 PPO：

- `dog_model`: 负责底盘 locomotion
- `arm_model`: 负责机械臂 joint action 与 planner body command

核心常量：

- `dog_policy_obs_dim = 56`
- `dog_privileged_obs_dim = 2`
- `arm_policy_obs_dim = 20`
- `arm_privileged_obs_dim = 9`
- `dog_action_dim = 12`
- `arm_action_dim = 6`
- `arm_plan_action_dim = 2`
- `full_action_dim = 18`
- `history_length = 30`
- `step_dt = 0.02`
- `episode_length = 1000 steps = 20s`
- `num_steps_per_env = 24`

#### Dog Model

- 历史输入长度：`56 * 30 = 1680`，即30帧的历史观测
- privileged 维度：`2`
- adaptation module：`1680 -> 256 -> 128 -> 2`
- actor MLP：`(1680 + 2) -> 512 -> 256 -> 128 -> 12`
- critic MLP：`(1680 + 2) -> 512 -> 256 -> 128 -> 1`
- 初始动作噪声标准差：`1.0`
- **也就是说狗的网络输入是本体感知的30帧历史信息总共1680维，再加上用这1680维拟合的特权信息adaptation module输出2维**

#### Arm Model

- 当前观测维度：`20`
- 历史输入长度：`20 * 30 = 600`
- privileged 维度：`9`
- adaptation module：`600 -> 256 -> 128 -> 9`
- actor history encoder：`(600 - 20) = 580 -> 512 -> 256 -> 128`
- actor MLP：`(20 + 9 + 128) -> 512 -> 256 -> 128 -> 8`
- critic history encoder：`580 -> 512 -> 256 -> 128`
- critic MLP：`(20 + 9 + 128) -> 512 -> 256 -> 128 -> 1`
- 输出 `8` 维：前 `6` 维是 arm joint action，后 `2` 维是 planner action
- 最后 `2` 维 planner action 会过 `tanh`
- 初始动作噪声标准差：`0.1`
-**手的网络输入不把全部历史信息600维塞入，而是用600维做特权拟合9维+历史编码128维，再把这一帧观测20D+9D+128D作为输入**

#### Stage Switch

- stage1：只训练 `dog PPO`
- stage1：环境执行的 arm action 恒为 `0`，且会在每个仿真物理步把机械臂拉回默认位置，清空速度
- stage2：dog/arm 同时工作

#### Final Action Execution

当前动作采用 `Go2ArmDefaultDeltaJointPositionActionCfg`：

- 动作顺序固定为：`12` 个腿关节 + `6` 个 arm 关节
- `action_scale = 0.25`
- 四个 hip 关节额外乘 `0.5`，即最终scale=0.125
- 最终目标形式为：`target_joint_pos = default_joint_pos + delta`
- stage1 冻结 `joint1..joint6`

arm policy 的最后 `2` 维 planner 输出不会直接下发到执行器，而是先映射成 dog command 里的 body `pitch/roll`：

- planner 输出先乘 `0.4`，对应前面stage1的pitch/roll命令范围
- 再按 `limit_body_pitch` / `limit_body_roll` 裁剪

### 2. Observations

#### Dog Policy Observation: 56D

| 项目 | 维度 | 说明 |
| --- | ---: | --- |
| `projected_gravity` | 3 | base frame 下重力投影 |
| `joint_pos` | 12 | 四条腿关节相对默认位姿 |
| `joint_vel` | 12 | 四条腿关节速度，额外乘 `0.05` |
| `actions` | 12 | 上一步有效执行的 dog action |
| `commands_dog` | 5 | `vx vy wz pitch roll`，带 `commands_scale_dog=(2,2,0.25,1,1)` |
| `commands_arm` | 6 | arm command observation；stage1 时为全零 |
| `roll_pitch` | 2 | 当前 base 的 roll / pitch |
| `clock_inputs` | 4 | 四条腿的步态相位时钟 `sin` 编码 |

总维度：`3 + 12 + 12 + 12 + 5 + 6 + 2 + 4 = 56`
**注意是手动用scale对维度尺度调整，没有打开归一化**

#### Dog Privileged Observation: 2D

| 项目 | 维度 | 说明 |
| --- | ---: | --- |
| `friction` | 1 | 足端平均摩擦系数，按 `[0,1]` 归一化区间做 scale-shift |
| `restitution` | 1 | 足端平均回弹系数，按 `[0,1]` 归一化区间做 scale-shift |

#### Arm Policy Observation: 20D

| 项目 | 维度 | 说明 |
| --- | ---: | --- |
| `joint_pos` | 6 | arm 关节相对默认位姿 |
| `actions` | 6 | 上一步有效执行的 arm action |
| `commands_arm` | 6 | `l/p/y + abg` |
| `roll_pitch` | 2 | 当前 base 的 roll / pitch |

总维度：`6 + 6 + 6 + 2 = 20`

#### Arm Privileged Observation: 9D

| 项目 | 维度 | 说明 |
| --- | ---: | --- |
| `friction` | 1 | 足端平均摩擦系数 |
| `restitution` | 1 | 足端平均回弹系数 |
| `lpy` | 3 | 当前末端在 yaw-aligned base frame 下的 `l/p/y` |
| `ee_quat_in_base` | 4 | 当前末端四元数；这里按 RoboDuet 原仓库的 privileged 读取方式实现 |

说明：

- 所有观测组都关闭了 corruption，即假设观测是准确的
- `actions` 读取的是 effective action，不是原始策略输出

### 3. Commands

当前环境的命令项是 `RoboDuetCommand`，同时维护三类量：

- 底盘 dog command
- 机械臂 arm target
- gait clock / desired contact states

#### Dog Command

内部 dog command 是 5 维：

- `vx`
- `vy`
- `wz`
- `body_pitch`
- `body_roll`

训练配置中的有效范围：

- `lin_vel_x = [-3.0, 3.0]`(原仓库`[-5,5]`)
- `lin_vel_y = [-0.6, 0.6]`
- `ang_vel_yaw = [-2.0, 2.0]`(原仓库`[-5,5]`)
- `body_pitch_range = [-0.4, 0.4]`
- `body_roll_range = [-0.4, 0.4]`

curriculum 离散 bin：

- `x`: `21`
- `y`: `1`
- `yaw`: `21`
- `body_pitch`: `1`
- `body_roll`: `1`

stage1 / stage2 的差别：

- stage1：`body_pitch/body_roll` 来自命令 curriculum
- stage2：`body_pitch/body_roll` 不再从 curriculum 采样，而是由 arm policy 的 `2` 维 planner action 生成

采样与过滤规则：

- locomotion command 每 `10s` 重采样一次
- 每次重采样有 `10%` 概率直接把 `(vx, vy, wz)` 置零
- 小命令阈值过滤：
  - `|vx| <= 0.07 -> 0`
  - `|vy| <= 0.07 -> 0`
  - `|wz| <= 0.10 -> 0`

commands scale：

- `commands_scale_dog = (2.0, 2.0, 0.25, 1.0, 1.0)`

#### Arm Command

arm command 在策略输入里是 6 维：

- 前 `3` 维：位置命令 `l/p/y`
- 后 `3` 维：姿态命令 `abg`(和论文中的6D不同)

位置采样范围：

- `l_range = [0.3, 0.7]`
- `p_range = [-0.45pi, 0.45pi]`
- `y_range = [-pi/2, pi/2]`

姿态采样范围：

- `roll_ee_range = [-0.45pi, 0.45pi]`
- `pitch_ee_range = [-60deg, 60deg]`
- `yaw_ee_range = [-75deg, 75deg]`

轨迹时长：

- `traj_time_range = [2.0s, 3.0s]`

碰撞 / 无效目标过滤（**原仓库中没有**）：

- arm target 会在局部 `xyz` 空间里进行排除盒检测
- `arm_collision_lower_limits = (-0.38, -0.16, -0.3)`
- `arm_collision_upper_limits = (0.3, 0.16, 0.10)`
- `arm_underground_limit = -0.38`

stage1 / stage2 的差别：

- stage1：arm command 不采样，`commands_arm_obs` 为零
- stage2：arm command 在 reset 后和轨迹结束时重采样

#### Gait Clock

当前步态时钟由命令项内部直接维护：

- `gait_frequency = 3.0`
- `gait_duration = 0.5`
- `gait_kappa = 0.04`（原文0.07，类初始化时覆盖）

输出包括：

- `foot_indices`: 4 条腿的相位索引
- `clock_inputs`: `sin(2*pi*phase)` 形式的 4 维时钟输入
- `desired_contact_states`: 用于接触 shaping reward 的软接触目标

### 4. Rewards

这一节分为两部分：

1. **RoboDuet 原仓库版本**
2. **本地--omni形式奖励版本**

#### 4.1 RoboDuet 原仓库对齐版本

当前 reward 主体在 `mdp/rewards.py::_compute_roboduet_reward_state()` 中统一计算。

默认不是简单线性求和，而是采用 RoboDuet / Ji22 风格的正负项组合：

```text
reward = reward_pos * exp(reward_neg / sigma_rew_neg)
```

当前默认：

- `only_positive_rewards = False`
- `only_positive_rewards_ji22_style = True`
- `sigma_rew_neg = 0.05`

其中：

- `tracking_sigma = 0.25`
- `tracking_sigma_yaw = 0.25`
- `gait_force_sigma = 100.0`
- `gait_vel_sigma = 10.0`

##### Stage 1 Reward Scales: `PRETRAINED_REWARD_SCALES`

| term | scale |
| --- | ---: |
| `tracking_lin_vel` | `1.0` |
| `tracking_ang_vel` | `0.5` |
| `lin_vel_z` | `-0.1` |
| `ang_vel_xy` | `-0.005` |
| `orientation_control` | `-5.0` |
| `loco_energy` | `-5e-6` |
| `feet_slip` | `-0.04` |
| `feet_clearance_cmd_linear` | `-100.0` |
| `tracking_contacts_shaped_force` | `10.0` |
| `tracking_contacts_shaped_vel` | `10.0` |
| `collision` | `-10.0` |
| `dof_vel` | `-1e-4` |
| `dof_acc` | `-2.5e-7` |
| `action_rate` | `-0.01` |
| `action_smoothness_1` | `-0.1` |
| `action_smoothness_2` | `-0.1` |
| `torques` | `-5e-5` |
| `dof_pos_limits` | `-10.0` |
| `raibert_heuristic` | `-10.0` |

##### Stage 2 Reward Scales: `HYBRID_REWARD_SCALES`

stage2 在 stage1 基础上做如下更新和扩展：

| term | scale |
| --- | ---: |
| `tracking_lin_vel` | `0.7` |
| `tracking_ang_vel` | `0.25` |
| `orientation_heuristic` | `-2.0` |
| `orientation_control` | `-10.0` |
| `raibert_heuristic` | `-0.0` |
| `arm_manip_commands_tracking_combine` | `1.0` |
| `vis_manip_commands_tracking_lpy` | `1.0` |
| `vis_manip_commands_tracking_rpy` | `1.0` |
| `hip_action_l2` | `-0.05` |
| `arm_energy` | `-4e-5` |
| `arm_dof_vel` | `-0.001` |
| `arm_dof_acc` | `-2.5e-6` |
| `arm_action_rate` | `-0.1` |
| `arm_action_smoothness_1` | `-0.5` |
| `arm_action_smoothness_2` | `-0.5` |
| `arm_control_smoothness_1` | `-0.1` |
| `arm_control_limits` | `-5.0` |

其它 stage1 中未改动的项默认保持不变。

##### Reward Term Definitions

主要 reward term 的实际形式如下。

Locomotion tracking：

- `tracking_lin_vel = exp(-||cmd_xy - v_xy_body||^2 / 0.25)`
- `tracking_ang_vel = exp(-(cmd_yaw - wz_body)^2 / 0.25)`
- `lin_vel_z = vz_body^2`
- `ang_vel_xy = wx_body^2 + wy_body^2`

姿态相关：

- `orientation_control`：当前 projected gravity 与目标 `pitch/roll` 对应 projected gravity 的 XY 差平方和
- `orientation_heuristic`：根据 arm target 的高低位置，对 base pitch 提供启发式引导

能耗与平滑：

- `loco_energy = sum((tau_leg * qd_leg)^2)`
- `arm_energy = sum((q_target_arm * qd_arm)^2)`
- `dof_vel` / `arm_dof_vel`：关节速度平方和
- `dof_acc` / `arm_dof_acc`：关节加速度平方和
- `action_rate` / `arm_action_rate`：相邻两步有效动作差平方和
- `action_smoothness_1/2`、`arm_action_smoothness_1/2`：基于关节目标的一阶/二阶平滑惩罚
- `arm_control_smoothness_1`：planner 两维输出的一阶平滑惩罚
- `arm_control_limits`：planner 输出越过 `pitch/roll` 范围的惩罚

接触与步态：

- `feet_slip`：接触脚的足端平面速度平方和
- `feet_clearance_cmd_linear`：摆动脚足端高度对目标高度 `0.06 * phase + 0.02` 的误差平方
- `tracking_contacts_shaped_force`：摆动期应尽量少受力
- `tracking_contacts_shaped_vel`：支撑期应尽量少移动
- `collision`：非法 body 接触计数
- `raibert_heuristic`：基于期望落脚点的启发式误差

arm manipulation：

- `arm_manip_commands_tracking_combine`
  - 先计算当前末端 `lpy` 和目标 `lpy` 的归一化误差
  - 再计算当前末端 `abg` 和目标 `abg` 的归一化误差
  - 当前默认权重从 `lpy:4.0 / rpy:0.0` 逐步过渡到 `lpy:3.0 / rpy:1.0`
  - 过渡时长 `5000` iterations
- `vis_manip_commands_tracking_lpy = exp(-lpy_error)`
- `vis_manip_commands_tracking_rpy = exp(-rpy_error)`

约束项：

- `torques`：腿部 torque target 平方和 + arm joint position target 平方和
- `hip_action_l2`：4 个 hip action 的平方和
- `dof_pos_limits`：越过 soft joint limits 的总量

##### Omni Reward Modes

训练脚本中还保留了两个可选开关：

- `--omni1`
- `--omni2`

它们不是默认模式，但如果打开，会使用 RoboDuet 风格的 omni 正项聚合：

- stage1 omni：强化 `tracking_lin_vel` 和 `tracking_ang_vel`
- stage2 omni：额外把 manipulation 正项并入 dog reward，同时单独塑造 arm reward

#### 4.2 本地修改 / IsaacLab 实现补丁

这一部分不是在改变“要对齐的 reward 目标”，而是在说明 **当前本地实现相对上游代码的实现层差异**。

##### 1. 足端接触的实现更严格

本地实现优先使用 `contact_forces` 共享传感器中“合法足端 patch”对应的精确接触力：

- `get_go2arm_precise_foot_contact_forces()`
- `get_go2arm_precise_foot_normal_forces()`
- `get_go2arm_precise_foot_contact_timers()`

因此：

- `feet_slip`
- `tracking_contacts_shaped_force`
- `tracking_contacts_shaped_vel`
- `feet_contact_state`
- `feet_air_time`

这些项都会优先基于“合法足端 patch 聚合结果”来计算，而不是简单按 rigid body 级别接触力来算。

##### 2. `torques` 项做了上游语义修复

本地实现里，`torques` 不只统计腿部 `applied_torque`，还会把 arm 的 joint position target 一起计入：

```text
torques = sum(leg_torque_target^2) + sum(arm_joint_pos_target^2)
```

这是为了更接近上游 `control_type="M"` 的 `auto_train` 语义。

##### 3. manipulation 权重做了阶段过渡

`arm_manip_commands_tracking_combine` 里，`lpy` 和 `rpy` 的权重不是固定的：

- 初始：`lpy = 4.0`, `rpy = 0.0`
- 结束：`lpy = 3.0`, `rpy = 1.0`
- 过渡长度：`5000` iterations

这属于本地实现里显式写出的细化逻辑。

##### 4. `vis_*` 项是日志项，不直接计入 manager reward

- `vis_manip_commands_tracking_lpy`
- `vis_manip_commands_tracking_rpy`

这两个项会被记录和导出，但不会像普通 reward term 一样直接贡献 manager 的最终 reward。

##### 5. 命令课程里接触 shaping 项有额外偏置写法

在命令 curriculum 日志累积时：

- `tracking_contacts_shaped_force`
- `tracking_contacts_shaped_vel`

会带一个与 scale 和 `dt` 相关的 offset 累积方式，以保持和当前课程成功阈值逻辑一致。

### 5. Terminations

当前 RoboDuet 对齐配置下，真正启用的 termination 比原始 `go2arm` 基础任务少很多。

#### Active

- `time_out`
  - 固定 episode 长度 `1000` steps
  - 对应 `20s`

- `base_height_termination`
  - 使用 `roboduet_body_height_termination`
  - 当前阈值：`minimum_height = 0.28`
  - 在 plane 上直接比较 `base` 的高度

- `reverse_termination`
  - 只在 stage2 打开后生效
  - 当前参数：
    - `roll_limit = 0.10`
    - `pitch_limit = 0.20`
    - `headupdown_thres = 0.10`
    - `use_roll = False`
    - `use_pitch = True`
  - 代码里还要求 `arm_time / T_traj > 0.6`
  - 也就是说当前更像是一个 **后半段轨迹中的 pitch 方向反向失败检测**

#### Disabled In Current Aligned Flat Config

以下通用终止在当前 RoboDuet 对齐配置中被显式关闭：

- `terrain_out_of_bounds`
- `non_foot_contact_termination`
- `base_orientation_termination`
- `joint_position_termination`
- `joint_velocity_termination`
- `joint_torque_termination`
- `task_success`

因此在 stage1，有效的非超时终止基本只剩：

- `base_height_termination`

### 6. Curriculum

当前任务中的 curriculum 需要分成两层理解。

#### Active Curriculum

当前真正启用的是：

- RoboDuet locomotion command curriculum
- RoboDuet stage switch curriculum

其中 command curriculum：

- 对 `x/y/yaw/pitch/roll` 的离散 bin 做阈值更新
- 成功阈值来自：
  - `tracking_lin_vel`
  - `tracking_ang_vel`
  - `tracking_contacts_shaped_force`
  - `tracking_contacts_shaped_vel`
- 当前阈值：
  - `tracking_lin_vel: 0.8`
  - `tracking_ang_vel: 0.7`
  - `tracking_contacts_shaped_force: 0.9`
  - `tracking_contacts_shaped_vel: 0.9`

stage switch curriculum：

- `switch_iteration = 10000`：默认从 scratch
- `switch_iteration = 2000`：当同时有 dog/arm pretrained checkpoint
- `switch_iteration = 0`：显式关闭两阶段

#### Disabled Curriculum

本地 `go2arm` 以前的 `go2arm_reaching_stages` curriculum 在当前 RoboDuet 对齐配置中是关闭的：

- `self.curriculum.go2arm_reaching_stages = None`

也就是说当前 README 里不再把 `ee_pose` staged curriculum 当作 RoboDuet 主线逻辑来介绍。

### 7. Events And Randomization

当前 RoboDuet 对齐 flat 配置里，启用的 event / randomization 如下。

#### Startup Events

`randomize_rigid_body_material`

- 函数：`randomize_rigid_body_material_consistent`
- 语义：每个并行环境采样一组统一的 friction / restitution，并应用到整台机器人
- 范围：
  - `friction_range = (0.1, 3.0)`
  - `restitution_range = (0.0, 0.4)`

`randomize_rigid_body_mass_base`

- 仅作用于 `base`
- `mass_distribution_params = (-2.0, 2.0)`
- `operation = "add"`
- `recompute_inertia = True`

#### Reset Events

`randomize_reset_joints`

- 当前函数被切换成 `reset_joints_by_scale`
- 语义更接近上游 `default_joint_pos * U(0.5, 1.5)`
- 参数：
  - `position_range = (0.5, 1.5)`
  - `velocity_range = (0.0, 0.0)`

`randomize_reset_base`

- `pose_range`
  - `x = (-0.2, 0.2)`
  - `y = (-0.2, 0.2)`
  - `yaw = (-pi, pi)`
- `velocity_range`
  - `x/y/z = (-0.5, 0.5)`
  - `roll/pitch/yaw = (-0.5, 0.5)`

#### Disabled By Default In Current Aligned Config

以下 event 当前默认关闭：

- `randomize_rigid_body_mass_ee`
- `randomize_apply_external_force_torque_base`
- `randomize_apply_external_force_torque_ee`
- `randomize_push_robot`

另外环境配置里虽然保留了这些开关字段，但默认也是关闭状态：

- `roboduet_randomize_gravity = False`
- `roboduet_randomize_motor_strength = False`
- `roboduet_randomize_motor_offset = False`

对应参数范围分别为：

- `roboduet_gravity_range = (-1.0, 1.0)`
- `roboduet_motor_strength_range = (0.9, 1.1)`
- `roboduet_motor_offset_range = (-0.02, 0.02)`

回放 `play.py` 中还会进一步关闭 eval-time DR：

- 材料随机化关闭
- base mass 随机化关闭
- 外部扰动关闭
- push 关闭

但会保留 reset 随机化。

## Key Files

最关键的文件如下：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/flat_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/rough_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/agents/rsl_rl_ppo_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/agents/automatic_models.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/locomanip/go2arm/agents/automatic_runner.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/observations.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terminations.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/events.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
- `scripts/reinforcement_learning/rsl_rl/train.py`
- `scripts/reinforcement_learning/rsl_rl/play.py`

## Notes

- 当前 README 的主叙述是：**对齐 RoboDuet，且当前只把 flat 任务作为主线**
- `rough` 任务 ID 目前更多是兼容入口，而不是一个独立完成的 rough-terrain RoboDuet 版本
- 如果后续开始真正做 rough terrain，对应说明应单独拆分，不建议继续把它和当前 flat 对齐说明混写

## Citation

This repository is a modified local variant of `robot_lab`.
