**Workspace test result: `reliable_fixed_mount_volume_m3 = 0.42144`, `reliable_expanded_volume_m3 = 0.746752219`, `reliable_gain_percent = 77.1906366`.**

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
- stage1：环境执行的 arm action 恒为 `0`，且会在**每个物理仿真步把机械臂拉回默认位置，清空速度**
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

默认采用 RoboDuet / Ji22 风格的正负项组合：

```text
reward = reward_pos * exp(reward_neg / sigma_rew_neg)
```

当前默认：

- `only_positive_rewards = False`
- `only_positive_rewards_ji22_style = True`
- `sigma_rew_neg = 0.05`

##### Stage 1 Reward Scales: `PRETRAINED_REWARD_SCALES`

注意：部分奖励进行了调整，例如调大了sigma_rew_neg，调小了loco_energy和torque的权重，增大了步态相关的权重

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
- `feet_clearance_cmd_linear`：摆动脚足端高度对目标高度 `0.06 * phase + 0.02` 的误差平方（0.06代表摆腿高度，原仓库0.04）
- `tracking_contacts_shaped_force`：摆动期应尽量少受力
- `tracking_contacts_shaped_vel`：支撑期应尽量少移动
- `collision`：非法 body 接触计数
- `raibert_heuristic`：基于期望落脚点的启发式误差

arm manipulation：

- `arm_manip_commands_tracking_combine`
  - 先计算当前末端 `lpy` 和目标 `lpy` 的归一化误差
  - 再计算当前末端 `abg` 和目标 `abg` 的归一化误差
  - 当前默认权重从 `lpy:4.0 / rpy:0.0` 逐步过渡到 `lpy:3.0 / rpy:1.0`（原仓库直接给3/1，这里做一个课程）
  - 过渡时长 `5000` iterations
- `vis_manip_commands_tracking_lpy = exp(-lpy_error)`（只用来看跟踪效果，不参与实际奖励计算）
- `vis_manip_commands_tracking_rpy = exp(-rpy_error)`（只用来看跟踪效果，不参与实际奖励计算）

约束项：

- `torques`：腿部 torque target 平方和 + arm joint position target 平方和
- `hip_action_l2`：4 个 hip action 的平方和（只在stage2启用）
- `dof_pos_limits`：越过 soft joint limits 的总量

#### Omni Reward Modes

训练脚本中保留了三个可选开关：

- `--omni1`
- `--omni2`
- `--omni`（等价于--omni1+omni2）

它们不是默认模式，但如果打开，会使用omni风格的正项聚合：

- stage1 omni：强化 `tracking_lin_vel` 和 `tracking_ang_vel`
- stage2 omni：额外把 manipulation 正项并入 dog reward，同时单独塑造 arm reward


`--omni` 不会替换掉原来的 reward term 列表，也不会改各个 scale；它改的是**最后一步的总奖励聚合形式**。

##### Stage1 Omni (`--omni1`)

stage1 下只对 dog reward 的正项聚合做增强，核心是把 tracking reward 从 `r` 改成 `r + r^5`：

`r_lin = exp(-||cmd_xy - v_xy_body||^2 / 0.25)`
`r_yaw = exp(-(cmd_yaw - wz_body)^2 / 0.25)`

`R_pos,dog^omni = [1.0 * (r_lin + r_lin^5) + 0.5 * (r_yaw + r_yaw^5)] * dt`

最终 dog reward 变成：

`R_dog = 0.4 * (dt + R_pos,dog^omni) * exp(R_neg,dog / 0.05)`

这里的关键变化只有两点：

- tracking 正项从 `r` 变成了 `r + r^5`
- 最外层多了一个 `0.4 * (dt + ...)`

stage1 下 arm 分支仍然按默认 Ji22 风格聚合；但训练语义上 stage1 仍然是 dog-only。

##### Stage2 Omni (`--omni2`)

stage2 omni 不只是增强 locomotion tracking，还把 manipulation 正项显式并入 dog reward。

先定义：

`r_lin = exp(-||cmd_xy - v_xy_body||^2 / 0.25)`
`r_yaw = exp(-(cmd_yaw - wz_body)^2 / 0.25)`

机械臂跟踪部分先计算归一化误差：

- `e_lpy`：末端 `l/p/y` 相对命令的归一化误差
- `e_rpy`：末端 `a/b/g` 相对命令的归一化误差

stage2 中 manipulation 权重不是常数，而是在进入 stage2 后的前 `5000` iterations 内渐变：

- `w_lpy: 4.0 -> 3.0`
- `w_rpy: 0.0 -> 1.0`

普通 stage2 reward term 里，manip tracking 项是：

`r_manip = exp(-(w_lpy * e_lpy + w_rpy * e_rpy))`

但在 omni 聚合里，用的是更强的形式：

`r_pos = exp(-w_lpy * e_lpy)`
`r_ori = exp(-w_rpy * e_rpy)`
`r_manip^omni = (r_pos + r_pos^5) + r_pos * (r_ori + r_ori^5)`

于是：

`R_pos,dog^omni = [0.7 * (r_lin + r_lin^5) + 0.25 * (r_yaw + r_yaw^5) + 1.0 * r_manip^omni] * dt`
`R_pos,arm^omni = [1.0 * r_manip^omni] * dt`

最终：

`R_dog = 0.3 * (dt + R_pos,dog^omni) * exp(R_neg,dog / 0.05)`
`R_arm = 0.3 * (dt + R_pos,arm^omni) * exp(R_neg,arm / 0.05)`

stage2 omni 的关键点是：

- locomotion tracking 仍然用 `r + r^5` 增强
- **arm tracking 也用 `r + r^5` 增强，且利用优先级使得位置跟踪优先被满足**。
- dog 和 arm 两个分支都套上了 `(dt + ...) * exp(...)` 的 omni 聚合
`--omni` 不是“多开几个 reward term”，而是把默认的正负项指数聚合，改成了带 tracking/manipulation 强化正项的 omni 聚合。

### Terminations

当前 RoboDuet 对齐配置下，真正启用的 termination 很少。
严格来说，活跃的终止项只有：

- stage1：`time_out` + `base_height_termination`
- stage2：`time_out` + `base_height_termination` + `reverse_termination`

#### `time_out`

- episode 长度固定为 `1000` steps
- `dt = 0.02`
- 所以单个 episode 最长 `20s`

#### `base_height_termination`

当前配置使用：

- `minimum_height = 0.28`
- `sensor_cfg = None`

因此在当前 flat / plane 对齐配置下，它就是一个非常直接的判定：

`z_base < 0.28 -> terminate`

也就是说，这里比较的是 base 在世界坐标系下的高度，而不是“相对地形高度”版本。

#### `reverse_termination`

这个终止不是一个泛化的“翻车检测”，而是一个**只在 stage2 后半段 arm 轨迹中才启用的 pitch 方向失败判定**。

当前配置是：

- `roll_limit = 0.10`
- `pitch_limit = 0.20`
- `headupdown_thres = 0.10`
- `use_roll = False`
- `use_pitch = True`

由于 `use_roll = False`，所以当前实际上**只检查 pitch，不检查 roll**。

代码里先构造：

`delta_z = l * sin(p) + 0.38 - z_base`

其中 `l, p` 来自当前 arm 的 `lpy` 位置命令。
然后只有在轨迹已经走到后半段时才可能终止：

`arm_time / T_traj > 0.6`

满足这个时间条件后，再检查：

- `pitch < -0.20` 且 `delta_z < -0.10` 时终止
- `pitch > 0.20` 且 `delta_z > 0.10` 时终止

所以这个 termination 更准确的描述应该是：

- 它只在 stage2 生效
- 它只在 arm 轨迹后半程生效
- 它当前只看 pitch
- 它本质上是在检查“base pitch 方向”和“当前 arm 目标相对 base 的垂向趋势”是否出现明显反向失配


### 6. Curriculum

当前启用的是：

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

- `switch_iteration = 10000`：从0开始训练
- `switch_iteration = 0`：显式关闭两阶段

### 7. Events

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

环境配置里保留了这些开关字段，但默认关闭状态：

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

## Citation

This repository is a modified local variant of `robot_lab`.
