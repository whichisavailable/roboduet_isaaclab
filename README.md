## Overview

**robot_lab** is a RL extension library for robots, based on IsaacLab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

## Roboduet Task

`Roboduet` 是一个四足机器人背载机械臂的 loco-manipulation 任务,当前工作主要复现文章 *RoboDuet: Learning a Cooperative Policy for  Whole-Body Legged Loco-Manipulation* (RAL)。


### Robot Setup

- 机器人主体：Unitree Go2 四足机器人
- 机械臂：Piper 6 自由度机械臂
- 安装方式：机械臂通过背部固定安装位 `arm_mount` 装到 Go2 机体上，URDF 里的链路顺序是 `arm_mount -> link1 -> link2 -> link3 -> link4 -> link5 -> link6`
- 末端执行器：`link6`

### Default Joint State

当前任务使用的默认初始关节位来自本地 Go2Arm 资产配置，默认值如下：

```text
四足关节
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

机械臂关节
joint1 = 0.0
joint2 = 0.314
joint3 = -0.2967
joint4 = 0.0
joint5 = 0.0
joint6 = 0.0
```

其中，机械臂默认姿态会尽量避免把 `joint2` 和 `joint3` 放在单侧关节极限附近，以减少训练初期随机重置时被卡住的风险。

### Local Version

本地使用的版本是：

- Isaac Sim 4.5
- Isaac Lab 2.2.1

### Training

严格对齐上游 `scripts/auto_train.py` 的本地入口是平地任务：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Flat-Go2Arm-v0 \
  --headless \
  --num_envs 2048
```

说明：

- `RobotLab-Isaac-Flat-Go2Arm-v0` 对齐上游 `auto_train` 默认使用的 `plane` 地形。
- `RobotLab-Isaac-Rough-Go2Arm-v0` 是本地扩展版本，不属于原始 `auto_train` 默认设置。
- 默认两阶段切换迭代与上游一致：从头训练时为 `10000`，若同时提供 dog/arm 预训练权重则默认切到 `2000`。
- 可在 `go2arm/agents/rsl_rl_ppo_cfg.py` 中通过以下字段覆盖：
  `roboduet_pretrained_dog_checkpoint`、`roboduet_pretrained_arm_checkpoint`、
  `roboduet_stage_switch_iteration`、`roboduet_disable_two_stage`。

训练日志会额外导出与上游兼容的部署产物：

- `checkpoints_dog/ac_weights_*.pt`
- `checkpoints_arm/ac_weights_*.pt`
- `deploy_model/adaptation_module_latest_dog.jit`
- `deploy_model/body_latest_dog.jit`
- `deploy_model/adaptation_module_latest_arm.jit`
- `deploy_model/body_latest_arm.jit`
- `deploy_model/history_latest_arm.jit`


### Play

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Flat-Go2Arm-v0 \
  --checkpoint <path-to-model_xxx.pt>
```

说明：

- `play.py` 对 RoboDuet 自动训练 runner 使用确定性 `act_inference()` 推理，而不是采样动作。
- 播放时会自动关闭训练阶段的机械臂冻结掩码，便于直接观察真实机械臂输出。
- 对 RoboDuet 自动训练策略，推荐直接使用训练阶段导出的 `deploy_model/` 产物；`play.py` 不再重复走通用导出流程。



### Note

- 机器人资产仍保持 `Go2 + Piper` 的本地 URDF 导入链路，不回退到上游 `arx5go2.urdf`。
- 当前迁移目标聚焦本地 `go2arm` 版本，不包含上游 `go1` 分支。


### Go2Arm Task Design Details



#### 动作设计



#### 观测设计


##### 本体观测（policy）


##### 特权观测（privileged）



#### 命令设计


#### 奖励设计



#### 课程设计



#### Events 与随机化




#### 接触建模与终止条件





### RoboDuet Alignment Notes

Checked against the local upstream clone at `_tmp_roboduet_upstream`.

- Training-stage defaults are aligned with upstream `scripts/auto_train.py`: the two-stage switch is `10000` iterations from scratch and `2000` when both dog/arm pretrained checkpoints are provided.
- The local port keeps the upstream `dog_policy` / `dog_privileged` / `arm_policy` / `arm_privileged` split, the same arm `6 + 2` output layout, and the same reward-scale tables.
- Two implementation details are intentionally not byte-identical:
  - stage1 arm freeze is implemented by masking arm action deltas to zero inside the IsaacLab action term, instead of directly overwriting DOF state every physics step as upstream does;
  - foot-contact handling is stricter locally because four dedicated foot sensors are used to recover precise legal support contacts.

#### 1. Privileged Information

For the RoboDuet task in `config/locomanip/go2arm/rough_env_cfg.py`:

- `dog_privileged`: 2 dims
  - mean foot friction
  - mean foot restitution
- `arm_privileged`: 9 dims
  - mean foot friction
  - mean foot restitution
  - current `l, p, y` of the grasper in the yaw-aligned base frame
  - current end-effector quaternion in the yaw-aligned base frame

This matches upstream `dog_num_privileged_obs = 2` and `arm_num_privileged_obs = 9`.

#### 2. Rewards

Stage1 uses `PRETRAINED_REWARD_SCALES`. After the RoboDuet stage switch, rewards change to `HYBRID_REWARD_SCALES`.

Dog-stage / shared terms:

- `tracking_lin_vel = exp(-||cmd_xy - v_xy_body||^2 / 0.25)`
- `tracking_ang_vel = exp(-(cmd_yaw - w_z_body)^2 / 0.25)`
- `lin_vel_z = vz_body^2`
- `ang_vel_xy = wx_body^2 + wy_body^2`
- `orientation_control`: projected-gravity mismatch to the commanded body pitch/roll
- `loco_energy`: sum of squared leg joint power
- `feet_slip`: support-foot planar velocity penalty
- `feet_clearance_cmd_linear`: swing-foot height tracking penalty
- `tracking_contacts_shaped_force`: swing phase should have low contact force
- `tracking_contacts_shaped_vel`: stance phase should have low foot velocity
- `collision`: illegal non-foot contact count
- `dof_vel`, `dof_acc`: leg joint velocity / acceleration penalties
- `action_rate`: leg action difference penalty
- `action_smoothness_1`, `action_smoothness_2`: first/second-order leg target smoothness penalties
- `torques`: squared leg torque penalty
- `hip_action_l2`: squared hip action penalty

Arm / hybrid-only extra terms:

- `arm_manip_commands_tracking_combine = exp(-(3 * lpy_error + 1 * rpy_error))`
  - `lpy_error` is normalized by sampled `l/p/y` range
  - `rpy_error` is normalized by sampled `roll/pitch/yaw` range after conversion to upstream `abg`
- `arm_energy`: squared arm joint power
- `arm_dof_vel`, `arm_dof_acc`: arm joint velocity / acceleration penalties
- `arm_action_rate`: arm action difference penalty
- `arm_action_smoothness_1`, `arm_action_smoothness_2`: first/second-order arm target smoothness penalties
- `arm_control_smoothness_1`: plan-action smoothness penalty on the extra `pitch/roll` planner outputs
- `arm_control_limits`: penalty when planner outputs exceed commanded body pitch/roll limits

Final aggregation follows upstream:

- every enabled metric is multiplied by its stage-specific scale and added to the dog reward
- all terms except `tracking_lin_vel` and `tracking_ang_vel` are also added to the arm reward
- final reward uses the Ji22-style positive/negative composition:
  - `reward = reward_pos * exp(reward_neg / sigma_rew_neg)`, with `sigma_rew_neg = 0.02`

#### 3. Command Sampling, Filtering, Resampling, Frames

Locomotion command:

- sampled from the RoboDuet curriculum bins over
  - `x in [-1.0, 1.0]`
  - `y in [-0.6, 0.6]`
  - `yaw in [-1.0, 1.0]`
  - stage1-only body pitch/roll bins in `[-0.4, 0.4]`
- 10% of sampled velocity commands are forced to zero
- tiny commands are thresholded to zero:
  - `|x| <= 0.07`
  - `|y| <= 0.07`
  - `|yaw| <= 0.10`
- resampled every `10.0s` inside an episode, and also on reset

Arm command:

- sampled uniformly, with no extra reject-cuboid or IK/workspace filter:
  - `l in [0.3, 0.77]`
  - `p in [-0.45pi, 0.45pi]`
  - `y in [-pi/2, pi/2]`
  - `roll in [-0.45pi, 0.45pi]`
  - `pitch in [-60deg, 60deg]`
  - `yaw in [-75deg, 75deg]`
- resampled on reset and then every `T_traj ~ U(2.0, 3.0)s`
- arm resampling is disabled before the RoboDuet stage switch; stage1 exposes zero arm command to the actor

Frames:

- dog velocity commands are compared against base-frame linear/angular velocity
- arm `l/p/y` is expressed in a yaw-aligned base frame, with `z` measured relative to ground height
- arm orientation command/observation uses the upstream `abg` parameterization of the end-effector quaternion in the same yaw-aligned base frame

#### 4. Stage1 Arm Freeze and Network / Controller I/O

Upstream behavior:

- `keep_arm_fixed = True`
- when `switch_open == False`, upstream directly resets arm DOF position to default and arm DOF velocity to zero every physics step

Local IsaacLab port:

- the runner resolves the same stage-switch iteration as upstream
- before switch:
  - the arm PPO is not stepped
  - the environment executes zero arm actions
  - the joint action term forces joints `joint1..joint6` to use zero delta action
- because the Go2Arm action term is default-centered, zero arm delta means the arm controller target stays at the default arm posture

Network I/O:

- dog policy input:
  - `projected_gravity(3) + leg_joint_pos(12) + leg_joint_vel(12) + leg_action(12) + dog_cmd(5) + arm_cmd_obs(6) + roll_pitch(2) + clock(4) = 56`
- dog privileged input:
  - `2`
- dog output:
  - `12` leg action dims
- arm policy input:
  - `arm_joint_pos(6) + arm_action(6) + arm_cmd_obs(6) + roll_pitch(2) = 20`
- arm privileged input:
  - `9`
- arm output:
  - `8 = 6 arm joint-action dims + 2 plan-action dims`

Executed controller chain:

1. arm network outputs `8` dims
2. first `6` dims become arm joint delta actions
3. last `2` dims become planner outputs and are mapped to commanded body `pitch/roll`
4. the joint action term rescales all joint deltas by `0.25`
5. hip joint deltas are additionally multiplied by `0.5`
6. final joint-position target is `default_joint_pos + delta`
7. the robot arm actuator is `DelayedPDActuatorCfg(joint1..joint6)`, so the low-level controller receives joint position targets rather than torque commands

Play-mode note:

- `scripts/reinforcement_learning/rsl_rl/play.py` explicitly sets `fixed_delta_action_until_iteration = 0`, so playback does not re-enable the stage1 arm freeze.

## Citation

This repository is a modified version of `robot_lab`.
