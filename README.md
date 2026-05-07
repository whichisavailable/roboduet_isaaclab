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





## Citation

This repository is a modified version of `robot_lab`.
