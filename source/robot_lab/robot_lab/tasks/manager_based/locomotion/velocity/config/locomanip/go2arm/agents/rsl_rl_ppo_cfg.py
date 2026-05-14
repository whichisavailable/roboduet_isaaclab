# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

_RUNNER_MODULE = (
    "robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.automatic_runner"
)
_MODEL_MODULE = (
    "robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.automatic_models"
)
_ALGORITHM_MODULE = (
    "robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.automatic_ppo"
)


@configclass
class RoboDuetAutomaticDogModelCfg:
    """RoboDuet `auto_train` 中 dog actor-critic 的配置。"""

    class_name: str = f"{_MODEL_MODULE}:DogActorCritic"
    history_length: int = 30
    num_actions: int = 12
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    adaptation_module_branch_hidden_dims: list[int] = [256, 128]


@configclass
class RoboDuetAutomaticArmModelCfg:
    """RoboDuet `auto_train` 中 arm actor-critic 的配置。"""

    class_name: str = f"{_MODEL_MODULE}:ArmActorCritic"
    history_length: int = 30
    num_actions: int = 6
    num_plan_actions: int = 2
    init_noise_std: float = 0.1
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    adaptation_module_branch_hidden_dims: list[int] = [256, 128]


@configclass
class RoboDuetAutomaticPpoAlgorithmCfg:
    """RoboDuet `auto_train` 的单策略 PPO 配置。"""

    class_name: str = f"{_ALGORITHM_MODULE}:AutomaticPPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 5.0e-4
    adaptation_module_learning_rate: float = 5.0e-4
    num_adaptation_module_substeps: int = 1
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    selective_adaptation_module_loss: bool = False
    rnd_cfg = None


@configclass
class UnitreeGo2ArmTeacherRoughPPORunnerCfg(RslRlBaseRunnerCfg):
    """go2arm 默认训练入口，严格对齐 Roboduet `auto_train`。"""

    class_name: str = f"{_RUNNER_MODULE}:RoboDuetAutomaticRunner"
    num_steps_per_env: int = 24
    max_iterations: int = 100000
    save_interval: int = 400
    seed: int = 42
    empirical_normalization: bool = False
    experiment_name: str = "roboduet_go2arm_rough"
    run_name: str = ""
    clip_actions: float | None = 10.0
    obs_groups: dict[str, list[str]] = {
        "dog_policy": ["dog_policy"],
        "dog_privileged": ["dog_privileged"],
        "arm_policy": ["arm_policy"],
        "arm_privileged": ["arm_privileged"],
    }
    algorithm: RoboDuetAutomaticPpoAlgorithmCfg = RoboDuetAutomaticPpoAlgorithmCfg()
    dog_model: RoboDuetAutomaticDogModelCfg = RoboDuetAutomaticDogModelCfg()
    arm_model: RoboDuetAutomaticArmModelCfg = RoboDuetAutomaticArmModelCfg()
    roboduet_pretrained_dog_checkpoint: str | None = None
    roboduet_pretrained_arm_checkpoint: str | None = None
    roboduet_stage_switch_iteration: int | None = None
    roboduet_disable_two_stage: bool = False
    roboduet_export_deploy_models: bool = True
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"


@configclass
class UnitreeGo2ArmFlatPPORunnerCfg(UnitreeGo2ArmTeacherRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "roboduet_go2arm_flat"
