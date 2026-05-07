# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""注册 Roboduet go2arm 的 flat / rough 环境。"""

import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-Flat-Go2Arm-v0",
    entry_point=f"{__name__}.env:Go2ArmManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2ArmFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArmFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Rough-Go2Arm-v0",
    entry_point=f"{__name__}.env:Go2ArmManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2ArmRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArmTeacherRoughPPORunnerCfg",
    },
)
