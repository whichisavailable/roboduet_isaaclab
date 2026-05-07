# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""go2arm 任务默认使用的训练配置导出。"""

from .rsl_rl_ppo_cfg import (
    UnitreeGo2ArmFlatPPORunnerCfg,
    UnitreeGo2ArmTeacherRoughPPORunnerCfg,
)

__all__ = [
    "UnitreeGo2ArmFlatPPORunnerCfg",
    "UnitreeGo2ArmTeacherRoughPPORunnerCfg",
]
