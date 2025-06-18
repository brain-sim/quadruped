# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Spot step navigation environments - with cuboid steps
gym.register(
    id="Spot-Velocity-Step-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:SpotVelocityStepEnvCfg",
    },
)


# Play versions for testing (with fewer environments and longer episodes)
gym.register(
    id="Spot-Velocity-Step-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:SpotVelocityStepEnvCfg_PLAY",
    },
)
