# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Spot velocity navigation environments - built on Isaac's Spot configs
gym.register(
    id="Spot-Velocity-Flat-Quadruped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:SpotVelocityFlatEnvCfg",
    },
)

gym.register(
    id="Spot-Velocity-Rough-Quadruped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:SpotVelocityRoughEnvCfg",  # Same as flat for now
    },
)

# Spot obstacle navigation environments - with cuboid obstacles
gym.register(
    id="Spot-Velocity-Flat-Obstacle-Quadruped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:SpotVelocityFlatObstacleEnvCfg",
    },
)

gym.register(
    id="Spot-Velocity-Rough-Obstacle-Quadruped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:SpotVelocityRoughObstacleEnvCfg",
    },
)

# Play versions for testing (with fewer environments and longer episodes)
gym.register(
    id="Spot-Velocity-Flat-Obstacle-Quadruped-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:SpotVelocityFlatObstacleEnvCfg_PLAY",
    },
)

gym.register(
    id="Spot-Velocity-Rough-Obstacle-Quadruped-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:SpotVelocityRoughObstacleEnvCfg_PLAY",
    },
)
