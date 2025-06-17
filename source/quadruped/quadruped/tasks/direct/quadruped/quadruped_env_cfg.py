# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg

# Import terrain configurations
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab_assets.robots.spot import SPOT_CFG


@configclass
class QuadrupedEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 12  # for quadruped
    observation_space = 48  # example, adjust as needed
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 250, render_interval=decimation)

    # robot(s)
    robot_type = "spot"  # options: 'spot', 'anymal', 'go1'
    robot_cfgs = {
        "spot": SPOT_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        "anymal": ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        "go1": ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot"),  # stand-in
    }
    robot_cfg: ArticulationCfg = robot_cfgs[robot_type]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=6.0, replicate_physics=True
    )

    # terrain - now using Isaac Lab's proper terrain system
    terrain_type = "flat"  # Switch back to rough terrain

    # Flat terrain configuration
    terrain_flat = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Rough terrain configuration with actual geometric variation
    terrain_rough = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,  # Lower than AnymalC for easier navigation
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # Select terrain based on terrain_type
    terrain = terrain_rough if terrain_type == "rough" else terrain_flat

    # Dense obstacle configuration for comprehensive training
    add_obstacles = True
    num_obstacles = 25  # Increased from 5 to fill the terrain
    obstacle_size = (0.2, 0.2, 0.2)  # Slightly smaller for more obstacles
    obstacle_height_range = (0.1, 0.4)  # Varied heights for climbing practice

    # Obstacle field parameters
    obstacle_area_size = 8.0  # 8m x 8m area filled with obstacles
    obstacle_grid_spacing = 1.5  # Average spacing between obstacles

    # - action scale
    action_scale = 1.0
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -10.0
    rew_scale_forward = 1.0
    rew_scale_energy = -0.01
    rew_scale_obstacle_cross = 10.0
