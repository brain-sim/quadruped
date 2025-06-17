# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg

# Import base Isaac Lab MDP functions
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Import Isaac's Spot configuration to extend it
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotFlatEnvCfg,
)

# Import Isaac's Spot-specific MDP functions (for extension)
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import *  # noqa: F401, F403

# ⭐ THIS IS THE MISSING IMPORT ⭐
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

from . import mdp as local_mdp

##
# Template Quadruped Environment (keeping for reference)
##


@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """Configuration for a basic quadruped scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot - placeholder for template
    robot: ArticulationCfg = None

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class QuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    """Template quadruped environment configuration."""

    # Scene settings
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(num_envs=4096, env_spacing=4.0)

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer = ViewerCfg(
            eye=(3.0, 3.0, 3.0),  # Higher camera position
            lookat=(0.0, 0.0, 0.0),  # Look at ground level
            origin_type="world",
            env_index=0,
        )
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


##
# Spot Obstacle Navigation Configurations
##


@configclass
class SpotObstacleRewardsCfg:
    """Additional rewards for obstacle navigation."""

    # Obstacle climbing reward
    obstacle_clearance = RewTerm(
        func=local_mdp.obstacle_clearance_reward,
        weight=5.0,
        params={
            "height_threshold": 0.15,  # minimum height above obstacle to get reward
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_cfg": SceneEntityCfg("obstacle"),
        },
    )

    # Reward for moving forward after climbing
    forward_progress = RewTerm(
        func=local_mdp.forward_progress_reward,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class SpotVelocityFlatNavEnvCfg(SpotFlatEnvCfg):
    """Spot velocity tracking on flat terrain - minimal extension of Isaac's config."""

    def __post_init__(self):
        # Initialize parent configuration
        super().__post_init__()

        # Adjust episode length for navigation
        self.episode_length_s = 30.0

        # Increase environment spacing for obstacles
        self.scene.num_envs = 1024
        self.scene.env_spacing = 8.0


@configclass
class SpotVelocityFlatObstacleNavEnvCfg(SpotVelocityFlatNavEnvCfg):
    """Spot velocity tracking with cuboid obstacles - extends Isaac's flat config."""

    def __post_init__(self):
        # Initialize parent configuration first
        super().__post_init__()

        # Import Spot config

        # Set robot to spawn at origin but at proper height above ground
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.3)  # 0.6m above ground
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Set joint positions for standing pose
        self.scene.robot.init_state.joint_pos = {
            "fl_hx": 0.0,
            "fl_hy": 0.9,
            "fl_kn": -1.8,
            "fr_hx": 0.0,
            "fr_hy": 0.9,
            "fr_kn": -1.8,
            "hl_hx": 0.0,
            "hl_hy": 0.9,
            "hl_kn": -1.8,
            "hr_hx": 0.0,
            "hr_hy": 0.9,
            "hr_kn": -1.8,
        }

        # Create MULTIPLE obstacles using RigidObjectCollectionCfg

        self.scene.obstacles = RigidObjectCollectionCfg(
            rigid_objects={
                f"obstacle_{i}": RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.4, 0.8, 0.5),  # width, length, height
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,  # Disable gravity to keep them stationary
                            kinematic_enabled=True,  # Make them kinematic (user-controlled, not physics-controlled)
                            sleep_threshold=0.0,  # Set to 0 for kinematic bodies
                            stabilization_threshold=0.0,  # Set to 0 for kinematic bodies
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(
                            mass=5000.0,  # 5000 kg - very heavy obstacle!
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                        ),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            static_friction=1.5,  # High friction to prevent sliding
                            dynamic_friction=1.2,  # High dynamic friction
                            restitution=0.1,  # Low bounce
                            friction_combine_mode="max",  # Use maximum friction
                            restitution_combine_mode="min",  # Use minimum restitution
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.8, 0.2 + i * 0.1, 0.2),  # Different colors
                            metallic=0.3,
                            roughness=0.8,  # Make them look heavy/concrete-like
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(
                            2.0 + i * 1.5,
                            (-1) ** i * (i * 0.8),
                            0.25,  # Half the height (0.5/2) to place bottom on ground
                        ),  # Spread them out
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                )
                for i in range(10)  # Create 5 obstacles
            }
        )

        # Randomize multiple obstacle positions
        self.events.randomize_obstacles = EventTerm(
            func=local_mdp.randomize_multiple_obstacles,
            mode="reset",
            params={
                "obstacle_cfg": SceneEntityCfg("obstacles"),
                "position_range": {"x": (2.0, 10.0), "y": (-4.0, 4.0)},
                "min_spacing": 1.2,
            },
        )

        # Update rewards for multiple obstacles
        self.rewards.obstacle_clearance = RewTerm(
            func=local_mdp.multi_obstacle_clearance_reward,
            weight=5.0,
            params={
                "height_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_cfg": SceneEntityCfg("obstacles"),
            },
        )

        self.rewards.forward_progress = RewTerm(
            func=local_mdp.forward_progress_reward,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Update viewer
        self.viewer = ViewerCfg(
            eye=(8.0, 8.0, 4.0),
            lookat=(0.0, 0.0, 0.5),
            origin_type="world",
            env_index=0,
            asset_name="robot",
        )


@configclass
class SpotVelocityRoughObstacleNavEnvCfg(SpotVelocityFlatObstacleNavEnvCfg):
    """Spot velocity tracking with obstacles on rough terrain."""

    def __post_init__(self):
        # Initialize parent configuration
        super().__post_init__()

        # Use rough terrain from Isaac's config but keep obstacle additions
        # The terrain is already set up by SpotFlatEnvCfg, we just need to make it rougher
        if hasattr(self.scene, "terrain") and hasattr(
            self.scene.terrain, "terrain_generator"
        ):
            # Increase terrain roughness
            terrain_gen = self.scene.terrain.terrain_generator
            if hasattr(terrain_gen, "sub_terrains"):
                if "random_rough" in terrain_gen.sub_terrains:
                    terrain_gen.sub_terrains["random_rough"].noise_range = (0.03, 0.08)
                    terrain_gen.sub_terrains["random_rough"].proportion = 0.4


##
# Play Configurations for Testing
##


@configclass
class SpotVelocityFlatObstacleNavEnvCfg_PLAY(SpotVelocityFlatObstacleNavEnvCfg):
    """Play configuration for Spot flat obstacle navigation - optimized for testing."""

    def __post_init__(self):
        # Initialize parent configuration
        super().__post_init__()

        # Reduce number of environments for better visualization
        self.scene.num_envs = 16
        self.scene.env_spacing = 10.0

        # Longer episodes for testing
        self.episode_length_s = 60.0

        # Reduce obstacle randomization for more predictable testing
        if hasattr(self.events, "randomize_obstacle"):
            self.events.randomize_obstacle.params["num_obstacles_range"] = (2, 4)


@configclass
class SpotVelocityRoughObstacleNavEnvCfg_PLAY(SpotVelocityRoughObstacleNavEnvCfg):
    """Play configuration for Spot rough obstacle navigation - optimized for testing."""

    def __post_init__(self):
        # Initialize parent configuration
        super().__post_init__()

        # Reduce number of environments for better visualization
        self.scene.num_envs = 16
        self.scene.env_spacing = 10.0

        # Longer episodes for testing
        self.episode_length_s = 60.0

        # Reduce obstacle randomization for more predictable testing
        if hasattr(self.events, "randomize_obstacle"):
            self.events.randomize_obstacle.params["num_obstacles_range"] = (2, 4)
