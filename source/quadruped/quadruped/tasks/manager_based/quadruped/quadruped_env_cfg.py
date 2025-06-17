# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import (
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)

# Import base Isaac Lab MDP functions
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# Import Isaac's Spot configuration to extend it
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (
    SpotFlatEnvCfg,
    SpotRewardsCfg,
)

# Import Isaac's Spot-specific MDP functions (for extension)
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import *  # noqa: F401, F403

from . import mdp as local_mdp


@configclass
class SpotObstacleRewardsCfg(SpotRewardsCfg):
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


@configclass
class SpotMultiObstacleRewardsCfg(SpotRewardsCfg):
    """Additional events for obstacle navigation."""

    # Obstacle climbing reward
    obstacle_clearance = RewTerm(
        func=local_mdp.multi_obstacle_clearance_reward,
        weight=5.0,
        params={
            "height_threshold": 0.15,  # minimum height above obstacle to get reward
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_cfg": SceneEntityCfg("obstacle"),
        },
    )


@configclass
class SpotVelocityFlatEnvCfg(SpotFlatEnvCfg):
    """Spot velocity tracking on flat terrain - minimal extension of Isaac's config."""


@configclass
class SpotVelocityRoughEnvCfg(SpotFlatEnvCfg):
    """Spot velocity tracking on flat terrain with random termination."""

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


@configclass
class SpotVelocityFlatObstacleEnvCfg(SpotVelocityFlatEnvCfg):
    """Spot velocity tracking with cuboid obstacles - extends Isaac's flat config."""

    rewards: SpotMultiObstacleRewardsCfg = SpotMultiObstacleRewardsCfg()

    def __post_init__(self):
        # Initialize parent configuration first
        super().__post_init__()

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


@configclass
class SpotVelocityRoughObstacleEnvCfg(SpotVelocityFlatObstacleEnvCfg):
    """Spot velocity tracking with obstacles on rough terrain."""

    rewards: SpotMultiObstacleRewardsCfg = SpotMultiObstacleRewardsCfg()

    def __post_init__(self):
        # Initialize parent configuration first
        super().__post_init__()

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


##
# Play Configurations for Testing
##


@configclass
class SpotVelocityFlatObstacleNavEnvCfg_PLAY(SpotVelocityFlatObstacleEnvCfg):
    """Play configuration for Spot flat obstacle navigation - optimized for testing."""

    def __post_init__(self):
        # Initialize parent configuration
        super().__post_init__()

        # Reduce number of environments for better visualization
        self.scene.num_envs = 16
        self.scene.env_spacing = 10.0

        # Longer episodes for testing
        self.episode_length_s = 30.0

        # Reduce obstacle randomization for more predictable testing
        if hasattr(self.events, "randomize_obstacle"):
            self.events.randomize_obstacle.params["num_obstacles_range"] = (2, 4)


@configclass
class SpotVelocityRoughObstacleEnvCfg_PLAY(SpotVelocityRoughObstacleEnvCfg):
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
