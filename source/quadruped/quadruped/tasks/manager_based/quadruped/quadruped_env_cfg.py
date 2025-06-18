# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Import base Isaac Lab MDP functions
import isaaclab.terrains as terrain_gen
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (  # noqa: F401, F403
    SpotFlatEnvCfg,
    SpotRewardsCfg,
    SpotTerminationsCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp import *  # noqa: F401, F403

from .mdp import *  # noqa: F401, F403


@configclass
class SpotVelocityTerminationCfg(SpotTerminationsCfg):
    """Spot velocity tracking on flat terrain - minimal extension of Isaac's config."""

    random_steps = DoneTerm(
        func=random_termination,
        time_out=True,
    )


@configclass
class SpotVelocityRewardsCfg(SpotRewardsCfg):
    """Spot velocity tracking on flat terrain - minimal extension of Isaac's config."""

    # body_contact = RewardTermCfg(
    #     func=body_contact_penalty,
    #     weight=-1.0e-3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[".*leg"],
    #         ),
    #         "threshold": 1.0,
    #     },
    # )

    def __post_init__(self):
        super().__post_init__()
        self.base_motion.weight = -3.0  # -2.0
        self.base_orientation.weight = -4.0  # -3.0
        self.foot_slip.weight = -1.0  # -0.5
        self.gait.weight = 15.0  # 10.0
        self.base_linear_velocity.weight = 10.0  # 5.0
        self.base_angular_velocity.weight = 7.5  # 5.0
        self.joint_acc.weight = -5.0e-4  # -1.0e-4
        self.joint_pos.weight = -1.0  # -0.7
        self.joint_vel.weight = -5.0e-2  # -1.0e-2
        self.joint_torques.weight = -1.0e-3  # -5.0e-4
        self.foot_clearance.weight = 2.0  # 0.5
        # self.body_contact.weight = -1.0e-4  # -1.0e-3


@configclass
class SpotVelocityStepEnvCfg(SpotFlatEnvCfg):
    """Spot velocity tracking with cuboid obstacles - extends Isaac's flat config."""

    rewards: SpotRewardsCfg = SpotVelocityRewardsCfg()
    # terminations: SpotTerminationsCfg = SpotVelocityTerminationCfg()

    def __post_init__(self):
        # Initialize parent configuration first
        super().__post_init__()

        if hasattr(self.scene, "terrain") and hasattr(
            self.scene.terrain, "terrain_generator"
        ):
            # Get existing sub_terrains or create new dict
            existing_sub_terrains = getattr(
                self.scene.terrain.terrain_generator, "sub_terrains", {}
            )

            # Add step terrain configuration
            step_terrain = {
                "random_steps": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=0.2,  # 30% of terrain will be steps
                    grid_width=0.45,  # Width of each step
                    grid_height_range=(0.05, 0.15),  # Step heights from 5cm to 25cm
                    platform_width=2.0,  # Platform width between steps
                ),
                "pyramid_steps": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.2,  # 30% of terrain will be steps
                    step_width=0.45,  # Width of each step
                    step_height_range=(0.05, 0.20),  # Step heights from 5cm to 10cm
                    platform_width=2.0,  # Platform width between steps
                ),
                "inverted_pyramid_steps": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.2,  # 30% of terrain will be steps
                    step_width=0.45,  # Width of each step
                    step_height_range=(0.05, 0.20),  # Step heights from 5cm to 10cm
                    platform_width=2.0,  # Platform width between steps
                ),
            }

            # Merge with existing sub_terrains
            updated_sub_terrains = {**existing_sub_terrains, **step_terrain}

            # Update the terrain generator
            self.scene.terrain.terrain_generator.sub_terrains = updated_sub_terrains
