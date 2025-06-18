# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Import base Isaac Lab MDP functions
import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
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
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotVelocityRewardsCfg(SpotRewardsCfg):
    """Spot velocity tracking simplified to match Go2 approach with boosted values."""

    def __post_init__(self):
        super().__post_init__()

        # KEEP and BOOST rewards that exist in both Go2 and Spot
        self.base_linear_velocity.weight = (
            25.0  # Boosted from default 5.0 (equivalent to track_lin_vel_xy_exp)
        )
        self.base_angular_velocity.weight = (
            15.0  # Boosted from default 5.0 (equivalent to track_ang_vel_z_exp)
        )
        # Note: air_time (default 5.0) is equivalent to feet_air_time - keeping default
        # Note: joint_torques, joint_acc equivalent to dof_torques_l2, dof_acc_l2

        # DISABLE Spot-specific rewards (not present in Go2)
        self.foot_clearance.weight = (
            0.0  # Disabled from default 0.5 (Go2 doesn't have this)
        )
        self.gait.weight = 0.0  # Disabled from default 10.0 (Go2 doesn't have this)
        self.action_smoothness.weight = (
            0.0  # Disabled from default -1.0 (Go2 uses action_rate_l2 instead)
        )
        self.air_time_variance.weight = (
            0.0  # Disabled from default -1.0 (Go2 doesn't have this)
        )
        self.base_motion.weight = 0.0  # Disabled from default -2.0 (Go2 uses separate lin_vel_z_l2, ang_vel_xy_l2)
        self.base_orientation.weight = (
            0.0  # Disabled from default -3.0 (Go2 doesn't have this)
        )
        self.foot_slip.weight = (
            0.0  # Disabled from default -0.5 (Go2 doesn't have this)
        )
        self.joint_pos.weight = (
            0.0  # Disabled from default -0.7 (Go2 doesn't have this)
        )
        self.joint_vel.weight = (
            0.0  # Disabled from default -0.01 (Go2 doesn't have this)
        )

        # Keep the joint penalties that exist in both (with your modified values)
        self.joint_torques.weight = (
            -2.0e-4  # Reduced from default -5.0e-4 (equivalent to dof_torques_l2)
        )
        self.joint_acc.weight = (
            -1.0e-4  # Same as default -1.0e-4 (equivalent to dof_acc_l2)
        )
        # Note: air_time weight=5.0 (keeping default, equivalent to feet_air_time)


@configclass
class SpotVelocityStepEnvCfg(SpotFlatEnvCfg):
    """Spot velocity tracking with cuboid obstacles - extends Isaac's flat config."""

    rewards: SpotRewardsCfg = SpotVelocityRewardsCfg()
    observations: SpotObservationsCfg = SpotObservationsCfg()

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
            self.scene.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/body",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
