# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Import base Isaac Lab MDP functions
import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.envs import ManagerBasedEnv, mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns

# Replace the terrain configuration completely
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass, math
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import (  # noqa: F401, F403
    SpotFlatEnvCfg,
)

from .mdp import *  # noqa: F401, F403

__all__ = [
    "SpotFlatEnvCfg",
    "SpotVelocityRoughEnvCfg",
    "SpotFlatEnvCfgv2",
]

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.03, 0.10), noise_step=0.03, border_width=0.25
        ),
    },
)


@configclass
class SpotFlatEnvCfgv2(SpotFlatEnvCfg):
    """Spot velocity tracking with custom terrain pattern and center spawning."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_linear_velocity = RewTerm(
            func=base_linear_velocity_reward,
            weight=5.0,
            params={"std": math.sqrt(1.0), "asset_cfg": SceneEntityCfg("robot")},
        )

        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 4.5)

        self.scene.terrain.terrain_generator = COBBLESTONE_ROAD_CFG


# @configclass
# class SpotVelocityRewardsCfg(SpotRewardsCfg):
#     """Spot velocity tracking simplified to match Go2 approach with boosted values."""

#     def __post_init__(self):
#         super().__post_init__()

#         # KEEP and BOOST rewards that exist in both Go2 and Spot
#         self.base_linear_velocity.weight = (
#             5.0  # Boosted from default 5.0 (equivalent to track_lin_vel_xy_exp)
#         )
#         self.base_angular_velocity.weight = (
#             5.0  # Boosted from default 5.0 (equivalent to track_ang_vel_z_exp)
#         )
#         # Note: air_time (default 5.0) is equivalent to feet_air_time - keeping default
#         # Note: joint_torques, joint_acc equivalent to dof_torques_l2, dof_acc_l2
#         self.air_time.weight = (
#             5.0  # Boosted from default 5.0 (equivalent to track_air_time_exp)
#         )

#         # DISABLE Spot-specific rewards (not present in Go2)
#         self.foot_clearance.weight = (
#             0.5  # Disabled from default 0.5 (Go2 doesn't have this)
#         )
#         self.gait.weight = 10.0  # Disabled from default 10.0 (Go2 doesn't have this)
#         self.action_smoothness.weight = (
#             -1.0  # Disabled from default -1.0 (Go2 uses action_rate_l2 instead)
#         )
#         self.air_time_variance.weight = (
#             -1.0  # Disabled from default -1.0 (Go2 doesn't have this)
#         )
#         self.foot_slip.weight = (
#             -0.5  # Disabled from default -0.5 (Go2 doesn't have this)
#         )
#         self.joint_pos.weight = (
#             -0.7  # Disabled from default -0.7 (Go2 doesn't have this)
#         )
#         self.joint_vel.weight = (
#             0.01  # Disabled from default -0.01 (Go2 doesn't have this)
#         )

#         # Keep the joint penalties that exist in both (with your modified values)
#         self.joint_torques.weight = (
#             -5.0e-4  # Reduced from default -5.0e-4 (equivalent to dof_torques_l2)
#         )
#         self.joint_acc.weight = (
#             -1.0e-4  # Same as default -1.0e-4 (equivalent to dof_acc_l2)
#         )
#         # Note: air_time weight=5.0 (keeping default, equivalent to feet_air_time)

#         self.base_motion.weight = (
#             -2.0
#         )  # Disabled from default -2.0 (Go2 uses separate lin_vel_z_l2, ang_vel_xy_l2)
#         self.base_orientation.weight = (
#             -3.0  # Disabled from default -3.0 (Go2 doesn't have this)
#         )


def height_scan(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return (
        sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    )


@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=velocity_mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=velocity_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]),
            "threshold": 1.0,
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=velocity_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
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
            func=custom_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotVelocityRoughEnvCfg(SpotFlatEnvCfg):
    """Spot velocity tracking with custom terrain pattern and center spawning."""

    # rewards: SpotRewardsCfg = SpotVelocityRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    observations: SpotObservationsCfg = SpotObservationsCfg()

    def __post_init__(self):
        # Initialize parent configuration first
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.5,  # 0.5m spacing between points
                size=[1.0, 1.0],  # 1m x 1m grid area
                direction=(0.0, 0.0, -1.0),  # Point directly downward
            ),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
            max_distance=50.0,  # Maximum detection distance
        )

        # Update sensor period to match simulation
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Create our custom terrain mix for the pattern
        custom_terrain_cfg = ROUGH_TERRAINS_CFG.copy()
        custom_terrain_cfg.sub_terrains = {
            "pyramid_steps": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2,  # P terrain
                step_width=0.45,
                step_height_range=(0.05, 0.15),
                platform_width=3.0,  # Each sub-terrain is 3x3
            ),
            "inverted_pyramid_steps": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,  # I terrain
                step_width=0.45,
                step_height_range=(0.05, 0.15),
                platform_width=3.0,  # Each sub-terrain is 3x3
            ),
            "random_steps": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2,  # R terrain
                grid_width=0.45,
                grid_height_range=(0.05, 0.10),
                platform_width=3.0,  # Each sub-terrain is 3x3
            ),
            "rough_terrain": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2,  # R terrain
                noise_range=(0.02, 0.05),
                noise_step=0.02,
                border_width=0.25,
            ),
        }

        # Replace the terrain generator and disable debug visualization
        if hasattr(self.scene, "terrain"):
            self.scene.terrain.terrain_generator = custom_terrain_cfg
            self.scene.terrain.debug_vis = False
