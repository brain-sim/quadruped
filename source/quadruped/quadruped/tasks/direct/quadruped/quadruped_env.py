# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.shapes import spawn_cuboid
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg

from .quadruped_env_cfg import QuadrupedEnvCfg


class QuadrupedEnv(DirectRLEnv):
    cfg: QuadrupedEnvCfg

    def __init__(self, cfg: QuadrupedEnvCfg, render_mode: str | None = None, **kwargs):
        # For obstacle reward
        self.obstacle_positions = []  # Will be filled in _setup_scene
        self.obstacle_crossed_buf = None  # Will be initialized in _setup_scene

        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # Setup terrain using Isaac Lab's terrain system FIRST
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Setup robot config based on type AFTER terrain
        self.robot_cfg = self.cfg.robot_cfgs[self.cfg.robot_type]
        self.robot = Articulation(self.robot_cfg)

        # Obstacles (optional, on top of terrain)
        self.obstacle_positions = []
        if self.cfg.add_obstacles:
            import numpy as np

            print(
                f"[DEBUG]: Creating dense obstacle field with {self.cfg.num_obstacles} obstacles"
            )

            # Create a systematic grid-based obstacle field with some randomness
            area_size = self.cfg.obstacle_area_size
            grid_spacing = self.cfg.obstacle_grid_spacing

            # Calculate grid dimensions
            grid_size = int(area_size / grid_spacing)
            total_grid_points = grid_size * grid_size

            # Create obstacles in a grid pattern with randomness
            obstacles_created = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if obstacles_created >= self.cfg.num_obstacles:
                        break

                    # Grid position with some random offset
                    base_x = (i - grid_size // 2) * grid_spacing
                    base_y = (j - grid_size // 2) * grid_spacing

                    # Add random offset within grid cell
                    x = base_x + np.random.uniform(
                        -grid_spacing * 0.3, grid_spacing * 0.3
                    )
                    y = base_y + np.random.uniform(
                        -grid_spacing * 0.3, grid_spacing * 0.3
                    )

                    # Random height for variety
                    height = np.random.uniform(*self.cfg.obstacle_height_range)
                    size = self.cfg.obstacle_size

                    # Position obstacle
                    pos = np.array([x, y, height / 2])

                    # Create cuboid configuration with varied properties
                    cuboid_cfg = CuboidCfg(
                        size=(size[0], size[1], height),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(
                            mass=np.random.uniform(0.5, 2.0)
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                    )

                    spawn_cuboid(
                        prim_path=f"/World/obstacle_{obstacles_created}",
                        cfg=cuboid_cfg,
                        translation=(pos[0], pos[1], pos[2]),
                    )
                    self.obstacle_positions.append(pos.copy())
                    obstacles_created += 1

                if obstacles_created >= self.cfg.num_obstacles:
                    break

            # Add some additional random obstacles in the forward path
            forward_obstacles = min(10, self.cfg.num_obstacles - obstacles_created)
            for i in range(forward_obstacles):
                # Create obstacles specifically in the robot's forward path
                x = np.random.uniform(1.0, 6.0)  # Forward of robot spawn
                y = np.random.uniform(-2.0, 2.0)  # Within robot's path width
                height = np.random.uniform(*self.cfg.obstacle_height_range)
                size = self.cfg.obstacle_size

                pos = np.array([x, y, height / 2])

                cuboid_cfg = CuboidCfg(
                    size=(size[0], size[1], height),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=np.random.uniform(0.5, 2.0)
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                )

                spawn_cuboid(
                    prim_path=f"/World/obstacle_{obstacles_created}",
                    cfg=cuboid_cfg,
                    translation=(pos[0], pos[1], pos[2]),
                )
                self.obstacle_positions.append(pos.copy())
                obstacles_created += 1

            print(
                f"[DEBUG]: Created {obstacles_created} obstacles in dense field pattern"
            )

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Initialize buffer for obstacle crossing (per env, per obstacle)
        num_envs = self.scene.num_envs if hasattr(self.scene, "num_envs") else 1
        # Get device from simulation or robot data
        device = "cuda:0"  # Default fallback
        if hasattr(self, "device"):
            device = self.device
        elif hasattr(self.sim, "device"):
            device = self.sim.device
        elif hasattr(self.cfg.sim, "device"):
            device = self.cfg.sim.device

        self.obstacle_crossed_buf = torch.zeros(
            (num_envs, len(self.obstacle_positions)), dtype=torch.bool, device=device
        )
        print("[DEBUG]: Scene setup complete")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions * self.cfg.action_scale)

    def _get_observations(self) -> dict:
        # Example: return joint positions and velocities
        obs = torch.cat((self.robot.data.joint_pos, self.robot.data.joint_vel), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Enhanced reward system for obstacle navigation

        # Basic locomotion rewards
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w
        root_pos = self.robot.data.root_pos_w

        # Forward velocity reward (primary objective)
        forward_vel = root_lin_vel[:, 0]  # X-axis velocity
        forward_reward = self.cfg.rew_scale_forward * torch.clamp(forward_vel, 0.0, 2.0)

        # Penalize lateral and vertical velocities (encourage straight movement)
        lateral_penalty = -0.2 * torch.abs(root_lin_vel[:, 1])  # Y-axis penalty
        vertical_penalty = -0.1 * torch.abs(root_lin_vel[:, 2])  # Z-axis penalty

        # Penalize excessive angular velocities (encourage stable movement)
        ang_vel_penalty = -0.1 * torch.sum(torch.abs(root_ang_vel), dim=-1)

        # Alive reward (basic survival)
        alive_reward = self.cfg.rew_scale_alive * torch.ones_like(forward_reward)

        # Energy efficiency penalty
        energy_penalty = self.cfg.rew_scale_energy * torch.sum(
            self.robot.data.joint_vel**2, dim=-1
        )

        # Height maintenance reward (encourage staying upright)
        target_height = 0.3  # Approximate robot standing height
        height_diff = torch.abs(root_pos[:, 2] - target_height)
        height_reward = -2.0 * height_diff

        # Orientation reward (encourage keeping robot upright)
        # This encourages the robot to maintain proper orientation
        roll_pitch_penalty = -5.0 * (
            torch.abs(self.robot.data.root_quat_w[:, 1])  # Roll
            + torch.abs(self.robot.data.root_quat_w[:, 2])  # Pitch
        )

        # Combine basic rewards
        total_reward = (
            forward_reward
            + lateral_penalty
            + vertical_penalty
            + ang_vel_penalty
            + alive_reward
            + energy_penalty
            + height_reward
            + roll_pitch_penalty
        )

        # Enhanced obstacle crossing rewards for dense obstacle field
        if self.cfg.add_obstacles and len(self.obstacle_positions) > 0:
            root_x = root_pos[:, 0]  # Current X position
            device = root_x.device

            # Track progress through the obstacle field
            max_x_progress = root_x.max() if root_x.numel() > 0 else 0.0
            progress_reward = 0.5 * max_x_progress  # Reward forward progress

            # Count nearby obstacles for navigation difficulty bonus
            nearby_obstacles = 0
            total_approach_reward = torch.zeros_like(root_x)
            total_climbing_reward = torch.zeros_like(root_x)
            total_crossing_bonus = torch.zeros_like(root_x)

            for i, obs_pos in enumerate(self.obstacle_positions):
                obs_x = torch.tensor(obs_pos[0], device=device, dtype=torch.float32)
                obs_y = torch.tensor(obs_pos[1], device=device, dtype=torch.float32)
                obs_z = torch.tensor(obs_pos[2], device=device, dtype=torch.float32)

                # Distance to obstacle (3D)
                robot_y = root_pos[:, 1]
                distance_x = torch.abs(root_x - obs_x)
                distance_y = torch.abs(robot_y - obs_y)
                distance_3d = torch.sqrt(distance_x**2 + distance_y**2)

                # Check if robot has crossed obstacle
                not_crossed = ~self.obstacle_crossed_buf[:, i]
                crossed = (
                    root_x > obs_x + 0.3
                ) & not_crossed  # Smaller buffer for dense field

                # Approach reward - encourage moving toward obstacles
                approach_distance_threshold = 3.0
                approach_reward = torch.where(
                    distance_3d < approach_distance_threshold,
                    (approach_distance_threshold - distance_3d)
                    * 0.5,  # Reduced scale for dense field
                    torch.zeros_like(distance_3d),
                )
                total_approach_reward += approach_reward

                # Navigation bonus for being near multiple obstacles (complexity reward)
                if distance_3d.min() < 1.5:  # Within 1.5m
                    nearby_obstacles += 1

                # Climbing reward - reward height when near obstacles
                near_obstacle = distance_3d < 1.0  # Within 1m of obstacle
                current_height = root_pos[:, 2]
                climbing_reward = torch.where(
                    near_obstacle & (current_height > target_height + 0.05),
                    3.0
                    * (current_height - target_height),  # Reduced but still significant
                    torch.zeros_like(current_height),
                )
                total_climbing_reward += climbing_reward

                # Crossing bonus - major reward for successful navigation
                cross_reward = 5.0  # Reduced from config value for dense field
                crossing_bonus = crossed.float() * cross_reward
                total_crossing_bonus += crossing_bonus

                # Update crossing buffer
                self.obstacle_crossed_buf[:, i] |= crossed

            # Complexity navigation bonus
            complexity_bonus = min(nearby_obstacles * 1.0, 5.0)  # Cap at 5.0

            # Speed bonus for maintaining velocity in complex terrain
            speed_bonus = torch.where(
                forward_vel > 0.5,  # Moving forward at reasonable speed
                2.0 * forward_vel,
                torch.zeros_like(forward_vel),
            )

            # Add all obstacle-related rewards
            total_reward = (
                total_reward
                + total_approach_reward
                + total_climbing_reward
                + total_crossing_bonus
                + complexity_bonus
                + speed_bonus
            )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Example: done if robot falls (z < threshold)
        z = self.robot.data.root_pos_w[:, 2]
        terminated = z < 0.2
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]

        # Important: Use terrain origins for proper positioning with rough terrain
        if hasattr(self, "_terrain") and hasattr(self._terrain, "env_origins"):
            # For rough terrain, use terrain origins which include height variations
            terrain_origins = self._terrain.env_origins[env_ids]
            default_root_state[:, :3] += terrain_origins
            # Add extra height offset to ensure robot spawns above terrain surface
            # default_root_state[:, 2] += 0.5  # Add 0.5m above terrain surface
        else:
            # Fallback to scene origins for flat terrain
            default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.data.joint_pos[env_ids] = joint_pos
        self.robot.data.joint_vel[env_ids] = joint_vel
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Reset obstacle crossed buffer for these envs
        if self.obstacle_crossed_buf is not None:
            # Ensure the buffer is on the same device as robot data
            device = self.robot.data.root_pos_w.device
            if self.obstacle_crossed_buf.device != device:
                self.obstacle_crossed_buf = self.obstacle_crossed_buf.to(device)
            self.obstacle_crossed_buf[env_ids, :] = False


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(
        torch.square(pole_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(
        torch.abs(cart_vel).unsqueeze(dim=1), dim=-1
    )
    rew_pole_vel = rew_scale_pole_vel * torch.sum(
        torch.abs(pole_vel).unsqueeze(dim=1), dim=-1
    )
    total_reward = (
        rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    )
    return total_reward
