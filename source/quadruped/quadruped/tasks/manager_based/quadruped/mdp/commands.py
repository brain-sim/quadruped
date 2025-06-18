"""Custom command generators for quadruped navigation tasks."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand


class Uniform3DVelocityCommand(UniformVelocityCommand):
    r"""Command generator that generates a 3D velocity command from uniform distribution.

    Extends the standard 2D velocity command to include linear velocity in the z-direction,
    enabling jumping and vertical movement tracking.

    The command comprises of:
    - Linear velocity in x, y, and z directions
    - Angular velocity around the z-axis

    It is given in the robot's base frame.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        """Initialize the 3D velocity command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        """
        # Initialize parent class
        super().__init__(cfg, env)

        # Extend command buffer to include z-velocity
        # -- command: x vel, y vel, z vel, yaw vel
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)

        # Update metrics to include z-velocity tracking
        self.metrics["error_vel_xyz"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_z"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the 3D velocity command generator."""
        msg = "Uniform3DVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += "\tIncludes z-velocity (jumping): True"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired 3D base velocity command in the base frame. Shape is (num_envs, 4)."""
        return self.vel_command_b

    def _update_metrics(self):
        """Update tracking metrics for 3D velocity."""
        # Time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        # Log 2D velocity error (xy)
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )

        # Log 3D velocity error (xyz)
        self.metrics["error_vel_xyz"] += (
            torch.norm(
                self.vel_command_b[:, :3] - self.robot.data.root_lin_vel_b[:, :3],
                dim=-1,
            )
            / max_command_step
        )

        # Log z-velocity error specifically
        self.metrics["error_vel_z"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_lin_vel_b[:, 2])
            / max_command_step
        )

        # Log angular velocity error
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 3] - self.robot.data.root_ang_vel_b[:, 2])
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample 3D velocity commands."""
        # Sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)

        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)

        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)

        # -- linear velocity - z direction (jumping)
        if hasattr(self.cfg.ranges, "lin_vel_z"):
            self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.lin_vel_z)
        else:
            # Default to no z-velocity if not configured
            self.vel_command_b[env_ids, 2] = 0.0

        # -- angular velocity - yaw (rotation around z)
        self.vel_command_b[env_ids, 3] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # Handle heading target if configured
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # Update heading envs
            self.is_heading_env[env_ids] = (
                r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )

        # Update standing envs
        self.is_standing_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

    def _update_command(self):
        """Post-process the 3D velocity command."""
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # Resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # Compute angular velocity
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 3] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _debug_vis_callback(self, event):
        """Debug visualization callback for 3D velocity commands."""
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return

        # Get marker location
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # Resolve the scales and quaternions for XY velocity visualization
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.command[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )

        # Display markers (existing XY visualization)
        if hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.visualize(
                base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale
            )
            self.current_vel_visualizer.visualize(
                base_pos_w, vel_arrow_quat, vel_arrow_scale
            )
