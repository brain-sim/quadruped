"""Configuration classes for custom command generators."""

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass
from omegaconf import MISSING

from .commands import Uniform3DVelocityCommand


@configclass
class Uniform3DVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the 3D uniform velocity command generator."""

    class_type: type = Uniform3DVelocityCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the 3D velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        lin_vel_z: tuple[float, float] = MISSING
        """Range for the linear-z velocity command (in m/s). This enables jumping."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~Uniform3DVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the 3D velocity commands."""
