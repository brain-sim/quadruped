# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations or rewards."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def zero_base_linear_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Penalty for non-zero base linear velocity when the command is zero."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm(
        (target - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    lin_vel_error = torch.where(
        (target == 0).all(dim=1),
        lin_vel_error,
        torch.zeros_like(lin_vel_error),
    )
    return torch.exp(-lin_vel_error / std) * lin_vel_error


def body_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for body contact if it contacts the ground."""
    # Get body contact forces from the scene data
    body_contact_forces = env.scene[sensor_cfg.name].data.net_forces_w[..., -1]
    # Apply penalty only when contact forces exceed threshold
    body_contact_forces_thresholded = torch.where(
        body_contact_forces >= threshold,
        body_contact_forces,
        torch.zeros_like(body_contact_forces),
    )
    return torch.sum(body_contact_forces_thresholded, dim=-1)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm(
        (target - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)
