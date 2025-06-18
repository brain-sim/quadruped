# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations or rewards."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


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
