# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    print(object_pos_b.dtype)
    return object_pos_b


def is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    vis = 0

    try:
        new_lifted_envs = torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0).unsqueeze(1).float()
        old_lifted_envs = torch.where(env.observation_manager.group_obs_term_history_buffer["policy"]["is_lifted"].buffer[:, 0] == 1.0, 1.0, 0.0).float()
        print(new_lifted_envs.dtype, old_lifted_envs.dtype)
        vis = (new_lifted_envs > 0.0) | (old_lifted_envs > 0.0)
        vis = vis.float()
        print(vis.shape)
        print(vis)
        count_ones = torch.sum(vis, dim=1).sum()
        print(count_ones)
    except AttributeError:
        print("No obs manager yet.")
        count_ones = 0
        print(count_ones)
        vis = torch.zeros(4096, 1)
    return vis
        
        


    
