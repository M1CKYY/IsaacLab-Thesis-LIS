# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms
import numpy as np

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]


    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return -object_ee_distance

def object_goal_frame_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_frame_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Reward the agent for the distance between the object and the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    goal_frame = env.scene[goal_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # print(cube_pos_w.shape)
    # # End-effector position: (num_envs, 3)
    # print(dir(goal_frame))
    # print(vars(goal_frame))
    # goal_frame_w = goal_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    # # hard coded goal_frame position
    frame_pos_w = goal_frame.data.root_pos_w
    object_goal_frame_vector = frame_pos_w - cube_pos_w
    object_goal_frame_distance = torch.norm(object_goal_frame_vector, dim=1)

    def smooth_signed_log(x, eps=1e-6):
        return torch.sign(x) * torch.log1p(torch.abs(x) + eps)

    def fast_growth(x, k=3.0, p=1.0):
        return torch.sign(x) * (torch.exp(k * torch.abs(x) ** p) - 1) / (torch.exp(torch.tensor(k)) - 1)

    reward = smooth_signed_log((torch.tensordot(object.data.root_lin_vel_w, object_goal_frame_vector)) * torch.norm(object.data.root_lin_vel_w))


    return torch.where(object_goal_frame_distance > 0.3, reward, 4)

def object_fail(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for the distance between the object and the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] < 0.6, -4, 0)
