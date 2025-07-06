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
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height, if it hasn't been lifted yet."""
    object: RigidObject = env.scene[object_cfg.name]

    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def close_gripper_near_object(
    env: ManagerBasedRLEnv,
    gripper_action_name: str,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalizes the agent for closing the gripper when it is far from the object."""
    gripper_action = env.action_manager.action[:, -1]
    is_closing_action = (gripper_action <= 0).bool().squeeze(-1)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    hand_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    object_pos_w = object_asset.data.root_pos_w
    distance = torch.norm(hand_pos_w - object_pos_w, dim=1)
    # 3. Implement the desired logic using torch.where
    # First, define the reward signal for when the agent is CLOSING
    # It gets +1 if near the object, otherwise 0.
    reward_when_closing = torch.where(distance < minimal_height, 1.0, 0.0)

    # The reward for OPENING the gripper is always -1.
    reward_when_opening = 0.1

    # Finally, use the `is_closing_action` boolean tensor as a switch:
    # If `is_closing_action` is True, it returns `reward_when_closing`.
    # If `is_closing_action` is False, it returns `reward_when_opening`.
    return torch.where(is_closing_action, reward_when_closing*5, reward_when_opening)



def distance(
    env: ManagerBasedRLEnv,
    std: float,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("key"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object1: RigidObject = env.scene[object1_cfg.name]
    object2: FrameTransformer | RigidObject = env.scene[object2_cfg.name]
    # Target object position: (num_envs, 3)
    object1_pos_w = object1.data.root_pos_w
    # End-effector position: (num_envs, 3)
    if (isinstance(object2, FrameTransformer)):
        object2_pos_w = object2.data.target_pos_w[..., 0, :]
    else:
        object2_pos_w = object2.data.root_pos_w

    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object1_pos_w - object2_pos_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def key_to_box_dist(env: ManagerBasedRLEnv, threshold: float,
                       object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
                       ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    handle_pos = object.data.root_pos_w
    ee_tcp_pos = ee_frame.data.target_pos_w[..., 0, :]

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)

def grasp_key(env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg,
                        object_cfg: SceneEntityCfg = SceneEntityCfg("key"),
                        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    handle_pos = object.data.root_pos_w
    ee_tcp_pos = ee_frame.data.target_pos_w[..., 0, :]

    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)



def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04,
                       object_cfg: SceneEntityCfg = SceneEntityCfg("key"),
                       ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
    ) -> torch.Tensor:
    """Reward the robot's gripper reaching the drawer handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """


    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Target object position: (num_envs, 3)
    handle_pos = object.data.root_pos_w
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = ee_frame.data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Compute the distance of each finger from the handle
    lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])

    # Check if hand is in a graspable pose
    #is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    return ((offset - lfinger_dist) + (offset - rfinger_dist))

def align_grasp_around_handle(env: ManagerBasedRLEnv,
                       object_cfg: SceneEntityCfg = SceneEntityCfg("key"),
                       ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
    ) -> torch.Tensor:
    """Bonus for correct hand orientation around the handle.

    The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
    """
    # Target object position: (num_envs, 3)
    handle_pos = object_cfg.data.target_pos_w
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = ee_frame_cfg.data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] > handle_pos[:, 2]) & (lfinger_pos[:, 2] < handle_pos[:, 2])

    # bonus if left finger is above the drawer handle and right below
    return is_graspable


def object_fail(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("key"),
) -> torch.Tensor:
    """Reward the agent for the distance between the object and the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] < 0.2, -2, 0)


def key_to_box_height(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("key"),
    object_cfg2: SceneEntityCfg = SceneEntityCfg("box")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    object2: RigidObject = env.scene[object_cfg2.name]
    dist_vec = object.data.root_pos_w[:, 2] - object2.data.root_pos_w[:, 2]
    return torch.tanh(dist_vec/std)


# def object_goal_frame_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("key"),
#     goal_frame_cfg: SceneEntityCfg = SceneEntityCfg("box"),
# ) -> torch.Tensor:
#     """Reward the agent for the distance between the object and the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     goal_frame = env.scene[goal_frame_cfg.name]
#     # Target object position: (num_envs, 3)
#     cube_pos_w = object.data.root_pos_w
#     frame_pos_w = goal_frame.data.root_pos_w
#     object_goal_frame_vector = frame_pos_w - cube_pos_w
#     object_goal_frame_distance = torch.norm(object_goal_frame_vector, dim=1)
#
#     def smooth_signed_log(x, eps=1e-6):
#         return torch.sign(x) * torch.log1p(torch.abs(x) + eps)
#
#     reward = smooth_signed_log((torch.tensordot(object_goal_frame_vector)) * torch.norm(object.data.root_lin_vel_w))
#
#
#     return torch.where(object_goal_frame_distance < 0.2, reward * key_to_box_height(env), 4 * key_to_box_height(env))


def align_head_box_inserting(env: ManagerBasedRLEnv,
                       object_cfg: SceneEntityCfg = SceneEntityCfg("box"),
                       head_frame: SceneEntityCfg = SceneEntityCfg("key_head_frame")
    ) -> torch.Tensor:

    head_frame: FrameTransformer = env.scene[head_frame.name]
    object: RigidObject = env.scene[object_cfg.name]


    head_quat = head_frame.data.target_quat_w[..., 0, :]
    box_quat = object.data.root_quat_w


    head_quat_rot_matrix = matrix_from_quat(head_quat)
    box_quat = matrix_from_quat(box_quat)


    box_z, box_x = box_quat[..., 2], box_quat[..., 0]
    head_x, head_z = head_quat_rot_matrix[..., 0], head_quat_rot_matrix[..., 2]

    align_z = torch.bmm(head_x.unsqueeze(1), box_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(head_z.unsqueeze(1), -box_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


# def align_head_box_while_inside(env: ManagerBasedRLEnv,
#                        object_cfg: SceneEntityCfg = SceneEntityCfg("key"),
#                        head_frame: SceneEntityCfg = SceneEntityCfg("head")
#     ) -> torch.Tensor:
#
#     head_frame: FrameTransformer = env.scene[head_frame.name]
#     object: RigidObject = env.scene[object_cfg.name]
#
#
#     head_quat = ee_frame.data.target_quat_w[..., 0, :]
#     box_quat = object.data.root_quat_w
#
#
#     head_quat_rot_matrix = matrix_from_quat(head_quat)
#     box_quat = matrix_from_quat(box_quat)
#
#
#     box_z, box_x = box_quat[..., 2], box_quat[..., 0]
#     head_x, head_z = head_quat_rot_matrix[..., 0], head_quat_rot_matrix[..., 2]
#
#     align_z = torch.bmm(head_x.unsqueeze(1), box_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#     align_x = torch.bmm(head_z.unsqueeze(1), -box_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#
#     def distance(
#             env: ManagerBasedRLEnv,
#             object1_cfg: SceneEntityCfg = SceneEntityCfg("key"),
#             object2_cfg: SceneEntityCfg = SceneEntityCfg("head"),
#     ) -> torch.Tensor:
#         """Reward the agent for reaching the object using tanh-kernel."""
#         # extract the used quantities (to enable type-hinting)
#         object: RigidObject = env.scene[object_cfg.name]
#         object2: FrameTransformer | RigidObject = env.scene[ee_frame_cfg.name]
#         # Target object position: (num_envs, 3)
#         object1_pos_w = object.data.root_pos_w
#         # End-effector position: (num_envs, 3)
#         if (isinstance(object2, FrameTransformer)):
#             object2_pos_w = object2.data.target_pos_w[..., 0, :]
#         else:
#             object2_pos_w = object2.data.root_pos_w
#
#         # Distance of the end-effector to the object: (num_envs,)
#         object_ee_distance = torch.norm(object1_pos_w - object2_pos_w, dim=1)
#
#         return torch.where(object_ee_distance < 0.03, 1.0, 0.0)
#     return (0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)) * distance()



def align_head_box_while_inside(env: ManagerBasedRLEnv,
                       object_cfg: SceneEntityCfg = SceneEntityCfg("key"),
                       ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("box")
    ) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_tcp_quat = ee_frame.data.target_quat_w[..., 0, :]
    handle_quat = object.data.root_quat_w

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)


    # get current x and y direction of the handle
    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    # get current x and z direction of the gripper
    ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]

    # make sure gripper aligns with the handle
    # in this case, the z direction of the gripper should be close to the -x direction of the handle
    # and the x direction of the gripper should be close to the -y direction of the handle
    # dot product of z and x should be large
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)
