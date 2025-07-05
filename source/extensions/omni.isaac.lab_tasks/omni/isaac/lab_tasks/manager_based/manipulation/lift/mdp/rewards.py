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
from omni.isaac.lab.utils.math import combine_frame_transforms, matrix_from_quat

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


def penalize_letting_go_of_lifted_object(
    env: ManagerBasedRLEnv,
    gripper_action_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalizes the agent for closing the gripper when it is far from the object."""
    # 1. Determine if the gripper action is "close"
    # 2. Use the indices to slice the action tensor.
    object_asset: RigidObject = env.scene[object_cfg.name]
    object = env.scene[object_cfg.name]
    gripper_action = env.action_manager.action[:, -1]
    # Squeeze the tensor to ensure it is 1D: [num_envs, 1] -> [num_envs]
    is_opening_action = (gripper_action < 0).float().squeeze(-1)

    return torch.where(env.obs_buf["policy"][:, -1] != 0 * is_opening_action, -1.0, 0.0)


def fingers_to_object_distance(
    env: ManagerBasedRLEnv,
    alpha: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward the agent for moving its end-effector fingers very close to the object.

    This version uses a squared exponential (Gaussian) kernel, which provides a much
    stronger reward signal when the fingers are extremely close to the object,
    and the reward falls off very quickly with distance.

    Args:
        env: The environment instance.
        alpha: A scaling factor for the exponential kernel. A larger value creates a
               steeper, more focused reward.
        object_cfg: The configuration for the target object. Defaults to "object".
        ee_frame_cfg: The configuration for the end-effector FrameTransformer.
                      Defaults to "ee_frame".

    Returns:
        A tensor containing the calculated reward for each environment.
    """
    # Extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Object position in the world frame: (num_envs, 3)
    object_pos_w = object_asset.data.root_pos_w
    # End-effector frame positions in the world frame
    ee_targets_w = ee_frame.data.target_pos_w
    num_targets = ee_targets_w.shape[1]

    # Check the number of tracked targets to avoid crashing
    if num_targets == 0:
        return torch.zeros(env.num_envs, device=env.device)
    elif num_targets == 1:
        hand_pos_w = ee_targets_w[:, 0, :]
        total_distance = torch.norm(object_pos_w - hand_pos_w, dim=1)
    else:
        left_finger_pos_w = ee_targets_w[:, 0, :]
        right_finger_pos_w = ee_targets_w[:, 1, :]
        dist_left = torch.norm(object_pos_w - left_finger_pos_w, dim=1)
        dist_right = torch.norm(object_pos_w - right_finger_pos_w, dim=1)
        total_distance = dist_left + dist_right

    # Use a squared exponential kernel for a more focused reward at close distances
    # reward = exp(-alpha * distance^2)
    return torch.exp(-alpha * total_distance.pow(2))



def scaled_lin_vel(env: ManagerBasedRLEnv, std: float, std_2: float,  command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    def object_goal_distance(
            env: ManagerBasedRLEnv,
            std = std_2,
            cmd = command_name,
            robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
            object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Reward the agent for tracking the goal pose using tanh-kernel."""
        # extract the used quantities (to enable type-hinting)
        robot: RigidObject = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(cmd)
        # compute the desired position in the world frame
        des_pos_b = command[:, :3]
        des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
        # distance of the end-effector to the object: (num_envs,)
        distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
        # rewarded if the object is lifted above the threshold
        return torch.tanh(distance / std)

    return torch.tanh((torch.square(asset.data.root_lin_vel_b[:, 2]) / std) * object_goal_distance(env))

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

    return 1 - torch.tanh(object_ee_distance / std)



def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))




# ----------------------------------------------------------------------------------------------------------------

def fingers_to_object_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward the agent for moving its end-effector fingers close to the object.

    This function is flexible based on the number of targets in the FrameTransformer:
    - If 1 target: Rewards moving that single point (e.g., hand center) to the object.
    - If 2+ targets: Rewards moving the first two points (e.g., fingertips) to the object.

    Args:
        env: The environment instance.
        std: The standard deviation for the tanh reward kernel. A smaller
             value makes the reward function steeper.
        object_cfg: The configuration for the target object. Defaults to "object".
        ee_frame_cfg: The configuration for the end-effector FrameTransformer.
                      Defaults to "ee_frame".

    Returns:
        A tensor containing the calculated reward for each environment.
    """
    # Extract the used quantities (to enable type-hinting)
    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Object position in the world frame: (num_envs, 3)
    object_pos_w = object_asset.data.root_pos_w
    # End-effector frame positions in the world frame
    ee_targets_w = ee_frame.data.target_pos_w
    num_targets = ee_targets_w.shape[1]

    # Check the number of tracked targets to avoid crashing
    if num_targets == 0:
        # No targets are tracked, return zero reward
        return torch.zeros(env.num_envs, device=env.device)
    elif num_targets == 1:
        # If only one target (e.g., the hand), calculate its distance to the object
        hand_pos_w = ee_targets_w[:, 0, :]
        total_distance = torch.norm(object_pos_w - hand_pos_w, dim=1)
    else:
        # If two or more targets, use the first two as left and right fingers
        left_finger_pos_w = ee_targets_w[:, 0, :]
        right_finger_pos_w = ee_targets_w[:, 1, :]

        # Calculate the distance from each finger to the object
        dist_left = torch.norm(object_pos_w - left_finger_pos_w, dim=1)
        dist_right = torch.norm(object_pos_w - right_finger_pos_w, dim=1)

        # The reward is based on the sum of the distances
        total_distance = dist_left + dist_right

    # Use a tanh kernel to create a reward that is 1 at 0 distance and falls off
    return 1.0 - torch.tanh(total_distance / std)

def stay_low(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w

    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    if isinstance(env.obs_buf["policy"], torch.Tensor):
        return torch.where(object.data.root_pos_w[:, 2] < minimal_height, 2.0, 0.0) * torch.where(env.obs_buf["policy"][:, -1]  == 1.0, 1.0, 0.0)
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def ee_base_distance(
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
    base_point : torch.Tensor = torch.zeros(4096, 3, device="cuda:0")
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    ee_base_distance = torch.norm(base_point - ee_w, dim=1)

    if isinstance(env.obs_buf["policy"], torch.Tensor):
        return 1 - torch.tanh(ee_base_distance / std) * torch.where(env.obs_buf["policy"][:, -1]  == 1.0, 1.0, 0.0)
    return 1 - torch.tanh(ee_base_distance / std)



def object_velocity_towards_goal_reward(
        env: ManagerBasedRLEnv,
        command_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward based on the object's velocity towards the goal."""
    # Extract assets and commands
    robot: RigidObject = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Goal position in the world frame
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7],
                                             goal_pos_b)

    # Object position and velocity in the world frame
    object_pos_w = object_asset.data.root_pos_w
    object_vel_w = object_asset.data.root_lin_vel_w

    # Vector from the object to the goal
    direction_to_goal = goal_pos_w - object_pos_w
    direction_to_goal = torch.nn.functional.normalize(direction_to_goal, p=2, dim=1)

    # Project the object's velocity onto the direction vector
    velocity_towards_goal = torch.sum(object_vel_w * direction_to_goal, dim=1)

    # Clamp to avoid huge rewards, rewarding only positive progress
    progress_reward = torch.clamp(velocity_towards_goal, min=0.0)

    return progress_reward



def encourage_lift_sequence(
        env: ManagerBasedRLEnv,
        ee_body_name: str,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        gripper_joint_cfg: SceneEntityCfg = SceneEntityCfg("robot", "panda_finger_joint.*"),
        reach_distance_threshold: float = 0.1,
        lift_velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """
    A staged reward to encourage the full sequence of lifting an object.

    This function provides a cascading reward for:
    1. Reaching the object.
    2. Aligning the fingers low and around the object.
    3. Closing the gripper.
    4. Moving the end-effector upwards with a closed gripper.
    """
    # -- Stage 1: Proximity to Object --
    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    hand_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # Assumes first target is the hand center
    object_pos_w = object_asset.data.root_pos_w

    # Distance from hand to object's xy-plane projection
    xy_dist = torch.norm(hand_pos_w[:, :2] - object_pos_w[:, :2], dim=1)
    # Vertical distance from hand to object
    z_dist = torch.abs(hand_pos_w[:, 2] - object_pos_w[:, 2])

    # Reward for being close in both xy and z
    reach_reward = (xy_dist < reach_distance_threshold).float() * (z_dist < reach_distance_threshold).float()

    # -- Stage 2: Gripper State --
    # Get gripper joint positions (we want them to be small, i.e., closed)
    gripper_joints = env.scene[gripper_joint_cfg.name]
    gripper_pos = torch.sum(gripper_joints.data.joint_pos, dim=1)  # Sum of finger joint positions
    # Condition: Is the gripper closed? (joint positions are near zero)
    is_gripper_closed = (gripper_pos < 0.01).float()

    # -- Stage 3: Upward Velocity --
    robot: Articulation = env.scene[robot_cfg.name]
    # Find the index of the end-effector body
    ee_body_indices = robot.find_bodies(ee_body_name)[0]
    # Get the linear velocity of the end-effector body, shape is (num_envs, 1, 3)
    ee_body_vel_w = robot.data.body_lin_vel_w[:, ee_body_indices, :]

    # FIX: Correctly index the tensor. We want the first (and only) body's velocity (index 0),
    # and then the z-component of that velocity vector (index 2).
    hand_vel_z = ee_body_vel_w[:, 0, 2]

    # Reward for positive upward velocity
    upward_vel_reward = (hand_vel_z > lift_velocity_threshold).float()

    # -- Combine Stages into a Cascading Reward --
    # The agent gets rewards sequentially. It must reach to get the grip reward,
    # and it must be gripping to get the lift reward.
    # Base reward for reaching
    total_reward = reach_reward
    # Add reward for gripping, but only if also reaching
    total_reward += reach_reward * is_gripper_closed
    # Add reward for lifting, but only if also reaching and gripping
    total_reward += reach_reward * is_gripper_closed * upward_vel_reward

    return total_reward


def object_distance_to_goal_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward based on the object's distance to the goal using a tanh kernel."""
    # Extract assets and commands
    robot: RigidObject = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Goal position in the world frame
    goal_pos_b = command[:, :3]
    goal_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], goal_pos_b)

    # Object position in the world frame
    object_pos_w = object_asset.data.root_pos_w

    # Calculate distance-based reward
    distance_to_goal = torch.norm(goal_pos_w - object_pos_w, dim=1)
    distance_reward = 1.0 - torch.tanh(distance_to_goal / distance_std)

    return distance_reward


def ee_object_z_axis_alignment(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward for aligning the end-effector's z-axis with the object's z-axis.

    This function computes the dot product between the z-axis vectors of the
    end-effector and the object. The reward is 1 for perfect alignment, -1 for
    perfect anti-alignment, and 0 for perpendicular axes. This is useful for
    encouraging a top-down grasp approach.

    Args:
        env: The environment instance.
        object_cfg: The configuration for the target object.
        ee_frame_cfg: The configuration for the end-effector FrameTransformer.

    Returns:
        A tensor containing the alignment reward for each environment.
    """
    # Get the orientation of the object
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object_asset.data.root_quat_w

    # Get the orientation of the end-effector (hand)
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    # FIX: Convert quaternions to 3x3 rotation matrices
    object_rot_matrix = matrix_from_quat(object_quat_w)
    ee_rot_matrix = matrix_from_quat(ee_quat_w)

    # The z-axis of a frame is the third column of its rotation matrix.
    # The shape of the matrix is (num_envs, 3, 3), so we take all rows for the 3rd column (index 2).
    object_z_axis = object_rot_matrix[:, :, 2]
    ee_z_axis = ee_rot_matrix[:, :, 2]

    # Compute the dot product between the two z-axis vectors
    # This gives a value between -1 (anti-aligned) and 1 (aligned).
    alignment_reward = torch.sum(object_z_axis * ee_z_axis, dim=1)

    gripper_action = env.action_manager.prev_action[:, -1]

    return alignment_reward * (gripper_action == 1.0).float()
