# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg, FrameTransformer
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

from . import mdp


##
# Scene definition
##


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class InsertKeySceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # # target object: will be populated by agent env cfg
    box: RigidObjectCfg | DeformableObjectCfg = MISSING
    #cube2: RigidObjectCfg | DeformableObjectCfg = MISSING
    key: RigidObjectCfg | DeformableObjectCfg = MISSING

    table: RigidObjectCfg | DeformableObjectCfg = MISSING


    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0)),
    )

    sphere_light = AssetBaseCfg(
        prim_path="/World/SphereLight",
        spawn=sim_utils.SphereLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.13920281220691166, 0.06265963052973753, 0.9272182128345804), rot=(0.5, 0.5, 0.5, 0.5))
    )

    distant_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.7, 0.4, 0.4, 0.0))
    )
    # marker_cfg = FRAME_MARKER_CFG.copy()
    # marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # marker_cfg.prim_path = "/Visuals/FrameTransformer"




##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointVelocityActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointVelocityActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        key_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("key")})
        key_orientation = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("key")})
        box_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("box")})
        box_orientation = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("box")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("cube1", body_names="cube1"),
    #     },
    # )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("key")}
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    ee_key_dist = RewTerm(func=mdp.distance, params={"std": 0.1, "object1_cfg": SceneEntityCfg("key"), "object2_cfg": SceneEntityCfg("ee_frame")}, weight=2.0)

    # grasp_key = RewTerm(
    #     func=mdp.grasp_key,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.03,
    #         "open_joint_pos": MISSING,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
    #     },
    # )
    #
    # key_to_box_height = RewTerm(
    #     func=mdp.key_to_box_height,
    #     params={"std": 0.3},
    #     weight=1.5,
    # )
    #
    # align_head_box_inserting = RewTerm(
    #     func=mdp.align_head_box_inserting,
    #     weight=1.5,
    # )
    #
    #
    # key_box_dist = RewTerm(func=mdp.distance2, params={"std": 0.1, "object1_cfg": SceneEntityCfg("box"),
    #                                                      "object2_cfg": SceneEntityCfg("key_head_frame")}, weight=2.0)


    #
    # object_goal_frame_distance = RewTerm(
    #     func=mdp.object_goal_frame_distance,
    #     params={"std": 0.3,  "object_cfg": SceneEntityCfg("box")},
    #     weight=1.0,
    # )

    #align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)


    key_fail = RewTerm(
        func=mdp.object_fail,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("key")},
        weight=1.0,
    )

    box_fail = RewTerm(
        func=mdp.object_fail,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("box")},
        weight=1.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

##
# Environment configuration
##


@configclass
class InsertKeyEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    scene: InsertKeySceneCfg = InsertKeySceneCfg(env_spacing=3)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


