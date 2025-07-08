# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab_tasks.manager_based.manipulation.insert_key import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.insert_key.insert_key_env_cfg import InsertKeyEnvCfg
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class FrankaInsertKeyEnvCfg(InsertKeyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                                          init_state=ArticulationCfg.InitialStateCfg(
                                                              joint_pos={
                                                                  "panda_joint1": 0.0,
                                                                  "panda_joint2": -0.569,
                                                                  "panda_joint3": 0.0,
                                                                  "panda_joint4": -2.810,
                                                                  "panda_joint5": 0.0,
                                                                  "panda_joint6": 3.037,
                                                                  "panda_joint7": 0.741,
                                                                  "panda_finger_joint.*": 0.04,
                                                              },
                                                              pos=(-6.477104761898872e-17, 0.3904529388479216,
                                                                   0.6810081391520464),
                                                              rot=(0.70711, 0, 0, -0.70711),
                                                          ))
        # Set actions for the specific Franka type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.3, use_default_offset=True
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
            debug_vis=True,
        )

        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table_low",
            spawn=UsdFileCfg(
                usd_path="/home/michias/Documents/Isaac Custom Environments/insert_key/table_low.usd",
                scale=(0.61808474, 1.4552572, 1),
            ),
        )

        self.scene.key = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Key",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, -0.0, 0.7),
            ),
            spawn=UsdFileCfg(
                usd_path="/home/michias/Documents/Isaac Custom Environments/insert_key/Key.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )


        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.0, -0.55, 0.8),
            ),
            spawn=UsdFileCfg(
                usd_path="/home/michias/Documents/Isaac Custom Environments/insert_key/Box.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )



        # override rewards
        self.rewards.grasp_key.params["open_joint_pos"] = 0.04
        self.rewards.grasp_key.params["asset_cfg"].joint_names = ["panda_finger_.*"]


@configclass
class FrankaInsertKeyEnvCfg_PLAY(FrankaInsertKeyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.rewards.grasp_key.params["open_joint_pos"] = 0.04
        self.rewards.grasp_key.params["asset_cfg"].joint_names = ["panda_finger_.*"]
