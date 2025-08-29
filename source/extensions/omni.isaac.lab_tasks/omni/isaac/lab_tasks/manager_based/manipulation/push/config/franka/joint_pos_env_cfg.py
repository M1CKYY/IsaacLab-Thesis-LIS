# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab_tasks.manager_based.manipulation.push import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.push.push_env_cfg import PushEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class FrankaCubePushEnvCfg(PushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # Set actions for the specific Franka type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
            debug_vis=True,
        )

        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=UsdFileCfg(
                usd_path="/home/michias/Documents/Isaac Custom Environments/box_push/table_low.usd",
                scale=(0.8, 0.8, 1),
            ),
        )

        self.scene.cube1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.27805966079156325, -0.11327031527026465, 0.7176599999999997),),
            spawn=UsdFileCfg(
                usd_path="/home/michias/Documents/Isaac Custom Environments/box_push/Cube1.usd",
                scale=(0.8, 0.8, 0.8),
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

        # self.scene.cube2 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Cube2",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.023207811638712883, -0.2540944678810327, 0.7176562547683716)),
        #     spawn=UsdFileCfg(
        #         usd_path="/home/michias/Documents/Isaac Custom Environments/box_push/Cube2.usd",
        #         scale=(0.8, 0.8, 0.8),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        # Listens to the required transforms


        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",  # Basis-Frame
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # Ziel-Frame
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )


        self.commands.object_pose.body_name = "panda_hand"





@configclass
class FrankaCubePushEnvPLAYCfg(FrankaCubePushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
