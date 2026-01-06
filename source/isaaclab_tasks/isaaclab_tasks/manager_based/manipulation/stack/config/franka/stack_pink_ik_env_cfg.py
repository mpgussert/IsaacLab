# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Franka Stack task using Pink IK - compatible with Newton physics engine.

Pink IK is a Pinocchio-based inverse kinematics solver that doesn't depend on
the physics engine's Jacobian API, making it compatible with Newton.
"""

import tempfile

from pink.tasks import DampingTask, FrameTask

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaCubeStackPinkIKEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """Configuration for Franka cube stacking with Pink IK (Newton-compatible)."""

    # Temporary directory for URDF files used by Pink IK
    temp_urdf_dir = tempfile.gettempdir()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot with high PD gains
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka) using Pink IK
        self.actions.arm_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
            hand_joint_names=[
                "panda_finger_joint1",
                "panda_finger_joint2",
            ],
            target_eef_link_names={
                "end_effector": "panda_hand",
            },
            asset_name="robot",
            controller=PinkIKControllerCfg(
                articulation_name="robot",
                base_link_name="panda_link0",
                num_hand_joints=2,
                show_ik_warnings=False,
                fail_on_joint_limit_violation=False,
                variable_input_tasks=[
                    FrameTask(
                        "panda_hand",
                        position_cost=8.0,
                        orientation_cost=1.0,
                        lm_damping=10,
                        gain=0.5,
                    ),
                    DampingTask(
                        cost=0.5,
                    ),
                    NullSpacePostureTask(
                        cost=0.2,
                        lm_damping=1,
                        controlled_frames=["panda_hand"],
                        controlled_joints=[
                            "panda_joint1",
                            "panda_joint2",
                            "panda_joint3",
                            "panda_joint4",
                            "panda_joint5",
                            "panda_joint6",
                            "panda_joint7",
                        ],
                    ),
                ],
                fixed_input_tasks=[],
                xr_enabled=False,
            ),
        )

        # Keep the binary gripper action
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Convert USD to URDF for Pink IK solver
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.arm_action.controller.urdf_path = temp_urdf_output_path
        self.actions.arm_action.controller.mesh_path = temp_urdf_meshes_output_path

        # Keyboard device for teleoperation
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

