# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports keyboard and spacemouse input devices for SE(3) control."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Teleop device. Supported: keyboard, spacemouse.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio for Pink IK controllers.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim. Pinocchio is required by the Pink IK controllers.
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import logging
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    # Create device callbacks
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
    }

    # Create teleop device
    teleop_interface = None
    try:
        sensitivity = args_cli.sensitivity
        if args_cli.teleop_device.lower() == "keyboard":
            teleop_interface = Se3Keyboard(
                Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
            )
        elif args_cli.teleop_device.lower() == "spacemouse":
            teleop_interface = Se3SpaceMouse(
                Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
            )
        else:
            logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
            logger.error("Supported devices: keyboard, spacemouse")
            env.close()
            simulation_app.close()
            return

        # Add callbacks to device
        for key, callback in teleoperation_callbacks.items():
            try:
                teleop_interface.add_callback(key, callback)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance()

                # process actions
                actions = action.repeat(env.num_envs, 1)
                # apply actions
                env.step(actions)

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
