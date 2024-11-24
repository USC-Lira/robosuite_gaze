"""
A script to collect human demonstration using Gamepad controller.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper, GazeDataCollectionWrapper

from robosuite.utils.gazepoint import GazepointClient


def collect_human_trajectory(
    env, device, arm, max_fr, gazepoint: GazepointClient, 
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()
    env.render()

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action()

        # get the latest gaze information
        gaze_x, gaze_y = gazepoint.get_latest_gaze()
        gaze_np_xy = np.array([gaze_x, gaze_y])

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action, gaze_np_xy)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory: str, out_dir: str, env_info: str) -> None:
    """
    Gathers demonstrations with gaze data into a single HDF5 file.

    Structure of the HDF5 file:
    data (group)
        ├── attributes
        │   ├── date
        │   ├── time
        │   ├── repository_version
        │   ├── env
        │   └── env_info
        └── demos
            ├── demo1 (group)
            │   ├── model_file (attribute)
            │   ├── states (dataset)
            │   ├── actions (dataset)
            │   └── gaze (dataset)
            └── demo2 (group)
                ...

    Args:
        directory: Path to directory containing raw demonstrations
        out_dir: Path to store the HDF5 file
        env_info: JSON-encoded string with environment information
    """
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, "demo.hdf5")

    with h5py.File(hdf5_path, "w") as f:
        # Create main data group
        grp = f.create_group("data")
        num_eps = 0
        env_name = None

        # Process each episode directory
        for ep_directory in os.listdir(directory):
            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            states = []
            actions = []
            gazes = []
            success = False

            # Collect all states, actions, and gaze data from episode files
            for state_file in sorted(glob(state_paths)):
                try:
                    data = np.load(state_file, allow_pickle=True)
                    env_name = str(data["env"])

                    # Collect states
                    states.extend(data["states"])

                    # Collect actions
                    for action_info in data["action_infos"]:
                        actions.append(action_info["actions"])

                    # Collect gaze data
                    for gaze_info in data["gaze_infos"]:
                        gazes.append(gaze_info["gaze"])

                    success = success or data["successful"]
                except Exception as e:
                    print(f"Error processing {state_file}: {e}")
                    continue

            # Skip empty episodes
            if not states:
                continue

            # Only save successful demonstrations
            if success:
                print(f"Saving successful demonstration from {ep_directory}")

                # Remove last state (matches action count)
                states.pop()

                # Verify data alignment
                if not (len(states) == len(actions) == len(gazes)):
                    print(f"Warning: Misaligned data in {ep_directory}")
                    print(f"States: {len(states)}, Actions: {len(actions)}, Gaze: {len(gazes)}")
                    continue

                # Create demonstration group
                num_eps += 1
                ep_data_grp = grp.create_group(f"demo_{num_eps}")

                # Store model XML
                xml_path = os.path.join(directory, ep_directory, "model.xml")
                try:
                    with open(xml_path, "r") as xml_file:
                        ep_data_grp.attrs["model_file"] = xml_file.read()
                except Exception as e:
                    print(f"Error reading model file: {e}")
                    continue

                # Create datasets
                ep_data_grp.create_dataset("states", data=np.array(states))
                ep_data_grp.create_dataset("actions", data=np.array(actions))
                ep_data_grp.create_dataset("gaze", data=np.array(gazes))
            else:
                print(f"Skipping unsuccessful demonstration from {ep_directory}")

        # Store metadata
        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = env_name
        grp.attrs["env_info"] = env_info

    print(f"Successfully saved {num_eps} demonstrations to {hdf5_path}")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="default", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="ps4_controller")  # keyboard
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = GazeDataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    elif args.device == "ps4_controller":
        from robosuite.devices import PS4Controller, WindowsPS4Controller

        device = WindowsPS4Controller(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)

    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # gaze data client
    gazepoint = GazepointClient()
    gazepoint.connect()
    
    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.max_fr, gazepoint)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
