# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates quadruped robot testing with keyboard controls and low-level policy actions.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p quadruped/tests/quadruped_test.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher
from pynput import keyboard

"""Rest everything follows."""


def launch_app():
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="This script demonstrates quadruped robot testing with keyboard controls."
    )
    parser.add_argument(
        "--random_actions",
        action="store_true",
        help="Use random actions instead of keyboard control",
    )
    parser.add_argument(
        "--low_level_policy",
        action="store_true",
        help="Enable low-level policy action mapping",
    )
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    return simulation_app, args_cli


try:
    from isaaclab.app import AppLauncher

    simulation_app, args_cli = launch_app()
except ImportError:
    raise ImportError("Isaac Lab is not installed. Please install it first.")


def make_isaaclab_env(
    task,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    video_length=200,
    *args,
    **kwargs,
):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import quadruped.tasks  # noqa: F401

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )
        env = gym.make(
            task,
            cfg=cfg,
            render_mode="rgb_array"
            if (capture_video and log_dir is not None)
            else None,
            debug=False,
        )
        return env

    return thunk


try:
    import msvcrt  # For Windows

    def get_key():
        return msvcrt.getch().decode("utf-8").lower()
except ImportError:
    import termios
    import tty

    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.lower()


# High-level keyboard mapping: (forward, left, yaw)
HIGH_LEVEL_MAPPING = {
    "w": (4.0, 0.0, 0.0),  # forward
    "s": (-1.0, 0.0, 0.0),  # backward
    "a": (0.0, 2.0, 0.0),  # left
    "d": (0.0, -2.0, 0.0),  # right
    "q": (0.0, 0.0, 3.0),  # yaw left
    "e": (0.0, 0.0, -3.0),  # yaw right
}

# Low-level policy keyboard mapping for individual joint/leg control
LOW_LEVEL_MAPPING = {
    # Front left leg
    "i": "fl_hip_pitch_up",
    "k": "fl_hip_pitch_down",
    "j": "fl_hip_roll_left",
    "l": "fl_hip_roll_right",
    "u": "fl_knee_extend",
    "o": "fl_knee_flex",
    # Front right leg
    "t": "fr_hip_pitch_up",
    "g": "fr_hip_pitch_down",
    "f": "fr_hip_roll_left",
    "h": "fr_hip_roll_right",
    "r": "fr_knee_extend",
    "y": "fr_knee_flex",
    # Rear left leg
    "8": "rl_hip_pitch_up",
    "2": "rl_hip_pitch_down",
    "4": "rl_hip_roll_left",
    "6": "rl_hip_roll_right",
    "7": "rl_knee_extend",
    "9": "rl_knee_flex",
    # Rear right leg
    "v": "rr_hip_pitch_up",
    "c": "rr_hip_pitch_down",
    "x": "rr_hip_roll_left",
    "n": "rr_hip_roll_right",
    "b": "rr_knee_extend",
    "m": "rr_knee_flex",
}

action = None
exit_flag = False
key_pressed = False
use_random_actions = True
use_low_level_policy = False


def apply_low_level_action(key, action_tensor):
    """Apply low-level policy actions based on keyboard input."""
    joint_mapping = {
        # Front left leg (indices 0-2)
        "fl_hip_pitch_up": (0, 0.5),
        "fl_hip_pitch_down": (0, -0.5),
        "fl_hip_roll_left": (1, 0.5),
        "fl_hip_roll_right": (1, -0.5),
        "fl_knee_extend": (2, 0.5),
        "fl_knee_flex": (2, -0.5),
        # Front right leg (indices 3-5)
        "fr_hip_pitch_up": (3, 0.5),
        "fr_hip_pitch_down": (3, -0.5),
        "fr_hip_roll_left": (4, 0.5),
        "fr_hip_roll_right": (4, -0.5),
        "fr_knee_extend": (5, 0.5),
        "fr_knee_flex": (5, -0.5),
        # Rear left leg (indices 6-8)
        "rl_hip_pitch_up": (6, 0.5),
        "rl_hip_pitch_down": (6, -0.5),
        "rl_hip_roll_left": (7, 0.5),
        "rl_hip_roll_right": (7, -0.5),
        "rl_knee_extend": (8, 0.5),
        "rl_knee_flex": (8, -0.5),
        # Rear right leg (indices 9-11)
        "rr_hip_pitch_up": (9, 0.5),
        "rr_hip_pitch_down": (9, -0.5),
        "rr_hip_roll_left": (10, 0.5),
        "rr_hip_roll_right": (10, -0.5),
        "rr_knee_extend": (11, 0.5),
        "rr_knee_flex": (11, -0.5),
    }

    if key in LOW_LEVEL_MAPPING:
        action_name = LOW_LEVEL_MAPPING[key]
        if action_name in joint_mapping:
            joint_idx, action_val = joint_mapping[action_name]
            if joint_idx < len(action_tensor):
                action_tensor[joint_idx] = action_val
                print(
                    f"[DEBUG]: Low-level action - {action_name}: joint {joint_idx} = {action_val}"
                )


def on_press(key):
    global action, exit_flag, key_pressed, use_random_actions
    try:
        k = key.char.lower()
        if k == "esc" or k == "\x1b":  # ESC key
            print("[INFO]: Exiting...")
            exit_flag = True
            return False  # Stop listener
        elif k == "p":
            # Toggle random actions
            use_random_actions = not use_random_actions
            print(
                f"[INFO]: Random actions {'enabled' if use_random_actions else 'disabled'}"
            )
            key_pressed = True
        elif k in HIGH_LEVEL_MAPPING and not use_low_level_policy:
            # High-level control (navigation commands)
            delta = torch.tensor(
                HIGH_LEVEL_MAPPING[k], dtype=torch.float32, device=action.device
            )
            if len(action) >= 3:
                action[:3] = delta
            key_pressed = True
            print(f"[DEBUG]: High-level action updated: {action[:3].cpu().tolist()}")
        elif use_low_level_policy:
            # Low-level policy control
            apply_low_level_action(k, action)
            key_pressed = True
        elif k == " ":
            action.zero_()
            key_pressed = True
            print("[DEBUG]: Action reset to zero.")
        elif k == "r":
            action.zero_()
            key_pressed = True
            print("[DEBUG]: Environment reset requested.")
    except AttributeError:
        # Handle special keys
        if key == keyboard.Key.esc:
            print("[INFO]: Exiting...")
            exit_flag = True
            return False


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


def generate_random_action(action_space_shape, device="cuda:0"):
    """Generate random actions for visualization."""
    return torch.randn(action_space_shape, dtype=torch.float32, device=device) * 0.5


def main():
    """Main function."""
    global action, key_pressed, use_random_actions, use_low_level_policy

    # Set modes based on command line arguments
    use_random_actions = args_cli.random_actions
    use_low_level_policy = args_cli.low_level_policy

    print("[INFO]: Creating quadruped environment...")

    # Create environment with full Isaac reward system + obstacle extensions
    device = "cuda:0"
    env = make_isaaclab_env(
        "Spot-Velocity-Flat-Obstacle-Quadruped-v0",
        device,
        128,
        False,
        False,
    )()

    obs, _ = env.reset()

    # Determine the device from the environment
    if hasattr(env, "device"):
        device = env.device
    elif hasattr(env.unwrapped, "device"):
        device = env.unwrapped.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print(f"[INFO]: Using device: {device}")

    action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=device)

    print(f"[INFO]: Action space shape: {env.action_space.shape}")
    print(f"[INFO]: Observation space shape: {env.observation_space.shape}")

    if use_random_actions:
        print("[INFO]: Random action mode enabled. Press 'p' to toggle, 'ESC' to exit.")
    elif use_low_level_policy:
        print("[INFO]: Low-level policy mode enabled.")
        print("[INFO]: Use IJKL/UO for front left leg, TFGH/RY for front right leg,")
        print("[INFO]: 8246/79 for rear left leg, VCXN/BM for rear right leg.")
        print("[INFO]: Press 'ESC' to exit, 'r' to reset, space to zero action.")
    else:
        print("[INFO]: High-level control mode enabled.")
        print(
            "[INFO]: Use WASD to move, Q/E to yaw. Press 'ESC' to exit, 'r' to reset, space to zero action."
        )
        print("[INFO]: Press 'p' to toggle random actions.")

    start_keyboard_listener()

    step_count = 0
    while not exit_flag:
        if use_random_actions:
            step_action = generate_random_action(env.action_space.shape, device)
            if step_count % 100 == 0:
                print(f"[DEBUG]: Random action: {step_action.cpu().tolist()}")
        elif key_pressed:
            step_action = action.clone()
            key_pressed = False
            action.zero_()
        else:
            step_action = torch.zeros_like(action, device=device)
        _, reward, _, _, info = env.step(step_action)

        if step_count % 100 == 0:
            # Convert tensor reward to scalar for formatting
            reward_val = (
                reward.sum().item() if hasattr(reward.sum(), "item") else float(reward)
            )
            print(f"[INFO]: Step {step_count}, Reward: {reward_val:.4f}")

        step_count += 1
        time.sleep(0.02)  # Match simulation dt

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception:", e)
        raise e
    finally:
        simulation_app.close()
