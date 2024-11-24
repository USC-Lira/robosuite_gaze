"""
Driver class for PS4 Gamepad controller for RoboSuite (Windows version).
"""

from typing import Dict, Optional
import pygame
import numpy as np
import time

from robosuite.devices import Device
from robosuite.gamepad import available
from robosuite.gamepad import PS4


class WindowsPS4Controller(Device):
    """
    A driver class for PS4 Gamepad control for RoboSuite on Windows.

    Args:
        env (RobotEnv): Environment containing the robot(s) to control
        pos_sensitivity (float): Position control sensitivity scaling
        rot_sensitivity (float): Rotation control sensitivity scaling
    """

    def __init__(self, env, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.0):
        super().__init__(env)

        # Initialize controller parameters
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # Initialize controller state
        self.controller = None
        self.axis_data = {}
        self.button_data = {}
        self._running = False
        self.active_robot = 0

        self._display_controls()

    @staticmethod
    def _display_controls() -> None:
        """Display gamepad control mapping."""

        def print_command(char: str, info: str) -> None:
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("\nGamepad Controls:")
        print_command("L2", "Close gripper")
        print_command("Left stick", "XY motion")
        print_command("Right stick Y", "Z motion")
        print_command("Circle", "Reset simulation")
        print("\n")

    def start_control(self) -> None:
        """Initialize controller and start background updates."""
        pygame.init()
        pygame.joystick.init()

        try:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
        except pygame.error as e:
            raise RuntimeError(f"Failed to connect to PS4 controller: {e}")

        # Start background thread
        self._running = True
        threading.Thread(target=self._update_controller_state, daemon=True).start()

        print("Gamepad connected")
        self._reset_internal_state()

    def _update_controller_state(self):
        """Update controller state in background."""
        while self._running and pygame.get_init():
            pygame.event.pump()

            # Update axes
            for i in range(self.controller.get_numaxes()):
                self.axis_data[i] = round(self.controller.get_axis(i), 2)

            # Update buttons
            for i in range(self.controller.get_numbuttons()):
                self.button_data[i] = self.controller.get_button(i)

            time.sleep(0.01)

    def get_controller_state(self) -> Dict:
        """
        Get the current state of the gamepad controller.

        Returns:
            dict: Current controller state
        """
        if not self.controller:
            raise RuntimeError("Controller not initialized")

        # Get position input
        dpos = np.zeros(3)

        # Apply deadzone
        def deadzone(value, threshold=0.05):
            return 0 if abs(value) < threshold else value

        # Map controller axes to movement
        dpos[0] = -0.1 * deadzone(self.axis_data.get(1, 0)) * self.pos_sensitivity  # Forward/Back
        dpos[1] = -0.1 * deadzone(self.axis_data.get(0, 0)) * self.pos_sensitivity  # Left/Right
        dpos[2] = -0.1 * deadzone(self.axis_data.get(3, 0)) * self.pos_sensitivity  # Up/Down

        # Default rotation
        rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        raw_drotation = np.zeros(3)

        # Get gripper and reset states
        grasp = 1 if self.axis_data.get(4, 0) > 0.5 else 0  # L2 trigger
        reset = 1 if self.button_data.get(1, False) else 0  # Circle button

        return {
            "dpos": dpos,
            "rotation": rotation,
            "raw_drotation": raw_drotation,
            "grasp": grasp,
            "reset": reset,
            "base_mode": 0,
        }

    def _reset_internal_state(self) -> None:
        """Reset the internal state of the controller."""
        super()._reset_internal_state()
        self.axis_data = {}
        self.button_data = {}

    def _postprocess_device_outputs(self, dpos, drotation):
        """Scale and clip outputs."""
        dpos = dpos * 2.0
        drotation = drotation * 1.5

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation

    def close(self):
        """Clean up controller resources."""
        self._running = False
        if self.controller:
            self.controller.quit()
        pygame.quit()
