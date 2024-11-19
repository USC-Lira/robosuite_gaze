"""
Driver class for PS4 Gamepad controller for RoboSuite.
"""

from typing import Dict, Optional
import numpy as np
import time

from robosuite.devices import Device
from robosuite.gamepad import available
from robosuite.gamepad import PS4


class PS4Controller(Device):
    """
    A driver class for PS4 Gamepad control for RoboSuite.

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

        # Initialize gamepad
        self.gamepad: Optional[PS4] = None
        self._enabled = False
        self._display_controls()

    @staticmethod
    def _display_controls() -> None:
        """Display gamepad control mapping."""

        def print_command(char: str, info: str) -> None:
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("\nGamepad Controls:")
        print_command("L2", "Close gripper")
        print_command("R2", "Enable control")
        print_command("Left stick", "XY motion")
        print_command("Right stick Y", "Z motion")
        print_command("Circle", "Reset simulation")
        print("\n")

    def start_control(self) -> None:
        """
        Initialize gamepad control by connecting to the device and
        starting background updates.
        """
        if not available():
            print("Please connect your gamepad...")
            while not available():
                time.sleep(1.0)

        self.gamepad = PS4()
        self.gamepad.startBackgroundUpdates()
        print("Gamepad connected")

        self._reset_internal_state()
        self._enabled = True

    def get_controller_state(self) -> Dict:
        """
        Get the current state of the gamepad controller.

        Returns:
            dict: Current controller state containing:
                - dpos (numpy.ndarray): Change in position (delta position)
                - rotation (numpy.ndarray): Current rotation matrix
                - raw_drotation (numpy.ndarray): Raw change in rotation
                - grasp (int): Gripper state (1 for close, -1 for open)
                - reset (int): Reset signal
                - base_mode (int): Base control mode
        """
        if not self.gamepad or not self.gamepad.isConnected():
            raise RuntimeError("Gamepad is not connected")

        # Get position input
        dpos = np.zeros(3)
        # TODO(dhanush): 0.1 found by trial and error. Expose.
        dpos[0] = 0.1 * self.gamepad.axis("LEFT-Y") * self.pos_sensitivity  # Forward/Back
        dpos[1] = 0.1 * self.gamepad.axis("LEFT-X") * self.pos_sensitivity  # Left/Right
        dpos[2] = -0.1 * self.gamepad.axis("RIGHT-Y") * self.pos_sensitivity  # Up/Down

        # No rotation input for now
        # TODO(dhanush): Implement rotation control
        rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        raw_drotation = np.zeros(3)

        # Get gripper and control states
        grasp = 1 if self.gamepad.axis("L2") == 1 else 0
        reset = 1 if self.gamepad.beenPressed("CIRCLE") else 0
        control_enabled = self.gamepad.axis("R2") == 1

        # Only return actions if control is enabled
        if not control_enabled:
            dpos = np.zeros(3)

        return {
            "dpos": dpos,
            "rotation": rotation,
            "raw_drotation": raw_drotation,
            "grasp": grasp,
            "reset": reset,
            "base_mode": 0,  # No base control for now
        }

    def _reset_internal_state(self) -> None:
        """Reset the internal state of the controller."""
        super()._reset_internal_state()
        self._enabled = False

    def _postprocess_device_outputs(self, dpos, drotation):
        """
        Post-process the device outputs to scale them appropriately.

        Args:
            dpos: Position change
            drotation: Rotation change

        Returns:
            Tuple of processed position and rotation changes
        """
        # Scale position and rotation
        # TODO(dhanush) : Adjust these multipliers as needed
        dpos = dpos * 2.0
        drotation = drotation * 1.5

        # Clip values
        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation
