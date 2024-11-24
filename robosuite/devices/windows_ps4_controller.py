"""
Driver class for PS4 controller on Windows using Pygame.
"""

import pygame
import numpy as np

from robosuite.devices import Device


class WindowsPS4Controller(Device):
    """
    A minimalistic driver class for PS4 controller using Pygame on Windows.
    Args:
        env (RobotEnv): Environment containing the robot(s) to control
        pos_sensitivity (float): Position control sensitivity scaling
        rot_sensitivity (float): Rotation control sensitivity scaling
    """

    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0):
        super().__init__(env)

        self._display_controls()
        self._reset_internal_state()

        # Control parameters
        self._reset_state = 0
        self._enabled = False
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # Initialize pygame and controller
        pygame.init()
        pygame.joystick.init()
        try:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print("PS4 Controller initialized")
        except pygame.error as e:
            print("Failed to initialize controller:", e)
            self.controller = None

    @staticmethod
    def _display_controls():
        """Method to pretty print controls."""
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Left stick", "move arm horizontally in x-y plane")
        print_command("Right stick", "move arm vertically")
        print_command("L2", "close gripper")
        print_command("Circle", "reset simulation")
        print_command("b", "toggle arm/base mode (if applicable)")
        print("")

    def _reset_internal_state(self):
        """Resets internal state of controller, except for the reset signal."""
        super()._reset_internal_state()
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    def start_control(self):
        """Method that should be called externally before controller can start receiving commands."""
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the controller.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        if not self._enabled or not self.controller:
            return self._zero_state()

        pygame.event.pump()

        def deadzone(value, threshold=0.05):
            return 0.0 if abs(value) < threshold else value

        # Get position (use deadzone to prevent drift)
        dpos = np.zeros(3)
        dpos[0] = +deadzone(self.controller.get_axis(1)) * 0.1 * self.pos_sensitivity  # Left stick Y
        dpos[1] = +deadzone(self.controller.get_axis(0)) * 0.1 * self.pos_sensitivity  # Left stick X
        dpos[2] = -deadzone(self.controller.get_axis(3)) * 0.1 * self.pos_sensitivity  # Right stick Y

        # Get other states
        grasp = 1 if self.controller.get_axis(4) > 0.5 else 0  # L2 trigger
        reset = 1 if self.controller.get_button(1) else 0      # Circle button

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.zeros(3),
            grasp=grasp,
            reset=reset,
            base_mode=int(self.base_mode),
        )

    def _zero_state(self):
        """Returns zero state when controller is disabled."""
        return dict(
            dpos=np.zeros(3),
            rotation=self.rotation,
            raw_drotation=np.zeros(3),
            grasp=0,
            reset=0,
            base_mode=0,
        )

    def _postprocess_device_outputs(self, dpos, drotation):
        """Post-process the device outputs to scale them appropriately."""
        dpos = dpos * 2.0
        drotation = drotation * 1.5

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation
