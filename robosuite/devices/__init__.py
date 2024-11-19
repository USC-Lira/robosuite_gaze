from .device import Device
from .keyboard import Keyboard
from .ps4_controller import PS4Controller

try:
    from .spacemouse import SpaceMouse
except ImportError as e:
    print("Exception!", e)
    print(
        """Unable to load module hid, required to interface with SpaceMouse.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )
