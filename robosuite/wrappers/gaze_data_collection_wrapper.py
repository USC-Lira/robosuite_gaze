"""
Wrapper for collecting demonstration data along with gaze information in RoboSuite.
Extends the base DataCollectionWrapper to include gaze tracking data.
"""

import os
import time
import json
import numpy as np
from typing import Dict, Tuple, Any, Optional

from robosuite.wrappers import Wrapper


class GazeDataCollectionWrapper(Wrapper):
    """
    A wrapper for collecting both demonstration and gaze data during robot manipulation.
    """

    def __init__(self, env, directory: str, collect_freq: int = 1, flush_freq: int = 100):
        """
        Initialize the gaze data collection wrapper.

        Args:
            env: The environment to monitor
            directory: Directory to store collected data
            collect_freq: Frequency of saving simulation state (in env steps)
            flush_freq: Frequency of writing to disk (in env steps)
        """
        super().__init__(env)

        # Setup logging directory
        self.directory = directory
        if not os.path.exists(directory):
            print(f"Creating data collection directory at {directory}")
            os.makedirs(directory)

        # Collection parameters
        self.collect_freq = collect_freq
        self.flush_freq = flush_freq

        # Episode data storage
        self._reset_storage()

        # Episode metadata
        self.ep_directory: Optional[str] = None
        self.has_interaction = False
        self._current_task_instance_state = None
        self._current_task_instance_xml = None

    def _reset_storage(self) -> None:
        """Reset all data storage containers."""
        self.states = []
        self.action_infos = []
        self.gaze_infos = []
        self.successful = False

    def _create_episode_directory(self) -> None:
        """Create a new directory for the current episode."""
        timestamp = f"ep_{int(time.time()*1000)}"
        self.ep_directory = os.path.join(self.directory, timestamp)
        os.makedirs(self.ep_directory)
        print(f"Created episode directory: {self.ep_directory}")

    def _save_episode_metadata(self) -> None:
        """Save the initial episode state and model information."""
        if not self.ep_directory:
            return

        # Save model XML
        xml_path = os.path.join(self.ep_directory, "model.xml")
        with open(xml_path, "w") as f:
            f.write(self._current_task_instance_xml)

        # Save metadata if available
        if hasattr(self.env, "get_ep_meta"):
            meta_path = os.path.join(self.ep_directory, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(self.env.get_ep_meta(), f)

    def _start_new_episode(self) -> None:
        """Initialize a new episode's data collection."""
        # Save any remaining data from previous episode
        if self.has_interaction:
            self._flush()

        # Reset episode state
        self.t = 0
        self.has_interaction = False
        self._reset_storage()

        # Save initial state
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())

        # Ensure deterministic playback
        self.env.reset_from_xml_string(self._current_task_instance_xml)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(self._current_task_instance_state)
        self.env.sim.forward()

    def _flush(self) -> None:
        """Write collected data to disk."""
        if not self.ep_directory:
            return

        timestamp = int(time.time() * 1000)
        filepath = os.path.join(self.ep_directory, f"state_{timestamp}.npz")

        # Get environment name
        env_name = (
            self.env.unwrapped.__class__.__name__ if hasattr(self.env, "unwrapped") else self.env.__class__.__name__
        )

        # Save collected data
        np.savez(
            filepath,
            states=np.array(self.states),
            action_infos=self.action_infos,
            gaze_infos=self.gaze_infos,
            successful=self.successful,
            env=env_name,
        )

        # Reset storage
        self._reset_storage()

    def step(self, action: np.ndarray, gaze_data: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute environment step and collect both action and gaze data.

        Args:
            action: Action to take in environment
            gaze_data: Gaze tracking data for the current timestep

        Returns:
            Tuple containing (observation, reward, done, info)
        """
        # Execute environment step
        obs, reward, done, info = super().step(action)
        self.t += 1

        # Initialize episode directory on first interaction
        if not self.has_interaction:
            self.has_interaction = True
            self._create_episode_directory()
            self._save_episode_metadata()
            self.states.append(self._current_task_instance_state)

        # Collect data at specified frequency
        if self.t % self.collect_freq == 0:
            # Save simulation state
            self.states.append(self.env.sim.get_state().flatten())

            # Save action and gaze data
            self.action_infos.append({"actions": np.array(action)})
            self.gaze_infos.append({"gaze": np.array(gaze_data)})

            # Check success
            if self.env._check_success():
                self.successful = True

        # Write to disk at flush frequency
        if self.t % self.flush_freq == 0:
            self._flush()

        return obs, reward, done, info

    def reset(self) -> Dict:
        """Reset environment and start new episode data collection."""
        obs = super().reset()
        self._start_new_episode()
        return obs

    def close(self) -> None:
        """Clean up and save any remaining data."""
        if self.has_interaction:
            self._flush()
        self.env.close()
