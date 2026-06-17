"""
nav/nav_interface.py — Abstract base for all navigation backends + PoseStamp.

Every nav backend (ROS2Nav, FileNav, future backends) must implement
NavInterface so the rest of the system never cares where pose comes from.

PoseStamp fields match what ros2_nav.py and file_nav.py already construct:
    PoseStamp(x=, y=, z=, yaw=, timestamp=, source=, valid=)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PoseStamp:
    """A single timestamped pose estimate, in base-station-relative meters."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0
    source: str = "none"     # e.g. "orb_slam3", "orb_slam3_odom", "file"
    valid: bool = False      # False = no recent pose, treat as stale/unknown


class NavInterface(ABC):
    """
    Abstract navigation source. Implementations must be non-blocking:
    get_pose() should return immediately with the latest cached pose,
    never wait on I/O or network.
    """

    def start(self) -> None:
        """Begin listening for pose updates (e.g. start ROS2 node thread).
        Default no-op — backends like FileNav poll lazily inside get_pose()
        and don't need a background thread."""
        pass

    @abstractmethod
    def get_pose(self) -> PoseStamp:
        """Return the most recently received pose. If no pose has been
        received yet, or the last pose is older than the configured
        timeout, return a PoseStamp with valid=False."""
        raise NotImplementedError

    def stop(self) -> None:
        """Optional: clean shutdown of background threads/nodes.
        Default no-op; override if your backend needs cleanup."""
        pass