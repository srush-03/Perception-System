"""
nav/file_nav.py — File-based pose reader for testing without ROS2.
VINS team writes current_pose.json, we read it every poll_interval_ms.
Used as fallback when ROS2 is unavailable.

Expected JSON format:
{
  "x": 3.42, "y": 7.15, "z": 2.80,
  "yaw": 1.57, "timestamp": 1748700000.123
}
"""
import json
import os
import time
import logging
from nav.nav_interface import NavInterface, PoseStamp

log = logging.getLogger(__name__)


class FileNav(NavInterface):

    def __init__(self, pose_file: str, poll_interval_ms: int = 50):
        self._pose_file = pose_file
        self._interval  = poll_interval_ms / 1000.0
        self._last_pose = PoseStamp(source="file", valid=False)
        self._last_mtime = -1.0

    def get_pose(self) -> PoseStamp:
        try:
            mtime = os.path.getmtime(self._pose_file)
            if mtime != self._last_mtime:
                with open(self._pose_file, "r") as f:
                    d = json.load(f)
                self._last_pose = PoseStamp(
                    x=float(d.get("x", 0.0)),
                    y=float(d.get("y", 0.0)),
                    z=float(d.get("z", 0.0)),
                    yaw=float(d.get("yaw", 0.0)),
                    timestamp=float(d.get("timestamp", time.monotonic())),
                    source="file",
                    valid=True,
                )
                self._last_mtime = mtime
        except FileNotFoundError:
            pass  # Return last known pose; valid=False on first call
        except Exception as e:
            log.warning(f"FileNav: read error: {e}")
        return self._last_pose

    def is_available(self) -> bool:
        return os.path.exists(self._pose_file)
