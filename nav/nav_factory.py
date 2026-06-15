"""
nav/nav_factory.py — Instantiates correct nav backend from config.
If ROS2 nav fails to start, automatically falls back to file-based nav.
"""
import logging
from nav.nav_interface import NavInterface

log = logging.getLogger(__name__)


def make_nav(nav_cfg: dict) -> NavInterface:
    nav_type = nav_cfg.get("type", "file").lower()

    if nav_type in ("orb_slam3_ros2", "vins_ros2"):
        try:
            from nav.ros2_nav import ROS2Nav
            # ORB-SLAM3 uses PoseStamped; legacy VINS used Odometry
            default_topic    = "/orbslam3/camera_pose" if nav_type == "orb_slam3_ros2" \
                               else "/vins_estimator/odometry"
            default_msg_type = "PoseStamped" if nav_type == "orb_slam3_ros2" \
                               else "Odometry"
            nav = ROS2Nav(
                topic=nav_cfg.get("topic", default_topic),
                msg_type=nav_cfg.get("msg_type", default_msg_type),
            )
            nav.start()
            log.info(f"NavFactory: ROS2Nav started ({nav_type})")
            return nav
        except Exception as e:
            log.warning(f"ROS2Nav failed ({e}), falling back to FileNav")
            return _make_file_nav(nav_cfg)

    elif nav_type == "file":
        return _make_file_nav(nav_cfg)

    else:
        log.warning(f"Unknown nav type '{nav_type}', using FileNav")
        return _make_file_nav(nav_cfg)


def _make_file_nav(nav_cfg: dict) -> NavInterface:
    from nav.file_nav import FileNav
    pose_file = nav_cfg.get("pose_file", "state/current_pose.json")
    poll_ms   = nav_cfg.get("poll_interval_ms", 50)
    log.info(f"NavFactory: FileNav watching {pose_file}")
    return FileNav(pose_file=pose_file, poll_interval_ms=poll_ms)