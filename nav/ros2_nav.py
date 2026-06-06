"""
nav/ros2_nav.py — ROS2 subscriber for VINS/SLAM odometry.
Runs a rclpy node in a background thread.
get_pose() returns latest cached pose immediately (non-blocking).

Compatible topics:
  nav_msgs/Odometry   (VINS-Mono, VINS-Fusion, ORB-SLAM3)
  geometry_msgs/PoseStamped

Set topic and msg_type in system_config.yaml:
  navigation:
    type: vins_ros2
    topic: /vins_estimator/odometry
    msg_type: Odometry          # Odometry | PoseStamped
"""
import threading
import time
import logging
from nav.nav_interface import NavInterface, PoseStamp
import math

log = logging.getLogger(__name__)


class ROS2Nav(NavInterface):

    def __init__(self, topic: str = "/vins_estimator/odometry",
                 msg_type: str = "Odometry"):
        self._topic    = topic
        self._msg_type = msg_type
        self._latest   = PoseStamp(source="ros2", valid=False)
        self._lock     = threading.Lock()
        self._node     = None
        self._thread   = None
        self._running  = False

    def start(self):
        """Call once before flight to start ROS2 subscriber thread."""
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        log.info(f"ROS2Nav: started subscriber on {self._topic}")

    def stop(self):
        self._running = False
        if self._node:
            try:
                import rclpy
                self._node.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass

    def _spin(self):
        try:
            import rclpy
            from rclpy.node import Node

            rclpy.init()
            self._node = _PoseListenerNode(
                self._topic, self._msg_type, self._on_pose
            )
            self._running = True
            rclpy.spin(self._node)
        except ImportError:
            log.error("rclpy not available — ROS2Nav cannot start. "
                      "Falling back to file-based nav.")
        except Exception as e:
            log.error(f"ROS2Nav spin error: {e}")

    def _on_pose(self, pose: PoseStamp):
        with self._lock:
            self._latest = pose

    def get_pose(self) -> PoseStamp:
        with self._lock:
            return self._latest

    def is_available(self) -> bool:
        with self._lock:
            age = time.monotonic() - self._latest.timestamp
            return self._latest.valid and age < 2.0


class _PoseListenerNode:
    """Inner ROS2 node — instantiated inside the spin thread."""

    def __init__(self, topic: str, msg_type: str, callback):
        from rclpy.node import Node
        import rclpy

        self._cb = callback

        class _Node(Node):
            def __init__(inner):
                super().__init__("ascend_nav_listener")

        self._node = _Node()

        if msg_type == "Odometry":
            from nav_msgs.msg import Odometry
            self._node.create_subscription(
                Odometry, topic, self._odom_cb, 10
            )
        elif msg_type == "PoseStamped":
            from geometry_msgs.msg import PoseStamped
            self._node.create_subscription(
                PoseStamped, topic, self._pose_stamped_cb, 10
            )
        else:
            log.error(f"Unsupported msg_type: {msg_type}")

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
        self._cb(PoseStamp(
            x=pos.x, y=pos.y, z=pos.z,
            yaw=yaw,
            timestamp=time.monotonic(),
            source="vins_ros2",
            valid=True,
        ))

    def _pose_stamped_cb(self, msg):
        pos = msg.pose.position
        ori = msg.pose.orientation
        yaw = self._quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
        self._cb(PoseStamp(
            x=pos.x, y=pos.y, z=pos.z,
            yaw=yaw,
            timestamp=time.monotonic(),
            source="ros2_pose",
            valid=True,
        ))

    @staticmethod
    def _quat_to_yaw(x, y, z, w) -> float:
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny, cosy)

    def destroy_node(self):
        self._node.destroy_node()
