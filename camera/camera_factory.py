"""
camera/camera_factory.py — Instantiates correct camera backend from config.

Mirrors nav/nav_factory.py pattern: read cfg["type"], dispatch to the
matching implementation. Add new hardware (CSI, RealSense, etc.) by
writing a new file in camera/ + registering its type string here —
no other file in the system needs to change.
"""
import logging
from camera.camera_interface import CameraInterface

log = logging.getLogger(__name__)


def make_camera(camera_cfg: dict) -> CameraInterface:
    cam_type = camera_cfg.get("type", "usb").lower()
    width  = camera_cfg.get("width",  1280)
    height = camera_cfg.get("height", 720)
    fps    = camera_cfg.get("fps",    10)

    if cam_type == "oak_d":
        from camera.oak_d_camera import OakDCamera
        log.info("CameraFactory: OakDCamera selected")
        return OakDCamera(width=width, height=height, fps=fps)

    elif cam_type == "usb":
        from camera.usb_camera import UsbCamera
        device_index = camera_cfg.get("device_index", 0)
        log.info(f"CameraFactory: UsbCamera selected (index={device_index})")
        return UsbCamera(device_index=device_index, width=width,
                          height=height, fps=fps)

    else:
        log.warning(f"CameraFactory: unknown type '{cam_type}', "
                     f"defaulting to UsbCamera index 0")
        from camera.usb_camera import UsbCamera
        return UsbCamera(device_index=0, width=width, height=height, fps=fps)