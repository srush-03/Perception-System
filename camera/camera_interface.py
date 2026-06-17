"""
camera/camera_interface.py — Abstract base for all camera backends.

Contract used throughout the system (flight_manager.py, main.py, tools/*):
    cam = make_camera(cfg)
    cam.open()              -> bool
    frame, ts = cam.get_frame()   -> (np.ndarray | None, float)
    cam.release()

Implementations: UsbCamera (USB/Logitech webcam), OakDCamera (OAK-D Lite
via DepthAI). Add new hardware by subclassing CameraInterface and
registering it in camera_factory.py — no changes needed anywhere else.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class CameraInterface(ABC):

    @abstractmethod
    def open(self) -> bool:
        """Open/initialize the camera. Return True on success, False on
        failure (never raise — callers check the bool and abort cleanly)."""
        raise NotImplementedError

    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (frame, timestamp). frame is a BGR np.ndarray (OpenCV
        convention) or None if no frame is currently available.
        timestamp is time.monotonic() at capture, NOT wall-clock time —
        this matters for matching against pose timestamps from nav."""
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Release hardware/driver resources. Must be safe to call even
        if open() was never called or failed."""
        raise NotImplementedError

    def is_open(self) -> bool:
        """Optional convenience check. Default: assume open if no error."""
        return True