"""
camera/usb_camera.py — USB/Logitech webcam backend (OpenCV VideoCapture).

Known platform fixes baked in from earlier debugging:
  - Windows: MSMF backend throws grab errors → use CAP_DSHOW
  - Linux/Jetson: use CAP_V4L2 explicitly (avoids GStreamer auto-pick issues)
"""
import cv2
import time
import logging
import platform
from typing import Optional, Tuple
import numpy as np

from camera.camera_interface import CameraInterface

log = logging.getLogger(__name__)


class UsbCamera(CameraInterface):

    def __init__(self, device_index: int = 0, width: int = 1280,
                 height: int = 720, fps: int = 10):
        self._index  = device_index
        self._width  = width
        self._height = height
        self._fps    = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        self._cap = cv2.VideoCapture(self._index, backend)

        if not self._cap.isOpened():
            log.warning(f"UsbCamera: backend {backend} failed, trying CAP_ANY")
            self._cap = cv2.VideoCapture(self._index, cv2.CAP_ANY)

        if not self._cap.isOpened():
            log.error(f"UsbCamera: could not open device index {self._index}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        log.info(f"UsbCamera opened: requested {self._width}x{self._height} "
                 f"got {actual_w:.0f}x{actual_h:.0f}")
        return True

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        if self._cap is None or not self._cap.isOpened():
            return None, time.monotonic()
        ok, frame = self._cap.read()
        ts = time.monotonic()
        if not ok:
            return None, ts
        return frame, ts

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()