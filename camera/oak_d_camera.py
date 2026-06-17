"""
camera/oak_d_camera.py — OAK-D Lite backend via DepthAI SDK.

Requires: pip install depthai (uncomment in requirements.txt)
Produces 1920x1080 RGB by default; downscaled to config width/height
on-device via ColorCamera.setIspScale() to reduce USB bandwidth and
match the rest of the pipeline's expected resolution (1280x720).

NOTE: this is the OAK-D Lite Auto-Focus variant by default. Drone
vibration can blur frames with AF hunting — swap to Fixed-Focus
hardware variant before competition if possible (flagged in project
notes as a pending hardware item).
"""
import time
import logging
from typing import Optional, Tuple
import numpy as np

from camera.camera_interface import CameraInterface

log = logging.getLogger(__name__)


class OakDCamera(CameraInterface):

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 10):
        self._width  = width
        self._height = height
        self._fps    = fps
        self._device = None
        self._queue  = None
        self._pipeline = None

    def open(self) -> bool:
        try:
            import depthai as dai
        except ImportError:
            log.error("OakDCamera: depthai not installed. "
                       "Run: pip install depthai --break-system-packages")
            return False

        try:
            pipeline = dai.Pipeline()
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setFps(self._fps)

            # Downscale on-device from 1920x1080 to target (e.g. 1280x720)
            # via ISP scale, reduces USB3 bandwidth vs full-res streaming.
            cam_rgb.setIspScale(2, 3)  # 1920x1080 -> 1280x720
            cam_rgb.setInterleaved(False)
            cam_rgb.initialControl.setManualFocus(130)  # mitigate AF hunting

            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("rgb")
            cam_rgb.isp.link(xout.input)

            self._pipeline = pipeline
            self._device   = dai.Device(pipeline)
            self._queue    = self._device.getOutputQueue(
                name="rgb", maxSize=4, blocking=False)

            log.info(f"OakDCamera opened: target {self._width}x{self._height} "
                      f"@ {self._fps}fps (ISP-scaled from 1920x1080)")
            return True

        except Exception as e:
            log.error(f"OakDCamera: failed to open — {e}")
            self._device = None
            return False

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        if self._queue is None:
            return None, time.monotonic()
        in_rgb = self._queue.tryGet()
        ts = time.monotonic()
        if in_rgb is None:
            return None, ts
        return in_rgb.getCvFrame(), ts

    def release(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            except Exception as e:
                log.warning(f"OakDCamera: release error — {e}")
            self._device = None
            self._queue  = None

    def is_open(self) -> bool:
        return self._device is not None