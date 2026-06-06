"""
flight/boundary_detector.py — Detects yellow boundary lines.
Runs on raw frames (not just keyframes) in a separate thread.
Generates boundary_warning alerts; never blocks capture pipeline.
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from alert_writer import alert_boundary_warning

log = logging.getLogger(__name__)


class BoundaryDetector:

    def __init__(self, cfg: dict):
        self.hsv_lower = np.array(cfg.get("hsv_lower", [18, 80, 80]),  np.uint8)
        self.hsv_upper = np.array(cfg.get("hsv_upper", [35, 255, 255]), np.uint8)
        self.min_ar    = cfg.get("min_aspect_ratio", 4.0)
        self.max_fill  = cfg.get("max_fill_ratio",   0.6)
        self.roi_frac  = cfg.get("roi_fraction",     0.5)
        self._frame_counter = 0

    def detect(self, frame: np.ndarray,
               frame_id: str = "") -> Tuple[bool, float, str, Optional[list]]:
        """
        Returns (detected, confidence, direction, bbox_or_None).
        direction: TURN_LEFT | TURN_RIGHT | TURN_BACK | CAUTION
        """
        h, w = frame.shape[:2]
        roi_y = int(h * (1.0 - self.roi_frac))
        roi = frame[roi_y:h, :]

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0.0, "", None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < 500:
            return False, 0.0, "", None

        rx, ry, rw, rh = cv2.boundingRect(best)
        ar   = rw / max(rh, 1)
        fill = area / max(rw * rh, 1)

        if ar < self.min_ar or fill > self.max_fill:
            return False, 0.0, "", None

        # Confidence = normalized area proportion of ROI
        conf = min(1.0, area / (roi.shape[0] * roi.shape[1] * 0.3))
        cx   = rx + rw // 2
        direction = self._get_direction(cx, w, ry)

        # Adjust bbox back to full-frame coordinates
        bbox = [rx, roi_y + ry, rx + rw, roi_y + ry + rh]

        if frame_id:
            alert_boundary_warning(frame_id, direction, conf, bbox)

        return True, conf, direction, bbox

    @staticmethod
    def _get_direction(cx: int, frame_w: int, ry: int) -> str:
        third = frame_w // 3
        if ry < 10:          # very close to top of ROI = near boundary ahead
            return "TURN_BACK"
        if cx < third:
            return "TURN_RIGHT"
        if cx > 2 * third:
            return "TURN_LEFT"
        return "CAUTION"
