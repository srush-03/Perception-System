"""
flight/flight_manager.py — Orchestrates the 4-thread flight pipeline.

Threads:
  1. CaptureThread  — grabs frames from camera → raw_queue
  2. FilterThread   — quality + keyframe selection → kf_queue
  3. StorageThread  — writes HD frame + metadata to disk
  4. BoundaryThread — yellow line detection on latest raw frame

All queues are bounded. No shared mutable state without locks.
"""
import cv2
import os
import json
import time
import queue
import threading
import logging
from typing import Optional

from camera.camera_interface import CameraInterface
from nav.nav_interface import NavInterface
from flight.quality_filter import QualityFilter
from flight.keyframe_selector import KeyframeSelector
from flight.boundary_detector import BoundaryDetector
from flight.storage_monitor import StorageMonitor

log = logging.getLogger(__name__)

_SENTINEL = None   # signals threads to stop


class FlightManager:

    def __init__(self, camera: CameraInterface, nav: NavInterface,
                 cfg: dict, sortie_id: str):
        self._cam       = camera
        self._nav       = nav
        self._sortie_id = sortie_id
        self._frames_dir = cfg["paths"]["frames_dir"]
        os.makedirs(self._frames_dir, exist_ok=True)

        self._q_raw = queue.Queue(maxsize=4)
        self._q_kf  = queue.Queue(maxsize=8)

        self._quality   = QualityFilter(cfg["quality"])
        self._selector  = KeyframeSelector(cfg["keyframe"])
        self._boundary  = BoundaryDetector(cfg["boundary_detector"])
        self._storage_mon = StorageMonitor(
            cfg["storage"], keyframe_selector=self._selector
        )

        self._running = False
        self._frame_count  = 0
        self._kf_count     = 0
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start all threads. Call after camera.open()."""
        self._running = True
        self._storage_mon.start()

        self._t_capture  = threading.Thread(target=self._capture_loop,  daemon=True, name="Capture")
        self._t_filter   = threading.Thread(target=self._filter_loop,   daemon=True, name="Filter")
        self._t_storage  = threading.Thread(target=self._storage_loop,  daemon=True, name="Storage")
        self._t_boundary = threading.Thread(target=self._boundary_loop, daemon=True, name="Boundary")

        for t in [self._t_capture, self._t_filter, self._t_storage, self._t_boundary]:
            t.start()

        log.info(f"FlightManager started — sortie {self._sortie_id}")

    def stop(self):
        """Signal all threads to stop cleanly."""
        log.info("FlightManager: stopping...")
        self._running = False
        # Drain and unblock queues
        for _ in range(4):
            try: self._q_raw.put_nowait(_SENTINEL)
            except queue.Full: pass
        for _ in range(4):
            try: self._q_kf.put_nowait(_SENTINEL)
            except queue.Full: pass

        self._t_capture.join(timeout=3)
        self._t_filter.join(timeout=5)
        self._t_storage.join(timeout=10)
        self._t_boundary.join(timeout=3)
        self._storage_mon.stop()

        log.info(f"FlightManager stopped. "
                 f"Captured {self._frame_count} frames, "
                 f"{self._kf_count} keyframes saved.")

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "frames_captured": self._frame_count,
                "keyframes_saved": self._kf_count,
                "sortie_id": self._sortie_id,
            }

    # ── Thread 1: Capture ─────────────────────────────────────────────────────

    def _capture_loop(self):
        log.info("CaptureThread: started")
        while self._running:
            if self._storage_mon.should_halt:
                log.critical("CaptureThread: storage halt — stopping capture")
                self._running = False
                break

            frame, ts = self._cam.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            pose = self._nav.get_pose()

            packet = {"frame": frame, "ts": ts, "pose": pose}

            try:
                self._q_raw.put_nowait(packet)
            except queue.Full:
                # Drop oldest raw frame (not a keyframe yet, acceptable)
                try: self._q_raw.get_nowait()
                except queue.Empty: pass
                self._q_raw.put_nowait(packet)

            with self._lock:
                self._frame_count += 1

        log.info("CaptureThread: stopped")

    # ── Thread 2: Filter ──────────────────────────────────────────────────────

    def _filter_loop(self):
        log.info("FilterThread: started")
        while True:
            try:
                pkt = self._q_raw.get(timeout=2.0)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if pkt is _SENTINEL:
                self._q_kf.put(_SENTINEL)
                break

            frame = pkt["frame"]
            ts    = pkt["ts"]
            pose  = pkt["pose"]

            passed, reason = self._quality.check(frame)
            if not passed:
                log.debug(f"Quality reject: {reason}")
                continue

            is_kf, kf_reason = self._selector.is_keyframe(frame, ts)
            if not is_kf:
                log.debug(f"KF reject: {kf_reason}")
                continue

            self._q_kf.put({"frame": frame, "ts": ts, "pose": pose,
                             "kf_reason": kf_reason})

        log.info("FilterThread: stopped")

    # ── Thread 3: Storage ─────────────────────────────────────────────────────

    def _storage_loop(self):
        log.info("StorageThread: started")
        while True:
            try:
                pkt = self._q_kf.get(timeout=2.0)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if pkt is _SENTINEL:
                break

            frame  = pkt["frame"]
            ts     = pkt["ts"]
            pose   = pkt["pose"]

            # Filename encodes timestamp + coordinates
            fname  = (f"frame_{ts:.3f}_"
                      f"x{pose.x:.3f}_y{pose.y:.3f}_z{pose.z:.3f}.jpg")
            fpath  = os.path.join(self._frames_dir, fname)

            cv2.imwrite(fpath, frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Sidecar metadata JSON
            meta = {
                "filename":  fname,
                "timestamp": ts,
                "pose": {"x": pose.x, "y": pose.y, "z": pose.z,
                         "yaw": pose.yaw, "source": pose.source},
                "kf_reason": pkt["kf_reason"],
                "sortie_id": self._sortie_id,
            }
            meta_path = fpath.replace(".jpg", ".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            with self._lock:
                self._kf_count += 1
            log.debug(f"Saved keyframe: {fname}")

            # Explicit release
            del frame

        log.info("StorageThread: stopped")

    # ── Thread 4: Boundary ───────────────────────────────────────────────────

    def _boundary_loop(self):
        log.info("BoundaryThread: started")
        _local_q: queue.Queue = self._q_raw

        # This thread reads from raw_queue using a SEPARATE listener
        # It does NOT drain from the main raw_queue.
        # Implementation: we watch the capture output by sampling latest frames
        # from a shadow reference (updated by CaptureThread)
        _latest_pkt = {}
        _orig_put = self._q_raw.put

        def _shadow_put(pkt, *args, **kwargs):
            if pkt is not _SENTINEL:
                _latest_pkt.clear()
                _latest_pkt.update(pkt if pkt else {})
            return _orig_put(pkt, *args, **kwargs)

        self._q_raw.put = _shadow_put

        frame_id = 0
        while self._running:
            time.sleep(0.2)     # check ~5 Hz — boundary doesn't need 8 Hz
            frame = _latest_pkt.get("frame")
            if frame is None:
                continue
            try:
                detected, conf, direction, bbox = self._boundary.detect(
                    frame, frame_id=f"frame_{frame_id:05d}"
                )
                if detected:
                    log.warning(f"BOUNDARY DETECTED: {direction} conf={conf:.2f}")
            except Exception as e:
                log.error(f"BoundaryThread error: {e}")
            frame_id += 1

        log.info("BoundaryThread: stopped")
