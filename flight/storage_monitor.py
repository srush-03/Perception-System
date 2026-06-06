"""
flight/storage_monitor.py — Background thread watching disk free space.
Emits alerts and can reduce keyframe rate or halt capture.
"""
import psutil
import os
import time
import threading
import logging
from alert_writer import alert_storage_warning

log = logging.getLogger(__name__)


class StorageMonitor:

    def __init__(self, cfg: dict, keyframe_selector=None):
        self.warn_gb     = cfg.get("warn_gb",     4.0)
        self.critical_gb = cfg.get("critical_gb", 2.0)
        self.halt_gb     = cfg.get("halt_gb",     1.0)
        self.interval    = cfg.get("check_interval_sec", 30)
        self.watch_path  = cfg.get("watch_path", ".")
        self._selector   = keyframe_selector   # optional: to tighten thresholds
        self._running    = False
        self._halt       = False
        self._thread     = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()
        log.info("StorageMonitor started")

    def stop(self):
        self._running = False

    @property
    def should_halt(self) -> bool:
        return self._halt

    def _run(self):
        while self._running:
            self._check()
            time.sleep(self.interval)

    def _check(self):
        try:
            usage = psutil.disk_usage(self.watch_path)
            free_gb = usage.free / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)

            if free_gb < self.halt_gb:
                log.critical(f"Storage HALT: {free_gb:.2f} GB free")
                self._halt = True
                alert_storage_warning("HALT", free_gb, used_gb,
                                      "CAPTURE_HALTED")

            elif free_gb < self.critical_gb:
                log.error(f"Storage CRITICAL: {free_gb:.2f} GB free")
                if self._selector:
                    self._selector.motion_thresh = 25   # tighten
                    self._selector.force_every_sec = 5.0
                alert_storage_warning("CRITICAL", free_gb, used_gb,
                                      "REDUCED_KEYFRAME_RATE")

            elif free_gb < self.warn_gb:
                log.warning(f"Storage WARN: {free_gb:.2f} GB free")
                alert_storage_warning("WARN", free_gb, used_gb, "")

        except Exception as e:
            log.error(f"StorageMonitor error: {e}")
