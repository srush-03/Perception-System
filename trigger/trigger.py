"""
trigger/ — Charging dock trigger abstraction.
Arduino Nano monitors a limit switch. When drone docks, switch closes,
Arduino sends 'DOCKED\n' over USB serial at 9600 baud.

If Arduino is unreachable, falls back to watching mission_state.json.
"""
import json
import os
import time
import logging
import threading
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


# ── Abstract Base ────────────────────────────────────────────────────────────

class TriggerInterface(ABC):
    @abstractmethod
    def wait_for_dock(self, timeout: float = 600.0) -> bool:
        """Block until DOCKED signal received or timeout (seconds). Returns True if docked."""

    @abstractmethod
    def is_docked(self) -> bool:
        """Non-blocking check — True if currently docked."""


# ── Arduino Serial Trigger ───────────────────────────────────────────────────

class ArduinoSerialTrigger(TriggerInterface):

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 9600,
                 dock_keyword: str = "DOCKED"):
        self._port    = port
        self._baud    = baud
        self._keyword = dock_keyword.strip().upper()
        self._docked  = False

    def wait_for_dock(self, timeout: float = 600.0) -> bool:
        try:
            import serial
        except ImportError:
            log.error("pyserial not installed: pip install pyserial")
            return False

        try:
            ser = serial.Serial(self._port, self._baud, timeout=1.0)
            log.info(f"Arduino trigger: listening on {self._port} @ {self._baud} baud")
            deadline = time.monotonic() + timeout

            while time.monotonic() < deadline:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip().upper()
                    if line:
                        log.debug(f"Arduino: '{line}'")
                    if self._keyword in line:
                        self._docked = True
                        ser.close()
                        log.info(f"Arduino trigger: DOCKED received")
                        return True
                except Exception as e:
                    log.warning(f"Arduino read error: {e}")
                    time.sleep(0.1)

            ser.close()
            log.warning("Arduino trigger: timeout waiting for DOCKED")
            return False

        except Exception as e:
            log.error(f"Arduino serial failed: {e}")
            return False

    def is_docked(self) -> bool:
        return self._docked


# ── File-Based Trigger (fallback) ─────────────────────────────────────────────

class FileTrigger(TriggerInterface):
    """
    Watches mission_state.json for state == 'DOCKED_FOR_CHARGING'.
    Used when Arduino is unavailable (testing, no hardware).
    """

    def __init__(self, state_file: str = "state/mission_state.json"):
        self._state_file = state_file
        self._docked = False

    def wait_for_dock(self, timeout: float = 600.0) -> bool:
        log.info(f"FileTrigger: watching {self._state_file}")
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self._check_file():
                self._docked = True
                return True
            time.sleep(0.5)

        log.warning("FileTrigger: timeout waiting for DOCKED state")
        return False

    def _check_file(self) -> bool:
        try:
            with open(self._state_file, "r") as f:
                d = json.load(f)
            return d.get("state", "").upper() == "DOCKED_FOR_CHARGING"
        except Exception:
            return False

    def is_docked(self) -> bool:
        return self._docked


# ── Factory ──────────────────────────────────────────────────────────────────

def make_trigger(trig_cfg: dict) -> TriggerInterface:
    ttype    = trig_cfg.get("type", "file").lower()
    fallback = trig_cfg.get("fallback", "file").lower()

    if ttype == "arduino_serial":
        try:
            import serial
            t = ArduinoSerialTrigger(
                port=trig_cfg.get("port", "/dev/ttyUSB0"),
                baud=trig_cfg.get("baud", 9600),
                dock_keyword=trig_cfg.get("dock_keyword", "DOCKED"),
            )
            log.info("TriggerFactory: ArduinoSerialTrigger ready")
            return t
        except ImportError:
            log.warning("pyserial missing — using FileTrigger fallback")

    # file fallback
    state_file = "state/mission_state.json"
    log.info(f"TriggerFactory: FileTrigger watching {state_file}")
    return FileTrigger(state_file=state_file)
