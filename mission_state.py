"""
mission_state.py — Central state machine + pipeline checkpoint manager.
All state transitions go through MissionState.transition().
"""
import json
import os
import time
import logging
from typing import Optional
from alert_writer import alert_mission_state

log = logging.getLogger(__name__)

STATES = [
    "IDLE",
    "PREFLIGHT_CHECK",
    "FLIGHT_ACTIVE",
    "LANDING_DETECTED",
    "DOCKED_FOR_CHARGING",
    "POST_FLIGHT_PROCESSING",
    "READY_FOR_TRANSFER",
    "REVALIDATION_COMPLETE",
    "ERROR",
]

STATE_FILE      = "state/mission_state.json"
CHECKPOINT_FILE = "state/pipeline_checkpoint.json"


class MissionState:

    def __init__(self, sortie_id: str = ""):
        self._state     = "IDLE"
        self._sortie_id = sortie_id or self._gen_sortie_id()
        self._history   = []
        os.makedirs("state", exist_ok=True)
        self._write_state_file()

    @property
    def state(self) -> str:
        return self._state

    @property
    def sortie_id(self) -> str:
        return self._sortie_id

    def transition(self, new_state: str, trigger_source: str = "",
                   notes: str = "") -> bool:
        if new_state not in STATES:
            log.error(f"Invalid state: {new_state}")
            return False
        prev = self._state
        self._state = new_state
        self._history.append({
            "from": prev, "to": new_state,
            "time": time.time(), "trigger": trigger_source
        })
        log.info(f"STATE: {prev} → {new_state}  [{trigger_source}]")
        self._write_state_file()
        alert_mission_state(new_state, prev, self._sortie_id,
                            trigger_source, notes)
        return True

    def _write_state_file(self):
        d = {
            "state":     self._state,
            "sortie_id": self._sortie_id,
            "timestamp": time.time(),
            "history":   self._history[-10:],  # last 10 transitions
        }
        with open(STATE_FILE, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def _gen_sortie_id() -> str:
        from datetime import datetime
        return "sortie_" + datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Pipeline Checkpoint ───────────────────────────────────────────────────────

class PipelineCheckpoint:
    """
    Tracks post-flight processing progress.
    On crash/restart, pipeline resumes from last completed stage.
    """

    _DEFAULTS = {
        "frame_selection_complete": False,
        "stitching_complete":       False,
        "stitching_output":         None,
        "matching_complete":        False,
        "matching_progress":        {"processed": 0, "total": 0},
        "dedup_complete":           False,
        "validation_complete":      False,
        "transfer_ready":           False,
    }

    def __init__(self, sortie_id: str):
        self._sortie_id = sortie_id
        self._data = {"sortie_id": sortie_id,
                      "checkpoint_time": time.time(),
                      **self._DEFAULTS}
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r") as f:
                    saved = json.load(f)
                if saved.get("sortie_id") == self._sortie_id:
                    self._data.update(saved)
                    log.info(f"Checkpoint loaded for {self._sortie_id}")
                    self._log_resume_point()
            except Exception as e:
                log.warning(f"Checkpoint load failed: {e}")

    def _log_resume_point(self):
        for stage in ["frame_selection_complete", "stitching_complete",
                       "matching_complete", "dedup_complete",
                       "validation_complete", "transfer_ready"]:
            if not self._data[stage]:
                log.info(f"Will resume from: {stage}")
                break

    def mark(self, stage: str, value=True, **extra):
        self._data[stage] = value
        self._data["checkpoint_time"] = time.time()
        for k, v in extra.items():
            self._data[k] = v
        self._save()

    def get(self, stage: str, default=None):
        return self._data.get(stage, default)

    def is_done(self, stage: str) -> bool:
        return bool(self._data.get(stage, False))

    def _save(self):
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(self._data, f, indent=2)
