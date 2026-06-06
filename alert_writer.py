"""
alert_writer.py — Writes JSON alert files to alerts/.
Telemetry team and Dashboard team watch this directory.
We write; they read. No sockets, no APIs.
"""
import json
import os
import time
import logging
from typing import Any, Dict

log = logging.getLogger(__name__)

_ALERTS_DIR = "alerts"


def _ensure_dir():
    os.makedirs(_ALERTS_DIR, exist_ok=True)


def write_alert(event_type: str, payload: Dict[str, Any]) -> str:
    """Write an alert JSON file. Returns the file path."""
    _ensure_dir()
    ts = time.time()
    ts_str = f"{ts:.3f}".replace(".", "_")
    fname = f"alert_{ts_str}_{event_type}.json"
    fpath = os.path.join(_ALERTS_DIR, fname)

    data = {"schema": f"{event_type}_v1",
            "timestamp": ts,
            "event_type": event_type,
            **payload}

    with open(fpath, "w") as f:
        json.dump(data, f, indent=2)
    log.debug(f"Alert written: {fname}")
    return fpath


# ── Convenience wrappers ─────────────────────────────────────────────────────

def alert_mission_state(state: str, prev_state: str, sortie_id: str,
                        trigger_source: str = "", notes: str = ""):
    return write_alert("mission_state", {
        "state": state,
        "prev_state": prev_state,
        "sortie_id": sortie_id,
        "trigger_source": trigger_source,
        "notes": notes,
    })


def alert_detection(sortie_id: str, feature_type: str, instance_id: str,
                    confidence: float, tier: str, scores: dict,
                    coords: dict, proof_image: str, frame_ts: float):
    return write_alert("detection", {
        "sortie_id": sortie_id,
        "feature_type": feature_type,
        "instance_id": instance_id,
        "confidence": round(confidence, 4),
        "confidence_tier": tier,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "coordinates": coords,
        "proof_image": proof_image,
        "frame_timestamp": frame_ts,
    })


def alert_boundary_warning(frame_id: str, direction: str,
                           confidence: float, bbox: list):
    return write_alert("boundary_warning", {
        "frame_id": frame_id,
        "detected": True,
        "direction": direction,
        "confidence": round(confidence, 3),
        "bbox": bbox,
    })


def alert_storage_warning(level: str, free_gb: float, used_gb: float,
                          action: str = ""):
    return write_alert("storage_warning", {
        "level": level,          # WARN | CRITICAL | HALT
        "free_gb": round(free_gb, 2),
        "used_gb": round(used_gb, 2),
        "action_taken": action,
    })


def alert_validation_result(sortie_id: str, result: str, checks: dict,
                             recommendation: str, target_features: list):
    return write_alert("validation_result", {
        "sortie_id": sortie_id,
        "result": result,
        "checks": checks,
        "recommendation": recommendation,
        "target_features": target_features,
    })


def alert_dino_fallback(reason: str):
    return write_alert("DINO_FALLBACK_ACTIVE", {
        "reason": reason,
        "fallback_model": "mobilenet_v2",
    })


def alert_startup_complete(sortie_id: str, camera_type: str, nav_type: str):
    return write_alert("startup_complete", {
        "sortie_id": sortie_id,
        "camera_type": camera_type,
        "nav_type": nav_type,
    })


def alert_zero_detections(sortie_id: str):
    return write_alert("ZERO_DETECTIONS", {
        "sortie_id": sortie_id,
        "recommendation": "RE_SORTIE_REQUIRED",
    })
