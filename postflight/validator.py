"""
postflight/validator.py — Onboard validation on Jetson.
Runs 5 checks on deduplicated detections. Outputs onboard_validation.json.
"""
import cv2
import os
import json
import logging
from typing import List, Dict, Any

from alert_writer import alert_validation_result

log = logging.getLogger(__name__)


class Validator:

    def __init__(self, arena_cfg: dict, matching_cfg: dict,
                 logs_dir: str, sortie_id: str):
        self._x_bounds  = arena_cfg.get("x_bounds", [-1.0, 11.0])
        self._y_bounds  = arena_cfg.get("y_bounds", [-1.0,  8.0])
        self._th_medium = matching_cfg.get("thresholds", {}).get("medium", 0.65)
        self._logs_dir  = logs_dir
        self._sortie_id = sortie_id

    def validate(self, detections: List[dict],
                 expected_types: List[str]) -> dict:
        """
        Runs all checks. Returns validation result dict.
        """
        checks = {}

        # ── Check 1: Coverage ───────────────────────────────────────────────
        detected_types = {d["feature_type"] for d in detections}
        missing = [ft for ft in expected_types if ft not in detected_types]
        checks["coverage"] = {
            "pass": len(missing) == 0,
            "detected": list(detected_types),
            "missing":  missing,
        }

        # ── Check 2: Confidence ─────────────────────────────────────────────
        conf_failures = []
        for ft in expected_types:
            ft_dets = [d for d in detections if d["feature_type"] == ft]
            has_good = any(d["confidence"] >= self._th_medium for d in ft_dets)
            if not has_good and ft_dets:
                conf_failures.append(ft)
        checks["confidence"] = {
            "pass": len(conf_failures) == 0,
            "low_confidence_types": conf_failures,
        }

        # ── Check 3: Coordinates ────────────────────────────────────────────
        bad_coords = []
        for d in detections:
            x = d["coordinates"].get("x", 0)
            y = d["coordinates"].get("y", 0)
            if not (self._x_bounds[0] <= x <= self._x_bounds[1] and
                    self._y_bounds[0] <= y <= self._y_bounds[1]):
                bad_coords.append(d["instance_id"])
        checks["coordinates"] = {
            "pass": len(bad_coords) == 0,
            "out_of_bounds": bad_coords,
        }

        # ── Check 4: Proof images ────────────────────────────────────────────
        bad_images = []
        for d in detections:
            proof = d.get("proof_image", "")
            if not os.path.exists(proof):
                bad_images.append(proof)
                continue
            img = cv2.imread(proof)
            if img is None:
                bad_images.append(proof)
                continue
            h, w = img.shape[:2]
            if w < 1280 or h < 720:
                bad_images.append(proof)
        checks["proof_images"] = {
            "pass": len(bad_images) == 0,
            "bad_images": bad_images,
        }

        # ── Check 5: Duplicate spacing ───────────────────────────────────────
        from collections import defaultdict
        import math
        dup_violations = []
        by_type = defaultdict(list)
        for d in detections:
            by_type[d["feature_type"]].append(d)
        for ft, dets in by_type.items():
            for i in range(len(dets)):
                for j in range(i+1, len(dets)):
                    xi = dets[i]["coordinates"]["x"]; yi = dets[i]["coordinates"]["y"]
                    xj = dets[j]["coordinates"]["x"]; yj = dets[j]["coordinates"]["y"]
                    dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if dist < 0.5:
                        dup_violations.append({
                            "a": dets[i]["instance_id"],
                            "b": dets[j]["instance_id"],
                            "dist_m": round(dist, 3)
                        })
        checks["duplicates"] = {
            "pass": len(dup_violations) == 0,
            "violations": dup_violations,
        }

        # ── Final result ─────────────────────────────────────────────────────
        all_pass = all(c["pass"] for c in checks.values())
        result   = "SUCCESS" if all_pass else "PARTIAL_SUCCESS"

        failed_types = set()
        if not checks["coverage"]["pass"]:
            failed_types.update(missing)
        if not checks["confidence"]["pass"]:
            failed_types.update(conf_failures)

        recommendation = "MISSION_SUCCESS" if all_pass else "RE_SORTIE_REQUIRED"
        target = sorted(failed_types)

        output = {
            "sortie_id":      self._sortie_id,
            "result":         result,
            "checks":         checks,
            "recommendation": recommendation,
            "target_features": target,
        }

        # Save to disk
        out_path = os.path.join(self._logs_dir, "onboard_validation.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        # Write alert
        alert_validation_result(
            self._sortie_id, result, checks, recommendation, target
        )

        log.info(f"Validation: {result} — {recommendation}")
        if target:
            log.warning(f"  Failed feature types: {target}")

        return output
