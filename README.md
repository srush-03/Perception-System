# ASCEND Perception System
### Team Marsvista | ISRO IRoC-U 2026 | Elimination Round

Onboard post-processing UAV perception stack.  
DINOv2 + LBP + HSV hybrid matching | Jetson Orin Nano | OAK-D Lite

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt --break-system-packages   # Jetson
pip install -r requirements.txt                           # PC

# 2. Add seed images to refs/ (see refs/*/README.txt)

# 3. System check
python tools/test_system.py

# 4. Run
python main.py
```

---

## Folder Structure

```
ascend_perception/
│
├── main.py                     ENTRY POINT — run this
├── config_loader.py            Singleton config reader
├── mission_state.py            State machine + checkpoint manager
├── alert_writer.py             JSON alert emitter (Telemetry/Dashboard reads these)
├── logger_setup.py             Logging config
├── requirements.txt            Python dependencies
│
├── config/
│   └── system_config.yaml      ALL tunable parameters — edit before flight
│
├── camera/                     CAMERA ABSTRACTION LAYER
│   ├── camera_interface.py     Abstract base class
│   ├── usb_camera.py           USB/Logitech webcam (used first)
│   ├── oak_d_camera.py         OAK-D Lite via DepthAI
│   └── camera_factory.py       Auto-selects from config
│
├── nav/                        NAVIGATION ABSTRACTION LAYER
│   ├── nav_interface.py        Abstract base + PoseStamp dataclass
│   ├── ros2_nav.py             ROS2 VINS/SLAM topic subscriber
│   ├── file_nav.py             File-based fallback (for testing)
│   └── nav_factory.py          Auto-selects; falls back to file if ROS2 fails
│
├── trigger/
│   └── trigger.py              Arduino serial + file fallback + factory
│       ArduinoSerialTrigger:   waits for "DOCKED" from Arduino Nano
│       FileTrigger:            watches state/mission_state.json (fallback)
│
├── flight/                     FLIGHT-PHASE MODULES (lightweight only)
│   ├── flight_manager.py       4-thread pipeline orchestrator
│   ├── quality_filter.py       Blur/exposure/entropy checks
│   ├── keyframe_selector.py    Motion/SSIM/histogram gating
│   ├── boundary_detector.py    Yellow line HSV detection
│   └── storage_monitor.py      Disk space watchdog thread
│
├── postflight/                 POST-FLIGHT PROCESSING (heavy, runs after dock)
│   ├── post_flight_pipeline.py Main orchestrator: stitch→match→dedup→validate
│   ├── stitcher.py             Sequential ORB homography mosaic
│   ├── matcher.py              Per-frame matching, LR in memory only
│   ├── dino_embedder.py        DINOv2 ViT-S/14 + MobileNetV2 fallback
│   ├── lbp_descriptor.py       Local Binary Pattern texture histogram
│   ├── hsv_descriptor.py       HSV color histogram
│   ├── fusion_scorer.py        Adaptive weighted fusion per feature type
│   ├── deduplicator.py         3-stage dedup: spatial + embedding + temporal
│   ├── reference_manager.py    Auto-discovers refs/, caches embeddings
│   └── validator.py            5-check onboard validation
│
├── revalidation/               BASE STATION (PC) — independent logic
│   └── revalidator.py          ORB + HSV + SHA256 + FP filter
│
├── arduino/
│   └── dock_trigger/
│       └── dock_trigger.ino    Arduino Nano firmware (upload once)
│
├── refs/                       ← PUT YOUR SEED IMAGES HERE
│   ├── layered_rock/           3-5 JPG/PNG, top-down, 1.5-2m height
│   ├── oxide_patch/            3-5 JPG/PNG, top-down, 1.5-2m height
│   └── reflective_ice/         3-5 JPG/PNG, top-down, 1.5-2m height
│       (add new feature type = create new subfolder, zero code changes)
│
├── tools/                      UTILITIES
│   ├── test_system.py          Pre-flight system check (run before every flight)
│   ├── test_matching.py        Test matcher on a single image or folder
│   ├── calibrate_arena.py      Live HSV calibration tool
│   ├── send_start.py           Trigger START without dashboard
│   └── simulate_dock.py        Trigger DOCKED without Arduino
│
├── data/                       RUNTIME DATA (auto-created)
│   ├── frames/                 HD keyframes saved during flight
│   ├── mosaic/                 Stitched mosaic (internal, not competition output)
│   ├── matches/                HD proof images for confirmed detections
│   ├── logs/                   JSON logs: raw_detections, dedup, validation
│   └── transfer/               Final package for base station
│
├── state/                      MISSION STATE FILES
│   ├── mission_state.json      Current state (read by Telemetry/Dashboard)
│   ├── pipeline_checkpoint.json Crash-recovery checkpoint
│   └── start_command.json      Write this to start mission (consumed once)
│
├── alerts/                     JSON EVENT ALERTS (read by Telemetry/Dashboard)
│   └── alert_<ts>_<TYPE>.json  mission_state, detection, boundary_warning, etc.
│
└── cache/                      PRECOMPUTED EMBEDDINGS
    └── ref_embeddings.pkl      Auto-built from refs/ at startup
```

---

## Competition Day Checklist

1. **Reseat seed images** — retake if lighting has changed
2. **Delete old cache**: `rm cache/ref_embeddings.pkl`
3. **Run system check**: `python tools/test_system.py`
4. **Calibrate arena** (optional): `python tools/calibrate_arena.py`
5. **Check Arduino** connected on `/dev/ttyUSB0`
6. **Update config** if camera changed: `config/system_config.yaml`
7. **Start system**: `python main.py`
8. **Send START**: dashboard OR `python tools/send_start.py`
9. After landing+docking: post-flight runs automatically
10. **Revalidate on PC**: `python main.py --revalidate data/transfer/manifest_*.json`

---

## Command Reference

| Command | Purpose |
|---|---|
| `python main.py` | Full mission |
| `python main.py --resume sortie_ID` | Resume crashed sortie |
| `python main.py --postflight-only sortie_ID` | Run post-flight on saved frames |
| `python main.py --revalidate manifest.json` | PC revalidation only |
| `python tools/test_system.py` | Pre-flight check |
| `python tools/test_matching.py --image x.jpg --show` | Test single image |
| `python tools/send_start.py` | Trigger flight start |
| `python tools/simulate_dock.py` | Simulate dock (no Arduino) |

---

## Nav Interface (for VINS team)

The nav team must write `state/current_pose.json` at ≥10 Hz:
```json
{ "x": 3.42, "y": 7.15, "z": 2.80, "yaw": 1.57, "timestamp": 1748700000.1 }
```

Or: set `navigation.type: vins_ros2` in config and publish to  
`/vins_estimator/odometry` as `nav_msgs/Odometry`.

---

## Alert Schema (for Telemetry/Dashboard team)

All alerts written to `alerts/` as JSON files.  
Event types: `mission_state`, `detection`, `boundary_warning`,  
`storage_warning`, `validation_result`, `startup_complete`, `DINO_FALLBACK_ACTIVE`
