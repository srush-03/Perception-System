"""
main.py — ASCEND Perception System entry point.
Single command: python main.py

Usage:
    python main.py                                           # normal run
    python main.py --config config/system_config.yaml
    python main.py --resume sortie_20260601_120000          # resume crashed sortie
    python main.py --postflight-only sortie_20260601_120000
    python main.py --revalidate data/transfer/manifest_sortie_xxx.json
"""
import argparse
import sys
import os
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger_setup import setup_logging, get_logger
from config_loader import load_config

log = None


def main():
    parser = argparse.ArgumentParser(description="ASCEND Perception System")
    parser.add_argument("--config",           default="config/system_config.yaml")
    parser.add_argument("--resume",           default=None, metavar="SORTIE_ID")
    parser.add_argument("--postflight-only",  default=None, metavar="SORTIE_ID",
                        dest="postflight_only")
    parser.add_argument("--revalidate",       default=None, metavar="MANIFEST_PATH")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(
        log_dir=cfg.paths.get("logs_dir", "data/logs"),
        level=cfg.logging.get("level", "INFO"),
    )
    global log
    log = get_logger("main")
    log.info("="*60)
    log.info("  ASCEND PERCEPTION SYSTEM  |  Team Marsvista")
    log.info("  ISRO IRoC-U 2026  |  Elimination Round")
    log.info("="*60)

    _ensure_dirs(cfg)

    if args.revalidate:
        _run_revalidation(args.revalidate, cfg); return
    if args.postflight_only:
        _run_postflight_only(args.postflight_only, cfg); return

    _run_full_mission(cfg, resume_sortie=args.resume)


# ── Full mission ──────────────────────────────────────────────────────────────

def _run_full_mission(cfg, resume_sortie=None):
    from mission_state import MissionState
    from camera.camera_factory import make_camera
    from nav.nav_factory import make_nav
    from trigger.trigger import make_trigger
    from flight.flight_manager import FlightManager
    from postflight.reference_manager import ReferenceManager
    from postflight.post_flight_pipeline import PostFlightPipeline
    from alert_writer import alert_startup_complete, alert_zero_detections

    ms = MissionState(sortie_id=resume_sortie or "")
    ms.transition("PREFLIGHT_CHECK", notes="System startup")

    _check_storage_on_startup(cfg)

    camera = make_camera(cfg.camera)
    if not camera.open():
        log.critical("Camera failed to open — aborting")
        ms.transition("ERROR", notes="camera_open_failed")
        sys.exit(1)

    nav     = make_nav(cfg.navigation)
    trigger = make_trigger(cfg.trigger)

    ref_mgr = ReferenceManager(
        refs_dir=cfg.paths.get("refs_dir", "refs"),
        cache_dir=cfg.paths.get("cache_dir", "cache"),
    )
    count = ref_mgr.load()
    if count == 0:
        log.critical("No reference images in refs/ — aborting")
        log.critical("Add seed images to refs/layered_rock/, refs/oxide_patch/, refs/reflective_ice/")
        ms.transition("ERROR", notes="no_references")
        sys.exit(1)

    alert_startup_complete(ms.sortie_id,
                           cfg.camera.get("type"),
                           cfg.navigation.get("type"))

    log.info("Waiting for START command → write state/start_command.json")
    _wait_for_start_command()
    ms.transition("FLIGHT_ACTIVE", trigger_source="start_command")

    flight = FlightManager(
        camera=camera,
        nav=nav,
        cfg={"paths": dict(cfg.paths), "quality": dict(cfg.quality),
             "keyframe": dict(cfg.keyframe),
             "boundary_detector": dict(cfg.boundary),
             "storage": dict(cfg.storage)},
        sortie_id=ms.sortie_id,
    )

    def _sig(sig, frame):
        log.warning("SIGINT — stopping flight")
        flight.stop()
    signal.signal(signal.SIGINT, _sig)

    flight.start()
    _monitor_flight(flight, nav, cfg)
    flight.stop()
    camera.release()

    log.info(f"Flight stats: {flight.get_stats()}")
    ms.transition("LANDING_DETECTED")

    log.info("Waiting for dock signal…")
    docked = trigger.wait_for_dock(timeout=cfg.trigger.get("timeout_sec", 600))
    if not docked:
        log.warning("Dock timeout — proceeding with post-flight")
    ms.transition("DOCKED_FOR_CHARGING", trigger_source="trigger_interface")
    ms.transition("POST_FLIGHT_PROCESSING")

    pipeline = PostFlightPipeline(
        cfg={"paths": dict(cfg.paths), "matching": dict(cfg.matching),
             "deduplication": dict(cfg.deduplication), "arena": dict(cfg.arena)},
        sortie_id=ms.sortie_id,
        ref_manager=ref_mgr,
    )
    result = pipeline.run()

    if result.get("result") == "FAILURE":
        alert_zero_detections(ms.sortie_id)

    ms.transition("READY_FOR_TRANSFER",
                  notes=result.get("recommendation", ""))
    log.info(f"DONE: {result.get('result')} — {result.get('recommendation')}")
    log.info("Transfer package in data/transfer/ — hand off to Telemetry team")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _monitor_flight(flight, nav, cfg):
    alt_min   = cfg.arena.get("altitude_min", 2.0)
    low_count = 0
    while True:
        time.sleep(0.5)
        pose = nav.get_pose()
        if pose.valid and pose.z < (alt_min - 0.3):
            low_count += 1
            if low_count >= 3:
                log.info(f"Landing detected: z={pose.z:.2f}m")
                break
        else:
            low_count = 0
        if not flight._running:
            break


def _wait_for_start_command(timeout: float = 3600):
    start_file = "state/start_command.json"
    deadline   = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(start_file):
            try: os.remove(start_file)
            except: pass
            return
        time.sleep(0.5)
    log.warning("Start command timeout — proceeding")


def _check_storage_on_startup(cfg):
    try:
        import psutil
        free_gb = psutil.disk_usage(".").free / (1024**3)
        need_gb = cfg.storage.get("halt_gb", 1.0) + 0.5
        if free_gb < need_gb:
            log.critical(f"Only {free_gb:.1f} GB free, need >{need_gb:.1f} GB")
            sys.exit(1)
        log.info(f"Storage: {free_gb:.1f} GB free — OK")
    except ImportError:
        pass


def _ensure_dirs(cfg):
    for d in [cfg.paths.get("frames_dir",  "data/frames"),
              cfg.paths.get("mosaic_dir",   "data/mosaic"),
              cfg.paths.get("matches_dir",  "data/matches"),
              cfg.paths.get("logs_dir",     "data/logs"),
              cfg.paths.get("alerts_dir",   "alerts"),
              cfg.paths.get("state_dir",    "state"),
              cfg.paths.get("cache_dir",    "cache"),
              cfg.paths.get("refs_dir",     "refs"),
              "data/transfer"]:
        os.makedirs(d, exist_ok=True)


def _run_postflight_only(sortie_id, cfg):
    from postflight.reference_manager import ReferenceManager
    from postflight.post_flight_pipeline import PostFlightPipeline
    log.info(f"Post-flight only: {sortie_id}")
    ref_mgr = ReferenceManager(refs_dir=cfg.paths.get("refs_dir", "refs"),
                               cache_dir=cfg.paths.get("cache_dir", "cache"))
    ref_mgr.load()
    PostFlightPipeline(
        cfg={"paths": dict(cfg.paths), "matching": dict(cfg.matching),
             "deduplication": dict(cfg.deduplication), "arena": dict(cfg.arena)},
        sortie_id=sortie_id, ref_manager=ref_mgr,
    ).run()


def _run_revalidation(manifest_path, cfg):
    from revalidation.revalidator import Revalidator
    log.info(f"Revalidation: {manifest_path}")
    rv = Revalidator(refs_dir=cfg.paths.get("refs_dir", "refs"),
                     arena_cfg=dict(cfg.arena))
    result = rv.run(manifest_path)
    log.info(f"Verdict: {result.get('verdict')}")


if __name__ == "__main__":
    main()
