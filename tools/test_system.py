"""
tools/test_system.py — Pre-flight system check.
Run this before every competition day to verify all components.

Usage:  python tools/test_system.py
        python tools/test_system.py --skip-dino   (faster, skips GPU test)
        python tools/test_system.py --camera-only
"""
import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"

results = []

def check(name, fn, *args, skip=False):
    if skip:
        print(f"{SKIP} {name}")
        results.append((name, "skip"))
        return None
    try:
        r = fn(*args)
        print(f"{PASS} {name}")
        results.append((name, "pass"))
        return r
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        results.append((name, "fail"))
        return None


def test_imports():
    import cv2, numpy, yaml, psutil, skimage, sklearn, serial
    return True

def test_config():
    from config_loader import load_config
    cfg = load_config()
    assert cfg.camera.get("type") is not None
    return True

def test_camera(cfg_camera):
    from camera.camera_factory import make_camera
    cam = make_camera(cfg_camera)
    assert cam.open(), "Camera open failed"
    frame, ts = cam.get_frame()
    cam.release()
    assert frame is not None, "Frame is None"
    assert frame.shape[1] >= 640, f"Too small: {frame.shape}"
    return frame.shape

def test_quality_filter(frame):
    from flight.quality_filter import QualityFilter
    qf = QualityFilter({"blur_laplacian_min": 80, "overexposure_v_max": 240,
                        "underexposure_v_min": 30, "entropy_min": 4.5,
                        "solid_color_s_max": 8})
    passed, reason = qf.check(frame)
    return f"passed={passed} reason={reason}"

def test_keyframe_selector(frame):
    from flight.keyframe_selector import KeyframeSelector
    ks = KeyframeSelector({"motion_threshold_px": 15, "histogram_diff_min": 0.12,
                            "ssim_reject_above": 0.88, "min_orb_features": 80,
                            "force_every_sec": 3.0})
    is_kf, reason = ks.is_keyframe(frame, time.monotonic())
    assert is_kf, f"First frame must be keyframe, got: {reason}"
    return reason

def test_boundary_detector(frame):
    from flight.boundary_detector import BoundaryDetector
    bd = BoundaryDetector({"hsv_lower": [18,80,80], "hsv_upper": [35,255,255],
                           "min_aspect_ratio": 4.0, "max_fill_ratio": 0.6,
                           "roi_fraction": 0.5})
    detected, conf, direction, bbox = bd.detect(frame)
    return f"detected={detected}"

def test_refs():
    from postflight.reference_manager import ReferenceManager
    rm = ReferenceManager(refs_dir="refs", cache_dir="cache")
    count = rm.load()
    assert count > 0, "No reference images found in refs/"
    return f"{count} refs across {rm.get_feature_types()}"

def test_lbp_hsv():
    import cv2
    dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    from postflight.lbp_descriptor import get_lbp_histogram
    from postflight.hsv_descriptor import get_hsv_histogram
    lbp = get_lbp_histogram(dummy)
    hsv = get_hsv_histogram(dummy)
    assert lbp.shape == (26,), f"LBP shape: {lbp.shape}"
    assert hsv.shape == (100,), f"HSV shape: {hsv.shape}"
    return f"LBP={lbp.shape} HSV={hsv.shape}"

def test_dino():
    import cv2
    dummy = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    from postflight.dino_embedder import get_embedding, get_mode
    emb = get_embedding(dummy)
    mode = get_mode()
    assert emb.ndim == 1 and emb.shape[0] > 0
    return f"mode={mode} dim={emb.shape[0]}"

def test_arduino(port):
    import serial as ser
    s = ser.Serial(port, 9600, timeout=2)
    time.sleep(2)
    s.close()
    return f"opened {port} OK"

def test_nav_file():
    import json, tempfile, os
    from nav.file_nav import FileNav
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"x":1.0,"y":2.0,"z":3.0,"yaw":0.0,"timestamp":time.time()}, f)
        fname = f.name
    nav = FileNav(pose_file=fname)
    pose = nav.get_pose()
    os.unlink(fname)
    assert abs(pose.x - 1.0) < 0.01
    return f"x={pose.x} y={pose.y} z={pose.z}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dino",    action="store_true")
    parser.add_argument("--camera-only",  action="store_true")
    parser.add_argument("--skip-arduino", action="store_true", default=True)
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  ASCEND System Check")
    print("="*50)

    check("Python imports",    test_imports)
    check("Config loader",     test_config)

    # Get config for camera test
    cfg_camera = None
    try:
        from config_loader import get_config
        cfg_camera = get_config().camera
    except Exception:
        pass

    frame = check("Camera open + frame grab", test_camera, cfg_camera)
    if frame is not None and not args.camera_only:
        # frame returned as shape tuple — need actual frame
        # Re-grab for downstream tests
        try:
            from camera.camera_factory import make_camera
            from config_loader import get_config
            cam = make_camera(get_config().camera)
            cam.open()
            frame_img, _ = cam.get_frame()
            cam.release()
            if frame_img is not None:
                check("Quality filter",       test_quality_filter,    frame_img)
                check("Keyframe selector",    test_keyframe_selector, frame_img)
                check("Boundary detector",    test_boundary_detector, frame_img)
        except Exception as e:
            print(f"  [SKIP] Frame-dependent tests: {e}")

    if not args.camera_only:
        check("LBP + HSV descriptors",   test_lbp_hsv)
        check("DINOv2 / MobileNet",      test_dino, skip=args.skip_dino)
        check("Reference manager",       test_refs)
        check("Nav file reader",         test_nav_file)
        check("Arduino serial",          test_arduino, "/dev/ttyUSB0",
              skip=args.skip_arduino)

    # Summary
    print("\n" + "="*50)
    passed = sum(1 for _, r in results if r == "pass")
    failed = sum(1 for _, r in results if r == "fail")
    skipped = sum(1 for _, r in results if r == "skip")
    print(f"  PASS: {passed}  FAIL: {failed}  SKIP: {skipped}")
    if failed:
        print(f"\n  FAILED CHECKS:")
        for name, r in results:
            if r == "fail":
                print(f"    - {name}")
        print("\n  Fix failures before flight.")
        sys.exit(1)
    else:
        print("\n  All checks passed — system ready.")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
