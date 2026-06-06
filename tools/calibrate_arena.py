"""
tools/calibrate_arena.py — Live arena background calibration tool.
Point camera at plain arena soil and press keys:
  C — capture calibration frame
  S — show current HSV stats
  Q — quit and print recommended config values

Run this on competition day before flight to get arena-specific values.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

def main():
    from config_loader import get_config
    from camera.camera_factory import make_camera

    cfg = get_config()
    cam = make_camera(cfg.camera)
    if not cam.open():
        print("Camera failed to open"); return

    samples = []
    print("\nArena Calibration Tool")
    print("  C = capture sample   S = show stats   Q = quit\n")

    while True:
        frame, _ = cam.get_frame()
        if frame is None:
            continue

        # Show live HSV overlay
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_mean = hsv[:,:,0].mean()
        s_mean = hsv[:,:,1].mean()
        v_mean = hsv[:,:,2].mean()

        disp = frame.copy()
        cv2.putText(disp, f"H:{h_mean:.1f} S:{s_mean:.1f} V:{v_mean:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(disp, f"Samples: {len(samples)}",
                    (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(disp, "C=capture  S=stats  Q=quit",
                    (10, disp.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)
        cv2.imshow("Arena Calibration", disp)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('c') or key == ord('C'):
            samples.append({'H': h_mean, 'S': s_mean, 'V': v_mean})
            print(f"Captured sample {len(samples)}: H={h_mean:.1f} S={s_mean:.1f} V={v_mean:.1f}")
        elif key == ord('s') or key == ord('S'):
            if samples:
                _print_stats(samples)
        elif key == ord('q') or key == ord('Q') or key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    if samples:
        print("\n--- FINAL STATS ---")
        _print_stats(samples)


def _print_stats(samples):
    H = np.array([s['H'] for s in samples])
    S = np.array([s['S'] for s in samples])
    V = np.array([s['V'] for s in samples])
    print(f"\n  Samples: {len(samples)}")
    print(f"  H: mean={H.mean():.1f}  std={H.std():.1f}  range=[{H.min():.1f}, {H.max():.1f}]")
    print(f"  S: mean={S.mean():.1f}  std={S.std():.1f}  range=[{S.min():.1f}, {S.max():.1f}]")
    print(f"  V: mean={V.mean():.1f}  std={V.std():.1f}  range=[{V.min():.1f}, {V.max():.1f}]")
    print(f"\n  NOTE: Update config if these differ significantly from previous values")


if __name__ == "__main__":
    main()
