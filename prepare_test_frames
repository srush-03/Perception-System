"""
tools/prepare_test_frames.py — Renames phone photos into the
frame_<ts>_x<x>_y<y>_z<z>.jpg format the matcher expects, so you can
test the post-flight pipeline (stitching + matching + validation)
without a real drone flight.

Usage:
    Put your phone photos in a folder, e.g. phone_photos/
    Then run:
        python tools/prepare_test_frames.py --src phone_photos --grid

    --grid auto-assigns x,y across a simple left-to-right grid spanning
    your arena width/height (from config). Good enough for algorithm
    testing — these are NOT real coordinates.

    Or assign your own approximate position per photo interactively:
        python tools/prepare_test_frames.py --src phone_photos --manual
"""
import argparse
import os
import shutil
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder with phone photos")
    parser.add_argument("--dst", default="data/frames", help="Destination (default: data/frames)")
    parser.add_argument("--grid", action="store_true",
                        help="Auto-assign positions across a grid (default if no flag given)")
    parser.add_argument("--manual", action="store_true",
                        help="Prompt for x,y,z per photo")
    parser.add_argument("--height", type=float, default=2.5,
                        help="Assumed flight height in meters (default 2.5)")
    parser.add_argument("--arena-w", type=float, default=10.7, help="Arena width meters")
    parser.add_argument("--arena-h", type=float, default=7.6,  help="Arena height meters")
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"ERROR: source folder not found: {args.src}")
        return

    os.makedirs(args.dst, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(f for f in os.listdir(args.src)
                   if os.path.splitext(f)[1].lower() in exts)

    if not files:
        print(f"No images found in {args.src}")
        return

    print(f"Found {len(files)} photos in {args.src}")
    base_ts = time.time()

    for i, fname in enumerate(files):
        src_path = os.path.join(args.src, fname)

        if args.manual:
            print(f"\n[{i+1}/{len(files)}] {fname}")
            x = float(input("  x (meters): ") or 0.0)
            y = float(input("  y (meters): ") or 0.0)
            z = float(input(f"  z (meters) [{args.height}]: ") or args.height)
        else:
            # Simple grid: spread photos left-to-right, wrapping rows
            cols = max(1, int(len(files) ** 0.5))
            row, col = divmod(i, cols)
            x = (col + 0.5) * (args.arena_w / cols)
            y = (row + 0.5) * (args.arena_h / max(1, (len(files) // cols) + 1))
            z = args.height

        ts = base_ts + i * 1.0  # 1 second apart, fake but monotonic
        new_name = f"frame_{ts:.3f}_x{x:.3f}_y{y:.3f}_z{z:.3f}.jpg"
        dst_path = os.path.join(args.dst, new_name)

        shutil.copy2(src_path, dst_path)
        print(f"  {fname}  ->  {new_name}")

    print(f"\nDone. {len(files)} test frames in {args.dst}/")
    print("NOTE: x,y,z are FAKE placeholder coordinates for algorithm testing only.")
    print("Run: python main.py --postflight-only test_sortie")


if __name__ == "__main__":
    main()