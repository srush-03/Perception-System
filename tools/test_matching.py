"""
tools/test_matching.py — Test the hybrid matcher on a single image or folder.
Use to tune thresholds and verify references are working correctly.

Usage:
    python tools/test_matching.py --image path/to/test.jpg
    python tools/test_matching.py --folder data/frames/
    python tools/test_matching.py --image test.jpg --show   (OpenCV window)
"""
import sys, os, argparse, glob, cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_loader import get_config
from logger_setup import setup_logging
from postflight.reference_manager import ReferenceManager
from postflight.dino_embedder import get_embedding, get_mode
from postflight.lbp_descriptor import get_lbp_histogram
from postflight.hsv_descriptor import get_hsv_histogram
from postflight.fusion_scorer import FusionScorer


def score_image(img_path: str, ref_mgr: ReferenceManager,
                scorer: FusionScorer, show: bool = False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Cannot read: {img_path}")
        return

    lr = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    q_dino = get_embedding(lr)
    q_lbp  = get_lbp_histogram(lr)
    q_hsv  = get_hsv_histogram(lr)

    print(f"\n{'─'*50}")
    print(f"  {os.path.basename(img_path)}")
    best_overall = 0.0
    best_ft      = "none"

    for ft in ref_mgr.get_feature_types():
        refs = ref_mgr.get_refs(ft)
        best_score = 0.0; best_bd = {}
        for ref in refs:
            score, bd = scorer.score(q_dino, q_lbp, q_hsv,
                                     ref.dino, ref.lbp, ref.hsv,
                                     feature_type=ft)
            if score > best_score:
                best_score = score; best_bd = bd
        tier = scorer.get_tier(best_score)
        marker = "★" if tier != "REJECT" else " "
        print(f"  {marker} {ft:<20} score={best_score:.4f}  [{tier}]")
        print(f"      dino={best_bd.get('dino',0):.3f}  "
              f"lbp={best_bd.get('lbp',0):.3f}  "
              f"hsv={best_bd.get('hsv',0):.3f}")
        if best_score > best_overall:
            best_overall = best_score; best_ft = ft

    print(f"  → Best match: {best_ft}  ({best_overall:.4f})")

    if show:
        disp = cv2.resize(img, (512, 512))
        cv2.putText(disp, f"{best_ft}: {best_overall:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if best_overall >= 0.65 else (0, 0, 255), 2)
        cv2.imshow("Match Result", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  default=None)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--show",   action="store_true")
    args = parser.parse_args()

    setup_logging(level="WARNING")  # quiet for testing
    cfg   = get_config()
    ref_mgr = ReferenceManager(refs_dir=cfg.paths.get("refs_dir","refs"),
                               cache_dir=cfg.paths.get("cache_dir","cache"))
    count = ref_mgr.load()
    if count == 0:
        print("ERROR: No references found. Add images to refs/"); return

    scorer = FusionScorer(dict(cfg.matching))
    print(f"DINOv2 mode: {get_mode()}")
    print(f"Feature types: {ref_mgr.get_feature_types()}")

    if args.image:
        score_image(args.image, ref_mgr, scorer, show=args.show)
    elif args.folder:
        paths = sorted(glob.glob(os.path.join(args.folder, "*.jpg")) +
                       glob.glob(os.path.join(args.folder, "*.png")))
        print(f"Testing {len(paths)} images in {args.folder}")
        for p in paths:
            score_image(p, ref_mgr, scorer, show=False)
    else:
        print("Specify --image or --folder"); return


if __name__ == "__main__":
    main()
