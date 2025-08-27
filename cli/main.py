"""
Command-line entrypoint for the training-free context-driven STR pipeline.

Two modes:
  (A) Folder mode (default):
      python -m cli.main --config configs/run_presets/demo_folder.yaml --images data/samples --out_dir outputs/demo

  (B) CSV dataset mode (image + optional GT):
      python -m cli.main --config configs/run_presets/icdar15.yaml --dataset_csv path/to/list.csv --out_dir outputs/icdar15

CSV format expected:
    image_path[, gt_text]

Artifacts written:
  - results.csv with per-image scores and decisions
  - masks/ and crops/ for qualitative inspection (if --save_artifacts)
"""

from __future__ import annotations
import argparse, csv, os, sys, json
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

try:
    from omegaconf import OmegaConf
except Exception as e:
    print("[fatal] omegaconf not installed. Please `pip install omegaconf`.", file=sys.stderr)
    raise

from pipeline.runner import PipelineRunner
from pipeline.io_utils import list_images, imread_rgb, imwrite, save_mask, ensure_dir, draw_boxes
from pipeline import set_global_seed


def _load_cfg(config_path: str) -> "OmegaConf":
    base = OmegaConf.load("configs/default.yaml")
    user = OmegaConf.load(config_path)
    return OmegaConf.merge(base, user)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_presets/demo_folder.yaml", help="YAML config to merge over default.yaml")
    ap.add_argument("--images", default=None, help="Folder of images (folder mode)")
    ap.add_argument("--dataset_csv", default=None, help="CSV with columns: image_path[,gt_text]")
    ap.add_argument("--out_dir", default="outputs/run")
    ap.add_argument("--save_artifacts", action="store_true", help="Save masks and crops")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def _iter_csv(rows_path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(rows_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if not parts:
                continue
            img = parts[0]
            gt = parts[1] if len(parts) > 1 else ""
            rows.append((img, gt))
    return rows


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    set_global_seed(args.seed, deterministic=True)

    cfg = _load_cfg(args.config)
    # Save the resolved config for reproducibility
    with open(os.path.join(args.out_dir, "config.resolved.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    runner = PipelineRunner(cfg)

    # Discover inputs
    items: List[Tuple[str, str]]
    if args.dataset_csv:
        items = _iter_csv(args.dataset_csv)
    elif args.images:
        items = [(p, "") for p in list_images(args.images)]
    else:
        raise SystemExit("Provide either --images (folder mode) or --dataset_csv (csv mode).")

    results_csv = os.path.join(args.out_dir, "results.csv")
    with open(results_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "image", "T1", "best_T3", "T2",
            "S1", "S3_best", "L1", "L3_best",
            "C1", "C3_best", "Cmax", "final", "used_fallback", "sec"
        ])

        for (img_path, gt) in tqdm(items, desc="Running pipeline"):
            try:
                img = imread_rgb(img_path)
                out = runner.run_image(img, image_path=img_path)

                # Optionally save artifacts
                base = os.path.splitext(os.path.basename(img_path))[0]
                if args.save_artifacts:
                    # mask
                    mask_path = os.path.join(args.out_dir, "masks", f"{base}_mask.png")
                    save_mask(out.mask, mask_path)
                    # crops
                    for i, ci in enumerate(out.crops):
                        crop_bgr = cv2.cvtColor(img[ci.box[1]:ci.box[3], ci.box[0]:ci.box[2]], cv2.COLOR_RGB2BGR)
                        imwrite(os.path.join(args.out_dir, "crops", f"{base}_crop{i}.png"), crop_bgr)
                    # debug overlay
                    dbg = draw_boxes(img, out.boxes)
                    imwrite(os.path.join(args.out_dir, "debug", f"{base}_boxes.png"), dbg)

                # Write CSV row
                w.writerow(out.as_row())

            except Exception as e:
                print(f"[warn] failed on {img_path}: {e}", file=sys.stderr)

    print(f"[ok] Wrote: {results_csv}")


if __name__ == "__main__":
    main()
