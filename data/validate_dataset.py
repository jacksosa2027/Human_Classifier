"""
Dataset Validation Utility
===========================
Run after prepare_dataset.py to sanity-check the unified dataset:
  - Counts images / labels per split
  - Verifies every image has a matching label and vice-versa
  - Checks label values are in [0, 1] and class ids are in range
  - Optionally renders a grid of random samples with boxes

Usage:
    python validate_dataset.py --data nomad_dataset
    python validate_dataset.py --data nomad_dataset --visualise --n 16
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

NUM_CLASSES = 1  # person — keep in sync with prepare_dataset.py CLASS_NAMES
COLORS = [(0, 255, 80)]  # person=green


def check_split(base: Path, split: str) -> dict:
    img_dir = base / "images" / split
    lbl_dir = base / "labels" / split

    images = {p.stem: p for p in img_dir.iterdir()
              if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")} \
        if img_dir.exists() else {}
    labels = {p.stem: p for p in lbl_dir.iterdir()
              if p.suffix == ".txt"} \
        if lbl_dir.exists() else {}

    missing_label = [s for s in images if s not in labels]
    missing_image = [s for s in labels if s not in images]
    empty_labels = [s for s, p in labels.items() if p.stat().st_size == 0]

    bad_values = []
    for stem, lp in labels.items():
        for line in lp.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                bad_values.append(stem)
                break
            try:
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= NUM_CLASSES:
                    bad_values.append(stem)
                    break
                vals = list(map(float, parts[1:]))
                if any(v < 0 or v > 1 for v in vals):
                    bad_values.append(stem)
                    break
            except ValueError:
                bad_values.append(stem)
                break

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_label": missing_label[:5],
        "missing_label_total": len(missing_label),
        "missing_image": missing_image[:5],
        "missing_image_total": len(missing_image),
        "empty_labels": empty_labels[:5],
        "empty_labels_total": len(empty_labels),
        "bad_values": bad_values[:5],
        "bad_values_total": len(bad_values),
    }


def draw_boxes(img: np.ndarray, label_path: Path) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, bw, bh = parts[0], *map(float, parts[1:])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        color = COLORS[int(cls_id) % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def make_grid(images: list, cols: int = 4, cell: int = 320) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (cell, cell))
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = resized
    return grid


def visualise(base: Path, n: int):
    all_pairs = []
    for split in ("train", "val", "test"):
        img_dir, lbl_dir = base / "images" / split, base / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                all_pairs.append((img_path, lbl_path))

    sample = random.sample(all_pairs, min(n, len(all_pairs)))
    annotated = []
    for img_path, lbl_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        annotated.append(draw_boxes(img, lbl_path))

    if annotated:
        grid = make_grid(annotated)
        out = base / "sample_grid.jpg"
        cv2.imwrite(str(out), grid)
        print(f"\nSample grid saved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="nomad_dataset")
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--n", type=int, default=16, help="Number of samples for grid visualisation")
    args = parser.parse_args()

    base = Path(args.data)
    if not base.exists():
        raise SystemExit(f"Error: dataset path '{base}' does not exist.")
    print(f"\n{'=' * 55}\n  Dataset Validation: {base.resolve()}\n{'=' * 55}")

    total_imgs = 0
    for split in ("train", "val", "test"):
        stats = check_split(base, split)
        total_imgs += stats["images"]
        print(f"\n[{split.upper()}]")
        print(f"  Images : {stats['images']}")
        print(f"  Labels : {stats['labels']}")
        if stats["missing_label"]:
            print(f"  Missing labels ({stats['missing_label_total']} total, first 5): {stats['missing_label']}")
        if stats["missing_image"]:
            print(f"  Missing images ({stats['missing_image_total']} total, first 5): {stats['missing_image']}")
        if stats["empty_labels"]:
            print(f"  Background frames (empty label, expected for occluded persons): {stats['empty_labels_total']}")
        if stats["bad_values"]:
            print(f"  Bad values    ({stats['bad_values_total']} total, first 5): {stats['bad_values']}")
        if not any([stats["missing_label"], stats["missing_image"], stats["bad_values"]]):
            print("  All checks passed")

    print(f"\nTotal images: {total_imgs}")

    if args.visualise:
        visualise(base, args.n)


if __name__ == "__main__":
    main()
