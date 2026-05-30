"""
Dataset Preparation Script
===========================
Unifies 3 datasets into a single YOLO format dataset:

1. scidb (Fixed-wing-UAV-A') - Pascal VOC XML annotations
   Structure: Top_Down/Infrared_Anns/*.xml + corresponding images

2. cybersimar08 (kaggle) - COCO JSON annotations
   Structure: test/train/valid folders with _annotations.coco.json

3. monkeyboy999 (kaggle) - YOLO txt annotations
   Structure: test/images + test/labels/*.txt

Output: unified_dataset/
    images/train/ images/val/ images/test/
    labels/train/ labels/val/ labels/test/

File must be run from terminal with the following command:
    $python3 guide.py --scidb /path/to/scidb --coco /path/to/coco --yolo /path/to/yolo

"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import argparse
from tqdm import tqdm
import cv2

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CLASS_MAP = {"Fixed_Wing_UAV": 0, "drone": 0, "airplane": 0}  # all → class 0
CLASS_NAME = "drone"
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# TEST_RATIO  = 0.1  (remainder)

SEED = 42
random.seed(SEED)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def make_dirs(base: Path):
    for split in ("train", "val", "test"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_pair(img_src: Path, lbl_content: str, base: Path, split: str, stem: str):
    """Copy image and write label file into the unified dataset."""
    ext = img_src.suffix
    dst_img = base / "images" / split / f"{stem}{ext}"
    dst_lbl = base / "labels" / split / f"{stem}.txt"
    shutil.copy2(img_src, dst_img)
    dst_lbl.write_text(lbl_content)

def assign_split(idx: int, total: int) -> str:
    r = idx / total
    if r < TRAIN_RATIO:
        return "train"
    elif r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    return "test"

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

# ─── DATASET 1: scidb  Pascal VOC XML ────────────────────────────────────────

def voc_xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> str:
    """Convert a Pascal VOC XML annotation to YOLO format string."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bb      = obj.find("bndbox")
        if name_el is None or name_el.text is None or bb is None:
            continue
        xmin_el = bb.find("xmin")
        xmax_el = bb.find("xmax")
        ymin_el = bb.find("ymin")
        ymax_el = bb.find("ymax")
        if (xmin_el is None or xmin_el.text is None or
                xmax_el is None or xmax_el.text is None or
                ymin_el is None or ymin_el.text is None or
                ymax_el is None or ymax_el.text is None):
            continue
        cls  = CLASS_MAP.get(name_el.text.strip(), 0)
        xmin = float(xmin_el.text)
        xmax = float(xmax_el.text)
        ymin = float(ymin_el.text)
        ymax = float(ymax_el.text)
        cx = clamp((xmin + xmax) / 2 / img_w)
        cy = clamp((ymin + ymax) / 2 / img_h)
        w  = clamp((xmax - xmin) / img_w)
        h  = clamp((ymax - ymin) / img_h)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def process_scidb(scidb_root: Path, out: Path):
    """
    scidb folder layout expected:
      <scidb_root>/
        Bottom_Up/  Horizontal/  Top_Down/
          Infrared_Anns/*.xml
          Infrared_Imgs/*.jpg   (same stem as xml)
    """
    print("\n[1/3] Processing scidb (Pascal VOC XML)...")
    xml_files = list(scidb_root.rglob("*.xml"))
    if not xml_files:
        print("  ⚠  No XML files found – check scidb_root path.")
        return

    random.shuffle(xml_files)
    for idx, xml_path in enumerate(tqdm(xml_files, desc="scidb")):
        # Find matching image (same stem, sibling *_Imgs folder)
        ann_dir = xml_path.parent          # …/Infrared_Anns
        img_dir = Path(str(ann_dir).replace("Infrared_Anns", "Infrared_Imgs"))
        stem    = xml_path.stem            # e.g. _0_0_000001
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label = voc_xml_to_yolo(xml_path, w, h)
        if not label.strip():
            continue

        split    = assign_split(idx, len(xml_files))
        view_dir = xml_path.parent.parent.name  # e.g. Top_Down, Bottom_Up
        uid      = f"scidb_{view_dir}_{stem}"
        copy_pair(img_path, label, out, split, uid)

# ─── DATASET 2: cybersimar08  COCO JSON ──────────────────────────────────────

def process_coco(coco_root: Path, out: Path):
    """
    cybersimar08 layout:
      <coco_root>/test/train/valid/
        _annotations.coco.json
        *.jpg / *.png
    """
    print("\n[2/3] Processing cybersimar08 (COCO JSON)...")
    split_map = {"train": "train", "valid": "val", "test": "test"}
    total_pairs = []

    for folder, split in split_map.items():
        folder_path = coco_root / folder
        json_path   = folder_path / "_annotations.coco.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            coco = json.load(f)

        id2img = {img["id"]: img for img in coco["images"]}
        # group annotations by image
        ann_by_img = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_id, anns in tqdm(ann_by_img.items(), desc=f"  coco/{folder}"):
            img_info = id2img[img_id]
            img_file = folder_path / img_info["file_name"]
            if not img_file.exists():
                continue

            iw = img_info["width"]
            ih = img_info["height"]
            lines = []
            for ann in anns:
                x, y, bw, bh = ann["bbox"]          # COCO: x_min, y_min, w, h
                cx = clamp((x + bw / 2) / iw)
                cy = clamp((y + bh / 2) / ih)
                nw = clamp(bw / iw)
                nh = clamp(bh / ih)
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            total_pairs.append((img_file, "\n".join(lines), split,
                                f"coco_{folder}_{img_id}"))

    for img_path, label, split, uid in tqdm(total_pairs, desc="coco copy"):
        copy_pair(img_path, label, out, split, uid)

# ─── DATASET 3: monkeyboy999  YOLO txt ───────────────────────────────────────

def process_yolo_txt(yolo_root: Path, out: Path):
    """
    monkeyboy999 layout:
      <yolo_root>/test/
        images/*.jpg
        labels/*.txt
    """
    print("\n[3/3] Processing monkeyboy999 (YOLO txt)...")
    pairs = []
    for split_dir in ("test", "train", "valid", "val"):
        img_dir = yolo_root / split_dir / "images"
        lbl_dir = yolo_root / split_dir / "labels"
        if not img_dir.exists():
            continue
        split = "val" if split_dir in ("valid", "val") else split_dir
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            pairs.append((img_path, lbl_path.read_text(), split,
                          f"yolo_{split_dir}_{img_path.stem}"))

    for img_path, label, split, uid in tqdm(pairs, desc="yolo copy"):
        copy_pair(img_path, label, out, split, uid)

# ─── dataset.yaml ────────────────────────────────────────────────────────────

def write_yaml(out: Path):
    yaml_content = f"""# Unified Drone Detection Dataset
path: {out.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['{CLASS_NAME}']
"""
    (out / "dataset.yaml").write_text(yaml_content)
    print(f"\n✅  dataset.yaml written → {out / 'dataset.yaml'}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unify drone detection datasets")
    parser.add_argument("--scidb",   required=True, help="Path to scidb Fixed-wing-UAV-A' root")
    parser.add_argument("--coco",    required=True, help="Path to cybersimar08 kaggle root")
    parser.add_argument("--yolo",    required=True, help="Path to monkeyboy999 kaggle root")
    parser.add_argument("--out",     default="unified_dataset", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    make_dirs(out)

    process_scidb(Path(args.scidb), out)
    process_coco(Path(args.coco), out)
    process_yolo_txt(Path(args.yolo), out)
    write_yaml(out)

    # Summary
    for split in ("train", "val", "test"):
        n = len(list((out / "images" / split).iterdir()))
        print(f"  {split:6s}: {n} images")
    print("\nDone! ✔")

if __name__ == "__main__":
    main()