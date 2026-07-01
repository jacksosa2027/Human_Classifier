"""
Training Script — YOLOv8n for Raspberry Pi 5
==============================================
Ultralytics YOLOv8 nano: best speed/accuracy trade-off for an RPi5 CPU with
no GPU/NPU. Single class (see data/prepare_dataset.py): 0 = person.

After training, run training/export.py to produce the NCNN export used
on-device (drone/detector.py loads that, not the raw .pt).

Install dependencies first:
    conda env create -f environment.yml
    conda activate nomad_detect

Then run:
    python training/train.py --data nomad_dataset/dataset.yaml --device 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULTS = dict(
    data="nomad_dataset/dataset.yaml",
    model="yolov8n.pt",       # nano — best for RPi5
    epochs=100,
    imgsz=416,                # 416 vs 640: ~2x faster on CPU at inference time
    batch=16,
    workers=4,
    device="cpu",             # set "0" if you have a GPU for training
    project="runs/train",
    name="nomad_person",
    patience=20,               # early stopping
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    # Augmentations — keep moderate, top-down aerial framing is already
    # rotation/scale-diverse from altitude variation alone.
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=15.0,   # nadir camera has no fixed "up" relative to subjects
    translate=0.1,
    scale=0.3,
    fliplr=0.5,
    flipud=0.5,     # valid for top-down imagery, unlike ground-level photos
    mosaic=1.0,
)


def main():
    parser = argparse.ArgumentParser(description="Train the NOMAD person detector")
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key}", default=default, type=type(default))
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Error: dataset config '{args.data}' not found. "
            "Run data/prepare_dataset.py and data/validate_dataset.py first."
        )

    model = YOLO(args.model)
    model.train(**{k: getattr(args, k) for k in DEFAULTS if k != "model"})

    print("\nTraining complete. Best weights:")
    print(f"  {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print("\nNext: python training/export.py --weights <path to best.pt>")


if __name__ == "__main__":
    main()
