"""
Export Script — .pt -> NCNN for Raspberry Pi 5
=================================================
NCNN is the fastest CPU inference backend on RPi5 for Ultralytics models
(no GPU/NPU required). Copy the exported folder to the Pi and point
drone/detector.py at it.

Usage:
    python training/export.py --weights runs/train/nomad_person/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--format", default="ncnn", choices=["ncnn", "onnx"])
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Error: weights file '{weights}' not found.")

    model = YOLO(str(weights))
    exported = model.export(format=args.format, imgsz=args.imgsz)

    print(f"\nExported to: {exported}")
    print("Copy this whole folder to the Raspberry Pi 5 as models/best_ncnn_model — "
          "configs/mission_config.yaml's detector.model_path already points there by default.")


if __name__ == "__main__":
    main()
