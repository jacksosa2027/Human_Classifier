"""
Training Script — YOLOv8n for Raspberry Pi 5
==============================================
Uses Ultralytics YOLOv8 nano: best speed/accuracy trade-off for
edge devices.  After training, exports to NCNN (fastest on RPi5 CPU)
and optionally ONNX.

Install dependencies first:
    Create environment: conda env create -f requirements_train.txt
    Activate:           conda activate drone_detection

    And then run:
    $python train.py --data unified_dataset/dataset.yaml --device 0
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

# ─── DEFAULTS ────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    data        = "unified_dataset/dataset.yaml",
    model       = "yolov8n.pt",         # nano – best for RPi5
    epochs      = 100,
    imgsz       = 416,                  # 416 vs 640: ~2× faster on CPU
    batch       = 16,
    workers     = 4,
    device      = "cpu",                # set "0" if you have a GPU for training
    project     = "runs/train",
    name        = "drone_rpi5",
    patience    = 20,                   # early stopping
    lr0         = 0.01,
    lrf         = 0.01,
    momentum    = 0.937,
    weight_decay= 0.0005,
    warmup_epochs = 3,
    # Augmentations  (keep moderate – dataset is already diverse)
    hsv_h       = 0.015,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    degrees     = 10.0,
    translate   = 0.1,
    scale       = 0.5,
    fliplr      = 0.5,
    mosaic      = 1.0,
    mixup       = 0.1,
)


def train(cfg: dict):
    model_path = cfg.pop("model")
    model = YOLO(model_path)

    print("=" * 60)
    print("  Drone Detection Training – YOLOv8n")
    print("  Target: Raspberry Pi 5 (NCNN export)")
    print("=" * 60)

    results = model.train(**cfg)
    assert results is not None, "Training failed: model.train() returned None"

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✅  Training complete.  Best weights → {best_weights}")
    return best_weights


def export_for_rpi5(weights: Path, imgsz: int):
    """
    Export pipeline for Raspberry Pi 5:
      1. NCNN  – fastest on ARM CPU, no extra runtime needed
      2. ONNX  – fallback / for ORT if you prefer
    """
    model = YOLO(weights)

    print("\n[Export 1/2] NCNN (primary – use this on RPi5)")
    model.export(
        format   = "ncnn",
        imgsz    = imgsz,
        optimize = True,      # ARM-specific optimisations
        half     = False,     # RPi5 CPU does NOT support FP16
    )

    print("\n[Export 2/2] ONNX (optional fallback)")
    model.export(
        format    = "onnx",
        imgsz     = imgsz,
        opset     = 12,
        simplify  = True,
        dynamic   = False,
    )

    ncnn_dir = weights.parent / "best_ncnn_model"
    onnx_path = weights.with_suffix(".onnx")
    print(f"\nExports ready:")
    print(f"  NCNN → {ncnn_dir}")
    print(f"  ONNX → {onnx_path}")


def validate(weights: Path, data: str, imgsz: int):
    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device="cpu")
    print(f"\nValidation mAP50:     {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95:  {metrics.box.map:.4f}")
    return metrics


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n drone detector")
    parser.add_argument("--data",    default=DEFAULTS["data"])
    parser.add_argument("--epochs",  type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--imgsz",   type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--batch",   type=int, default=DEFAULTS["batch"])
    parser.add_argument("--device",   default=DEFAULTS["device"],
                        help="'cpu' or GPU id e.g. '0'")
    parser.add_argument("--workers",  type=int, default=DEFAULTS["workers"])
    parser.add_argument("--name",     default=DEFAULTS["name"])
    parser.add_argument("--project",  default=DEFAULTS["project"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--no-export", action="store_true",
                        help="Skip export step after training")
    args = parser.parse_args()

    cfg = {**DEFAULTS}
    cfg["data"]     = args.data
    cfg["epochs"]   = args.epochs
    cfg["imgsz"]    = args.imgsz
    cfg["batch"]    = args.batch
    cfg["device"]   = args.device
    cfg["workers"]  = args.workers
    cfg["name"]     = args.name
    cfg["project"]  = args.project
    cfg["patience"] = args.patience

    best = train(cfg)
    validate(best, args.data, args.imgsz)
    if not args.no_export:
        export_for_rpi5(best, args.imgsz)


if __name__ == "__main__":
    main()