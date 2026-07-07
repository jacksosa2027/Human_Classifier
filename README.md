# SAM Drone Model

A YOLOv8n-based aerial person detector for search-and-rescue (SAR) drone operations. The model is trained on the [NOMAD dataset](https://github.com/artruss/nomad) — a top-down aerial dataset captured at multiple altitudes over forested environments — and exported to NCNN for CPU inference on a Raspberry Pi 5.

---

## Environment setup

**Training machine** (requires a CUDA-capable GPU):

```bash
conda env create -f environment.yml
conda activate nomad_detect
```

**Raspberry Pi 5** (inference only, no GPU):

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements_rpi5.txt
```

---

## Data pipeline

### 1. Download the NOMAD dataset

NOMAD is hosted on Google Drive and requires a one-time manual access grant before rclone can reach it.

**Prerequisites:**

```bash
sudo apt install rclone
rclone config   # create a remote named "gdrive" -> type: Google Drive
```

Accept the dataset access agreement at the link below (required once, in a browser):  
[https://github.com/artruss/nomad](https://github.com/artruss/nomad)

**Run the download script:**

```bash
bash data/download_nomad.sh
```

By default this downloads `Actor001`–`Actor095` (images + labels only) into `./NOMAD/`. Edit `FIRST_ACTOR` / `LAST_ACTOR` at the top of the script to change the range.

> **Disk space:** each actor's folder is several GB. The full dataset (95 actors) requires approximately 200–250 GB of free space before preparation. Use `--delete-raw` in the next step to reclaim raw files as they are processed.

---

### 2. Prepare the dataset

Converts the raw NOMAD layout into YOLO-format train/val/test splits.

```bash
python data/prepare_dataset.py --root ./NOMAD
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--root` | *(required)* | Path to the extracted NOMAD download |
| `--out` | `nomad_dataset` | Output directory for the prepared dataset |
| `--val-ratio` | `0.1` | Fraction of actors held out for validation |
| `--test-ratio` | `0.1` | Fraction of actors held out for test |
| `--delete-raw` | off | Delete each actor's raw folder after copying, to reclaim disk space |
| `--seed` | `42` | Random seed for the actor split |

Splitting is done **by actor**, not by frame — every frame from a given actor goes entirely into one split, so the same person's appearance never leaks between train and val/test.

The output directory follows the standard YOLO layout:

```
nomad_dataset/
├── images/train/  images/val/  images/test/
├── labels/train/  labels/val/  labels/test/
└── dataset.yaml
```

---

### 3. Validate the dataset

Sanity-checks the prepared dataset for missing labels, mismatched image/label pairs, and out-of-range bounding box values.

```bash
python data/validate_dataset.py --data nomad_dataset
```

Add `--visualise` to write a grid of annotated sample images to `nomad_dataset/sample_grid.jpg`:

```bash
python data/validate_dataset.py --data nomad_dataset --visualise --n 16
```

> **Note:** frames where the person is fully occluded have intentionally empty label files (valid negative training examples). These are reported as "Background frames" and are not a data integrity issue.

---

## Training

### 4. Train the model

```bash
python training/train.py --device 0
```

Pass `--device cpu` to train without a GPU (significantly slower). All standard Ultralytics training parameters can be overridden as flags — the most useful ones:

| Flag | Default | Description |
|---|---|---|
| `--data` | `nomad_dataset/dataset.yaml` | Dataset config |
| `--model` | `yolov8n.pt` | Base weights (downloads automatically on first run) |
| `--epochs` | `100` | Maximum training epochs |
| `--imgsz` | `416` | Input resolution (416 is ~2× faster than 640 on CPU at inference) |
| `--batch` | `16` | Batch size |
| `--device` | `cpu` | `cpu` or GPU index e.g. `0` |
| `--patience` | `20` | Early stopping patience (epochs without improvement) |

Best weights are saved to `runs/train/nomad_person/weights/best.pt`.

**Augmentation notes:** `flipud=0.5` is enabled because top-down imagery has no fixed orientation. `degrees=15.0` applies moderate rotation since altitude variation already provides scale diversity in the training data.

---

### 5. Export for Raspberry Pi 5

Converts `best.pt` to NCNN format — the fastest CPU inference backend for the RPi5 with no GPU or NPU required.

```bash
python training/export.py --weights runs/train/nomad_person/weights/best.pt
```

| Flag | Default | Description |
|---|---|---|
| `--weights` | *(required)* | Path to trained `.pt` weights file |
| `--imgsz` | `416` | Must match the resolution used during training |
| `--format` | `ncnn` | `ncnn` (recommended for RPi5) or `onnx` |

Copy the exported folder to the Raspberry Pi and set `model_path` in `configs/mission_config.yaml` to point at it.

---

## Dataset

**NOMAD** — *A Natural, Occluded, Multi-scale Aerial Dataset for Emergency Response Scenarios*  
- 42,825 frames from 5.4K resolution video, 100 actors  
- Captured at five aerial distances over natural environments  
- Annotations: YOLO bounding boxes, 10 visibility levels, activity labels (Walking / Hiding / Laying)  
- License: see [https://github.com/artruss/nomad](https://github.com/artruss/nomad)  
- Access: [Google Drive](https://drive.google.com/drive/folders/1zRiOzedR-PzO1bps5I1vb6jtVoQHFWzg)

This pipeline uses YOLO bounding-box labels only. Visibility level and activity metadata are available in `annotations/` but are not used during training — the model detects people regardless of posture, and incapacitation is determined at inference time by a separate temporal heuristic.
