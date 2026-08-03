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

This downloads and converts `Actor001`–`Actor100` (all 100 actors, images + labels), **one actor at a time**: download → convert+resize+altitude-filter into `nomad_dataset/` → delete the raw copy → next actor. Edit `FIRST_ACTOR` / `LAST_ACTOR` at the top of the script to change the actor range, and `ALTITUDES` to change which aerial-distance tiers are kept (default `10,30` — see `--altitudes` below for why the other tiers are skipped by default).

> **Disk space:** the raw NOMAD dataset is roughly 230–280 GB for all 100 actors (~2.2–2.6 GB/actor) at native 5.4K resolution — too much to keep around twice over on a 350 GB budget. Because the script streams one actor at a time and `prepare_dataset.py` resizes images, drops irrelevant altitude tiers, and deletes each actor's raw folder as it goes (see below), peak usage stays at roughly "prepared output so far + one actor's raw" rather than "full raw dataset + full prepared copy." With the default `10,30` altitude filter, the prepared output for all 100 actors lands around 8–10 GB. If you're re-running `prepare_dataset.py` manually against an already-fully-downloaded `./NOMAD/` folder, budget for the full raw size up front instead.

---

### 2. Prepare the dataset

`download_nomad.sh` calls this automatically per-actor in streaming mode. Run it directly only if you already have a full raw NOMAD download and want to (re-)convert it in one pass:

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
| `--max-size` | `1280` | Downscale images so the long edge is at most this many pixels (`0` = native resolution). Labels are normalized so no label math is needed. Default is above the `imgsz=640` training resolution but cuts dataset size by >10x vs. the raw 5.4K frames. |
| `--keep-raw` | off | Keep each actor's raw folder after copying instead of deleting it (raw is deleted by default to keep peak disk usage down) |
| `--single-actor` / `--actor-range` | — | Streaming mode used internally by `download_nomad.sh`: convert one already-downloaded actor, with the split computed over the full actor range so it's identical across calls |
| `--altitudes` | all (`10/30/50/70/90`) | Comma-separated aerial distances (meters) to keep, e.g. `10,30`. **Match this to your mission's flight altitude.** People shrink fast with altitude — at `imgsz=640`, `a30` boxes have a median height of only ~8px (right at YOLO's finest detection stride), and `a50`/`a70`/`a90` are mostly sub-pixel and contribute near-zero training signal while diluting the dataset with unlearnable examples. |
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
| `--imgsz` | `640` | Input resolution. 416 is faster on the RPi5 CPU but shrinks people at real flight altitudes (e.g. 30m) to a few pixels — below YOLO's finest detection stride — so accuracy at mission-relevant range wins out over raw speed here. Lower it only if your timing budget can't absorb the ~2x inference cost vs. 416, and only after filtering the dataset to altitudes where 416 still resolves a usable box (see `--altitudes` above). |
| `--batch` | `16` | Batch size |
| `--device` | `cpu` | `cpu` or GPU index e.g. `0` |
| `--patience` | `20` | Early stopping patience (epochs without improvement) |

Best weights are saved under `runs/train/nomad_person/weights/best.pt` — Ultralytics appends a numeric suffix (`nomad_person2`, `nomad_person3`, ...) if that folder already exists from a previous run, so `train.py` prints the exact resolved path at the end rather than assuming the unsuffixed one.

**Augmentation notes:** `flipud=0.5` is enabled because top-down imagery has no fixed orientation. `degrees=15.0` applies moderate rotation since altitude variation already provides scale diversity in the training data.

---

### 5. Export for Raspberry Pi 5

Converts `best.pt` to NCNN format — the fastest CPU inference backend for the RPi5 with no GPU or NPU required.

```bash
python training/export.py --weights runs/train/nomad_person/weights/best.pt
```

(use the exact path `train.py` printed if the run folder got a numeric suffix)

| Flag | Default | Description |
|---|---|---|
| `--weights` | *(required)* | Path to trained `.pt` weights file |
| `--imgsz` | `640` | Must match the resolution used during training |
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
