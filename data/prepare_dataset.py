"""
NOMAD Dataset Preparation Script
===================================
Converts a raw NOMAD download into YOLO-detection format for training a
top-down person detector. NOMAD ("A Natural, Occluded, Multi-scale Aerial
Dataset, for Emergency Response Scenarios") is shot from actual aerial
distances, which is the right viewpoint for a bottom-mounted nadir drone
camera, unlike ground-level "UAV-mimicking" datasets.

Source: https://github.com/artruss/nomad
Download: https://drive.google.com/drive/folders/1zRiOzedR-PzO1bps5I1vb6jtVoQHFWzg
(Google Drive only — no CLI download script is provided by the dataset
authors. Download and extract it manually, then point --root at it.)

Expected raw layout (already YOLO-annotated by the dataset authors):
    <root>/images/Actor###/Actor###_a##/*.jpg   (a## = aerial distance, e.g. a10 = 10m)
    <root>/labels/Actor###/Actor###_a##/*.txt   (class cx cy w h, normalized)

NOMAD ships no train/val/test split — it's organized by actor and distance
instead. Splitting is done here by *actor*, not by frame: every frame from
a given actor goes entirely into one split, so the same person's appearance
never leaks between train and val/test.

Single class: every YOLO class id in the source labels is collapsed to
0 -> person. NOMAD's source labels are single-class already; the actor
activity (Walking/Hiding/Laying) lives in annotations/activityLabels.json
and visibility level in annotations/annotations.json, neither of which is
used here since this detector only needs to find people, not classify
posture (drone/incapacitation.py determines incapacitation from movement
over time instead).

Output: nomad_dataset/
    images/train/ images/val/ images/test/
    labels/train/ labels/val/ labels/test/
    dataset.yaml

Usage:
    python prepare_dataset.py --root /path/to/NOMAD
    python prepare_dataset.py --root /path/to/NOMAD --out nomad_dataset --val-ratio 0.1 --test-ratio 0.1
"""

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm

CLASS_NAMES = ["person"]
SEED = 42


def make_dirs(base: Path):
    for split in ("train", "val", "test"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)


def _force_class_zero(label_text: str) -> str:
    """Source labels may carry their own class ids — collapse them all
    to class 0 since this is a single "person" class."""
    lines = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            lines.append(" ".join(["0", *parts[1:]]))
    return "\n".join(lines)


def split_actors(actor_ids: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    """Assigns each actor entirely to one split, so no actor's appearance
    leaks across splits."""
    shuffled = sorted(actor_ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    split_by_actor = {}
    for i, actor_id in enumerate(shuffled):
        if i < n_test:
            split_by_actor[actor_id] = "test"
        elif i < n_test + n_val:
            split_by_actor[actor_id] = "val"
        else:
            split_by_actor[actor_id] = "train"
    return split_by_actor


def process_nomad(root: Path, out: Path, val_ratio: float, test_ratio: float, seed: int):
    images_root = root / "images"
    labels_root = root / "labels"
    if not images_root.exists():
        raise SystemExit(f"Error: '{images_root}' not found — check --root points at the extracted NOMAD folder.")
    if not labels_root.exists():
        raise SystemExit(f"Error: '{labels_root}' not found — download the labels/ folder alongside images/.")

    actor_dirs = sorted(p for p in images_root.iterdir() if p.is_dir())
    if not actor_dirs:
        raise SystemExit(f"Error: no Actor### subdirectories found under {images_root}")

    split_by_actor = split_actors([p.name for p in actor_dirs], val_ratio, test_ratio, seed)

    counts = {"train": 0, "val": 0, "test": 0}
    for actor_dir in actor_dirs:
        split = split_by_actor[actor_dir.name]
        distance_dirs = sorted(p for p in actor_dir.iterdir() if p.is_dir())
        for distance_dir in tqdm(distance_dirs, desc=f"{actor_dir.name} ({split})"):
            rel = distance_dir.relative_to(images_root)
            label_dir = labels_root / rel
            for img_path in sorted(distance_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                lbl_path = label_dir / (img_path.stem + ".txt")
                label = _force_class_zero(lbl_path.read_text()) if lbl_path.exists() else ""

                stem = f"{rel.as_posix().replace('/', '_')}_{img_path.stem}"
                dst_img = out / "images" / split / f"{stem}{img_path.suffix}"
                dst_lbl = out / "labels" / split / f"{stem}.txt"
                shutil.copy2(img_path, dst_img)
                dst_lbl.write_text(label)
                counts[split] += 1

    return counts


def write_yaml(out: Path):
    yaml_content = f"""# NOMAD aerial person-detection dataset
path: {out.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (out / "dataset.yaml").write_text(yaml_content)
    print(f"\ndataset.yaml written -> {out / 'dataset.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the NOMAD dataset for training")
    parser.add_argument("--root", required=True, help="Path to the extracted NOMAD download")
    parser.add_argument("--out", default="nomad_dataset", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of actors held out for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of actors held out for testing")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        raise SystemExit(
            f"Error: val_ratio ({args.val_ratio}) + test_ratio ({args.test_ratio}) "
            "must be < 1.0 or nothing will be left for training."
        )

    root, out = Path(args.root), Path(args.out)
    if not root.exists():
        raise SystemExit(f"Error: --root '{root}' does not exist.")
    make_dirs(out)

    counts = process_nomad(root, out, args.val_ratio, args.test_ratio, args.seed)
    write_yaml(out)

    for split, n in counts.items():
        print(f"  {split:6s}: {n} images")
    print("\nDone. Run data/validate_dataset.py next to sanity-check the output.")


if __name__ == "__main__":
    main()
