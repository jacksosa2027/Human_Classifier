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

Images are re-encoded to --max-size (long edge, default 1280) on the way in.
Training runs at imgsz=416 (see training/train.py), so storing the source
5.4K frames verbatim would waste >10x the disk space for no accuracy gain.
Labels are normalized [0,1] YOLO coords, so no label math is needed for the
resize. Pass --max-size 0 to copy images verbatim instead.

Raw files are deleted per-actor as soon as they're converted (see
--keep-raw to disable) — this is what lets data/download_nomad.sh stream
one actor at a time instead of needing the whole raw dataset on disk at once.

Usage:
    python prepare_dataset.py --root /path/to/NOMAD
    python prepare_dataset.py --root /path/to/NOMAD --out nomad_dataset --val-ratio 0.1 --test-ratio 0.1

    # Streaming mode (used internally by download_nomad.sh): convert+delete
    # one already-downloaded actor at a time, with the split assignment
    # computed over the full actor range so it's identical across calls.
    python prepare_dataset.py --root ./NOMAD --single-actor Actor007 --actor-range 1 100
"""

import argparse
import random
import re
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

CLASS_NAMES = ["person"]
SEED = 42
DEFAULT_MAX_SIZE = 1280
DISTANCE_RE = re.compile(r"_a(\d+)$")


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


def _write_image(img_path: Path, dst_img: Path, max_size: int):
    """Copies img_path to dst_img, downscaling to max_size on the long edge.
    Labels are normalized [0,1] YOLO coords, so they're unaffected by the resize."""
    if max_size <= 0:
        shutil.copy2(img_path, dst_img)
        return
    img = cv2.imread(str(img_path))
    if img is None:
        shutil.copy2(img_path, dst_img)  # unreadable by cv2 (unexpected format) — fall back verbatim
        return
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if dst_img.suffix.lower() in (".jpg", ".jpeg") else []
    cv2.imwrite(str(dst_img), img, encode_params)


def actor_id_range(first: int, last: int) -> list[str]:
    return [f"Actor{i:03d}" for i in range(first, last + 1)]


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


def _distance_of(distance_dir: Path) -> int | None:
    """Extracts the aerial distance in meters from a '..._a##' directory name."""
    m = DISTANCE_RE.search(distance_dir.name)
    return int(m.group(1)) if m else None


def _process_actor(actor_dir: Path, images_root: Path, labels_root: Path, out: Path,
                    split: str, max_size: int, delete_raw: bool, altitudes: set[int] | None = None) -> int:
    n = 0
    distance_dirs = sorted(p for p in actor_dir.iterdir() if p.is_dir())
    if altitudes is not None:
        distance_dirs = [d for d in distance_dirs if _distance_of(d) in altitudes]
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
            _write_image(img_path, dst_img, max_size)
            dst_lbl.write_text(label)
            n += 1

    if delete_raw:
        shutil.rmtree(actor_dir)
        raw_labels_dir = labels_root / actor_dir.name
        if raw_labels_dir.exists():
            shutil.rmtree(raw_labels_dir)
        print(f"  Deleted raw: {actor_dir.name}")

    return n


def process_nomad(root: Path, out: Path, val_ratio: float, test_ratio: float, seed: int,
                   max_size: int = DEFAULT_MAX_SIZE, delete_raw: bool = True,
                   altitudes: set[int] | None = None):
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
        counts[split] += _process_actor(actor_dir, images_root, labels_root, out, split,
                                         max_size, delete_raw, altitudes)

    return counts


def process_single_actor(root: Path, out: Path, actor_id: str, actor_range: tuple[int, int],
                          val_ratio: float, test_ratio: float, seed: int,
                          max_size: int = DEFAULT_MAX_SIZE, delete_raw: bool = True,
                          altitudes: set[int] | None = None) -> dict:
    """Converts one already-downloaded actor and (by default) deletes its raw
    folder immediately — lets download_nomad.sh stream actors one at a time
    instead of needing the whole raw NOMAD dataset on disk at once. The split
    assignment is recomputed from the full actor range each call so it's
    identical regardless of which actor is processed first."""
    images_root = root / "images"
    labels_root = root / "labels"
    actor_dir = images_root / actor_id
    if not actor_dir.exists():
        raise SystemExit(f"Error: '{actor_dir}' not found — did the download for {actor_id} succeed?")

    all_actor_ids = actor_id_range(*actor_range)
    split_by_actor = split_actors(all_actor_ids, val_ratio, test_ratio, seed)
    if actor_id not in split_by_actor:
        raise SystemExit(f"Error: {actor_id} is outside --actor-range {actor_range}")
    split = split_by_actor[actor_id]

    n = _process_actor(actor_dir, images_root, labels_root, out, split, max_size, delete_raw, altitudes)
    return {"train": 0, "val": 0, "test": 0, split: n}


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
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE,
                        help="Downscale images so their long edge is at most this many pixels "
                             "(0 = copy at native resolution). Labels are normalized so they "
                             "need no adjustment. Default keeps well above the imgsz=416 training "
                             "resolution while cutting dataset size by >10x vs. the raw 5.4K frames.")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep each actor's raw NOMAD folder after copying instead of deleting it. "
                             "Off by default because raw+prepared together can exceed disk budgets; "
                             "only use this if you have space to spare.")
    parser.add_argument("--single-actor", metavar="ActorNNN",
                        help="Streaming mode: convert only this one already-downloaded actor, "
                             "then apply --keep-raw/delete as usual. Requires --actor-range so the "
                             "split assignment is computed the same way on every call.")
    parser.add_argument("--actor-range", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="Full Actor### range (e.g. 1 100) used to compute the deterministic "
                             "train/val/test split. Required with --single-actor.")
    parser.add_argument("--altitudes", default=None,
                        help="Comma-separated aerial distances in meters to keep (e.g. '10,30'). "
                             "Default keeps all five NOMAD tiers (10/30/50/70/90). Frames get "
                             "smaller as altitude increases — at imgsz=416/640 training resolution, "
                             "a70/a90 boxes are mostly sub-pixel and contribute near-zero signal. "
                             "Filter to the altitude range your mission actually flies at.")
    args = parser.parse_args()
    altitudes = {int(a) for a in args.altitudes.split(",")} if args.altitudes else None

    if args.val_ratio + args.test_ratio >= 1.0:
        raise SystemExit(
            f"Error: val_ratio ({args.val_ratio}) + test_ratio ({args.test_ratio}) "
            "must be < 1.0 or nothing will be left for training."
        )
    if args.single_actor and not args.actor_range:
        raise SystemExit("Error: --single-actor requires --actor-range FIRST LAST.")

    root, out = Path(args.root), Path(args.out)
    if not root.exists():
        raise SystemExit(f"Error: --root '{root}' does not exist.")
    make_dirs(out)
    delete_raw = not args.keep_raw

    if args.single_actor:
        counts = process_single_actor(root, out, args.single_actor, tuple(args.actor_range),
                                       args.val_ratio, args.test_ratio, args.seed,
                                       args.max_size, delete_raw, altitudes)
    else:
        counts = process_nomad(root, out, args.val_ratio, args.test_ratio, args.seed,
                                args.max_size, delete_raw, altitudes)
    write_yaml(out)

    for split, n in counts.items():
        print(f"  {split:6s}: {n} images")
    print("\nDone. Run data/validate_dataset.py next to sanity-check the output.")


if __name__ == "__main__":
    main()
