"""Validate fog F_<id>_<L|M|H> and clear <id> pairs; emit JSONL and YOLO YAML."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_index(directory):
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"No images in {directory}")
    result = {}
    for path in files:
        if path.stem in result:
            raise ValueError(f"Duplicate image id: {path}")
        result[path.stem] = path
    return result


def read_labels(path, nc):
    if not path.is_file():
        raise FileNotFoundError(f"Missing label (empty objects require an empty file): {path}")
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    labels = np.asarray(rows, dtype=np.float64)
    if not rows:
        return np.empty((0, 5))
    if labels.ndim != 2 or labels.shape[1] != 5 or not np.isfinite(labels).all():
        raise ValueError(f"Expected YOLO class cx cy w h: {path}")
    cls, xywh = labels[:, 0], labels[:, 1:]
    if ((cls < 0) | (cls >= nc) | (cls != cls.astype(int))).any():
        raise ValueError(f"Invalid class id: {path}")
    if (xywh < 0).any() or (xywh > 1).any() or (xywh[:, 2:] <= 0).any():
        raise ValueError(f"Invalid normalized bounding box: {path}")
    return labels


def annotation(image, labels, size):
    w, h = size
    boxes = []
    for cls, cx, cy, bw, bh in labels:
        boxes.append([max(0, (cx - bw / 2) * w), max(0, (cy - bh / 2) * h),
                      min(w, (cx + bw / 2) * w), min(h, (cy + bh / 2) * h), int(cls)])
    return {"image": image.as_posix(), "boxes": boxes}


def prepare(root, output, levels=("L", "M", "H")):
    root, output = Path(root).resolve(), Path(output).resolve()
    levels = tuple(levels)
    if not levels or len(set(levels)) != len(levels) or set(levels) - {"L", "M", "H"}:
        raise ValueError("Select unique levels from L M H")
    if any(char.isspace() for char in str(output)):
        raise ValueError("Prepared output path must not contain spaces (native YOLO validation)")
    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8-sig"))
    names = config["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if not names or config.get("nc", len(names)) != len(names):
        raise ValueError("Invalid class names/nc in data.yaml")
    clear_root = root / "clear images" / "vesion2"
    report = {"classes": names, "levels": levels, "splits": {}}
    split_ids, records, native_lists = {}, {}, {}
    # Validate everything before writing annotations; never pair by sorted position.
    for split in ("train", "val", "test"):
        clear = image_index(clear_root / split / "images")
        fog = image_index(root / split / "images")
        split_ids[split] = set(clear)
        records[split] = {"fog": [], "paired_clear": [], "clear": []}
        native_lists[split] = {level: [] for level in ("L", "M", "H", "mixed", "clear")}
        clear_labels, clear_sizes = {}, {}
        for key, image in clear.items():
            labels = read_labels(clear_root / split / "labels" / f"{key}.txt", len(names))
            with Image.open(image) as im:
                size = im.size
            clear_labels[key], clear_sizes[key] = labels, size
            records[split]["clear"].append(annotation(image, labels, size))
            native_lists[split]["clear"].append(image.as_posix())
        counts, actual_pairs = Counter(), set()
        for stem, image in fog.items():
            match = re.fullmatch(r"F_(.+)_([LMH])", stem)
            if not match:
                raise ValueError(f"Expected F_<clear id>_<L|M|H>: {image}")
            key, level = match.groups()
            if key not in clear:
                raise ValueError(f"No clear pair in the same split for {image}")
            labels = read_labels(root / split / "labels" / f"{stem}.txt", len(names))
            with Image.open(image) as im:
                size = im.size
            if size != clear_sizes[key]:
                raise ValueError(f"Clear/fog size mismatch: {image}")
            # Fog synthesis should preserve boxes. Ignore label row ordering.
            a = np.asarray(sorted(labels.tolist())).reshape(-1, 5)
            b = np.asarray(sorted(clear_labels[key].tolist())).reshape(-1, 5)
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-5, rtol=0):
                raise ValueError(f"Clear/fog labels differ: {image}")
            counts[level] += 1
            actual_pairs.add((key, level))
            native_lists[split][level].append(image.as_posix())
            if level in levels:
                records[split]["fog"].append(annotation(image, labels, size))
                records[split]["paired_clear"].append(annotation(clear[key], clear_labels[key], size))
                native_lists[split]["mixed"].append(image.as_posix())
        missing = {(key, level) for key in clear for level in levels} - actual_pairs
        if missing:
            raise ValueError(f"Missing {len(missing)} selected fog pairs in {split}: {sorted(missing)[:5]}")
        report["splits"][split] = {"clear": len(clear), "fog_by_level": dict(counts),
                                   "selected_fog": len(records[split]["fog"])}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_ids[a] & split_ids[b]
        if overlap:
            raise ValueError(f"Scene ids leak across {a}/{b}: {sorted(overlap)[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    for split in records:
        for name, rows in records[split].items():
            (output / f"{split}_{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        for level, images in native_lists[split].items():
            (output / f"{split}_{level}_images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    for level in ("L", "M", "H", "mixed", "clear"):
        spec = {"path": output.as_posix(), "names": names,
                **{split: (output / f"{split}_{level}_images.txt").as_posix()
                   for split in ("train", "val", "test")}}
        (output / f"{level}.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="Directory containing data.yaml and train/val/test")
    p.add_argument("--output", type=Path, default=Path("prepared"))
    p.add_argument("--levels", nargs="+", choices=("L", "M", "H"), default=["L", "M", "H"])
    args = p.parse_args()
    prepare(args.root, args.output, args.levels)


if __name__ == "__main__":
    main()
