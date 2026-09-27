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
VALID_LEVELS = ("L", "M", "H")
VALID_SPLITS = ("train", "val", "test")


def image_index(directory: Path):
    """Recursively index images by filename stem."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No images in {directory}")

    result = {}
    for path in files:
        if path.stem in result:
            raise ValueError(
                f"Duplicate image id '{path.stem}' under {directory}: "
                f"{result[path.stem]} and {path}"
            )
        result[path.stem] = path
    return result


def read_labels(path: Path, nc: int):
    """Read a YOLO txt label file and validate class ids / normalized xywh."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing label (empty objects require an empty file): {path}"
        )

    rows = [
        list(map(float, line.split()))
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if not rows:
        return np.empty((0, 5), dtype=np.float64)

    labels = np.asarray(rows, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 5 or not np.isfinite(labels).all():
        raise ValueError(f"Expected YOLO class cx cy w h: {path}")

    cls, xywh = labels[:, 0], labels[:, 1:]
    if ((cls < 0) | (cls >= nc) | (cls != cls.astype(int))).any():
        raise ValueError(f"Invalid class id: {path}")
    if (xywh < 0).any() or (xywh > 1).any() or (xywh[:, 2:] <= 0).any():
        raise ValueError(f"Invalid normalized bounding box: {path}")

    return labels


def annotation(image: Path, labels: np.ndarray, size):
    """Convert normalized YOLO xywh labels to pixel xyxy JSON records."""
    w, h = size
    boxes = []
    for cls, cx, cy, bw, bh in labels:
        boxes.append(
            [
                max(0.0, (cx - bw / 2) * w),
                max(0.0, (cy - bh / 2) * h),
                min(float(w), (cx + bw / 2) * w),
                min(float(h), (cy + bh / 2) * h),
                int(cls),
            ]
        )
    return {"image": image.as_posix(), "boxes": boxes}


def _normalize_names(config: dict, config_path: Path):
    names = config.get("names", [])
    if isinstance(names, dict):
        try:
            keys = sorted(names, key=lambda k: int(k))
        except (TypeError, ValueError):
            keys = sorted(names)
        names = [names[k] for k in keys]
    elif isinstance(names, (tuple, list)):
        names = list(names)
    else:
        raise ValueError(f"Invalid 'names' in {config_path}: expected list or dict")

    if not names:
        raise ValueError(f"No class names found in {config_path}")

    nc = config.get("nc", len(names))
    try:
        nc = int(nc)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid nc in {config_path}: {nc!r}") from exc

    if nc != len(names):
        raise ValueError(
            f"Invalid class names/nc in {config_path}: nc={nc}, names={len(names)}"
        )
    return names


def _resolve_clear_root(root: Path):
    """Support the dataset's actual typo 'vesion2' and the corrected 'version2'."""
    candidates = [
        root / "clear images" / "vesion2",
        root / "clear images" / "version2",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find clear dataset root. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _resolve_fog_root(root: Path):
    """Find the fog dataset directory that contains train/val/test and data.yaml."""
    candidates = [
        root / "fog" / "fog",
        root / "fog",
        root,
    ]
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "data.yaml").is_file()
            and any((candidate / split).is_dir() for split in VALID_SPLITS)
        ):
            return candidate

    found = sorted(root.rglob("data.yaml")) if root.is_dir() else []
    viable = [
        p.parent
        for p in found
        if any((p.parent / split).is_dir() for split in VALID_SPLITS)
    ]
    if len(viable) == 1:
        return viable[0]

    raise FileNotFoundError(
        f"Cannot uniquely determine fog root under {root}. Candidates: {viable or found}"
    )


def _label_path(split_root: Path, stem: str):
    """Return labels/<stem>.txt for a split and fail with a clear message if absent."""
    return split_root / "labels" / f"{stem}.txt"


def prepare(root, output, levels=("L", "M", "H")):
    root = Path(root).resolve()
    output = Path(output).resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    levels = tuple(levels)
    if not levels or len(set(levels)) != len(levels) or set(levels) - set(VALID_LEVELS):
        raise ValueError("Select unique levels from L M H")
    if any(char.isspace() for char in str(output)):
        raise ValueError("Prepared output path must not contain spaces (native YOLO validation)")

    # Keep --root unchanged. Resolve the two dataset branches independently.
    clear_root = _resolve_clear_root(root)
    fog_root = _resolve_fog_root(root)
    config_path = fog_root / "data.yaml"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}")
    names = _normalize_names(config, config_path)

    print(f"Dataset root : {root}")
    print(f"Clear root   : {clear_root}")
    print(f"Fog root     : {fog_root}")
    print(f"data.yaml    : {config_path}")
    print(f"Classes      : {len(names)}")
    print(f"Levels       : {', '.join(levels)}")

    report = {"classes": names, "levels": list(levels), "splits": {}}
    split_ids = {}
    records = {}
    native_lists = {}

    # Validate everything before writing annotations; never pair by sorted position.
    for split in VALID_SPLITS:
        clear_split = clear_root / split
        fog_split = fog_root / split

        # Clear images may be directly under split or under split/images; recursive indexing handles both.
        clear = image_index(clear_split)
        fog = image_index(fog_split / "images")

        split_ids[split] = set(clear)
        records[split] = {"fog": [], "paired_clear": [], "clear": []}
        native_lists[split] = {
            level: [] for level in (*VALID_LEVELS, "mixed", "clear")
        }

        clear_labels = {}
        clear_sizes = {}

        for key, image in clear.items():
            labels = read_labels(_label_path(clear_split, key), len(names))
            with Image.open(image) as im:
                size = im.size
            clear_labels[key] = labels
            clear_sizes[key] = size
            records[split]["clear"].append(annotation(image, labels, size))
            native_lists[split]["clear"].append(image.as_posix())

        counts = Counter()
        actual_pairs = set()

        for stem, image in fog.items():
            match = re.fullmatch(r"F_(.+)_([LMH])", stem)
            if not match:
                raise ValueError(f"Expected F_<clear id>_<L|M|H>: {image}")

            key, level = match.groups()
            if key not in clear:
                raise ValueError(f"No clear pair in the same split for {image}")

            labels = read_labels(_label_path(fog_split, stem), len(names))
            with Image.open(image) as im:
                size = im.size

            if size != clear_sizes[key]:
                raise ValueError(
                    f"Clear/fog size mismatch for id={key}: "
                    f"clear={clear_sizes[key]}, fog={size}, image={image}"
                )

            # Fog synthesis should preserve boxes. Ignore label-row ordering.
            a = np.asarray(sorted(labels.tolist()), dtype=np.float64).reshape(-1, 5)
            b = np.asarray(sorted(clear_labels[key].tolist()), dtype=np.float64).reshape(-1, 5)
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-5, rtol=0):
                raise ValueError(f"Clear/fog labels differ: {image}")

            counts[level] += 1
            actual_pairs.add((key, level))
            native_lists[split][level].append(image.as_posix())

            if level in levels:
                records[split]["fog"].append(annotation(image, labels, size))
                records[split]["paired_clear"].append(
                    annotation(clear[key], clear_labels[key], size)
                )
                native_lists[split]["mixed"].append(image.as_posix())

        missing = {(key, level) for key in clear for level in levels} - actual_pairs
        if missing:
            preview = sorted(missing)[:5]
            raise ValueError(
                f"Missing {len(missing)} selected fog pairs in {split}: {preview}"
            )

        report["splits"][split] = {
            "clear": len(clear),
            "fog_by_level": {level: counts.get(level, 0) for level in VALID_LEVELS},
            "selected_fog": len(records[split]["fog"]),
        }

    # Prevent scene leakage across splits.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_ids[a] & split_ids[b]
        if overlap:
            raise ValueError(f"Scene ids leak across {a}/{b}: {sorted(overlap)[:5]}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    for split in records:
        for name, rows in records[split].items():
            (output / f"{split}_{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

        for level, images in native_lists[split].items():
            (output / f"{split}_{level}_images.txt").write_text(
                "\n".join(images) + ("\n" if images else ""),
                encoding="utf-8",
            )

    # Ultralytics YAMLs for native validation/inference lists.
    for level in (*VALID_LEVELS, "mixed", "clear"):
        spec = {
            "path": output.as_posix(),
            "names": names,
            **{
                split: (output / f"{split}_{level}_images.txt").as_posix()
                for split in VALID_SPLITS
            },
        }
        (output / f"{level}.yaml").write_text(
            yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Dataset parent containing 'clear images/<vesion2|version2>' and "
            "the fog dataset (usually 'fog/fog')."
        ),
    )
    p.add_argument("--output", type=Path, default=Path("prepared"))
    p.add_argument(
        "--levels",
        nargs="+",
        choices=VALID_LEVELS,
        default=list(VALID_LEVELS),
    )
    args = p.parse_args()
    prepare(args.root, args.output, args.levels)


if __name__ == "__main__":
    main()
