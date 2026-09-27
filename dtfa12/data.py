"""DTFA text annotations with synchronized clean/adverse letterboxing and flips."""
from pathlib import Path
import json
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


def read_lines(path):
    lines = [s.strip() for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not lines:
        raise ValueError(f"Empty annotation file: {path}")
    return lines


def parse_annotation(line):
    """JSONL supports paths containing spaces; legacy DTFA TXT remains valid."""
    if line.startswith("{"):
        record = json.loads(line)
        return record["image"], record.get("boxes", [])
    image, *labels = line.split()
    return image, [list(map(float, label.split(","))) for label in labels]


class PairedDataset(Dataset):
    def __init__(self, annotations, names, size=640, clean=None, augment=False):
        self.lines = read_lines(annotations)
        self.clean = read_lines(clean) if clean else self.lines
        if len(self.lines) != len(self.clean):
            raise ValueError("Clean/adverse annotation files must have the same number of rows")
        if size < 64 or size % 32:
            raise ValueError("Image size must be >=64 and divisible by 32")
        self.size, self.nc, self.augment = size, len(names), augment

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        adverse, labels = parse_annotation(self.lines[index])
        clear, _ = parse_annotation(self.clean[index])
        with Image.open(adverse) as source:
            image = source.convert("RGB")
        with Image.open(clear) as source:
            clean = source.convert("RGB")
        if image.size != clean.size:
            raise ValueError(f"Pair {index} must have matching image dimensions")
        w, h = image.size
        ratio = min(self.size / w, self.size / h)
        nw, nh = round(w * ratio), round(h * ratio)
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        def letterbox(im):
            canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
            canvas.paste(im.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
            return canvas
        image, clean = letterbox(image), letterbox(clean)
        boxes = np.asarray(labels, dtype=np.float32).reshape(-1, 5)
        if len(boxes):
            if not np.isfinite(boxes).all():
                raise ValueError(f"Non-finite annotation at row {index}")
            classes = boxes[:, 4]
            if ((classes < 0) | (classes >= self.nc) | (classes != classes.astype(int))).any():
                raise ValueError(f"Invalid class id at row {index}")
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w) * nw / w + left
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h) * nh / h + top
            boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        if self.augment and random.random() < 0.5:
            image, clean = ImageOps.mirror(image), ImageOps.mirror(clean)
            boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        if len(boxes):
            boxes[:, 2:4] -= boxes[:, :2]
            boxes[:, :2] += boxes[:, 2:4] / 2
            boxes[:, :4] /= self.size
        def tensor(im):
            return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
        return tensor(image), tensor(clean), torch.from_numpy(boxes)


def collate(samples):
    images, clean, boxes = zip(*samples)
    return {"img": torch.stack(images), "clean": torch.stack(clean),
            "bboxes": torch.cat([x[:, :4] for x in boxes]),
            "cls": torch.cat([x[:, 4:5] for x in boxes]),
            "batch_idx": torch.cat([torch.full((len(x),), i, dtype=torch.long)
                                    for i, x in enumerate(boxes)])}
