"""Native YOLOv12 detection plus DTFA's shallow (stride 8, 4) alignment."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, yaml_load

from nets.AWD.shallow_darknet import ShallowNet
from utils.AFB.loss import AFBLoss


def build_detector(names, scale="s", weights=None):
    config = Path(ultralytics.__file__).parent / "cfg/models/v12/yolov12.yaml"
    if not config.is_file():
        raise RuntimeError("Install the pinned official YOLOv12 fork from requirements-yolov12.txt")
    cfg = yaml_load(config)
    cfg["scale"] = scale
    model = DetectionModel(cfg, nc=len(names), verbose=False)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    model.names = dict(enumerate(names))
    if weights:
        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(path)
        source = YOLO(str(path)).model
        if source.yaml.get("scale") != scale:
            raise ValueError("Pretrained model scale differs from requested YOLOv12 scale")
        model.load(source)
    return model


def forward_features(model, images):
    """Capture backbone outputs, not Detect outputs; remove hooks even on failure."""
    features = {}
    handles = []
    for index in (2, 4):
        def capture(module, args, output, key=index):
            features[key] = output
        handles.append(model.model[index].register_forward_hook(capture))
    try:
        predictions = model(images)
    finally:
        for handle in handles:
            handle.remove()
    return predictions, (features[4], features[2])  # stride 8, stride 4


def feature_channels(model):
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, features = forward_features(model, torch.zeros(1, 3, 64, 64,
                device=next(model.parameters()).device))
        return tuple(x.shape[1] for x in features)
    finally:
        model.train(training)


class IRTFeatures(nn.Module):
    """The original IRT encoder, truncated after dark3 as in the AFB objective."""
    def __init__(self, checkpoint):
        super().__init__()
        original = ShallowNet().backbone
        self.stem, self.dark2, self.dark3 = original.stem, original.dark2, original.dark3
        # Original DTFA checkpoints are plain state_dicts, not pickled modules.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        expected = self.state_dict()
        selected = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            for prefix in ("backbone.backbone.", "backbone.", ""):
                candidate = key[len(prefix):] if key.startswith(prefix) else None
                if candidate in expected and value.shape == expected[candidate].shape:
                    selected[candidate] = value
                    break
        missing = set(expected) - set(selected)
        if missing:
            raise ValueError(f"IRT checkpoint is incompatible: {len(missing)} missing tensors, "
                             f"e.g. {sorted(missing)[:5]}")
        self.load_state_dict(selected, strict=True)
        self.requires_grad_(False).eval()

    def forward(self, rgb):
        # YOLOv12 uses RGB [0,1]; original IRT uses ImageNet normalization.
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        p2 = self.dark2(self.stem((rgb - mean) / std))
        return self.dark3(p2), p2


class Bridge(nn.Module):
    """Adapt channels, then retain the original shared pyramid and masked loss."""
    def __init__(self, student_channels, teacher_channels, alpha=1e-7):
        super().__init__()
        def project(channels):
            return nn.ModuleList(nn.Identity() if src == dst else nn.Conv2d(src, dst, 1)
                                 for src, dst in zip(channels, (128, 64)))
        self.student = project(student_channels)
        self.teacher = project(teacher_channels)
        self.afb = AFBLoss(alpha_mgd=alpha)

    def forward(self, student, teacher):
        s = tuple(layer(x) for layer, x in zip(self.student, student))
        t = tuple(layer(x.detach()) for layer, x in zip(self.teacher, teacher))
        return self.afb(s, t)


def export_detector(model, path):
    """A native Ultralytics checkpoint: no teachers, adapters, or persistent hooks."""
    detector = deepcopy(model).cpu().float().eval()
    if hasattr(detector, "criterion"):
        del detector.criterion
    # Native YOLO loading expects dict metadata, whereas loss uses a namespace.
    detector.args = vars(detector.args) if isinstance(detector.args, SimpleNamespace) else detector.args
    torch.save({"model": detector, "train_args": dict(DEFAULT_CFG_DICT)}, path)
