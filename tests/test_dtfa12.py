"""CPU integration tests; synthetic/random weights are used only in tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from dtfa12.data import PairedDataset, collate
from dtfa12.models import Bridge, IRTFeatures, build_detector, feature_channels, forward_features
from dtfa12.train import main
from nets.AWD.shallow_darknet import ShallowNet


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ["object"]
        self.classes = self.root / "classes.txt"
        self.classes.write_text("object\n")
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :40, 0] = 255
        self.image = self.root / "image.png"
        Image.fromarray(image).save(self.image)
        self.annotations = self.root / "annotations.txt"
        self.annotations.write_text(f"{self.image.as_posix()} 0,0,40,40,0\n{self.image.as_posix()}\n")

    def test_pair_geometry_and_empty_labels(self):
        data = PairedDataset(self.annotations, self.names, 64, augment=True)
        with patch("dtfa12.data.random.random", return_value=0):
            image, clean, boxes = data[0]
        torch.testing.assert_close(image, clean)
        torch.testing.assert_close(boxes, torch.tensor([[0.75, 0.5, 0.5, 0.5, 0]]))
        batch = collate([data[1], data[1]])
        self.assertEqual(batch["bboxes"].shape, (0, 4))
        self.assertEqual(batch["cls"].shape, (0, 1))
        self.assertEqual(batch["batch_idx"].numel(), 0)
        self.assertTrue(0 <= image.min() <= image.max() <= 1)

    def test_incomplete_irt_rejected(self):
        path = self.root / "bad.pth"
        torch.save({}, path)
        with self.assertRaisesRegex(ValueError, "missing tensors"):
            IRTFeatures(path)

    def test_alignment_backprop_and_teacher_freeze(self):
        student = build_detector(self.names)
        teacher = build_detector(self.names).eval().requires_grad_(False)
        self.assertEqual(feature_channels(student), (256, 128))
        bridge = Bridge(feature_channels(student), feature_channels(teacher))
        _, s = forward_features(student, torch.rand(2, 3, 64, 64))
        with torch.no_grad():
            _, t = forward_features(teacher, torch.rand(2, 3, 64, 64))
        loss = bridge(s, t)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is not None for p in bridge.parameters()))
        self.assertTrue(all(not m._forward_hooks for m in student.modules()))

    def test_spt_awd_export_and_empty_target_loss(self):
        common = ["--scale", "n", "--classes", str(self.classes), "--train", str(self.annotations),
                  "--val", str(self.annotations), "--imgsz", "64", "--batch", "2",
                  "--epochs", "1", "--device", "cpu"]
        spt_dir, awd_dir = self.root / "spt", self.root / "awd"
        main(["--stage", "spt", "--output", str(spt_dir), *common])
        irt = self.root / "irt.pth"
        torch.save(ShallowNet().state_dict(), irt)
        main(["--stage", "awd", "--output", str(awd_dir), "--spt", str(spt_dir / "best.pt"),
              "--irt", str(irt), "--clean-train", str(self.annotations), *common])
        saved = torch.load(awd_dir / "last.pt", weights_only=True)
        self.assertTrue(saved["bridges"])
        self.assertTrue(np.isfinite(saved["best"]))
        model = YOLO(str(awd_dir / "detector.pt"))
        result = model.predict(str(self.image), imgsz=64, device="cpu", verbose=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(model.names, {0: "object"})
        self.assertFalse(any("bridge" in key or "teacher" in key for key in model.model.state_dict()))
        detector = build_detector(self.names)
        batch = collate([PairedDataset(self.annotations, self.names, 64)[1]] * 2)
        loss, _ = detector.loss(batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_baseline_resume_after_interrupted_epoch(self):
        output = self.root / "baseline"
        args = ["--stage", "baseline", "--output", str(output),
                "--classes", str(self.classes), "--train", str(self.annotations),
                "--val", str(self.annotations), "--imgsz", "64", "--batch", "2",
                "--epochs", "2", "--device", "cpu"]
        original_step = torch.optim.AdamW.step
        calls = 0
        def interrupt(optimizer, *a, **kw):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return original_step(optimizer, *a, **kw)
        with patch.object(torch.optim.AdamW, "step", interrupt):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                main(args)
        saved = torch.load(output / "last.pt", weights_only=True)
        self.assertEqual(saved["epoch"], 0)
        main([*args, "--resume", str(output / "last.pt")])
        resumed = torch.load(output / "last.pt", weights_only=True)
        self.assertEqual(resumed["epoch"], 1)
        self.assertEqual(resumed["scheduler"]["last_epoch"], 2)
        self.assertEqual(len((output / "metrics.csv").read_text().splitlines()), 3)
        self.assertTrue((output / "detector.pt").is_file())


if __name__ == "__main__":
    unittest.main()
