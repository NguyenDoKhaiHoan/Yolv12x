import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from dtfa12.data import PairedDataset
from dtfa12.prepare_data import prepare


class Preparation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "fog"
        self.output = Path(self.tmp.name) / "prepared"
        self.root.mkdir()
        (self.root / "data.yaml").write_text("names: [object]\nnc: 1\n")
        for i, split in enumerate(("train", "val", "test")):
            key = f"{i:04}"
            for base in (self.root / split, self.root / "clear images/vesion2" / split):
                (base / "images").mkdir(parents=True)
                (base / "labels").mkdir()
            for level in ("L", "M", "H", "clear"):
                base = self.root / split if level != "clear" else self.root / "clear images/vesion2" / split
                stem = f"F_{key}_{level}" if level != "clear" else key
                Image.new("RGB", (80, 40), "white").save(base / "images" / f"{stem}.jpg")
                (base / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n")

    def prepare(self, levels=("L", "M", "H")):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.root, self.output, levels)

    def test_pairs_levels_and_space_in_clear_path(self):
        report = self.prepare(("H",))
        self.assertEqual(report["splits"]["train"]["selected_fog"], 1)
        row = json.loads((self.output / "train_fog.jsonl").read_text())
        self.assertTrue(row["image"].endswith("F_0000_H.jpg"))
        self.assertEqual(row["boxes"], [[20, 10, 60, 30, 0]])
        data = PairedDataset(self.output / "train_fog.jsonl", ["object"], 64,
                             clean=self.output / "train_paired_clear.jsonl")
        image, clear, boxes = data[0]
        self.assertEqual(image.shape, clear.shape)
        self.assertEqual(tuple(boxes.shape), (1, 5))
        spec = yaml.safe_load((self.output / "H.yaml").read_text())
        self.assertTrue(Path(spec["test"]).is_file())

    def test_accepts_archive_parent_with_nested_data_yaml(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = prepare(self.root.parent, self.output, ("L",))
        self.assertEqual(report["splits"]["val"]["selected_fog"], 1)

    def test_rejects_missing_pair(self):
        (self.root / "train/images/F_0000_L.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "Missing 1 selected fog pairs"):
            self.prepare()

    def test_rejects_conflicting_labels(self):
        (self.root / "train/labels/F_0000_L.txt").write_text("0 0.4 0.5 0.5 0.5\n")
        with self.assertRaisesRegex(ValueError, "labels differ"):
            self.prepare()

    def test_rejects_split_leakage(self):
        for directory in ("val", "clear images/vesion2/val"):
            for folder, suffix in (("images", ".jpg"), ("labels", ".txt")):
                for path in (self.root / directory / folder).glob(f"*{suffix}"):
                    path.rename(path.with_name(path.name.replace("0001", "0000")))
        with self.assertRaisesRegex(ValueError, "leak across"):
            self.prepare()

    def test_native_evaluation_from_prepared_yaml(self):
        import torch
        from dtfa12.evaluate import main
        from dtfa12.models import build_detector, export_detector
        torch.set_num_threads(1)
        self.prepare()
        weights = self.output / "detector.pt"
        export_detector(build_detector(["object"], scale="n"), weights)
        target = self.output / "evaluation"
        args = ["evaluate", "--weights", str(weights), "--data", str(self.output / "L.yaml"),
                "--split", "test", "--device", "cpu", "--batch", "1", "--imgsz", "64",
                "--output", str(target)]
        # Font download is unrelated to metric correctness and disabled in offline tests.
        with patch("sys.argv", args), patch("ultralytics.data.utils.check_font"):
            main()
        metrics = json.loads((target / "metrics.json").read_text())
        self.assertIn("metrics/mAP50(B)", metrics)
        self.assertIn("metrics/mAP50-95(B)", metrics)


if __name__ == "__main__":
    unittest.main()
