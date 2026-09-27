"""Native YOLOv12 mAP evaluation of an exported detector."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="Exported detector.pt")
    p.add_argument("--data", required=True, help="Prepared mixed/L/M/H/clear.yaml")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    metrics = YOLO(args.weights).val(data=args.data, split=args.split, device=args.device,
        batch=args.batch, imgsz=args.imgsz, workers=0, conf=0.001, iou=0.7,
        project=str(args.output.parent), name=args.output.name, exist_ok=False, plots=False)
    result = {key: float(value) for key, value in metrics.results_dict.items()}
    (Path(metrics.save_dir) / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
