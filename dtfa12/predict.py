"""Student-only inference using native YOLOv12 decoding and NMS."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="detector.pt from dtfa12.train")
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", default="runs/dtfa12-predict")
    args = parser.parse_args()
    model = YOLO(args.weights, task="detect")
    for _ in model.predict(source=args.source, device=args.device, imgsz=args.imgsz,
                           conf=args.conf, save=True, save_txt=True, save_conf=True,
                           project=args.output, name="predict", stream=True):
        pass


if __name__ == "__main__":
    main()
