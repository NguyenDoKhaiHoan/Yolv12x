"""Run with python -m dtfa12.train; single-device FP32 reference training.

This version adds live tqdm progress bars for train/validation, current loss,
learning rate, GPU memory usage and ETA while preserving the original training
logic and checkpoint format.
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import PairedDataset, collate
from .models import (
    Bridge,
    IRTFeatures,
    build_detector,
    export_detector,
    feature_channels,
    forward_features,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("spt", "baseline", "awd"), required=True)
    p.add_argument("--classes", required=True, help="One class per line, in label-id order")
    p.add_argument("--train", required=True, help="DTFA annotation txt/jsonl")
    p.add_argument("--val", required=True, help="DTFA annotation txt/jsonl")
    p.add_argument("--clean-train", help="Aligned clean rows for AWD")
    p.add_argument("--spt", help="best.pt from stage spt")
    p.add_argument("--irt", help="Original IRT or IRT.pth exported by dtfa12.train_irt")
    p.add_argument("--weights", help="Optional local official YOLOv12 pretrained .pt")
    p.add_argument("--scale", choices=list("nslmx"), default="s")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1e-7)
    p.add_argument("--det-weight", type=float, default=0.2, help="AWD detection multiplier")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path, help="last.pt from this script")
    return p


def load_run(path, names, scale, stage=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["names"] != names or ckpt["scale"] != scale:
        raise ValueError("Checkpoint class names/order or model scale do not match")
    if stage and ckpt["stage"] != stage:
        raise ValueError(f"Expected {stage} checkpoint, got {ckpt['stage']}")
    return ckpt


def teacher_hashes(args):
    if args.stage != "awd":
        return {}
    hashes = {}
    for key in ("spt", "irt"):
        with open(getattr(args, key), "rb") as stream:
            hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def gpu_memory_text(device):
    """Return a compact GPU-memory string for the tqdm postfix."""
    if device.type != "cuda":
        return "cpu"
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    return f"{allocated:.2f}/{reserved:.2f}G"


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)

    if min(args.batch, args.epochs) < 1 or args.lr <= 0 or args.alpha < 0 or args.det_weight <= 0:
        p.error("batch/epochs/lr/det-weight must be positive and alpha nonnegative")
    if args.stage == "awd" and not all((args.clean_train, args.spt, args.irt)):
        p.error("AWD requires --clean-train, --spt and --irt")

    names = [
        s.strip()
        for s in Path(args.classes).read_text(encoding="utf-8-sig").splitlines()
        if s.strip()
    ]
    if not names or len(names) != len(set(names)):
        p.error("Classes must be nonempty and unique")

    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error("Output directory is not empty; choose a new directory or use --resume")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    teacher_digests = teacher_hashes(args)

    student = build_detector(names, args.scale, args.weights).to(device)
    if not args.weights and not args.resume:
        print("No --weights: student starts from random initialization.", flush=True)

    train = PairedDataset(
        args.train,
        names,
        args.imgsz,
        args.clean_train if args.stage == "awd" else None,
        augment=True,
    )
    val = PairedDataset(args.val, names, args.imgsz)

    loader_args = dict(
        batch_size=args.batch,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)

    print(
        f"Stage={args.stage} | device={device} | train={len(train)} | val={len(val)} | "
        f"batch={args.batch} | train_batches={len(train_loader)} | val_batches={len(val_loader)}",
        flush=True,
    )

    bridges = nn.ModuleDict()
    spt = irt = None

    if args.stage == "awd":
        spt = build_detector(names, args.scale).to(device)
        spt.load_state_dict(load_run(args.spt, names, args.scale, "spt")["student"], strict=True)
        spt.requires_grad_(False).eval()

        irt = IRTFeatures(args.irt).to(device)
        irt.requires_grad_(False).eval()

        channels = feature_channels(student)
        bridges["spt"] = Bridge(channels, feature_channels(spt), args.alpha)
        bridges["irt"] = Bridge(channels, (128, 64), args.alpha)
        bridges.to(device)

    parameters = [
        p
        for module in (student, bridges)
        for p in module.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: 0.01
        + 0.99 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2,
    )

    start, best = 0, float("inf")

    if args.resume:
        ckpt = load_run(args.resume, names, args.scale, args.stage)

        # Loss scale and schedule must not silently change when resuming.
        for key in ("epochs", "imgsz", "batch", "alpha", "det_weight", "lr"):
            if ckpt["config"][key] != getattr(args, key):
                raise ValueError(f"Resume requires the original --{key.replace('_', '-')}")

        if "teacher_hashes" in ckpt:
            if ckpt["teacher_hashes"] != teacher_digests:
                raise ValueError("Resume requires identical teacher checkpoint contents")
        elif any(ckpt["config"][key] != getattr(args, key) for key in ("spt", "irt")):
            raise ValueError("Legacy checkpoint requires original teacher paths")

        student.load_state_dict(ckpt["student"], strict=True)
        bridges.load_state_dict(ckpt["bridges"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start, best = ckpt["epoch"] + 1, ckpt["best"]
        torch.set_rng_state(ckpt["torch_rng"])
        random.setstate(ckpt["python_rng"])
        if device.type == "cuda" and ckpt["cuda_rng"]:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

        print(f"Resuming from epoch {start + 1}/{args.epochs}", flush=True)

    if start >= args.epochs:
        p.error("Checkpoint already reached the requested number of epochs")

    args.output.mkdir(parents=True, exist_ok=True)
    history = args.output / "metrics.csv"

    with history.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(
                ["epoch", "train_total", "train_detection", "irt", "spt", "val_detection"]
            )
            stream.flush()

        epoch_bar = tqdm(
            range(start, args.epochs),
            desc="Epochs",
            unit="epoch",
            dynamic_ncols=True,
            position=0,
        )

        for epoch in epoch_bar:
            student.train()
            bridges.train()
            totals = torch.zeros(4, device=device)

            train_bar = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Train {epoch + 1}/{args.epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
                position=1,
            )

            for batch in train_bar:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)

                if args.stage == "awd":
                    preds, student_features = forward_features(student, batch["img"])
                    with torch.no_grad():
                        _, spt_features = forward_features(spt, batch["clean"])
                        irt_features = irt(batch["clean"])
                    rloss = bridges["irt"](student_features, irt_features)
                    dloss = bridges["spt"](student_features, spt_features)
                else:
                    preds = student(batch["img"])
                    rloss = dloss = batch["img"].new_zeros(())

                det, _ = student.loss(batch, preds)

                # Both the native detection objective and AFB sum over the batch.
                count = batch["img"].shape[0]
                total = (
                    (args.det_weight if args.stage == "awd" else 1.0) * det
                    + rloss
                    + dloss
                ) / count

                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite training loss")

                total.backward()
                nn.utils.clip_grad_norm_(parameters, 10.0)
                optimizer.step()

                totals += torch.stack(
                    (
                        total.detach() * count,
                        det.detach(),
                        rloss.detach(),
                        dloss.detach(),
                    )
                )

                lr_now = optimizer.param_groups[0]["lr"]
                postfix = {
                    "loss": f"{total.detach().item():.4f}",
                    "det": f"{(det.detach() / count).item():.4f}",
                    "lr": f"{lr_now:.2e}",
                    "mem": gpu_memory_text(device),
                }
                if args.stage == "awd":
                    postfix["irt"] = f"{(rloss.detach() / count).item():.4f}"
                    postfix["spt"] = f"{(dloss.detach() / count).item():.4f}"
                train_bar.set_postfix(postfix, refresh=False)

            train_bar.close()

            student.eval()
            val_sum = 0.0

            val_bar = tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Val   {epoch + 1}/{args.epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
                position=1,
            )

            with torch.no_grad():
                for batch in val_bar:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    loss, _ = student.loss(batch)
                    val_sum += loss.item()

                    val_bar.set_postfix(
                        {
                            "det": f"{(loss.item() / batch['img'].shape[0]):.4f}",
                            "mem": gpu_memory_text(device),
                        },
                        refresh=False,
                    )

            val_bar.close()

            val_loss = val_sum / len(val)
            if not math.isfinite(val_loss):
                raise FloatingPointError("Non-finite validation loss")

            scheduler.step()
            values = (totals / len(train)).tolist()

            writer.writerow([epoch + 1, *values, val_loss])
            stream.flush()

            lr_now = optimizer.param_groups[0]["lr"]
            epoch_bar.set_postfix(
                {
                    "train": f"{values[0]:.5f}",
                    "val": f"{val_loss:.5f}",
                    "lr": f"{lr_now:.2e}",
                },
                refresh=True,
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train={values[0]:.5f}, det={values[1]:.5f}, "
                f"irt={values[2]:.5f}, spt={values[3]:.5f}, "
                f"val_det={val_loss:.5f}, lr={lr_now:.3e}",
                flush=True,
            )

            improved = val_loss < best
            best = min(best, val_loss)
            config = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
            }
            checkpoint = dict(
                student=student.state_dict(),
                bridges=bridges.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=epoch,
                best=best,
                names=names,
                scale=args.scale,
                stage=args.stage,
                config=config,
                teacher_hashes=teacher_digests,
                torch_rng=torch.get_rng_state(),
                python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            )

            torch.save(checkpoint, args.output / "last.pt")
            if improved:
                torch.save(checkpoint, args.output / "best.pt")
                export_detector(student, args.output / "detector.pt")

    print(f"Training complete. Best validation detection loss: {best:.6f}", flush=True)
    print(f"Outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
