"""Pretrain a compatible IRT locally from paired fog/clear; no external weights."""
import argparse
import csv
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PairedDataset, collate
from .restoration import RestorationTeacher, psnr, reconstruction_loss

RECIPE = "dtfa12-shallow-reconstruction-v1"


def save_checkpoint(state, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("classes", "train", "clean-train", "val", "clean-val"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--identity-weight", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if min(args.epochs, args.batch) < 1 or args.lr <= 0 or args.identity_weight < 0 or args.workers < 0:
        p.error("Positive epochs/batch/lr and nonnegative identity-weight/workers required")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        p.error("Output is nonempty; choose a new directory or --resume")
    names = [s.strip() for s in Path(args.classes).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not names or len(names) != len(set(names)):
        p.error("Classes must be nonempty and unique")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = PairedDataset(args.train, names, args.imgsz, args.clean_train, augment=True)
    val = PairedDataset(args.val, names, args.imgsz, args.clean_val)
    device = torch.device(args.device)
    loader_args = dict(batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                       pin_memory=device.type == "cuda")
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    model = RestorationTeacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda epoch: 0.01 + 0.99 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2)
    start, best = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if ckpt["recipe"] != RECIPE or ckpt["names"] != names:
            raise ValueError("IRT recipe/classes differ from checkpoint")
        for key in ("epochs", "batch", "imgsz", "lr", "identity_weight", "seed"):
            if ckpt["config"][key] != getattr(args, key):
                raise ValueError(f"Resume requires original --{key.replace('_', '-')}")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start, best = ckpt["epoch"] + 1, ckpt["best"]
        torch.set_rng_state(ckpt["torch_rng"])
        random.setstate(ckpt["python_rng"])
        if device.type == "cuda" and ckpt["cuda_rng"]:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    if start >= args.epochs:
        p.error("Checkpoint already reached requested epochs")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with (args.output / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(["epoch", "train_l1", "train_identity_l1", "val_l1",
                             "val_psnr", "val_fog_psnr"])
        for epoch in range(start, args.epochs):
            model.train()
            train_sums = torch.zeros(2, device=device)
            for batch in tqdm(train_loader, desc=f"IRT {epoch + 1}/{args.epochs}"):
                fog, clear = batch["img"].to(device), batch["clean"].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, restoration, identity = reconstruction_loss(model, fog, clear, args.identity_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite IRT training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                train_sums += torch.stack((restoration.detach(), identity.detach())) * len(fog)
            model.eval()
            val_sums = torch.zeros(3, device=device)
            with torch.no_grad():
                for batch in val_loader:
                    fog, clear = batch["img"].to(device), batch["clean"].to(device)
                    restored = model(fog)
                    val_sums[0] += F.l1_loss(restored, clear) * len(fog)
                    val_sums[1] += psnr(restored, clear).sum()
                    val_sums[2] += psnr(fog, clear).sum()
            if not torch.isfinite(val_sums).all():
                raise FloatingPointError("Non-finite IRT validation metrics")
            values = (val_sums / len(val)).tolist()
            scheduler.step()
            writer.writerow([epoch + 1, *(train_sums / len(train)).tolist(), *values])
            stream.flush()
            print(f"IRT epoch {epoch + 1}: val_L1={values[0]:.5f}, "
                  f"restored_PSNR={values[1]:.2f}, fog_PSNR={values[2]:.2f}", flush=True)
            improved = values[0] < best
            best = min(best, values[0])
            state = dict(recipe=RECIPE, model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, best=best, names=names, config=config,
                torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            if improved:
                save_checkpoint(state, args.output / "best.pt")
                save_checkpoint({"state_dict": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
                    "recipe": RECIPE, "epoch": epoch, "val_l1": values[0], "val_psnr": values[1],
                    "val_fog_psnr": values[2]}, args.output / "IRT.pth")
            save_checkpoint(state, args.output / "last.pt")
    print(f"AWD teacher: {args.output / 'IRT.pth'}; best validation L1={best:.5f}")


if __name__ == "__main__":
    main()
