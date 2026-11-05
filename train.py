import argparse
import torch


from pathlib import Path
from torch.utils.data import DataLoader, Subset
from lane_detection.deep_learning.dataset import LaneDataset
from lane_detection.deep_learning.losses import BCEDiceLoss, batch_iou
from lane_detection.deep_learning.model import UNet


def make_loaders(root, size, batch_size, val_fraction, workers):
    train_ds = LaneDataset(root, size, augment=True)
    if len(train_ds) < 20:
        raise SystemExit(
            f"Only {len(train_ds)} images found in {root}/images. "
            "Need at least ~20 (ideally 1000+). Rebuild the dataset first.")
    val_ds = LaneDataset(root, size, augment=False)
    order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(42)).tolist()
    n_val = max(1, int(len(order) * val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]
    return (DataLoader(Subset(train_ds, train_idx), batch_size, shuffle=True, num_workers=workers),
            DataLoader(Subset(val_ds, val_idx), batch_size, shuffle=False, num_workers=workers))


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_iou = count = 0
    with torch.set_grad_enabled(training):
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n = imgs.size(0)
            total_loss += loss.item() * n
            total_iou += batch_iou(logits, masks) * n
            count += n
    return total_loss / count, total_iou / count


def main():
    p = argparse.ArgumentParser(description="Train lane U-Net")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("weights/unet_lane.pt"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0, help="keep 0 on Windows")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    size = (args.height, args.width)

    train_loader, val_loader = make_loaders(
        args.data, size, args.batch_size, args.val_fraction, args.workers)
    model = UNet(base=args.base).to(device)
    loss_fn = BCEDiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        tl, ti = run_epoch(model, train_loader, loss_fn, device, optimizer)
        vl, vi = run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch {epoch:02d}/{args.epochs} | train loss {tl:.3f} IoU {ti:.3f} "
              f"| val loss {vl:.3f} IoU {vi:.3f}")
        if vi > best_iou:                       # keep only the best student
            best_iou = vi
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "base": args.base,
                        "size": list(size)}, args.out)
            print(f"  ↳ saved best model to {args.out}")

if __name__ == "__main__":
    main()
