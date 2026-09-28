"""
Train the baseline CNN on COUGHVID train/val, with the model selected
purely from validation performance — ICBHI is never touched here
(Phase completion rule: a stable model is selected without looking at
ICBHI test performance).

Run from the repo root:
    python src/train.py
    python src/train.py --epochs 30 --batch-size 32 --lr 1e-3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_SEED
from dataset import AeroSenseDataset, build_label_map
from model import build_model, count_parameters
from augmentation import class_weights_from_counts


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            logits = model(specs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * specs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += specs.size(0)

    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DATA_PROCESSED_DIR / "manifest.csv"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    manifest = pd.read_csv(args.manifest)
    # Training/validation only use COUGHVID here — ICBHI rows (split="test")
    # are excluded at this stage entirely; they're read only in Phase 5.
    train_val = manifest[manifest["dataset"] == "coughvid"]
    label_map = build_label_map(train_val["label"])
    print(f"[labels] {label_map}")

    train_ds = AeroSenseDataset(train_val, label_map, split="train")
    val_ds = AeroSenseDataset(train_val, label_map, split="val")
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    counts = train_val[train_val["split"] == "train"]["label"].value_counts().to_dict()
    weights_by_label = class_weights_from_counts(counts)
    weight_tensor = torch.tensor(
        [weights_by_label[label] for label, _ in sorted(label_map.items(), key=lambda kv: kv[1])],
        dtype=torch.float32,
    ).to(device)
    print(f"[class weights] {dict(zip(label_map, weight_tensor.tolist()))}")

    model = build_model(n_classes=len(label_map)).to(device)
    print(f"[model] {count_parameters(model):,} trainable parameters")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc, epochs_no_improve = 0.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "label_map": label_map,
                "val_acc": val_acc,
                "epoch": epoch,
            }, MODELS_DIR / "baseline_cnn_best.pt")
            print(f"  -> new best (val_acc={val_acc:.3f}), checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs")
                break

    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best val_acc={best_val_acc:.3f} | "
          f"checkpoint -> {MODELS_DIR / 'baseline_cnn_best.pt'} | "
          f"history -> {history_path}")


if __name__ == "__main__":
    main()
