import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    def col(name):
        return [float(row[name]) for row in rows]

    epochs = [int(row["epoch"]) for row in rows]
    return {
        "epochs": epochs,
        "train_loss": col("train_loss"),
        "val_loss": col("val_loss"),
        "train_top1": col("train_top1"),
        "val_top1": col("val_top1"),
        "train_top3": col("train_top3"),
        "val_top3": col("val_top3"),
        "train_macro_f1": col("train_macro_f1"),
        "val_macro_f1": col("val_macro_f1"),
    }


def plot_curves(metrics, out_path: Path):
    epochs = metrics["epochs"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, metrics["train_loss"], label="Train")
    axes[0, 0].plot(epochs, metrics["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, metrics["train_top1"], label="Train")
    axes[0, 1].plot(epochs, metrics["val_top1"], label="Val")
    axes[0, 1].set_title("Top-1 Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, metrics["train_top3"], label="Train")
    axes[1, 0].plot(epochs, metrics["val_top3"], label="Val")
    axes[1, 0].set_title("Top-3 Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, metrics["train_macro_f1"], label="Train")
    axes[1, 1].plot(epochs, metrics["val_macro_f1"], label="Val")
    axes[1, 1].set_title("Macro-F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--csv", type=str, default="checkpoints/metrics.csv")
    parser.add_argument("--out", type=str, default="checkpoints/metrics.png")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    metrics = read_metrics(csv_path)
    plot_curves(metrics, out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
