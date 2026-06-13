import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_quickdraw import ConvGRUClassifier, compute_macro_f1, update_confusion, accuracy_topk


class QuickDrawEvalDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        with np.load(npz_path, allow_pickle=False) as data:
            if "X_test" not in data or "y_test" not in data or "ratios_test" not in data:
                raise KeyError("Missing X_test, y_test, or ratios_test in npz file")
            x_array = data["X_test"].astype(np.float32, copy=False)
            y_array = data["y_test"].astype(np.int64, copy=False)
            r_array = data["ratios_test"].astype(np.float32, copy=False)

        self.x_tensor = torch.from_numpy(x_array)
        self.y_tensor = torch.from_numpy(y_array)
        self.r_tensor = torch.from_numpy(r_array)

    def __len__(self) -> int:
        return self.y_tensor.shape[0]

    def __getitem__(self, index: int):
        return self.x_tensor[index], self.y_tensor[index], self.r_tensor[index]


def build_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def compute_lengths(inputs: torch.Tensor) -> torch.Tensor:
    nonzero_mask = torch.any(inputs != 0, dim=2)
    lengths = nonzero_mask.sum(dim=1)
    lengths = torch.clamp(lengths, min=1, max=inputs.size(1))
    return lengths


def init_group_metrics(num_classes: int):
    return {
        "count": 0,
        "top1": 0,
        "top3": 0,
        "confusion": torch.zeros(num_classes, num_classes, dtype=torch.int64),
    }


def update_group(group: dict, logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> None:
    if labels.numel() == 0:
        return
    group["count"] += labels.size(0)
    group["top1"] += (logits.argmax(dim=1) == labels).sum().item()
    group["top3"] += accuracy_topk(logits, labels, k=3)
    preds = logits.argmax(dim=1).detach().cpu()
    update_confusion(group["confusion"], preds, labels.detach().cpu(), num_classes)


def finalize_group(group: dict) -> dict:
    count = max(group["count"], 1)
    return {
        "top1": group["top1"] / count,
        "top3": group["top3"] / count,
        "macro_f1": compute_macro_f1(group["confusion"]),
        "count": group["count"],
    }


def plot_bar(metrics: dict, title: str, overall: dict, ax) -> None:
    labels = ["Top-1", "Top-3", "Macro-F1"]
    values = [metrics["top1"], metrics["top3"], metrics["macro_f1"]]
    overall_values = [overall["top1"], overall["top3"], overall["macro_f1"]]

    ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    for idx, value in enumerate(overall_values):
        ax.hlines(value, idx - 0.35, idx + 0.35, colors="#444444", linestyles="dashed", linewidth=1.2)
        ax.text(idx, value + 0.02, f"Overall {value:.3f}", ha="center", va="bottom", fontsize=8, color="#444444")

    ax.set_xlabel("Metric")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate truncation-specific metrics")
    parser.add_argument("--npz-path", type=str, default="quickdraw_processed.npz")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for evaluation.")

    dataset = QuickDrawEvalDataset(args.npz_path)
    loader = build_loader(dataset, args.batch_size, args.num_workers)

    model = ConvGRUClassifier(
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
        dropout=args.dropout,
        gru_layers=args.gru_layers,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    targets = {
        "10pct": 0.10,
        "20pct": 0.20,
        "30pct": 0.30,
        "40pct": 0.40,
        "50pct": 0.50,
        "60pct": 0.60,
        "75pct": 0.75,
        "90pct": 0.90,
        "100pct": 1.0,
    }

    groups = {"overall": init_group_metrics(args.num_classes)}
    for key in targets:
        groups[key] = init_group_metrics(args.num_classes)

    with torch.no_grad():
        for inputs, labels, ratios in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ratios = ratios.to(device, non_blocking=True)

            lengths = compute_lengths(inputs)
            logits = model(inputs, lengths)

            update_group(groups["overall"], logits, labels, args.num_classes)

            for key, target in targets.items():
                mask = torch.isclose(ratios, torch.tensor(target, device=ratios.device), atol=1e-4)
                if mask.any():
                    update_group(groups[key], logits[mask], labels[mask], args.num_classes)

    results = {name: finalize_group(group) for name, group in groups.items()}

    print("\nTest metrics by truncation:")
    for name, metrics in results.items():
        print(
            f"{name:>7} | n={metrics['count']:<7} "
            f"top1={metrics['top1']:.4f} "
            f"top3={metrics['top3']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_metrics = results["overall"]
    title_map = {
        "10pct": "Test Performance: 10% Completion",
        "20pct": "Test Performance: 20% Completion",
        "30pct": "Test Performance: 30% Completion",
        "40pct": "Test Performance: 40% Completion",
        "50pct": "Test Performance: 50% Completion",
        "60pct": "Test Performance: 60% Completion",
        "75pct": "Test Performance: 75% Completion",
        "90pct": "Test Performance: 90% Completion",
        "100pct": "Test Performance: 100% Completion",
    }

    order = ["overall", "10pct", "20pct", "30pct", "40pct", "50pct", "60pct", "75pct", "90pct", "100pct"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    plot_bar(overall_metrics, "Test Performance: Overall", overall_metrics, axes[0])
    for idx, key in enumerate(order[1:], start=1):
        plot_bar(results[key], title_map[key], overall_metrics, axes[idx])

    fig.suptitle("Test Performance by Truncation", y=1.02)
    fig.tight_layout()
    combined_path = out_dir / "test_bar_all.png"
    fig.savefig(combined_path, dpi=200)
    print(f"Saved plot to {combined_path}")

    truncation_order = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]
    truncation_keys = ["10pct", "20pct", "30pct", "40pct", "50pct", "60pct", "75pct", "90pct", "100pct"]
    top1_values = [results[key]["top1"] for key in truncation_keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(truncation_order, top1_values, marker="o", color="#4C72B0")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Truncation Ratio")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Top-1 Accuracy vs. Truncation")
    ax.set_xticks(truncation_order)
    ax.set_xticklabels(["10%", "20%", "30%", "40%", "50%", "60%", "75%", "90%", "100%"])
    fig.tight_layout()
    curve_path = out_dir / "test_top1_curve.png"
    fig.savefig(curve_path, dpi=200)
    print(f"Saved plot to {curve_path}")


if __name__ == "__main__":
    main()
