import argparse
import csv
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class TrainConfig:
    npz_path: str
    batch_size: int
    epochs: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    gru_layers: int
    dropout: float
    label_smoothing: float
    scheduler_t0: int
    scheduler_t_mult: int
    patience: int
    num_classes: int
    device: str
    save_dir: str
    grad_clip: float


class QuickDrawDataset(Dataset):
    def __init__(self, npz_path: str, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        with np.load(npz_path, allow_pickle=False) as data:
            x_key = f"X_{split}"
            y_key = f"y_{split}"
            r_key = f"ratios_{split}"
            if x_key not in data or y_key not in data:
                raise KeyError(f"Missing keys in npz: {x_key}, {y_key}")
            x_array = data[x_key].astype(np.float32, copy=False)
            y_array = data[y_key].astype(np.int64, copy=False)
            if r_key in data:
                r_array = data[r_key].astype(np.float32, copy=False)
            else:
                raise KeyError(f"Missing key in npz: {r_key}")

        ratio_mask = r_array >= 0.40
        x_array = x_array[ratio_mask]
        y_array = y_array[ratio_mask]

        nonzero_mask = np.any(x_array != 0, axis=2)
        lengths = nonzero_mask.sum(axis=1).astype(np.int64, copy=False)
        lengths = np.clip(lengths, 1, x_array.shape[1])

        self.x_tensor = torch.from_numpy(x_array)
        self.y_tensor = torch.from_numpy(y_array)
        self.len_tensor = torch.from_numpy(lengths)

    def __len__(self) -> int:
        return self.y_tensor.shape[0]

    def __getitem__(self, index: int):
        return self.x_tensor[index], self.y_tensor[index], self.len_tensor[index]


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = self.act(out + identity)
        return out


class ConvGRUClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float, gru_layers: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.block1 = ResidualBlock1D(64, 128, stride=2)
        self.block2 = ResidualBlock1D(128, 256, stride=2)
        self.block3 = ResidualBlock1D(256, 256, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.3 if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=8, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    @staticmethod
    def _conv_out_length(lengths: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int = 1) -> torch.Tensor:
        lengths = (lengths + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        return torch.clamp(lengths, min=1)

    @staticmethod
    def _downsample_lengths(lengths: torch.Tensor, stride: int) -> torch.Tensor:
        lengths = (lengths - 1) // stride + 1
        return torch.clamp(lengths, min=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.block1(x)
        lengths = self._downsample_lengths(lengths, stride=2)
        x = self.block2(x)
        lengths = self._downsample_lengths(lengths, stride=2)
        x = self.block3(x)
        x = x.transpose(1, 2)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        max_len = gru_out.size(1)
        padding_mask = torch.arange(max_len, device=gru_out.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_out, _ = self.attn(
            gru_out,
            gru_out,
            gru_out,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        valid_mask = (~padding_mask).unsqueeze(-1).to(attn_out.dtype)
        summed = (attn_out * valid_mask).sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp(min=1.0)
        features = summed / denom
        features = self.dropout(features)
        logits = self.head(features)
        return logits


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_confusion(confusion: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    preds = preds.view(-1).to(torch.int64)
    labels = labels.view(-1).to(torch.int64)
    indices = labels * num_classes + preds
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion += counts.view(num_classes, num_classes)
    return confusion


def compute_macro_f1(confusion: torch.Tensor, eps: float = 1e-9) -> float:
    confusion = confusion.to(torch.float32)
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.mean().item()


def accuracy_topk(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    topk = torch.topk(logits, k=k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1).sum().item()
    return int(correct)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                    device: torch.device, num_classes: int, grad_clip: float, label_smoothing: float,
                    scheduler: torch.optim.lr_scheduler._LRScheduler | None, epoch_progress: float) -> dict:
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top3 = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    total_batches = max(len(loader), 1)
    for batch_idx, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        inputs, labels, lengths = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(inputs, lengths)
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step(epoch_progress + (batch_idx + 1) / total_batches)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        correct_top1 += (logits.argmax(dim=1) == labels).sum().item()
        correct_top3 += accuracy_topk(logits, labels, k=3)

        preds = logits.argmax(dim=1).detach().cpu()
        update_confusion(confusion, preds, labels.detach().cpu(), num_classes)

    return {
        "loss": running_loss / max(total_samples, 1),
        "top1": correct_top1 / max(total_samples, 1),
        "top3": correct_top3 / max(total_samples, 1),
        "macro_f1": compute_macro_f1(confusion),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, label_smoothing: float) -> dict:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top3 = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            inputs, labels, lengths = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            logits = model(inputs, lengths)
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            correct_top1 += (logits.argmax(dim=1) == labels).sum().item()
            correct_top3 += accuracy_topk(logits, labels, k=3)

            preds = logits.argmax(dim=1).detach().cpu()
            update_confusion(confusion, preds, labels.detach().cpu(), num_classes)

    return {
        "loss": running_loss / max(total_samples, 1),
        "top1": correct_top1 / max(total_samples, 1),
        "top3": correct_top3 / max(total_samples, 1),
        "macro_f1": compute_macro_f1(confusion),
    }


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def format_metrics(prefix: str, metrics: dict) -> str:
    return (
        f"{prefix} loss={metrics['loss']:.4f} "
        f"top1={metrics['top1']:.4f} "
        f"top3={metrics['top3']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


def train(config: TrainConfig) -> None:
    os.makedirs(config.save_dir, exist_ok=True)
    device = torch.device(config.device)

    overall_start = time.time()

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training. Please run with a CUDA-capable device.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    total_mem_gb = gpu_props.total_memory / (1024 ** 3)
    print(f"Using GPU: {gpu_name} | VRAM: {total_mem_gb:.2f} GB")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_set = QuickDrawDataset(config.npz_path, "train")
    val_set = QuickDrawDataset(config.npz_path, "val")

    train_loader = build_loader(train_set, config.batch_size, config.num_workers, shuffle=True)
    val_loader = build_loader(val_set, config.batch_size, config.num_workers, shuffle=False)

    model = ConvGRUClassifier(
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
        dropout=config.dropout,
        gru_layers=config.gru_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_t0,
        T_mult=config.scheduler_t_mult,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_f1 = -1.0
    epochs_without_improve = 0

    metrics_path = os.path.join(config.save_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_top1",
            "train_top3",
            "train_macro_f1",
            "val_loss",
            "val_top1",
            "val_top3",
            "val_macro_f1",
            "epoch_seconds",
        ])

        for epoch in range(1, config.epochs + 1):
            start_time = time.time()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                config.num_classes,
                config.grad_clip,
                config.label_smoothing,
                scheduler,
                epoch - 1,
            )
            val_metrics = evaluate(model, val_loader, device, config.num_classes, config.label_smoothing)
            elapsed = time.time() - start_time

            print(f"Epoch {epoch}/{config.epochs} - {elapsed:.1f}s")
            print(format_metrics("Train", train_metrics))
            print(format_metrics("Val  ", val_metrics))

            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["top1"],
                train_metrics["top3"],
                train_metrics["macro_f1"],
                val_metrics["loss"],
                val_metrics["top1"],
                val_metrics["top3"],
                val_metrics["macro_f1"],
                elapsed,
            ])
            metrics_file.flush()

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                epochs_without_improve = 0
                best_path = os.path.join(config.save_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model to {best_path}")
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= config.patience:
                print("Early stopping triggered.")
                break

    last_path = os.path.join(config.save_dir, "last_model.pt")
    torch.save(model.state_dict(), last_path)
    print(f"Saved last model to {last_path}")

    total_elapsed = time.time() - overall_start
    print(f"Total training time: {total_elapsed / 60:.2f} minutes")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Quick Draw CNN+GRU training")
    parser.add_argument("--npz-path", type=str, default="quickdraw_processed.npz")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--scheduler-t0", type=int, default=5)
    parser.add_argument("--scheduler-t-mult", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA GPU is required. Please set --device cuda and ensure drivers are installed.")

    return TrainConfig(
        npz_path=args.npz_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        scheduler_t0=args.scheduler_t0,
        scheduler_t_mult=args.scheduler_t_mult,
        patience=args.patience,
        num_classes=args.num_classes,
        device=args.device,
        save_dir=args.save_dir,
        grad_clip=args.grad_clip,
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
