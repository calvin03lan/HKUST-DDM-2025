import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

WINDOW_SIZE = 200
BATCH_SIZE = 256
VAL_TARGET_POINTS = 50000
MAX_TRAIN_TARGET_POINTS = 50000
EPOCHS = 3
EARLY_STOP_PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


class DipeptideDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        dat = self.data[idx]
        lab = self.labels[idx]

        return dat, lab


class WindowedSeriesDataset(Dataset):
    def __init__(self, series, window_size, start_idx=0, end_idx=None):
        self.series = series
        self.window_size = window_size
        self.start_idx = int(start_idx)
        self.end_idx = int(series.shape[0] if end_idx is None else end_idx)
        self.length = max(0, self.end_idx - self.start_idx - self.window_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = self.start_idx + idx
        end = start + self.window_size
        return self.series[start:end], self.series[end]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_norm = nn.LayerNorm(30)
        self.gru = nn.GRU(
            input_size=30,
            hidden_size=192,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )
        self.delta_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Linear(128, 30),
        )
        # Start close to persistence baseline; then learn residual correction.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, x):
        """
        Input x is a [batchSize, 200, 30] array;
        The return should be a [batchSize, 30] array.
        """
        x = self.input_norm(x)
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        last_step = x[:, -1, :]
        delta = self.delta_head(last_hidden)
        return last_step + delta


def train(net, train_loader, test_loader):
    """
    Train your model
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        net.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = net(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        net.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = net(batch_x)
                val_losses.append(criterion(pred, batch_y).item())

        train_mse = float(np.mean(train_losses)) if train_losses else float("inf")
        val_mse = float(np.mean(val_losses)) if val_losses else float("inf")
        scheduler.step(val_mse)
        print(f"epoch {epoch:02d} | train_mse {train_mse:.6f} | val_mse {val_mse:.6f}")

        if val_mse < best_loss:
            best_loss = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOP_PATIENCE:
                print(f"early stop at epoch {epoch}, best_val_mse={best_loss:.6f}")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    series = torch.from_numpy(np.load("./trainDipeptide.npy")).float()
    n_points = series.shape[0]
    split_idx = n_points - VAL_TARGET_POINTS

    train_start = max(0, split_idx - MAX_TRAIN_TARGET_POINTS - WINDOW_SIZE)
    trainset = WindowedSeriesDataset(
        series,
        window_size=WINDOW_SIZE,
        start_idx=train_start,
        end_idx=split_idx,
    )
    testset = WindowedSeriesDataset(
        series,
        window_size=WINDOW_SIZE,
        start_idx=max(0, split_idx - WINDOW_SIZE),
        end_idx=n_points,
    )

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    net = NeuralNetwork()
    print(net)
    train(net, train_loader, test_loader)
    NeuralNetwork.__module__ = "sequential"
    torch.save(net, "sequential.pth")
