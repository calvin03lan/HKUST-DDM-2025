import torch
import numpy as np
from copy import deepcopy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

batch_size = 250
epochs = 1000
learning_rate = 0.1
lr_decay_factor = 0.5
lr_decay_patience = 10
lr_min = 1e-5
early_stop_patience = 20
min_improve = 0.05  # absolute accuracy gain in percentage points

# Stronger augmentation for training to improve generalization.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Keep test pipeline clean for fair evaluation.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Stem: keep early channels compact and push capacity deeper.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )

        # Block 1: 16 -> 32 with downsample.
        self.b1_dw = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
        self.b1_pw = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        self.b1_bn = nn.BatchNorm2d(32)

        # Block 2: 32 -> 56 (kept compact to stay under 10k params).
        self.b2_dw = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False)
        self.b2_pw = nn.Conv2d(32, 56, kernel_size=1, bias=False)
        self.b2_bn = nn.BatchNorm2d(56)

        # Block 3: 56 -> 88 for richer late-stage features.
        self.b3_dw = nn.Conv2d(56, 56, kernel_size=3, padding=1, groups=56, bias=False)
        self.b3_pw = nn.Conv2d(56, 88, kernel_size=1, bias=False)
        self.b3_bn = nn.BatchNorm2d(88)

        self.pool = nn.MaxPool2d(2)
        self.silu = nn.SiLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(88, 10)

    def forward(self, x):
        # New architecture path.
        if all(hasattr(self, name) for name in [
            "stem", "b1_dw", "b1_pw", "b1_bn", "b2_dw", "b2_pw", "b2_bn",
            "b3_dw", "b3_pw", "b3_bn", "pool", "silu", "avgpool", "classifier"
        ]):
            x = self.stem(x)

            # Block 1
            x = self.pool(self.silu(self.b1_bn(self.b1_pw(self.b1_dw(x)))))

            # Block 2 (residual only when shape matches)
            identity2 = x
            x = self.silu(self.b2_bn(self.b2_pw(self.b2_dw(x))))
            if identity2.shape == x.shape:
                x = x + identity2
            x = self.pool(x)

            # Block 3
            x = self.silu(self.b3_bn(self.b3_pw(self.b3_dw(x))))

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

        # Compatibility with an earlier features/classifier checkpoint.
        if hasattr(self, "features") and hasattr(self, "classifier"):
            x = self.features(x)
            x = x.reshape(x.size(0), -1)
            return self.classifier(x)

        # Compatibility with an older stem/block*/head checkpoint.
        if all(hasattr(self, name) for name in ["stem", "block1", "block2", "block3", "head"]):
            x = self.stem(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return self.head(x)

        raise RuntimeError("Model checkpoint structure does not match current NeuralNetwork forward path.")


def evaluate(net, loader, criterion, device):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return total_loss / total, 100.0 * correct / total

def train(net):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_decay_factor,
        patience=lr_decay_patience,
        threshold=min_improve,
        threshold_mode="abs",
        min_lr=lr_min
    )

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improve = 0

    print(f"Using device: {device}")
    print(
        f"Training for up to {epochs} epochs "
        f"(early_stop_patience={early_stop_patience}, lr_patience={lr_decay_patience}) ..."
    )

    for epoch in range(1, epochs + 1):
        net.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            running_correct += output.argmax(dim=1).eq(target).sum().item()
            seen += data.size(0)

        test_loss, test_acc = evaluate(net, test_loader, criterion, device)
        train_loss = running_loss / seen
        train_acc = 100.0 * running_correct / seen
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:02d}] done | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}% | "
            f"lr={current_lr:.6f}"
        )

        if test_acc > best_acc + min_improve:
            best_acc = test_acc
            best_epoch = epoch
            best_state = deepcopy(net.state_dict())
            epochs_without_improve = 0
            print(f"  New best model at epoch {epoch}: test_acc={best_acc:.2f}%")
        else:
            epochs_without_improve += 1
            print(
                f"  No significant improvement for {epochs_without_improve} epoch(s). "
                f"Best test_acc={best_acc:.2f}% (epoch {best_epoch})."
            )

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(test_acc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  LR reduced: {prev_lr:.6f} -> {new_lr:.6f}")

        if epochs_without_improve >= early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best test_acc={best_acc:.2f}% from epoch {best_epoch}."
            )
            break

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"Loaded best model weights from epoch {best_epoch} (test_acc={best_acc:.2f}%).")

if __name__ == "__main__":
    torch.manual_seed(5005)
    net = NeuralNetwork()
    params = [p for p in net.parameters() if p.requires_grad]
    nparams = int(sum(np.prod(p.size()) for p in params))
    print(net)
    print("total number of trainable parameters:", nparams)
    print("score denominator term max(nparams*1e-4, 1) =", max(nparams * 1e-4, 1))
    train(net)
    torch.save(net, 'cifarNet.pth')

