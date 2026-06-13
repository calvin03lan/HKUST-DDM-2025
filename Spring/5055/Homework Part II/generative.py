import bz2
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dfile = bz2.BZ2File("./xyData.bz2")
data = torch.from_numpy(np.load(dfile)).to(torch.float32)
dfile.close()


ADJ_ELE = np.array([
    -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1,
    1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1,
    -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1,
    -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1,
    1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1,
    -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1,
    1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1,
    1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
    1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1,
    1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1,
    -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1,
    -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1,
    1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1,
    1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1,
    -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1,
    -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1,
    1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1,
    -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1,
    1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1,
    -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1,
    1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1,
    1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1,
    -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1,
    1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1,
    1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1,
    -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1,
    -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1,
    1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1,
    1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1,
    -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1,
    -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1,
    -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1,
    -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1,
    -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1,
    -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1,
    1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1,
    -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1,
    1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1,
    1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1,
    -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1,
    -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1,
    -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1,
    -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1,
    1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1,
    -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1,
    -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1,
    -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1,
    -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1,
    1, 1, -1, 1, 1, 1
], dtype=np.int64)


def wrap_angle(angle):
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class Lattice:
    def __init__(self, length, dim, bc="periodic"):
        self.length = length
        self.dim = dim
        self.nsite = length**dim
        self.bc = bc

    def move(self, idx, dim, shift):
        coord = self.index2coord(idx)
        coord[dim] += shift

        if self.bc != "periodic":
            if coord[dim] >= self.length or coord[dim] < 0:
                return None

        if coord[dim] >= self.length:
            coord[dim] -= self.length
        if coord[dim] < 0:
            coord[dim] += self.length
        return self.coord2index(coord)

    def index2coord(self, idx):
        coord = np.zeros(self.dim, dtype=int)
        for d in range(self.dim):
            coord[self.dim - d - 1] = idx % self.length
            idx //= self.length
        return coord

    def coord2index(self, coord):
        idx = coord[0]
        for d in range(1, self.dim):
            idx *= self.length
            idx += coord[d]
        return idx


class Hypercube(Lattice):
    def __init__(self, length, dim, bc="periodic"):
        super().__init__(length, dim, bc)
        self.adj = np.zeros((self.nsite, self.nsite), dtype=int)
        for i in range(self.nsite):
            for d in range(self.dim):
                j = self.move(i, d, 1)
                if j is not None:
                    self.adj[i, j] = 1
                    self.adj[j, i] = 1


def build_adjacency_matrix():
    lattice = Hypercube(16, 2)
    adj_mask = torch.from_numpy(lattice.adj).bool()
    adj = torch.zeros(adj_mask.shape, dtype=torch.long)
    adj = adj.masked_scatter(adj_mask, torch.from_numpy(ADJ_ELE).long())
    return adj.to(torch.float32)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_angles = nn.Parameter(torch.empty(1, 1, 16, 16))
        nn.init.uniform_(self.base_angles, -math.pi, math.pi)

        smooth_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ) / 5.0
        self.register_buffer("smooth_kernel", smooth_kernel.view(1, 1, 3, 3))
        self.register_buffer("adjacency", build_adjacency_matrix())
        self.noise_scale = 0.05
        self.smoothing_steps = 3

    def energy(self, batch):
        spins = torch.cat([torch.cos(batch), torch.sin(batch)], dim=1)
        flat_spins = spins.flatten(-2)
        return -((flat_spins @ self.adjacency) * flat_spins).sum((-2, -1)) / 2.0

    def smooth_noise(self, noise):
        for _ in range(self.smoothing_steps):
            padded = F.pad(noise, (1, 1, 1, 1), mode="circular")
            noise = F.conv2d(padded, self.smooth_kernel)
        return noise

    def sample(self, batchSize):
        base = self.base_angles.expand(batchSize, -1, -1, -1)
        noise = torch.randn(
            batchSize, 1, 16, 16, device=self.base_angles.device, dtype=self.base_angles.dtype
        )
        noise = self.smooth_noise(noise) * self.noise_scale

        # A global rotation preserves the pairwise XY interaction energy.
        global_phase = torch.empty(
            batchSize, 1, 1, 1, device=self.base_angles.device, dtype=self.base_angles.dtype
        ).uniform_(-math.pi, math.pi)
        return wrap_angle(base + noise + global_phase)

    def implement_your_method_if_needed(self):
        return self.energy(self.base_angles).mean()


def train(net):
    net.to(device)

    with torch.no_grad():
        candidate_count = min(4096, data.shape[0])
        candidate_batch = data[:candidate_count].to(device)
        candidate_energy = net.energy(candidate_batch)
        best_idx = torch.argmin(candidate_energy)
        net.base_angles.copy_(candidate_batch[best_idx : best_idx + 1])

    optimizer = torch.optim.Adam([net.base_angles], lr=0.08)
    train_steps = 1500

    for step in range(train_steps):
        generated = net.sample(128)
        loss = net.energy(generated).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            net.base_angles.copy_(wrap_angle(net.base_angles))

        if step % 300 == 0 or step == train_steps - 1:
            print(f"step {step:4d} | mean energy {loss.item():.4f}")

    net.cpu()


if __name__ == "__main__":
    net = NeuralNetwork()
    print(net)
    train(net)
    torch.save(net, "generative.pth")
