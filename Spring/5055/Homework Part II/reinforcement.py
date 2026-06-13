import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch import nn

try:
    import ale_py

    gym.register_envs(ale_py)
except Exception:
    # Keep compatibility for setups where Atari envs are pre-registered.
    pass

# To properly use gym, you should run
# pip install "gymnasium[atari, accept-rom-license]"

SEED = 42
STATE_DIM = 128
ACTION_DIM = 7


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(STATE_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
        )

    def forward(self, state):
        features = self.feature(state)
        values = self.value_stream(features)
        advantages = self.adv_stream(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)

    def action(self, state):
        """
        Generate one greedy action compatible with the checker.
        """
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.to(dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # Keep this robust for direct RAM input without preprocessing.
        if torch.max(state).item() > 1.0:
            state = state / 255.0
        q_values = self.forward(state)
        return int(torch.argmax(q_values, dim=1)[0].item())

    def implement_your_method_if_needed(self):
        return self.feature


def _set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _soft_update(source, target, tau):
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def train(env, net):
    """
    Train Dueling Double DQN on Assault RAM.
    """
    _set_seed(SEED)

    device = torch.device("cpu")
    net.to(device)
    net.train()

    target_net = NeuralNetwork().to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(net.parameters(), lr=1e-4, eps=1.5e-4)
    replay = deque(maxlen=200000)

    batch_size = 128
    gamma = 0.99
    warmup_steps = 6000
    total_steps = 420000
    train_freq = 4
    tau = 0.005
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 320000

    state, _ = env.reset(seed=SEED)
    episode_reward = 0.0
    reward_window = deque(maxlen=30)
    best_seen_episode_reward = float("-inf")

    for step in range(1, total_steps + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0, (epsilon_decay_steps - step) / float(epsilon_decay_steps)
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = (
                    torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                )
                action = net.action(state_tensor)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay.append((state, action, float(reward), next_state, done))

        state = next_state
        episode_reward += reward

        if done:
            reward_window.append(episode_reward)
            if episode_reward > best_seen_episode_reward:
                best_seen_episode_reward = episode_reward
            state, _ = env.reset()
            episode_reward = 0.0

        if len(replay) >= warmup_steps and step % train_freq == 0:
            batch = random.sample(replay, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.as_tensor(np.array(states), dtype=torch.float32, device=device) / 255.0
            next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device) / 255.0
            actions = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = net(states).gather(1, actions)
            with torch.no_grad():
                next_actions = net(next_states).argmax(dim=1, keepdim=True)
                next_q = target_net(next_states).gather(1, next_actions)
                target = rewards + gamma * (1.0 - dones) * next_q

            loss = nn.functional.smooth_l1_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            _soft_update(net, target_net, tau=tau)

        if step % 10000 == 0:
            mean_recent = float(np.mean(reward_window)) if reward_window else 0.0
            print(
                f"step {step:6d} | epsilon {epsilon:.3f} | "
                f"recent_mean {mean_recent:.1f} | best_episode {best_seen_episode_reward:.1f}"
            )

    net.eval()


if __name__ == "__main__":
    environment = gym.make("Assault-v4", obs_type="ram")
    model = NeuralNetwork()
    print(model)
    train(environment, model)
    environment.close()
    torch.save(model, "reinforcement.pth")
