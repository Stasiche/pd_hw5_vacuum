import os.path

from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy
from buffer import Buffer


class DQN:
    def __init__(self, action_dim, buffer_size, batch_size, config):
        self.config = config
        self.batch_size = batch_size
        self.steps = 0
        self.action_dim = action_dim

        self.device = torch.device("cuda:0" if config['cuda'] else "cpu")

        self.model = nn.Sequential(
            nn.Conv2d(3, 4, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(4, 8, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )

        self.target_model = copy.deepcopy(self.model)

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.buffer = Buffer(buffer_size)
        self.optimizer = Adam(self.model.parameters(), lr=config['lr'])

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.push(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        return self.buffer.sample(self.batch_size)

    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, action, next_state, reward, done = batch
        state = torch.tensor(state).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()
        reward = torch.tensor(reward).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        done = torch.tensor(done).to(self.device, int)

        with torch.no_grad():
            target_q = self.target_model(next_state).max(1)[0].view(-1)

        target_q = reward + target_q * self.config['gamma'] * (1 - done)
        q = self.model(state).gather(1, action.unsqueeze(1))
        loss = F.mse_loss(q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def act(self, state, eps, steps, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        if random.random() < eps / (steps + 1):
            return random.randint(0, self.action_dim - 1)
        model = self.target_model if target else self.model
        return model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

    def soft_update(self):
        tau = self.config['tau']
        with torch.no_grad():
            for sp, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.data.mul_(1 - tau)
                tp.data.add_(tau * sp.data)

    def update(self, transition):
        self.consume_transition(transition)
        batch = self.sample_batch()
        self.train_step(batch)
        self.soft_update()
        self.steps += 1

    def save(self, step, metric_mean, save_dir):
        torch.save(
            {
                'epoch': step + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metric': metric_mean,
            },
            os.path.join(save_dir, 'model.pth')
        )
