import random
import copy
from collections import namedtuple, deque

import torch
from torch import nn, optim
import numpy as np
import torch.nn.utils.rnn as rnn_utils

from utils import get_masks

UPDATE_EVERY = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Refer to the code in https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition.
EPS_START = 5.0   # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 500  # episode to end the noise decay process
EPS_FINAL = 0     # final value for epsilon after decay

class Actor(nn.Module):
    def __init__(self, state_size=24, action_size=2, embed_dim=128, seed=2020):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_layers = 1
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(nn.Linear(state_size, self.embed_dim),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Linear(self.embed_dim, action_size))
        self.tanh = nn.Tanh()
        self.hidden_states = None

    def forward(self, states):
        """Build a network that maps state -> action values."""
        return self.tanh(self.mlp(states).squeeze())


class Critic(nn.Module):
    def __init__(self, state_size=24, action_size=2, embed_dim=128, seed=2020):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.embed_dim = embed_dim

        self.fc1_agent0 = nn.Linear(state_size, self.embed_dim)
        self.fc1_agent1 = nn.Linear(state_size, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim + action_size, self.embed_dim)
        self.fc3 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc4 = nn.Linear(self.embed_dim, 1)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, states, actions, agent=0):
        """Build a network that maps observations -> values."""
        if agent == 0:
            x_cat = torch.cat([self.relu(self.fc1_agent0(states)), actions], dim=2)
        if agent == 1:
            x_cat = torch.cat([self.relu(self.fc1_agent1(states)), actions], dim=2)
        x = self.relu(self.fc2(x_cat))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class Agent(nn.Module):
    def __init__(self, state_size=24, action_size=2, embed_dim=64, buffer_size=256, batch_size=32, lr=0.001, gamma=0.99,
                 pre_trained=False, seed=2020, debug=True):
        super(Agent, self).__init__()
        self.batch_size = batch_size
        self.gamma = gamma
        self.t_step = 0
        self.pre_trained = pre_trained
        self.debug = debug

        self.actor_local_0 = Actor(state_size, action_size, embed_dim, seed).to(
            device)
        self.actor_target_0 = Actor(state_size, action_size, embed_dim, seed).to(
            device)
        self.actor_optimizer_0 = optim.Adam(self.actor_local_0.parameters(), lr=lr)

        self.actor_local_1 = Actor(state_size, action_size, embed_dim, seed).to(
            device)
        self.actor_target_1 = Actor(state_size, action_size, embed_dim, seed).to(
            device)
        self.actor_optimizer_1 = optim.Adam(self.actor_local_1.parameters(), lr=lr)

        self.critic_local = Critic(2 * state_size, 2 * action_size, embed_dim, seed).to(
            device)
        self.critic_target = Critic(2 * state_size, 2 * action_size, embed_dim, seed).to(
            device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr)

        self.memory_0 = ReplayBuffer(action_size, buffer_size, seed)
        self.memory_1 = ReplayBuffer(action_size, buffer_size, seed)

        self.noise = OUNoise(action_size, mu=0, theta=0.15, sigma=0.2)
        self.eps = EPS_START
        self.eps_decay = 1 / EPS_EP_END

    def act(self, states, isnoise=True):
        states = torch.from_numpy(states).float().to(device)
        actions = []
        with torch.no_grad():
            actions.append(self.actor_local_0(states[0].unsqueeze(0).unsqueeze(1)).cpu().data.numpy())
            actions.append(self.actor_local_1(states[1].unsqueeze(0).unsqueeze(1)).cpu().data.numpy())
        actions = np.stack(actions, axis=0)
        if isnoise:
            actions += self.eps * self.noise.sample()
        return np.clip(actions, -1, 1)

    def step(self, batch_size=64, learn_num=1):
        self.t_step += 1

        # update noise decay parameter
        self.eps -= self.eps_decay
        self.eps = max(self.eps, EPS_FINAL)

        if self.t_step % UPDATE_EVERY == 0:
            for _ in range(learn_num):
                self.learn(batch_size, agent=0)
                self.learn(batch_size, agent=1)

    def learn(self, batch_size=64, agent=0):
        # DDPG implementation
        if agent == 0:
            experiences = self.memory_0.sample(batch_size)
        if agent == 1:
            experiences = self.memory_1.sample(batch_size)
        states, actions, rewards, next_states, dones, batch_lengths = experiences
        # Because different batch data is variable length, mask is used for cover the unused part.
        masks = get_masks(batch_lengths)

        # update critic
        with torch.no_grad():
            if agent == 0:
                next_actions = torch.cat([self.actor_target_0(next_states[:, :, :24]), actions[:, :, 2:]], dim=2)
            if agent == 1:
                next_actions = torch.cat([actions[:, :, :2], self.actor_target_1(next_states[:, :, 24:])], dim=2)
            q_value_prime = self.critic_target(next_states, next_actions, agent).squeeze(2) * masks
        q_value = self.critic_local(states.requires_grad_(True), actions, agent).squeeze(2) * masks
        td_error = (rewards + self.gamma * q_value_prime * (1 - dones) - q_value) * masks  # one-step estimate

        critic_loss = ((td_error ** 2).sum(1) / batch_lengths).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target, tau=0.01)

        # update actor
        if agent == 0:
            actions_pred = torch.cat([self.actor_local_0(states[:, :, :24]), actions[:, :, 2:]], dim=2)
        if agent == 1:
            actions_pred = torch.cat([actions[:, :, :2], self.actor_local_1(states[:, :, 24:])], dim=2)

        q_value = self.critic_local(states, actions_pred, agent)
        actor_loss = - ((q_value.squeeze() * masks).sum(1) / batch_lengths).mean()

        if agent == 0:
            self.actor_optimizer_0.zero_grad()
            (actor_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor_local_0.parameters(), 1)
            self.actor_optimizer_0.step()
            self.soft_update(self.actor_local_0, self.actor_target_0, tau=0.01)
        if agent == 1:
            self.actor_optimizer_1.zero_grad()
            (actor_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor_local_1.parameters(), 1)
            self.actor_optimizer_1.step()
            self.soft_update(self.actor_local_1, self.actor_target_1, tau=0.01)

        self.noise.reset()

        if self.t_step % 128 == 0:
            print(f'q_value: {q_value.detach().mean().item():.5f}, '
                  f'critic_loss: {critic_loss.detach().mean().item():.5f}, '
                  f'actor_loss: {actor_loss.detach().mean().item():.5f} ')

    def soft_update(self, local_model, target_model, tau=0.001):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed=2020):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience",
                                     field_names=["observations", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.length = 0

    def add(self, observations, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(observations, actions, rewards, next_states, dones)
        self.memory.append(e)
        if self.length + 1 < self.buffer_size:
            self.length += 1

    def sample(self, batch_size=32, seq_len=512):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)
        valid_seq_len = min(max(list(map(lambda e: e.dones.shape[0], experiences))), seq_len)

        observations, actions, rewards, next_states, dones = [], [], [], [], []
        lengths = []
        for e in experiences:
            start_index = random.randint(0, max(len(e.dones) - valid_seq_len, 0))
            end_index = start_index + valid_seq_len
            e_seq_len = e.dones[start_index: end_index].shape[0]
            lengths.append(e_seq_len)
            observations.append(torch.FloatTensor(e.observations[start_index: end_index]))
            actions.append(torch.FloatTensor(e.actions[start_index: end_index]))
            rewards.append(torch.FloatTensor(e.rewards[start_index: end_index]))
            next_states.append(torch.FloatTensor(e.next_states[start_index: end_index]))
            dones.append(torch.LongTensor(e.dones[start_index: end_index]))

        batch_lengths, idx_sort = torch.sort(torch.LongTensor(lengths), dim=0, descending=True)
        observations = rnn_utils.pad_sequence(observations, batch_first=True).index_select(0, idx_sort)
        actions = rnn_utils.pad_sequence(actions, batch_first=True).index_select(0, idx_sort)
        rewards = rnn_utils.pad_sequence(rewards, batch_first=True).index_select(0, idx_sort)
        next_states = rnn_utils.pad_sequence(next_states, batch_first=True).index_select(0, idx_sort)
        dones = rnn_utils.pad_sequence(dones, batch_first=True).index_select(0, idx_sort)

        return (observations, actions, rewards, next_states, dones, batch_lengths)

    def __len__(self):
        return self.length


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, seed=2020):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
