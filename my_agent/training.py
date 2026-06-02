"""
training.py — Training-only components for my_agent.

Keeps submission `agent.py` focused on inference/runtime safety.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import BomberDQN

NUM_ACTIONS = 6


class ReplayBuffer:
    """Replay buffer with optional prioritized sampling (PER)."""

    def __init__(
        self,
        capacity: int,
        map_shape: tuple,
        aux_dim: int,
        prioritized: bool = False,
        alpha: float = 0.6,
        priority_eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.priority_eps = float(priority_eps)

        self.map_s = np.zeros((capacity, *map_shape), dtype=np.float32)
        self.aux_s = np.zeros((capacity, aux_dim), dtype=np.float32)
        self.map_ns = np.zeros((capacity, *map_shape), dtype=np.float32)
        self.aux_ns = np.zeros((capacity, aux_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def push(self, map_s, aux_s, action, reward, map_ns, aux_ns, done):
        i = self.pos
        self.map_s[i] = map_s
        self.aux_s[i] = aux_s
        self.map_ns[i] = map_ns
        self.aux_ns[i] = aux_ns
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        max_p = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
        self.priorities[i] = max(max_p, 1.0)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        if not self.prioritized:
            idx = np.random.randint(0, self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            p = self.priorities[: self.size]
            p = np.power(p + self.priority_eps, self.alpha)
            p_sum = float(p.sum())
            if p_sum <= 0.0:
                probs = np.full(self.size, 1.0 / self.size, dtype=np.float32)
            else:
                probs = (p / p_sum).astype(np.float32)
            idx = np.random.choice(self.size, size=batch_size, p=probs)
            weights = np.power(self.size * probs[idx], -float(beta))
            weights /= max(float(weights.max()), 1e-8)
            weights = weights.astype(np.float32)

        batch = (
            self.map_s[idx],
            self.aux_s[idx],
            self.map_ns[idx],
            self.aux_ns[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )
        return batch, idx, weights

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.prioritized or idx is None:
            return
        self.priorities[idx] = np.abs(td_errors).astype(np.float32) + self.priority_eps


class TrainingAgent:
    """Double DQN training agent."""

    def __init__(
        self,
        agent_id: int,
        map_shape: tuple,
        aux_dim: int,
        num_actions: int = NUM_ACTIONS,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
        dueling: bool = True,
        pretrained_path: str | None = None,
    ):
        self.agent_id = agent_id
        self.map_shape = tuple(map_shape)
        self.aux_dim = int(aux_dim)
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device
        self.epsilon = 1.0
        self.global_step = 0

        ckpt = None
        if pretrained_path:
            from utils import load_checkpoint

            ckpt = load_checkpoint(pretrained_path, device)
            inp = ckpt["input_spec"]
            self.map_shape = tuple(inp[0])
            self.aux_dim = int(inp[1])
            self.num_actions = ckpt["num_actions"]
            self.global_step = ckpt.get("global_step", 0)
            self.epsilon = ckpt.get("epsilon", 0.05)
            lr = ckpt.get("lr", lr)

        self.q_net = BomberDQN(self.map_shape, self.aux_dim, self.num_actions, dueling=dueling).to(device)
        self.target_net = BomberDQN(self.map_shape, self.aux_dim, self.num_actions, dueling=dueling).to(device)

        if ckpt is not None:
            self.q_net.load_state_dict(ckpt["model_state_dict"])

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, eps=1e-8, weight_decay=1e-5)
        if ckpt is not None and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def act(
        self,
        map_state: np.ndarray,
        aux_state: np.ndarray,
        epsilon: float = 0.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        valid = np.flatnonzero(action_mask) if action_mask is not None else np.arange(self.num_actions)
        if valid.size == 0:
            return 0

        if random.random() < epsilon:
            return int(np.random.choice(valid))

        mt = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        at = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(mt, at).squeeze(0).detach().cpu().numpy()
            if action_mask is not None:
                q = q.copy()
                q[~action_mask] = -1e9
            return int(np.argmax(q))

    def train_step(self, batch, importance_weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        ms, as_, nms, nas, act, rew, don = batch
        dev = self.device

        ms_t = torch.from_numpy(ms).to(dev)
        as_t = torch.from_numpy(as_).to(dev)
        nms_t = torch.from_numpy(nms).to(dev)
        nas_t = torch.from_numpy(nas).to(dev)
        act_t = torch.from_numpy(act).unsqueeze(1).to(dev)
        rew_t = torch.from_numpy(rew).unsqueeze(1).to(dev)
        don_t = torch.from_numpy(don).unsqueeze(1).to(dev)

        q_vals = self.q_net(ms_t, as_t).gather(1, act_t)

        with torch.no_grad():
            best_next_a = self.q_net(nms_t, nas_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(nms_t, nas_t).gather(1, best_next_a)
            target_q = rew_t + self.gamma * max_next_q * (1.0 - don_t)

        td_error = target_q - q_vals
        per_sample_loss = self.loss_fn(q_vals, target_q)
        if importance_weights is not None:
            w_t = torch.from_numpy(importance_weights).unsqueeze(1).to(dev)
            per_sample_loss = per_sample_loss * w_t
        loss = per_sample_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.item()), np.abs(td_error.detach().cpu().numpy().reshape(-1))

    def sync_target(self, tau: float | None = None):
        """
        Sync target network.
        - tau is None or >=1.0: hard update
        - 0 < tau < 1: soft Polyak update
        """
        if tau is None or tau >= 1.0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            return

        tau = float(tau)
        with torch.no_grad():
            for t_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                t_param.data.mul_(1.0 - tau).add_(tau * q_param.data)
