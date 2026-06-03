"""
training.py — Training-only components for my_agent_2.

Upgraded to Rainbow DQN. Includes C51 Distributional RL categorical projection
and Noisy Nets (exploration without epsilon).
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.masks_ns = np.zeros((capacity, NUM_ACTIONS), dtype=bool)
        self.priorities = np.ones(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def push(self, map_s, aux_s, action, reward, map_ns, aux_ns, done, mask_ns):
        i = self.pos
        self.map_s[i] = map_s
        self.aux_s[i] = aux_s
        self.map_ns[i] = map_ns
        self.aux_ns[i] = aux_ns
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.masks_ns[i] = mask_ns
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
            self.masks_ns[idx],
        )
        return batch, idx, weights

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.prioritized or idx is None:
            return
        self.priorities[idx] = np.abs(td_errors).astype(np.float32) + self.priority_eps


class TrainingAgent:
    """Rainbow DQN training agent."""

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
        v_min: float = -20.0,
        v_max: float = 20.0,
        n_atoms: int = 51,
        n_step: int = 3,
        pretrained_path: str | None = None,
    ):
        self.agent_id = agent_id
        self.map_shape = tuple(map_shape)
        self.aux_dim = int(aux_dim)
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step
        self.device = device
        
        # Distributional RL parameters
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)

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
            lr = ckpt.get("lr", lr)
            self.v_min = ckpt.get("v_min", self.v_min)
            self.v_max = ckpt.get("v_max", self.v_max)
            self.n_atoms = ckpt.get("n_atoms", self.n_atoms)
            self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
            self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(device)

        self.q_net = BomberDQN(self.map_shape, self.aux_dim, self.num_actions, dueling=dueling, n_atoms=self.n_atoms).to(device)
        self.target_net = BomberDQN(self.map_shape, self.aux_dim, self.num_actions, dueling=dueling, n_atoms=self.n_atoms).to(device)

        if ckpt is not None:
            self.q_net.load_state_dict(ckpt["model_state_dict"])

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, eps=1.5e-4, weight_decay=1e-5) # Adam epsilon adjusted for Rainbow
        if ckpt is not None and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def act(
        self,
        map_state: np.ndarray,
        aux_state: np.ndarray,
        action_mask: np.ndarray | None = None,
        epsilon: float = 0.0,
    ) -> tuple[int, bool]:
        valid = np.flatnonzero(action_mask) if action_mask is not None else np.arange(self.num_actions)
        if valid.size == 0:
            return 0, False

        if epsilon > 0.0 and np.random.rand() < epsilon:
            return int(np.random.choice(valid)), True

        mt = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        at = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
        # Keep q_net in train() mode during experience collection so NoisyNet injects exploration noise.
        with torch.no_grad():
            logits = self.q_net(mt, at).squeeze(0) # (A, N)
            probs = F.softmax(logits, dim=1)
            # Q-value is the expected value of the distribution
            q = (probs * self.support).sum(dim=1).cpu().numpy() # (A,)

            if action_mask is not None:
                q = q.copy()
                q[~action_mask] = -1e9
            action = int(np.argmax(q))
        return action, False

    def train_step(self, batch, importance_weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        # Ensure online net is in train mode for NoisyNet gradient flow.
        self.q_net.train()
        ms, as_, nms, nas, act, rew, don, mask_ns = batch
        dev = self.device

        ms_t = torch.from_numpy(ms).to(dev)
        as_t = torch.from_numpy(as_).to(dev)
        nms_t = torch.from_numpy(nms).to(dev)
        nas_t = torch.from_numpy(nas).to(dev)
        act_t = torch.from_numpy(act).to(dev)
        rew_t = torch.from_numpy(rew).to(dev)
        don_t = torch.from_numpy(don).to(dev)
        mask_ns_t = torch.from_numpy(mask_ns).to(dev)

        batch_size = ms_t.size(0)

        # 1. Online network: compute log probabilities of the selected actions
        logits = self.q_net(ms_t, as_t)  # (B, A, N)
        log_probs = F.log_softmax(logits, dim=2)
        action_log_probs = log_probs[range(batch_size), act_t]  # (B, N)

        # 2. Target network: compute projected target distribution
        with torch.no_grad():
            target_logits = self.target_net(nms_t, nas_t)  # (B, A, N)
            target_probs = F.softmax(target_logits, dim=2)
            
            # Double DQN action selection: use online network
            next_logits = self.q_net(nms_t, nas_t)
            next_probs = F.softmax(next_logits, dim=2)
            next_q = (next_probs * self.support).sum(dim=2)  # (B, A)
            next_q[~mask_ns_t] = -1e9
            best_next_a = next_q.argmax(dim=1)  # (B,)

            # Get target distribution for the best next action
            next_target_probs = target_probs[range(batch_size), best_next_a]  # (B, N)

            # Compute projected distribution
            Tz = rew_t.unsqueeze(1) + (1.0 - don_t.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)  # (B, N)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Fix edge cases where l == u
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) * (l == u)] += 1

            m = torch.zeros(batch_size, self.n_atoms, device=dev)
            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size, dtype=torch.long, device=dev).unsqueeze(1)
            
            # Project probabilities
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_target_probs * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_target_probs * (b - l.float())).view(-1))

        # 3. Cross Entropy Loss
        # td_error here is the KL divergence loss per sample
        kl_div = -(m * action_log_probs).sum(dim=1)  # (B,)
        
        per_sample_loss = kl_div
        if importance_weights is not None:
            w_t = torch.from_numpy(importance_weights).to(dev)
            per_sample_loss = per_sample_loss * w_t
            
        loss = per_sample_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Reset noise for the NEXT forward pass (exploration diversity).
        # q_net is still in train() mode; target_net stays in eval().
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        
        self.global_step += 1
        
        # Return loss and errors for PER update
        return float(loss.item()), kl_div.detach().cpu().numpy()

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
