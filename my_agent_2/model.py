"""
model.py — Neural network architecture for Bomberland AI agent.

Upgraded to Rainbow DQN (Distributional RL + Noisy Nets).
Supports two modes via the `dueling` flag:
  - Standard DQN: head outputs logits for distributions
  - Dueling DQN:  head splits into Value V(s) and Advantage A(s,a),
                  combined as Q = V + (A - mean(A)) for better stability.

Two-branch design:
  Branch 1 (CNN): Processes the spatial 11-channel map grid (C, 13, 13)
  Branch 2 (MLP): Processes the auxiliary scalar feature vector
Both are concatenated and passed to the NoisyLinear head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet (Rainbow DQN)."""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class BomberDQN(nn.Module):
    """
    Two-branch Deep Q-Network for Bomberland (4-player).
    Upgraded with NoisyLinear and Distributional Output (n_atoms).

    Args:
        map_shape  (tuple): Shape of the spatial input, e.g. (11, 13, 13).
        aux_dim    (int):   Dimensionality of the scalar auxiliary vector.
        num_actions (int):  Number of discrete actions (6 in Bomberland).
        dueling    (bool):  If True, uses Dueling architecture.
        n_atoms    (int):   Number of atoms for Distributional RL (C51).
    """

    def __init__(
        self,
        map_shape: tuple,
        aux_dim: int,
        num_actions: int = 6,
        dueling: bool = True,
        n_atoms: int = 51,
    ):
        super().__init__()
        self.dueling = dueling
        self.num_actions = num_actions
        self.n_atoms = n_atoms
        c, h, w = map_shape

        # ── Branch 1: CNN for spatial map ─────────────────────────────────
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Compute flattened CNN output size dynamically
        with torch.no_grad():
            _dummy = torch.zeros(1, c, h, w)
            _conv_out = self.map_encoder(_dummy).reshape(1, -1)
            conv_out_dim = _conv_out.size(1)

        # ── Branch 2: MLP for scalar auxiliary features ───────────────────
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        fusion_dim = conv_out_dim + 64

        # ── Head (Using Noisy Networks for Exploration) ────────────────────
        if dueling:
            # Value stream: estimates V(s) distribution
            self.value_stream = nn.Sequential(
                NoisyLinear(fusion_dim, 256),
                nn.ReLU(inplace=True),
                NoisyLinear(256, n_atoms),
            )
            # Advantage stream: estimates A(s, a) distribution for each action
            self.advantage_stream = nn.Sequential(
                NoisyLinear(fusion_dim, 256),
                nn.ReLU(inplace=True),
                NoisyLinear(256, num_actions * n_atoms),
            )
        else:
            self.head = nn.Sequential(
                NoisyLinear(fusion_dim, 256),
                nn.ReLU(inplace=True),
                NoisyLinear(256, 128),
                nn.ReLU(inplace=True),
                NoisyLinear(128, num_actions * n_atoms),
            )

    def forward(self, map_x: torch.Tensor, aux_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_x: (B, C, H, W) spatial grid tensor
            aux_x: (B, aux_dim) scalar feature tensor
        Returns:
            Logits: (B, num_actions, n_atoms)
        """
        map_feat = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        aux_feat = self.aux_encoder(aux_x)
        fusion = torch.cat([map_feat, aux_feat], dim=1)

        if self.dueling:
            value = self.value_stream(fusion).view(-1, 1, self.n_atoms)                   # (B, 1, N)
            advantage = self.advantage_stream(fusion).view(-1, self.num_actions, self.n_atoms)  # (B, A, N)
            # Combine: Q = V + A - mean(A)
            q_logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_logits = self.head(fusion).view(-1, self.num_actions, self.n_atoms)

        return q_logits

    def reset_noise(self):
        """Reset noise for all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
