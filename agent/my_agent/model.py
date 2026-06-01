"""
model.py — Neural network architecture for Bomberland AI agent.

Supports two modes via the `dueling` flag:
  - Standard DQN: head outputs Q-values directly
  - Dueling DQN:  head splits into Value V(s) and Advantage A(s,a),
                  combined as Q = V + (A - mean(A)) for better stability.

Two-branch design:
  Branch 1 (CNN): Processes the spatial 11-channel map grid (C, 13, 13)
  Branch 2 (MLP): Processes the auxiliary scalar feature vector
Both are concatenated and passed to the head.
"""

import torch
import torch.nn as nn


class BomberDQN(nn.Module):
    """
    Two-branch Deep Q-Network for Bomberland (4-player).

    Args:
        map_shape  (tuple): Shape of the spatial input, e.g. (11, 13, 13).
        aux_dim    (int):   Dimensionality of the scalar auxiliary vector.
        num_actions (int):  Number of discrete actions (6 in Bomberland).
        dueling    (bool):  If True, uses Dueling architecture.
    """

    def __init__(
        self,
        map_shape: tuple,
        aux_dim: int,
        num_actions: int = 6,
        dueling: bool = True,
    ):
        super().__init__()
        self.dueling = dueling
        self.num_actions = num_actions
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

        # ── Head ──────────────────────────────────────────────────────────
        if dueling:
            # Value stream: estimates V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            )
            # Advantage stream: estimates A(s, a) for each action
            self.advantage_stream = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_actions),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_actions),
            )

    def forward(self, map_x: torch.Tensor, aux_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_x: (B, C, H, W) spatial grid tensor
            aux_x: (B, aux_dim) scalar feature tensor
        Returns:
            Q-values: (B, num_actions)
        """
        map_feat = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        aux_feat = self.aux_encoder(aux_x)
        fusion = torch.cat([map_feat, aux_feat], dim=1)

        if self.dueling:
            value = self.value_stream(fusion)                   # (B, 1)
            advantage = self.advantage_stream(fusion)           # (B, A)
            # Combine: Q = V + A - mean(A) — removes arbitrary scale of A
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q = self.head(fusion)

        return q
