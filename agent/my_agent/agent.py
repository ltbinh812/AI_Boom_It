"""
agent.py — Core DQN agent for Bomberland (4-player).

Components:
  encode_obs()      — Convert raw obs dict to (map_tensor, aux_tensor)
  ReplayBuffer      — Fast pre-allocated numpy circular buffer
  TrainingAgent     — Double DQN training logic
  Agent             — Submit-ready inference class (loaded at init from .pth)

Design choices:
  • 11-channel spatial map: 5 terrain channels + 4 player pos channels +
    bomb-timer channel + bomb-owner channel  →  (11, 13, 13)
  • Auxiliary vector (5 scalars): bombs_left, radius_bonus, n_enemies_alive,
    bomb_timer_nearest (self), distance_nearest_enemy
  • Double DQN: online net selects action; target net evaluates Q-value.
  • Dueling architecture in model.py.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Resolve root so we can import engine and model ────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in [str(_HERE), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import BomberDQN  # noqa: E402  (local import after path fix)

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_ACTIONS    = 6    # 0:STOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:PLACE_BOMB
NUM_PLAYERS    = 4
BOMB_MAX_TIMER = 7
MAX_BOMB_RADIUS = 5

# Map channels breakdown (11 channels total):
#   0: GRASS  1: WALL  2: BOX  3: ITEM_RADIUS  4: ITEM_CAPACITY
#   5: my position
#   6: combined enemy positions (all alive opponents)
#   7: bomb timer   (normalised 0-1, max value = most urgent)
#   8: bomb owned   (1 if I own it, 0 otherwise)
#   9: enemy-1 position  (distinguishes two closest opponents)
#  10: enemy-2 position
MAP_CHANNELS = 11

# Auxiliary vector (5 scalars):
#   [0] bombs_left / MAX_BOMB_CAPACITY
#   [1] radius_bonus / MAX_BOMB_RADIUS
#   [2] n_alive_enemies / (NUM_PLAYERS - 1)
#   [3] nearest_own_bomb_timer / BOMB_MAX_TIMER  (0 if none)
#   [4] nearest_enemy_manhattan / max_dist       (0 if none)
AUX_DIM = 5


# ── Observation Encoding ──────────────────────────────────────────────────────

def encode_obs(obs: dict, agent_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a raw observation into tensors for the neural network.

    Returns:
        map_feat : np.ndarray  shape (MAP_CHANNELS, H, W)  dtype float32
        aux_feat : np.ndarray  shape (AUX_DIM,)            dtype float32
    """
    grid    = obs["map"]            # (H, W)  int8
    players = np.asarray(obs["players"])  # (N, 5)  [x, y, alive, bombs_left, radius_bonus]
    bombs   = obs.get("bombs")

    H, W = grid.shape

    # ── Terrain channels (0-4) ────────────────────────────────────────────
    terrain_vals = [0, 1, 2, 3, 4]
    channels = [(grid == v).astype(np.float32) for v in terrain_vals]

    # ── My position (channel 5) ───────────────────────────────────────────
    my_x, my_y = int(players[agent_id][0]), int(players[agent_id][1])
    my_alive    = int(players[agent_id][2])
    my_pos = np.zeros((H, W), dtype=np.float32)
    if my_alive:
        my_pos[my_x, my_y] = 1.0
    channels.append(my_pos)

    # ── Enemy positions — split into "combined" and two individual ────────
    all_enemy_pos   = np.zeros((H, W), dtype=np.float32)
    enemy1_pos      = np.zeros((H, W), dtype=np.float32)
    enemy2_pos      = np.zeros((H, W), dtype=np.float32)

    alive_enemies = [
        i for i in range(len(players))
        if i != agent_id and int(players[i][2]) == 1
    ]
    # Sort by Manhattan distance so enemy1/enemy2 are consistent closest foes
    if my_alive:
        alive_enemies.sort(
            key=lambda e: abs(my_x - int(players[e][0])) + abs(my_y - int(players[e][1]))
        )
    for rank, eid in enumerate(alive_enemies):
        ex, ey = int(players[eid][0]), int(players[eid][1])
        all_enemy_pos[ex, ey] = 1.0
        if rank == 0:
            enemy1_pos[ex, ey] = 1.0
        elif rank == 1:
            enemy2_pos[ex, ey] = 1.0

    channels.append(all_enemy_pos)     # ch 6

    # ── Bomb channels (7-8) ───────────────────────────────────────────────
    bomb_timer_ch = np.zeros((H, W), dtype=np.float32)
    bomb_owned_ch = np.zeros((H, W), dtype=np.float32)
    nearest_own_timer = BOMB_MAX_TIMER + 1  # sentinel "no bomb"

    if bombs is not None:
        arr = np.asarray(bombs, dtype=np.float64)
        if arr.ndim == 1 and arr.size > 0:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] > 0:
            for b in arr:
                bx, by = int(b[0]), int(b[1])
                timer   = float(b[2])
                owner   = int(b[3])
                norm_t  = timer / BOMB_MAX_TIMER
                # Keep the most urgent (lowest norm_t) for each cell
                if bomb_timer_ch[bx, by] == 0.0 or norm_t < bomb_timer_ch[bx, by]:
                    bomb_timer_ch[bx, by] = norm_t
                if owner == agent_id:
                    bomb_owned_ch[bx, by] = 1.0
                    if timer < nearest_own_timer:
                        nearest_own_timer = timer

    channels.append(bomb_timer_ch)     # ch 7
    channels.append(bomb_owned_ch)     # ch 8
    channels.append(enemy1_pos)        # ch 9
    channels.append(enemy2_pos)        # ch 10

    map_feat = np.stack(channels, axis=0).astype(np.float32)   # (11, H, W)

    # ── Auxiliary scalars ─────────────────────────────────────────────────
    max_capacity = 5.0
    bombs_left   = float(players[agent_id][3]) / max_capacity
    radius_bonus = float(players[agent_id][4]) / float(MAX_BOMB_RADIUS)
    n_alive_foes = len(alive_enemies) / float(NUM_PLAYERS - 1)

    own_timer_norm = (
        0.0 if nearest_own_timer > BOMB_MAX_TIMER
        else float(nearest_own_timer) / BOMB_MAX_TIMER
    )

    max_dist = float(H + W - 2)
    if alive_enemies and my_alive:
        nearest_eid = alive_enemies[0]
        nd = abs(my_x - int(players[nearest_eid][0])) + abs(my_y - int(players[nearest_eid][1]))
        nearest_dist_norm = nd / max_dist
    else:
        nearest_dist_norm = 0.0

    aux_feat = np.array(
        [bombs_left, radius_bonus, n_alive_foes, own_timer_norm, nearest_dist_norm],
        dtype=np.float32,
    )
    return map_feat, aux_feat


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Pre-allocated circular numpy buffer.
    sample() is pure array indexing — no Python object overhead.
    """
    def __init__(self, capacity: int, map_shape: tuple, aux_dim: int):
        self.capacity   = capacity
        self.pos        = 0
        self.size       = 0
        self.map_shape  = tuple(map_shape)

        self.map_s  = np.zeros((capacity, *map_shape), dtype=np.float32)
        self.aux_s  = np.zeros((capacity, aux_dim),    dtype=np.float32)
        self.map_ns = np.zeros((capacity, *map_shape), dtype=np.float32)
        self.aux_ns = np.zeros((capacity, aux_dim),    dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def push(self, map_s, aux_s, action, reward, map_ns, aux_ns, done):
        i = self.pos
        self.map_s[i]  = map_s
        self.aux_s[i]  = aux_s
        self.map_ns[i] = map_ns
        self.aux_ns[i] = aux_ns
        self.actions[i] = action
        self.rewards[i]  = reward
        self.dones[i]    = float(done)
        self.pos  = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.map_s[idx], self.aux_s[idx],
            self.map_ns[idx], self.aux_ns[idx],
            self.actions[idx], self.rewards[idx], self.dones[idx],
        )


# ── Training Agent ────────────────────────────────────────────────────────────

class TrainingAgent:
    """
    Double DQN (Dueling architecture optional) training agent.

    Double DQN update:
        a* = argmax_a  Q_online(s', a)        # online net selects action
        target = r + γ * Q_target(s', a*)     # target net evaluates value
    """

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
        self.agent_id    = agent_id
        self.map_shape   = tuple(map_shape)
        self.aux_dim     = int(aux_dim)
        self.num_actions = num_actions
        self.gamma       = gamma
        self.device      = device
        self.epsilon     = 1.0
        self.global_step = 0

        if pretrained_path:
            from utils import load_checkpoint
            ckpt = load_checkpoint(pretrained_path, device)
            inp  = ckpt["input_spec"]
            self.map_shape   = tuple(inp[0])
            self.aux_dim     = int(inp[1])
            self.num_actions = ckpt["num_actions"]
            self.global_step = ckpt.get("global_step", 0)
            self.epsilon     = ckpt.get("epsilon", 0.05)
            lr               = ckpt.get("lr", lr)

        self.q_net = BomberDQN(self.map_shape, self.aux_dim, self.num_actions,
                               dueling=dueling).to(device)
        self.target_net = BomberDQN(self.map_shape, self.aux_dim, self.num_actions,
                                    dueling=dueling).to(device)

        if pretrained_path:
            self.q_net.load_state_dict(ckpt["model_state_dict"])

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr,
                                    eps=1e-8, weight_decay=1e-5)
        if pretrained_path and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.loss_fn = nn.SmoothL1Loss()   # Huber loss — more robust than MSE

    # ── Action selection ──────────────────────────────────────────────────

    def act(self, map_state: np.ndarray, aux_state: np.ndarray,
            epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        mt = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        at = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(mt, at).argmax(dim=1).item())

    # ── Double DQN update ─────────────────────────────────────────────────

    def train_step(self, batch) -> float:
        """
        Single gradient step.  batch = output of ReplayBuffer.sample().
        Returns the scalar loss value.
        """
        ms, as_, nms, nas, act, rew, don = batch
        dev = self.device

        ms_t  = torch.from_numpy(ms).to(dev)
        as_t  = torch.from_numpy(as_).to(dev)
        nms_t = torch.from_numpy(nms).to(dev)
        nas_t = torch.from_numpy(nas).to(dev)
        act_t = torch.from_numpy(act).unsqueeze(1).to(dev)
        rew_t = torch.from_numpy(rew).unsqueeze(1).to(dev)
        don_t = torch.from_numpy(don).unsqueeze(1).to(dev)

        # Current Q(s, a)
        q_vals = self.q_net(ms_t, as_t).gather(1, act_t)

        # Double DQN target
        with torch.no_grad():
            # 1. Online net picks best next action
            best_next_a = self.q_net(nms_t, nas_t).argmax(dim=1, keepdim=True)
            # 2. Target net evaluates that action
            max_next_q  = self.target_net(nms_t, nas_t).gather(1, best_next_a)
            target_q    = rew_t + self.gamma * max_next_q * (1.0 - don_t)

        loss = self.loss_fn(q_vals, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.item())

    def sync_target(self):
        """Hard-copy online weights into target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ── Submit-ready Agent class ──────────────────────────────────────────────────

class Agent:
    """
    Submission-ready agent class.
    Loads trained weights at __init__ and exposes act(obs) → int.
    CPU-only; no network access; no file writing.
    """

    team_id = "AI_BoomIt"

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device   = torch.device("cpu")
        self.q_net    = None
        self.map_shape = (MAP_CHANNELS, 13, 13)
        self.aux_dim   = AUX_DIM

        # Load checkpoint from same directory as this file
        ckpt_path = _HERE / "best_model.pth"
        if ckpt_path.exists():
            self._load(str(ckpt_path))
        else:
            # Fallback: search for any .pth in the same directory
            pths = sorted(_HERE.glob("*.pth"))
            if pths:
                self._load(str(pths[-1]))   # pick latest alphabetically
            else:
                # No weights found — build untrained network (random play)
                print("[Agent] WARNING: no .pth checkpoint found, using random network.")
                self.q_net = BomberDQN(self.map_shape, self.aux_dim, NUM_ACTIONS,
                                       dueling=True).to(self.device)
        self.q_net.eval()

    def _load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        inp  = ckpt.get("input_spec", ckpt.get("input_shape", ckpt.get("input_dim")))
        self.map_shape = tuple(inp[0])
        self.aux_dim   = int(inp[1])
        self.q_net = BomberDQN(self.map_shape, self.aux_dim, ckpt["num_actions"],
                               dueling=True).to(self.device)
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        print(f"[Agent] Loaded checkpoint from {path}")

    def act(self, obs: dict) -> int:
        """
        Args:
            obs: dict with keys 'map', 'players', 'bombs'
        Returns:
            action int in [0, 5]
        """
        try:
            map_s, aux_s = encode_obs(obs, self.agent_id)
            mt = torch.from_numpy(map_s).unsqueeze(0)
            at = torch.from_numpy(aux_s).unsqueeze(0)
            with torch.no_grad():
                return int(self.q_net(mt, at).argmax(dim=1).item())
        except Exception as e:
            print(f"[Agent] act() error: {e}")
            return 0   # fallback: STOP
