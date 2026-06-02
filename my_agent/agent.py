"""
agent.py — Core DQN agent for Bomberland (4-player).

Components:
  encode_obs()      — Convert raw obs dict to (map_tensor, aux_tensor)
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

import sys
from pathlib import Path

import numpy as np
import torch

# ── Resolve root so we can import engine and model ────────────────────────────
_HERE = Path(__file__).resolve().parent   # d:\Antigravity\AI_Boom_It\my_agent\
_ROOT = _HERE.parent                      # d:\Antigravity\AI_Boom_It\
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
_TERRAIN_VALUES = (0, 1, 2, 3, 4)
_MOVE_DELTAS = (
    (0, 0),   # STOP
    (0, -1),  # LEFT
    (0, 1),   # RIGHT
    (-1, 0),  # UP
    (1, 0),   # DOWN
)


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

    map_feat = np.zeros((MAP_CHANNELS, H, W), dtype=np.float32)
    for ch, v in enumerate(_TERRAIN_VALUES):
        map_feat[ch] = (grid == v)

    # ── My position (channel 5) ───────────────────────────────────────────
    my_x, my_y = int(players[agent_id][0]), int(players[agent_id][1])
    my_alive    = int(players[agent_id][2])
    my_pos = np.zeros((H, W), dtype=np.float32)
    if my_alive:
        my_pos[my_x, my_y] = 1.0
    map_feat[5] = my_pos

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

    map_feat[6] = all_enemy_pos

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

    map_feat[7] = bomb_timer_ch
    map_feat[8] = bomb_owned_ch
    map_feat[9] = enemy1_pos
    map_feat[10] = enemy2_pos

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


def valid_action_mask(obs: dict, agent_id: int) -> np.ndarray:
    """
    Return a boolean mask of valid actions for [STOP, LEFT, RIGHT, UP, DOWN, PLACE_BOMB].
    """
    grid = obs["map"]
    players = np.asarray(obs["players"])
    bombs = obs.get("bombs")
    h, w = grid.shape

    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[0] = True  # STOP is always valid

    x = int(players[agent_id][0])
    y = int(players[agent_id][1])
    bombs_left = int(players[agent_id][3])

    bomb_cells = set()
    if bombs is not None:
        arr = np.asarray(bombs, dtype=np.int64)
        if arr.ndim == 1 and arr.size > 0:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] > 0:
            for b in arr:
                bomb_cells.add((int(b[0]), int(b[1])))

    # Movement actions: cannot walk into wall/box or existing bomb.
    for action in (1, 2, 3, 4):
        dx, dy = _MOVE_DELTAS[action]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < h and 0 <= ny < w):
            continue
        cell = int(grid[nx, ny])
        if cell in (1, 2):  # wall/box
            continue
        # Exception: moving off your current bomb tile is allowed by engine rules.
        if (nx, ny) in bomb_cells and (nx, ny) != (x, y):
            continue
        mask[action] = True

    # PLACE_BOMB: bombs available and no bomb already at current tile.
    mask[5] = bombs_left > 0 and ((x, y) not in bomb_cells)
    return mask

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
        torch.set_num_threads(1)
        self.q_net    = None
        self.map_shape = (MAP_CHANNELS, 13, 13)
        self.aux_dim   = AUX_DIM

        # Load checkpoint from same directory as this file
        ckpt_path = _HERE / "model.pth"
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
        # keep init output quiet for faster and cleaner runtime guard logs

    def act(self, obs: dict) -> int:
        """
        Args:
            obs: dict with keys 'map', 'players', 'bombs'
        Returns:
            action int in [0, 5]
        """
        try:
            map_s, aux_s = encode_obs(obs, self.agent_id)
            action_mask = valid_action_mask(obs, self.agent_id)
            mt = torch.from_numpy(map_s).unsqueeze(0)
            at = torch.from_numpy(aux_s).unsqueeze(0)
            with torch.inference_mode():
                q = self.q_net(mt, at).squeeze(0).cpu().numpy()
                if not action_mask.any():
                    return 0
                q[~action_mask] = -1e9
                return int(np.argmax(q))
        except Exception as e:
            print(f"[Agent] act() error: {e}")
            return 0   # fallback: STOP
