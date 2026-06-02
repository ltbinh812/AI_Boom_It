"""
reward.py — Reward shaping for 4-player Bomberland training.

Extended from the BTC baseline reward to correctly handle 4 agents:
  - Win/loss conditions are relative to all 3 opponents.
  - Enemy death reward scales with number of kills in a single step.
  - Approach reward tracks nearest living opponent (any of 3).
  - Danger evasion uses blast prediction on full bomb list.
"""

from __future__ import annotations
import numpy as np

# ── Map cell constants (mirror engine/map.py) ────────────────────────────────
GRASS    = 0
WALL     = 1
BOX      = 2
ITEM_RADIUS   = 3
ITEM_CAPACITY = 4

BOMB_MAX_TIMER = 7
MAX_BOMB_RADIUS = 5

# ── Reward table (tweak to shape behaviour) ──────────────────────────────────
REWARDS = {
    "win":              3.0,   # Last agent standing
    "enemy_death":      1.0,   # Per kill
    "agent_death":     -5.0,   # Self-elimination (increased penalty for safety)
    "time_penalty":    -0.005, # Every step to encourage speed
    "standing_still":  -0.01,  # Penalise camping
    "plant_near_box":   0.15,  # Place bomb adjacent to a box (increased reward)
    "item_collection":  0.25,  # Pick up an item (increased reward)
    "danger_evasion":   0.20,  # Step out of blast zone (increased reward)
    "danger_enter":    -0.06,  # Step into blast zone (when moving)
    "own_blast_loiter": -0.04, # Remain in own blast zone per tick urgency
    "approach_enemy":   0.02,  # Per unit of Manhattan distance closed
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_bombs(obs: dict) -> np.ndarray:
    """Return bombs as (N, 4) array [x, y, timer, owner]. Empty → (0, 4)."""
    b = obs.get("bombs")
    if b is None or (hasattr(b, "__len__") and len(b) == 0):
        return np.empty((0, 4), dtype=np.float64)
    arr = np.asarray(b, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _explosion_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> set:
    """Cross-shaped blast, blocked by walls, stopped (inclusive) by boxes."""
    h, w = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < h and 0 <= ty < w):
                break
            cell = int(grid[tx, ty])
            if cell == WALL:
                break
            tiles.add((tx, ty))
            if cell == BOX:
                break
    return tiles


def _blast_at(obs: dict, x: int, y: int, own_only: bool = False, agent_id: int = -1):
    """
    Returns (in_blast: bool, min_timer: int | None).
    If own_only=True, only considers bombs placed by agent_id.
    """
    bombs = _parse_bombs(obs)
    if bombs.shape[0] == 0:
        return False, None

    players = obs["players"]
    grid    = obs["map"]
    in_blast, min_timer = False, None

    for b in bombs:
        bx, by, timer, owner_id = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        if own_only and owner_id != agent_id:
            continue
        # Radius = 1 + radius_bonus of the owner at placement time
        radius = 1 + int(players[owner_id][4])
        radius = min(radius, MAX_BOMB_RADIUS)
        if (x, y) in _explosion_tiles(grid, bx, by, radius):
            in_blast = True
            min_timer = timer if min_timer is None else min(min_timer, timer)

    return in_blast, min_timer


def _alive_enemies(players_arr: np.ndarray, agent_id: int) -> list[int]:
    """Return list of player IDs that are alive and not agent_id."""
    return [i for i in range(len(players_arr)) if i != agent_id and int(players_arr[i][2]) == 1]


def _nearest_enemy_dist(players_arr: np.ndarray, agent_id: int, x: int, y: int) -> int | None:
    alive = _alive_enemies(players_arr, agent_id)
    if not alive:
        return None
    return min(abs(x - int(players_arr[e][0])) + abs(y - int(players_arr[e][1])) for e in alive)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_reward(prev_obs: dict | None, curr_obs: dict, agent_id: int) -> float:
    """
    Compute the shaped reward for agent_id between two consecutive observations.

    Args:
        prev_obs:  Observation before the step (None on the very first call).
        curr_obs:  Observation after the step.
        agent_id:  Integer ID of the agent being trained.

    Returns:
        Scalar float reward.
    """
    if prev_obs is None:
        return 0.0

    prev_p = np.asarray(prev_obs["players"])
    curr_p = np.asarray(curr_obs["players"])

    prev_alive = int(prev_p[agent_id][2])
    curr_alive  = int(curr_p[agent_id][2])

    # ── 1. Death / win ────────────────────────────────────────────────────
    if prev_alive == 1 and curr_alive == 0:
        return float(REWARDS["agent_death"])

    reward = 0.0

    prev_enemies = _alive_enemies(prev_p, agent_id)
    curr_enemies = _alive_enemies(curr_p, agent_id)

    killed = len(prev_enemies) - len(curr_enemies)
    if killed > 0:
        reward += REWARDS["enemy_death"] * killed
    if len(curr_enemies) == 0 and len(prev_enemies) > 0:
        reward += REWARDS["win"]

    # ── 2. Movement ───────────────────────────────────────────────────────
    px, py = int(prev_p[agent_id][0]), int(prev_p[agent_id][1])
    cx, cy = int(curr_p[agent_id][0]), int(curr_p[agent_id][1])

    if px == cx and py == cy:
        reward += REWARDS["standing_still"]
    else:
        reward -= REWARDS["standing_still"]   # small bonus for moving

    reward += REWARDS["time_penalty"]

    # ── 3. Danger evasion / entering blast ────────────────────────────────
    any_bombs = (_parse_bombs(prev_obs).shape[0] > 0 or _parse_bombs(curr_obs).shape[0] > 0)
    if any_bombs:
        prev_in, prev_timer = _blast_at(prev_obs, px, py)
        curr_in,  _         = _blast_at(curr_obs, cx, cy)

        if prev_in and not curr_in:
            urgency = 1.5 if (prev_timer is not None and prev_timer <= 3) else 1.0
            reward += REWARDS["danger_evasion"] * urgency
        elif not prev_in and curr_in and (px != cx or py != cy):
            reward += REWARDS["danger_enter"]

    # Own blast loiter penalty (scales with urgency as fuse burns down)
    _, own_timer = _blast_at(curr_obs, cx, cy, own_only=True, agent_id=agent_id)
    if curr_alive == 1 and own_timer is not None:
        urgency = max(1, BOMB_MAX_TIMER + 1 - own_timer)
        reward += REWARDS["own_blast_loiter"] * float(urgency)

    # ── 4. Approach nearest enemy ─────────────────────────────────────────
    if curr_alive == 1 and curr_enemies:
        prev_d = _nearest_enemy_dist(prev_p, agent_id, px, py)
        curr_d = _nearest_enemy_dist(curr_p, agent_id, cx, cy)
        if prev_d is not None and curr_d is not None:
            reward += REWARDS["approach_enemy"] * (prev_d - curr_d)

    # ── 5. Item collection ────────────────────────────────────────────────
    stepped_on = int(prev_obs["map"][cx, cy]) if (0 <= cx < prev_obs["map"].shape[0] and
                                                    0 <= cy < prev_obs["map"].shape[1]) else GRASS
    if stepped_on in (ITEM_RADIUS, ITEM_CAPACITY):
        reward += REWARDS["item_collection"]
    else:
        # Fallback: radius bonus increased
        if int(curr_p[agent_id][4]) > int(prev_p[agent_id][4]):
            reward += REWARDS["item_collection"]

    # ── 6. Bomb placed near box (encourages strategic bombing) ────────────
    prev_bombs_left = int(prev_p[agent_id][3])
    curr_bombs_left = int(curr_p[agent_id][3])
    if curr_bombs_left < prev_bombs_left:
        grid = prev_obs["map"]
        H, W = grid.shape
        adjacent = [
            grid[max(0, cx - 1), cy],
            grid[min(H - 1, cx + 1), cy],
            grid[cx, max(0, cy - 1)],
            grid[cx, min(W - 1, cy + 1)],
        ]
        if BOX in [int(v) for v in adjacent]:
            reward += REWARDS["plant_near_box"]

    return float(reward)
