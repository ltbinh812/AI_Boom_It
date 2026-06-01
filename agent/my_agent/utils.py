"""
utils.py — Shared training utilities for my_agent.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, global_step: int, epsilon: float,
                    lr: float, input_spec: tuple, num_actions: int,
                    path: str) -> None:
    """Save a training checkpoint to *path*."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step":          global_step,
            "epsilon":              epsilon,
            "lr":                   lr,
            "input_spec":           input_spec,   # (map_shape, aux_dim)
            "num_actions":          num_actions,
        },
        path,
    )
    print(f"[ckpt] Saved → {path}  (step={global_step}, ε={epsilon:.4f})")


def load_checkpoint(path: str, device: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict from *path*."""
    ckpt = torch.load(path, map_location=device)
    # backward-compat with BTC's key naming
    if "input_spec" not in ckpt:
        ckpt["input_spec"] = ckpt.get("input_shape", ckpt.get("input_dim"))
    return ckpt


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_or_show(fig, path: str | None) -> None:
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_loss(loss_history: list[float], save_path: str | None = None,
              title: str = "Training Loss") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, linewidth=0.8, label="MSE Loss")
    ax.set(xlabel="Update step", ylabel="Loss", title=title)
    ax.legend(); ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_rewards(reward_history: list[float], save_path: str | None = None,
                 window: int = 50) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reward_history, linewidth=0.4, alpha=0.5, label="Raw reward")
    if len(reward_history) >= window:
        ma = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(reward_history)), ma, linewidth=1.5,
                label=f"MA-{window}")
    ax.set(xlabel="Step", ylabel="Reward", title="Episode Rewards")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_win_rates(win_history: list[int], save_path: str | None = None,
                   window: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(win_history) >= window:
        wr = np.convolve(win_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(win_history)), wr, linewidth=1.5)
    else:
        ax.plot(win_history)
    ax.set(xlabel="Episode", ylabel="Win rate", title=f"Win Rate (MA-{window})",
           ylim=[0, 1])
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_epsilon(eps_history: list[float], save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(eps_history, linewidth=1.0, color="orange")
    ax.set(xlabel="Episode", ylabel="ε", title="Epsilon Decay")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)
