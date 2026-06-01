"""
train.py — 4-Player Self-Play training for Bomberland DQN agent.

Training curriculum (controlled by --mode flag):
  1. "rule"       — Agent vs three rule-based bots (warm-up phase)
  2. "selfplay"   — Agent vs three copies of itself (older checkpoints)
  3. "mixed"      — 50% rule, 50% self-play opponents (recommended for final training)

Key design choices:
  • All 4 agent slots share experience into a *single* Replay Buffer.
  • The online agent being updated is fixed at ID=0; opponents are separate
    instances sharing the same (or older) weights.
  • Target network sync: hard copy every `target_sync_every` episodes.
  • Checkpoints saved every `save_every` episodes to `ckpts/` subfolder.
  • Epsilon anneals over episodes; loaded agents use a fixed low epsilon.

Usage examples:
  # Warm-up: 2000 episodes vs tactical bots
  python train.py --mode rule --enemy_type tactical --num_episodes 2000 --save_model

  # Self-play: continue from checkpoint
  python train.py --mode selfplay --num_episodes 5000 --load_model ckpts/.../best.pth --save_model

  # Mixed curriculum
  python train.py --mode mixed --num_episodes 10000 --save_model
"""

import argparse
import sys
import random
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # d:\Antigravity\AI_Boom_It\my_agent\
_ROOT = _HERE.parent                      # d:\Antigravity\AI_Boom_It\
for _p in [str(_HERE), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load local modules explicitly by file path to avoid name collision
# with the root-level `agent/` package (BTC rule-based bots)
import importlib.util as _ilu

def _load_local(name: str, filepath):
    spec = _ilu.spec_from_file_location(name, str(filepath))
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_agent_mod = _load_local("_my_agent",  _HERE / "agent.py")
_reward_mod = _load_local("_my_reward", _HERE / "reward.py")
_utils_mod  = _load_local("_my_utils",  _HERE / "utils.py")

TrainingAgent  = _agent_mod.TrainingAgent
ReplayBuffer   = _agent_mod.ReplayBuffer
encode_obs     = _agent_mod.encode_obs
NUM_ACTIONS    = _agent_mod.NUM_ACTIONS
MAP_CHANNELS   = _agent_mod.MAP_CHANNELS
AUX_DIM        = _agent_mod.AUX_DIM

compute_reward = _reward_mod.compute_reward

seed_everything = _utils_mod.seed_everything
save_checkpoint = _utils_mod.save_checkpoint
plot_loss       = _utils_mod.plot_loss
plot_rewards    = _utils_mod.plot_rewards
plot_win_rates  = _utils_mod.plot_win_rates
plot_epsilon    = _utils_mod.plot_epsilon

from engine.game import BomberEnv

# Root-level agent package contains rule-based bots.
# We import it as a proper package (with submodule_search_locations set)
# so that its relative imports (.random_agent, etc.) work correctly.
def _load_root_agent_module():
    """Load the root-level agent package without shadowing local modules."""
    agent_dir  = str(_ROOT / "agent")
    agent_init = str(_ROOT / "agent" / "__init__.py")
    spec = _ilu.spec_from_file_location(
        "agent",                        # must match the package name used in relative imports
        agent_init,
        submodule_search_locations=[agent_dir],
    )
    mod = _ilu.module_from_spec(spec)
    import sys as _sys
    _sys.modules["agent"] = mod         # register as 'agent' so sub-imports resolve
    spec.loader.exec_module(mod)
    return mod

_ragent = _load_root_agent_module()
RandomAgent       = _ragent.RandomAgent
SimpleRuleAgent   = _ragent.SimpleRuleAgent
SmarterRuleAgent  = _ragent.SmarterRuleAgent
TacticalRuleAgent = _ragent.TacticalRuleAgent
GeniusRuleAgent   = _ragent.GeniusRuleAgent
BoxFarmerAgent    = _ragent.BoxFarmerAgent


# ── Helper: build opponent agent ─────────────────────────────────────────────

def _make_rule_agent(agent_id: int, enemy_type: str):
    mapping = {
        "random":     RandomAgent,
        "simple":     SimpleRuleAgent,
        "smarter":    SmarterRuleAgent,
        "tactical":   TacticalRuleAgent,
        "genius":     GeniusRuleAgent,
        "box_farmer": BoxFarmerAgent,
    }
    cls = mapping.get(enemy_type, TacticalRuleAgent)
    return cls(agent_id)


def _make_opponents_rule(enemy_type: str, n: int = 3):
    return [_make_rule_agent(i + 1, enemy_type) for i in range(n)]


def _make_opponents_selfplay(
    map_shape, aux_dim, pretrained_path=None, device="cpu", n: int = 3
):
    """Return n TrainingAgents that share (or start from) the same weights."""
    agents = []
    for i in range(n):
        a = TrainingAgent(
            agent_id    = i + 1,
            map_shape   = map_shape,
            aux_dim     = aux_dim,
            device      = device,
            dueling     = True,
            pretrained_path = pretrained_path,
        )
        a.epsilon = 0.05   # mostly greedy (act as pseudo-fixed opponents)
        agents.append(a)
    return agents


def _act_rule(agent, obs: dict) -> int:
    try:
        return int(agent.act(obs))
    except Exception:
        return 0


def _act_dqn(agent: TrainingAgent, obs: dict) -> int:
    ms, aux = encode_obs(obs, agent.agent_id)
    return agent.act(ms, aux, epsilon=agent.epsilon)


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    mode: str = "mixed",
    enemy_type: str = "tactical",
    num_episodes: int = 5000,
    max_steps: int = 500,
    seed: int = 42,
    save_model: bool = True,
    load_model: str | None = None,
    dueling: bool = True,
    # Hyperparameters
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9995,
    batch_size: int = 128,
    lr: float = 5e-4,
    buffer_capacity: int = 50_000,
    target_sync_every: int = 20,
    save_every: int = 500,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        torch.set_num_threads(2)

    seed_everything(seed)
    print(f"[train] mode={mode}  enemy={enemy_type}  episodes={num_episodes}  device={device}")

    env = BomberEnv(max_steps=max_steps, seed=seed)

    # ── Bootstrap observation to infer shapes ─────────────────────────────
    dummy_obs = env.reset(seed=seed)
    sample_ms, sample_aux = encode_obs(dummy_obs, agent_id=0)
    map_shape = sample_ms.shape     # (11, 13, 13)
    aux_dim   = sample_aux.shape[0] # 5

    print(f"[train] map_shape={map_shape}  aux_dim={aux_dim}")

    # ── Initialise learner (agent 0) ──────────────────────────────────────
    learner = TrainingAgent(
        agent_id        = 0,
        map_shape       = map_shape,
        aux_dim         = aux_dim,
        lr              = lr,
        device          = device,
        dueling         = dueling,
        pretrained_path = load_model,
    )
    epsilon = learner.epsilon if load_model else epsilon_start

    # ── Replay Buffer ─────────────────────────────────────────────────────
    buffer = ReplayBuffer(capacity=buffer_capacity, map_shape=map_shape, aux_dim=aux_dim)

    # ── Output directory ──────────────────────────────────────────────────
    run_name = f"ckpts/{mode}_{enemy_type}_{num_episodes}ep_{seed}seed"
    Path(run_name).mkdir(parents=True, exist_ok=True)

    # ── Logging containers ────────────────────────────────────────────────
    loss_history    = []
    reward_history  = []
    win_history     = []
    eps_history     = []
    step_times      = deque(maxlen=200)

    # ── Training episodes ─────────────────────────────────────────────────
    best_win_rate = 0.0
    input_spec = (map_shape, aux_dim)

    with tqdm(total=num_episodes, desc="Training", unit="ep") as pbar:
        for ep in range(num_episodes):
            ep_seed = seed + ep

            # Decide opponent composition for this episode
            if mode == "rule":
                use_selfplay = False
            elif mode == "selfplay":
                use_selfplay = True
            else:   # mixed
                use_selfplay = (ep % 2 == 1)

            if use_selfplay:
                opponents = _make_opponents_selfplay(
                    map_shape, aux_dim,
                    pretrained_path = (
                        f"{run_name}/latest.pth"
                        if Path(f"{run_name}/latest.pth").exists()
                        else load_model
                    ),
                    device = device,
                )
                is_rule_opp = False
            else:
                opponents = _make_opponents_rule(enemy_type)
                is_rule_opp = True

            obs      = env.reset(seed=ep_seed)
            prev_obs = None
            done     = False
            ep_reward = 0.0

            map_s, aux_s = encode_obs(obs, agent_id=0)

            for _ in range(max_steps):
                t0 = time.perf_counter()

                # ── Collect actions from all 4 agents ─────────────────────
                learner_action = learner.act(map_s, aux_s, epsilon=epsilon)
                actions = [learner_action]
                for opp in opponents:
                    if is_rule_opp:
                        actions.append(_act_rule(opp, obs))
                    else:
                        actions.append(_act_dqn(opp, obs))

                # ── Step environment ──────────────────────────────────────
                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated

                # ── Compute reward for learner (agent 0) ──────────────────
                r = compute_reward(prev_obs, next_obs, agent_id=0)
                ep_reward += r
                reward_history.append(r)

                # ── Store transition ──────────────────────────────────────
                next_map_s, next_aux_s = encode_obs(next_obs, agent_id=0)
                buffer.push(map_s, aux_s, learner_action, r, next_map_s, next_aux_s, done)

                # ── Learn ─────────────────────────────────────────────────
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    loss  = learner.train_step(batch)
                    loss_history.append(loss)

                # ── Next step ─────────────────────────────────────────────
                prev_obs = obs
                obs      = next_obs
                map_s    = next_map_s
                aux_s    = next_aux_s

                step_times.append((time.perf_counter() - t0) * 1000)

                if done:
                    break

            # ── Win tracking ──────────────────────────────────────────────
            final_players = np.asarray(obs["players"])
            won = int(final_players[0][2]) == 1 and sum(
                int(final_players[i][2]) for i in range(1, 4)
            ) == 0
            win_history.append(int(won))

            # ── Epsilon decay ─────────────────────────────────────────────
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            eps_history.append(epsilon)

            # ── Target sync ───────────────────────────────────────────────
            if (ep + 1) % target_sync_every == 0:
                learner.sync_target()

            # ── Periodic checkpoint ───────────────────────────────────────
            if save_model and (ep + 1) % save_every == 0:
                save_checkpoint(
                    learner.q_net, learner.optimizer,
                    learner.global_step, epsilon, lr,
                    input_spec, NUM_ACTIONS,
                    f"{run_name}/step_{learner.global_step}.pth",
                )
                # Always overwrite "latest.pth" for self-play opponent loading
                save_checkpoint(
                    learner.q_net, learner.optimizer,
                    learner.global_step, epsilon, lr,
                    input_spec, NUM_ACTIONS,
                    f"{run_name}/latest.pth",
                )

            # ── Best model ────────────────────────────────────────────────
            if len(win_history) >= 100:
                wr = float(np.mean(win_history[-100:]))
                if wr > best_win_rate:
                    best_win_rate = wr
                    if save_model:
                        save_checkpoint(
                            learner.q_net, learner.optimizer,
                            learner.global_step, epsilon, lr,
                            input_spec, NUM_ACTIONS,
                            f"{run_name}/best_model.pth",
                        )

            # ── Progress bar ──────────────────────────────────────────────
            avg_ms = np.mean(step_times) if step_times else 0.0
            pbar.set_postfix(
                reward   = f"{ep_reward:.2f}",
                epsilon  = f"{epsilon:.4f}",
                win_rate = f"{np.mean(win_history[-100:]) if win_history else 0:.2%}",
                ms_step  = f"{avg_ms:.1f}",
            )
            pbar.update(1)
            import gc
            gc.collect()

    # ── Final checkpoint ──────────────────────────────────────────────────
    if save_model:
        save_checkpoint(
            learner.q_net, learner.optimizer,
            learner.global_step, epsilon, lr,
            input_spec, NUM_ACTIONS,
            f"{run_name}/final_{learner.global_step}_steps.pth",
        )
        # Also save to my_agent/ so Agent() finds it automatically
        save_checkpoint(
            learner.q_net, learner.optimizer,
            learner.global_step, epsilon, lr,
            input_spec, NUM_ACTIONS,
            str(_HERE / "model.pth"),
        )

    # ── Plot training curves ──────────────────────────────────────────────
    plot_loss(loss_history,   save_path=f"{run_name}/loss.png")
    plot_rewards(reward_history, save_path=f"{run_name}/rewards.png")
    plot_win_rates(win_history,  save_path=f"{run_name}/win_rate.png")
    plot_epsilon(eps_history,    save_path=f"{run_name}/epsilon.png")

    print(f"\n[train] Done. Best win rate (last-100 MA): {best_win_rate:.2%}")
    print(f"[train] Avg inference time: {np.mean(step_times):.2f} ms/step")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bomberland DQN agent")
    parser.add_argument("--mode",          default="mixed",
                        choices=["rule", "selfplay", "mixed"],
                        help="Training mode (default: mixed)")
    parser.add_argument("--enemy_type",    default="tactical",
                        choices=["random", "simple", "smarter", "tactical",
                                 "genius", "box_farmer"],
                        help="Rule-based opponent type for 'rule' mode")
    parser.add_argument("--num_episodes",  type=int,   default=5000)
    parser.add_argument("--max_steps",     type=int,   default=500)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--save_model",    action="store_true")
    parser.add_argument("--load_model",    default=None,
                        help="Path to a .pth checkpoint to continue training")
    parser.add_argument("--no_dueling",    action="store_true",
                        help="Disable Dueling DQN (use standard head instead)")
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--buffer",        type=int,   default=50_000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_min",   type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.9995)
    parser.add_argument("--target_sync",   type=int,   default=20,
                        help="Sync target network every N episodes")
    parser.add_argument("--save_every",    type=int,   default=500)
    args = parser.parse_args()

    train(
        mode            = args.mode,
        enemy_type      = args.enemy_type,
        num_episodes    = args.num_episodes,
        max_steps       = args.max_steps,
        seed            = args.seed,
        save_model      = args.save_model,
        load_model      = args.load_model,
        dueling         = not args.no_dueling,
        lr              = args.lr,
        batch_size      = args.batch_size,
        buffer_capacity = args.buffer,
        epsilon_start   = args.epsilon_start,
        epsilon_min     = args.epsilon_min,
        epsilon_decay   = args.epsilon_decay,
        target_sync_every = args.target_sync,
        save_every      = args.save_every,
    )
