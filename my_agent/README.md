# Team agent (`my_agent`)

Canonical folder for development and submission. Baseline bots from the organizer live under `agent/` (do not submit those).

## Layout

| File | Role |
|------|------|
| `agent.py` | `Agent` class for the game + training helpers |
| `model.py` | `BomberDQN` network (imported by `agent.py`) |
| `model.pth` | Weights loaded at inference (keep this name) |
| `reward.py` | Reward shaping (training only) |
| `train.py` | Training script |
| `utils.py` | Checkpoints and plots (training only) |
| `ckpts/` | Training runs (gitignored) |

## Local test

```powershell
py -3.13 -m scripts.participant.run_local_match --agent_paths my_agent None None None --visualize true
py -3.13 -m scripts.participant.estimate_agent_time my_agent --opponents None None None
```

## Submit zip

From repo root:

```powershell
.\scripts\participant\build_submission_zip.ps1
```

Upload `submission.zip`. Root must contain `agent.py` only (no parent folder, no `requirements.txt`).
