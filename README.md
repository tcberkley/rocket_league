# Rocket League Dribble Bot

This repo trains a Rocket League bot with PPO using `rlgym` + `RocketSim`, so it can learn entirely in simulation without running the actual game client.

The project currently has two modes:

- `standard`: a simple 1v1 self-play setup
- `dribble`: a single-car dribble task where the car spawns on an elliptical loop lane with the ball on its hood, and chases breadcrumb waypoints around the loop while maintaining ball control

## Requirements

- Python 3.11 is recommended
- macOS, Linux, or Windows should work in principle, but this repo has mainly been developed on Apple Silicon macOS

## Setup

Clone the repo, then create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

Basic training:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train
```

Train the dribble bot with the live dashboard:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train --scenario dribble --dashboard
```

Start a fresh run instead of resuming the latest checkpoint:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train --scenario dribble --dashboard --fresh
```

Useful options:

- `--timesteps 2000000`: total timestep budget
- `--n-proc 4`: number of parallel simulation workers
- `--dashboard-update-seconds 1`: dashboard refresh rate
- `--train-segment-hours 3`: train for this long before pausing
- `--cooldown-minutes 15`: cooldown duration between training segments
- `--wandb`: enable Weights & Biases logging

By default, training resumes from the latest checkpoint in `models/` unless `--fresh` is passed.

## Watching The Bot

Watch the latest checkpoint in the built-in sandbox viewer:

```bash
python main.py watch --scenario dribble --renderer sandbox
```

Useful options:

- `--checkpoint latest`: load the most recent checkpoint
- `--episodes 3`: number of episodes to watch
- `--max-steps 300`: cap episode length
- `--renderer headless`: run without opening a viewer

## Dashboard

When training dribble mode with `--dashboard`, a local Tk window opens with:

- 4 charts: carry time, carry distance, loop progress, breadcrumbs reached
- a live field minimap showing the car trail, loop lane, waypoint, and ball from the last completed episode

The plots show 50-episode rolling averages with 5th/95th percentile bands.

Every 10,000 timesteps, a GIF of a 95th percentile episode is saved to `artifacts/periodic_p95/`.

The dashboard reads from `metrics/dribble_episode_metrics.csv` (and `metrics/dribble_phase4_metrics.csv` for loop-specific stats). Additional metrics like wall approach penalty and correct-turn yaw are still logged to CSV but not displayed on the dashboard.

`distance traveled` is the total ground-path distance of the car during an episode, measured by summing step-to-step `x/y` movement.

## Checkpoints

Checkpoints are saved under `models/` every 50k timesteps.

Examples:

```bash
models/50000
models/100000
models/150000
```

These contain the PPO policy and running stats needed to resume training or watch the bot later.

## Rendering GIFs

Record a rollout to a GIF:

```bash
python record_rollout.py --scenario dribble --checkpoint latest --max-steps 180 --output artifacts/dribble_attempt.gif
```

Generated GIFs are saved in `artifacts/`. The dribble scenario overlays the loop lane ellipse and current waypoint marker on frames.

## Notes For Apple Silicon Macs

If PyTorch tries to use MPS and some ops are unsupported, keep this enabled:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1
```

If MPS is flaky on a specific machine, training can also be forced to CPU with:

```bash
python main.py train --device cpu
```

## Quick Start

If someone just wants to try it:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train --scenario dribble --dashboard
python main.py watch --scenario dribble --renderer sandbox
```
