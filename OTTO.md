# Hey Otto 👋

This is a Rocket League dribbling bot trained with reinforcement learning (~500 million timesteps). No Rocket League installation needed — it runs on a headless physics engine called RocketSim.

---

## What you need

- **Python 3.11** — must be 3.11 specifically (not 3.12+)
  - Download: https://www.python.org/downloads/release/python-3119/
  - During install, check **"Add Python to PATH"**
- **Git** — to clone the repo
- A terminal (PowerShell or Command Prompt)

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/tcberkley/rocket_league.git
cd rocket_league

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Watching the bot (2D top-down view)

This works immediately with no extra setup:

```bash
python main.py watch --scenario dribble --renderer sandbox
```

A window will open showing a top-down field view. The bot will play 3 episodes back-to-back. It spawns outside the ellipse loop, dribbles in, and tries to hit as many waypoints as possible before dropping the ball.

To watch more episodes:
```bash
python main.py watch --scenario dribble --renderer sandbox --episodes 10
```

---

## Watching the bot (3D view)

For the full 3D view you need the **RLViser** viewer binary:

1. Go to: https://github.com/VirxEC/rlviser/releases
2. Download the latest `rlviser.exe` (Windows release)
3. Place `rlviser.exe` in the root of this repo (same folder as `main.py`)

Then run:
```bash
python main.py watch --scenario dribble --renderer rlviser
```

A 3D Rocket League field will open with the bot playing live.

> **Note:** If the rlviser window doesn't open, make sure `rlviser.exe` is in the repo root (not in a subfolder).

---

## What you're looking at

The bot is trained on a single task: carry the ball on the car's hood and hit a series of waypoints. The waypoints form a loop around the field, then random targets after the loop is complete.

- **Best run so far:** 47 waypoints, ~110 seconds of carry time
- **Average:** ~7 waypoints per episode
- The episode ends immediately when the ball touches the ground

Check out `artifacts/periodic_p95/record_ep1792698_47crumbs_110.40s.gif` for a replay of the best run, and `artifacts/periodic_p95/screenshot_2026-04-03_best_47crumbs.png` for a dashboard snapshot of where training is at.

---

## Troubleshooting

**"No checkpoint found"** — make sure you cloned the full repo including the `models/` folder. Run `git lfs pull` if the model files are empty (they may be stored with Git LFS).

**`pip install` fails on `rocketsim`** — you need Python 3.11. Check with `python --version`.

**Import errors on torch** — if you have an NVIDIA GPU and want to use it: `pip install torch --index-url https://download.pytorch.org/whl/cu121` before running requirements.

**The window is very slow** — add `--render-delay 0.1` to slow down playback, or `--render-delay 0` to run as fast as possible.
