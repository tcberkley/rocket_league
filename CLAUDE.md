# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rocket League 1v1 bot trained with PPO. Uses RLGym v2 with RocketSim as a headless physics engine (no game client required). Collision meshes are bundled in the `rlgym` pip package — no manual asset setup needed.

Libraries: `rlgym` v2 (environment API), `rlgym-ppo` (PPO training loop), `rocketsim` (C++ physics via Python bindings), `torch`.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running Training

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train
```

Useful variants:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train --scenario dribble --dashboard
python main.py watch --scenario dribble --renderer sandbox
```

Checkpoints save to `models/` every 50k timesteps and training resumes from the latest checkpoint by default unless `--fresh` is passed. The CLI also supports timed cooldowns between training segments via `--train-segment-hours` and `--cooldown-minutes`.

## Architecture

```
main.py           Entry point; defines standard/dribble env factories, watch mode, and CLI wiring
rewards.py        Custom VelocityBallToGoalReward (v2 API); TouchReward is from rlgym built-ins
learner.py        run_learner(env_create_func): configures and starts rlgym_ppo.Learner
dribble.py        Dribble scenario mutator, reward shaping, and termination logic
dribble_metrics.py Live dribble dashboard, CSV episode metrics logger, minimap, and record-breaker GIFs
record_rollout.py  Record rollout GIFs with loop lane and waypoint overlays
models/           Saved PPO checkpoints (gitignored)
metrics/          Episode CSV logs and marker annotations
artifacts/        Generated GIFs (rollouts and record-breakers)
```

### Data flow
`rlgym_ppo.Learner` spawns `n_proc` worker processes. Each worker calls `env_create_func()` independently, so:

**The env factory function (`build_rlgym_v2_env` in `main.py`) must contain all its imports inside the function body.** It gets pickled and sent to worker processes.

### RLGym v2 vs v1 (rlgym-sim) differences
- v2 uses `RLGym(...)` constructor instead of `rlgym_sim.make()`
- `tick_skip` is handled via `RepeatAction(LookupTableAction(), repeats=tick_skip)` wrapping the action parser
- Reward functions return `Dict[AgentID, float]` instead of a single float
- Termination and truncation are separate conditions
- `RLGymV2GymWrapper` bridges v2's dict-based API to rlgym-ppo's flat gym API
- Game state uses `state.cars[agent_id]` dict instead of `game_state.players` list
- `state.cars[agent].physics.position`, `.linear_velocity` etc. instead of old PlayerData fields

## Reward Functions (`rewards.py`)

- `VelocityBallToGoalReward` (custom): reward in [-1, 1] proportional to ball velocity toward opponent's goal
- `TouchReward` (built-in from rlgym): +1.0 on any touch
- Combined weight: VelocityBallToGoal x 1.0, TouchReward x 0.5

## Dribble Scenario

- `main.py --scenario dribble` trains a single-car carry task instead of standard 1v1
- `DribbleStartMutator` spawns the car on an elliptical loop lane with the ball on its hood, tangent-aligned to the lane direction
- A **breadcrumb waypoint** system chains targets along the loop; reaching one spawns the next, rewarding continuous turning while carrying
- **Curriculum difficulty** ramps from easy (wide loops, gentle turns) through medium to hard (tighter loops, more waypoints) based on episode count (`EASY_CURRICULUM_EPISODES`, `MIXED_CURRICULUM_EPISODES`)
- Wall, corner, and wall-approach penalties discourage drifting to field edges during carries
- Goal mouths are still treated as failure zones — driving into the goal ends the episode
- The dribble reward combines carry quality, forward movement, loop progress (yaw delta + ellipse arc), tangent alignment, breadcrumb approach, and the anti-stall mechanism
- Global state accessors `get_current_loop_state()` / `get_current_turn_target_xy()` expose loop info for visualization

## Dashboard Metrics

- `--dashboard` enables the local Tk dashboard during dribble training
- `--dashboard-update-seconds` controls refresh cadence; the current default is `1.0`
- The dashboard loads history from `metrics/dribble_episode_metrics.csv`; phase 4 loop metrics go to `metrics/dribble_phase4_metrics.csv`
- Charts display 50-episode rolling averages with 5th/95th percentile bands, compressed into at most 420 plotted bins
- Dashboard includes a live field minimap showing car position, ball, loop lane ellipse, and current waypoint
- Record-breaking episodes automatically generate animated GIF replays saved to `artifacts/record_breakers/`
- Annotation markers (stored in `metrics/dribble_markers.json`) can be placed on charts to mark training milestones
- The plotted `distance_traveled_uu` metric is episode path length in the ground plane (`x/y` step-to-step distance), not straight-line displacement from spawn

## macOS M-Series Notes

- This repo is currently being developed on a `MacBook Air (MacBookAir10,1)` with an `Apple M1` chip, `8 CPU cores`, and `8 GB` RAM
- `rocketsim` requires macOS 14+ for the ARM64 wheel
- PyTorch MPS backend for GPU acceleration (`device="auto"` in `learner.py`)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported MPS ops; or change to `device="cpu"` if MPS causes issues
- Tune `n_proc` in `learner.py`: 4 for M1/M2, 6-8 for Pro/Max/Ultra
