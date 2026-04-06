# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
# Dribble (ball-carry) training
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train --scenario dribble --dashboard --timesteps 0 --train-segment-hours 2 --cooldown-minutes 15

# Power shot training (fresh start)
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train --scenario power_shot --fresh --timesteps 0 --train-segment-hours 2 --cooldown-minutes 15 --dashboard

# Watch a checkpoint
python main.py watch --scenario dribble --renderer sandbox
python main.py watch --scenario power_shot --renderer sandbox
```

Checkpoints save to a per-scenario directory every 50k timesteps and resume from the latest by default unless `--fresh` is passed. `--timesteps 0` means no limit. The CLI supports timed cooldowns via `--train-segment-hours` and `--cooldown-minutes`.

## Architecture

```
main.py            Entry point; defines standard/dribble/power_shot env factories, watch mode, and CLI wiring
rewards.py         Custom VelocityBallToGoalReward (v2 API); TouchReward is from rlgym built-ins
learner.py         run_learner(env_create_func): configures and starts rlgym_ppo.Learner
dribble.py         Dribble scenario mutator, reward shaping, and termination logic
dribble_metrics.py Live dribble dashboard, CSV episode metrics logger, minimap, and record-breaker GIFs
power_shot.py      Power Shot scenario: 3-stage curriculum, mutator, reward, termination, obs (95 features)
record_rollout.py  Record rollout GIFs with loop lane and waypoint overlays
models-dribble/    Saved dribble PPO checkpoints (gitignored)
models-power-shot/ Saved power_shot PPO checkpoints (gitignored)
metrics/           Episode CSV logs and marker annotations
artifacts/         Generated GIFs (rollouts and record-breakers)
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

- `main.py --scenario dribble` trains a single-car ball-carry task (no opponent)
- `DribbleStartMutator` spawns the car on a tangent line outside the ellipse loop with the ball on the hood
- **Termination**: ball touches the ground, or car/ball enters a goal mouth
- **Reward**: carry quality, breadcrumb waypoint approach/success bonuses, speed comfort zone, wall penalties, terminal drop penalty
- Curriculum ramps easy → hard based on episode count (spawn speed, ellipse size, loop lane margins)

## Power Shot Scenario

- `main.py --scenario power_shot` trains a single blue car to shoot with maximum speed into the orange goal
- **Observation**: 95 features (DefaultObs 92 + goal_dir_x, goal_dir_y, ball_vel_toward_goal)
- **Curriculum**: 3 stages, gated by rolling 50% goal rate — not episode count
  - Stage 0: ball drops in place ahead of car (bouncing), car in attacking half
  - Stage 1: ball crosses car's path laterally
  - Stage 2: full variety (wide angles, heights, speeds)
- **Reward**: inverse-distance approach (pre-touch), large first-touch and goal bonuses; approach uses `1 - dist/3000` not closing-speed to prevent oscillation hacking

## Dashboard Metrics

- `--dashboard` enables the local Tk dashboard during dribble training
- `--dashboard-update-seconds` controls refresh cadence; the current default is `1.0`
- The dashboard loads history from `metrics/dribble_episode_metrics.csv`
- Charts display 50-episode rolling averages compressed into at most 480 plotted bins so long runs stay readable
- The plotted `distance_traveled_uu` metric is episode path length in the ground plane (`x/y` step-to-step distance), not straight-line displacement from spawn

## macOS M-Series Notes

- This repo is currently being developed on a `MacBook Air (MacBookAir10,1)` with an `Apple M1` chip, `8 CPU cores`, and `8 GB` RAM
- `rocketsim` requires macOS 14+ for the ARM64 wheel
- PyTorch MPS backend for GPU acceleration (`device="auto"` in `learner.py`)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported MPS ops; or change to `device="cpu"` if MPS causes issues
- Tune `n_proc` in `learner.py`: 4 for M1/M2, 6-8 for Pro/Max/Ultra
