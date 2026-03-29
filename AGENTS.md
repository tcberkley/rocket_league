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
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py
```

Checkpoints save to `models/` every 50k timesteps. To resume, set `checkpoint_load_folder="models/<number>"` in the `Learner` constructor in `learner.py`.

## Architecture

```
main.py           Entry point; defines build_rlgym_v2_env() factory + calls run_learner()
rewards.py        Custom VelocityBallToGoalReward (v2 API); TouchReward is from rlgym built-ins
learner.py        run_learner(env_create_func): configures and starts rlgym_ppo.Learner
models/           Saved PPO checkpoints (gitignored)
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

## macOS M-Series Notes

- This repo is currently being developed on a `MacBook Air (MacBookAir10,1)` with an `Apple M1` chip, `8 CPU cores`, and `8 GB` RAM
- `rocketsim` requires macOS 14+ for the ARM64 wheel
- PyTorch MPS backend for GPU acceleration (`device="auto"` in `learner.py`)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported MPS ops; or change to `device="cpu"` if MPS causes issues
- Tune `n_proc` in `learner.py`: 4 for M1/M2, 6-8 for Pro/Max/Ultra
