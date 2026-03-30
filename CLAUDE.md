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
# Indefinite dribble run with dashboard, 2h train / 15min cooldown
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train --scenario dribble --dashboard --timesteps 0 --train-segment-hours 2 --cooldown-minutes 15

# Fresh run
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py train --scenario dribble --dashboard --fresh --timesteps 0 --train-segment-hours 2 --cooldown-minutes 15

# Watch a saved checkpoint
python main.py watch --scenario dribble --renderer sandbox
```

Checkpoints save to `models/` every 50k timesteps and training resumes from the latest checkpoint by default unless `--fresh` is passed.

**`--timesteps 0` = no limit** (internally maps to 5 billion, the rlgym_ppo Learner maximum). Any positive value is a hard cap. The default is 10M — if a checkpoint is already at 10M the learner exits immediately, so always pass `--timesteps` explicitly when resuming.

**Thermal management**: `--train-segment-hours` and `--cooldown-minutes` control the train/rest cycle. The process saves a checkpoint, sleeps for the cooldown period, then resumes automatically. Recommended: `--train-segment-hours 2 --cooldown-minutes 15` on M1.

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
- **Termination**: ball touching ground (`ball_pos[2] < 100.0`), or car/ball entering a goal mouth
- **Reward** (6 components): carry quality (always), breadcrumb approach (while carrying), breadcrumb success bonus (1.26–1.68 by difficulty), speed comfort zone (while carrying), wall/corner/approach penalties, terminal penalty (-1.0 on drop)

### Spawn Geometry

`DribbleStartMutator` spawns using a strict tangent-line setup each episode:

1. A random `spawn_angle` is sampled on the ellipse
2. `tangent_point = ellipse_point(spawn_angle)` — the intersection point on the ellipse
3. `tangent_dir = ellipse_tangent(spawn_angle)` — the true tangent direction at that point
4. Car is placed 2000–3500 uu back along the tangent line: `spawn_xy = tangent_point - tangent_dir * spawn_distance`
5. `_max_safe_spawn_distance()` caps the distance so the car never needs to be clamped off the line
6. **First waypoint = `tangent_point`** — exactly where the tangent line meets the ellipse
7. Car velocity is along `tangent_dir` (exact); car facing has small yaw noise (±5–9°) for variety
8. Ball is placed on the hood with random noise offsets

This means the agent always starts with a long straight-line dribble before entering the ellipse loop.

### Two-Phase Breadcrumb System

Each episode has two phases:

**Phase A — Ellipse loop (breadcrumbs 1–5):** Deterministic type schedule teaches turning in progressively tighter arcs. Total arc ≈ 2π (full loop).

| Crumb | Type | Arc (rad) | Distance (uu) | Purpose |
|-------|------|-----------|---------------|---------|
| 1 | `first` | 0.01–0.12 | 350–750 | Tangent entry point — straight ahead |
| 2 | `moderate` | 0.30–0.60 | 800–1600 | Close along the circle after entering |
| 3 | `hard_turn` | 1.20–1.75 | 2200–3800 | Sharp turn |
| 4 | `hard_turn` | 1.20–1.75 | 2200–3800 | Sharp turn |
| 5 | `closing` | 2.09–3.09 | 2000–4500 | Large arc to close the loop |

**Phase B — Random field (breadcrumb 6+):** After completing the loop, waypoints are sampled randomly across the field (`RANDOM_FIELD_SAFE_X/Y` margins), 1500–4000 uu from the car. The loop lane ellipse is hidden in the visualizer when phase B begins.

- **Curriculum difficulty** ramps from easy → medium → hard based on episode count (affects spawn speed, ellipse size, and loop lane margins)

| Difficulty | Episodes | Speed min (uu/s) | Speed max (uu/s) |
|------------|----------|------------------|------------------|
| Easy       | 0–2500   | 200              | 400              |
| Medium     | 2500–8000| 250              | 480              |
| Hard       | 8000+    | 300              | 550              |
- Global state accessor `get_current_loop_state()` exposes loop info for visualization; `loop_x=0/loop_y=0` signals Phase B to renderers

## Dashboard Metrics

- `--dashboard` enables the local Tk dashboard during dribble training
- `--dashboard-update-seconds` controls refresh cadence; the current default is `1.0`
- The dashboard loads history from `metrics/dribble_episode_metrics.csv`; episode stats go to `metrics/dribble_phase4_metrics.csv`
- Dashboard layout: 2x2 chart grid + live field minimap on the right

### Dashboard Charts

| Position | Metric | Color | Notes |
|----------|--------|-------|-------|
| Top-left | Carry Time Per Episode (s) | Blue | All episodes |
| Top-right | Breadcrumbs Reached Per Episode | Brown/Amber | Tracked episodes; amber step line = all-time best |
| Bottom-left | Avg Car Speed Per Episode (uu/s) | Purple | In-memory only, resets on restart |
| Bottom-right | Wall Approach Penalty Per Episode | Teal | Tracked episodes |

All charts show 50-episode rolling mean with 5th/95th percentile bands, compressed to at most 420 plotted bins.

### Minimap
Shows the last completed episode: car trail (blue line), loop lane ellipse (amber, hidden in Phase B), active waypoint (magenta crosshair), reached breadcrumbs (green circles with ✓), ball (white), car (blue triangle). Breadcrumb count shown top-center.

### Record GIFs
A GIF is saved to `artifacts/periodic_p95/` **only when a new all-time breadcrumb count record is set**. Filename: `record_ep{N}_{crumbs}crumbs_{carry}s.gif`. Each frame shows the car trail, reached crumbs (green), and active waypoint (magenta) at game-unit-proportional scale.

- Annotation markers (stored in `metrics/dribble_markers.json`) can be placed on charts to mark training milestones
- Direction flips and near-wall carry seconds are logged to CSV but not shown on the dashboard

### Worker metrics serialization note
`_collect_metrics` returns two arrays: a fixed 20-float metric array and a fixed-size crumb buffer (`1 + MAX_TRACKED_CRUMBS * 2 = 21` floats). The crumb buffer is always the same size so that `rlgym_ppo`'s `shm_view` (shared memory, only reallocated on agent-count change) is never overrun by a growing array.

**Important**: `env.reset()` is called by rlgym_ppo workers *before* `collect_metrics(info["state"])` on terminal steps. This clears module-level globals (`REACHED_BREADCRUMB_POSITIONS`, `CURRENT_LOOP_STATE`). To work around this:
- Reached crumb positions are tracked in `_consume_metric` by detecting when `breadcrumbs_reached` count increases (car position recorded at that step), rather than reading the global.
- `episode_reward_sum` is not overwritten when it drops to 0 (terminal reset artifact); the last non-zero value is preserved until episode finalization.
- The terminal step frame is trimmed from episode frame lists before saving GIFs or updating the minimap.

## macOS M-Series Notes

- This repo is currently being developed on a `MacBook Air (MacBookAir10,1)` with an `Apple M1` chip, `8 CPU cores`, and `8 GB` RAM
- `rocketsim` requires macOS 14+ for the ARM64 wheel
- PyTorch MPS backend for GPU acceleration (`device="auto"` in `learner.py`)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported MPS ops; or change to `device="cpu"` if MPS causes issues
- Tune `n_proc` in `learner.py`: 4 for M1/M2, 6-8 for Pro/Max/Ultra
