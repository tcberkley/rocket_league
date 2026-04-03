"""
Power Shot scenario: bot receives varied passes and is judged on shot speed into the opponent's goal.

6-stage curriculum (per-worker thresholds; multiply by ~4 for global episode count):
  Stage 0 (0–25k/w,  ~0–100k global):   ball 80–300 uu in front, near-stationary. Just touch and score.
  Stage 1 (25–50k/w, ~100–200k global): short rolling passes, ±25°.
  Stage 2 (50–75k/w, ~200–300k global): medium passes, ±45°, slight height.
  Stage 3 (75–100k/w,~300–400k global): longer arcing passes, ±70°.
  Stage 4 (100–125k/w,~400–500k global):hard passes, ±90°, significant height.
  Stage 5 (125k+/w,  ~500k+ global):    full chaos — 360°, 0–800 uu, 500–3000 uu/s.
"""

import math
import random
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POWER_SHOT_EPISODE_SECONDS = 12
POWER_SHOT_TICK_SKIP = 8
POWER_SHOT_NO_TOUCH_TIMEOUT = 6

CAR_SPAWN_X_RANGE = 2500
CAR_SPAWN_Y_MIN = -2000
CAR_SPAWN_Y_MAX = 2000

BACK_WALL_Y = 5120
GOAL_CENTER_TO_POST = 892.755
SIDE_WALL_X = 4096
BALL_MAX_SPEED = 6000
ORANGE_GOAL_Y = BACK_WALL_Y
GRAVITY = 650.0

# ---------------------------------------------------------------------------
# Unified curriculum
# Each row: (per_worker_threshold, spawn_cfg, reward_cfg)
#
# spawn_cfg keys:
#   angle  — half-arc from car forward direction (radians)
#   h_min/h_max — ball height range (uu)
#   s_min/s_max — ball launch speed range (uu/s); 0 = near-stationary
#   d_min/d_max — ball distance from car (uu)
#
# reward_cfg keys:
#   ball_vel_w        — weight on per-step ball-velocity-toward-goal / 6000
#   approach_w        — weight on approach reward (1 - dist/5000) before first touch
#   first_touch       — one-time bonus on first contact
#   subseq_touch      — bonus on each subsequent contact
#   goal_base         — flat bonus on scoring regardless of speed
#   shot_speed_w      — multiplier on speed/6000 bonus on goal
#   no_touch_pen      — penalty when episode times out with no touch
#   own_goal_pen      — penalty when bot scores in own goal
# ---------------------------------------------------------------------------
_CURRICULUM = [
    # Stage 0: ~0–100k global — ball RIGHT in front, near-stationary. Just nudge it.
    (25_000,
     dict(angle=math.radians(10), h_min=93, h_max=93,  s_min=0,   s_max=30,   d_min=80,   d_max=250),
     dict(ball_vel_w=6.0, approach_w=0.0, first_touch=3.0, subseq_touch=1.0,
          goal_base=10.0, shot_speed_w=1.0, no_touch_pen=-0.1, own_goal_pen=-1.0)),

    # Stage 1: ~100–200k global — short rolls, ±25°.
    (50_000,
     dict(angle=math.radians(25), h_min=93, h_max=93,  s_min=50,  s_max=350,  d_min=200,  d_max=800),
     dict(ball_vel_w=4.0, approach_w=2.0, first_touch=2.0, subseq_touch=0.5,
          goal_base=6.0,  shot_speed_w=2.0, no_touch_pen=-0.3, own_goal_pen=-2.0)),

    # Stage 2: ~200–300k global — medium passes, ±45°, slight height.
    (75_000,
     dict(angle=math.radians(45), h_min=93, h_max=150, s_min=150, s_max=700,  d_min=500,  d_max=1500),
     dict(ball_vel_w=2.5, approach_w=1.5, first_touch=1.5, subseq_touch=0.5,
          goal_base=4.0,  shot_speed_w=3.0, no_touch_pen=-0.4, own_goal_pen=-2.0)),

    # Stage 3: ~300–400k global — longer arcing passes, ±70°.
    (100_000,
     dict(angle=math.radians(70), h_min=93, h_max=300, s_min=250, s_max=1200, d_min=800,  d_max=2500),
     dict(ball_vel_w=1.5, approach_w=1.0, first_touch=1.0, subseq_touch=0.5,
          goal_base=2.0,  shot_speed_w=4.0, no_touch_pen=-0.5, own_goal_pen=-3.0)),

    # Stage 4: ~400–500k global — hard passes, ±90°, significant height.
    (125_000,
     dict(angle=math.radians(90), h_min=93, h_max=500, s_min=400, s_max=2000, d_min=1200, d_max=3500),
     dict(ball_vel_w=1.0, approach_w=0.5, first_touch=0.5, subseq_touch=0.3,
          goal_base=1.0,  shot_speed_w=5.0, no_touch_pen=-0.5, own_goal_pen=-3.0)),

    # Stage 5: ~500k+ global — full chaos.
    (None,
     dict(angle=math.pi,           h_min=93, h_max=800, s_min=500, s_max=3000, d_min=1500, d_max=4000),
     dict(ball_vel_w=1.0, approach_w=0.2, first_touch=1.0, subseq_touch=0.5,
          goal_base=0.0,  shot_speed_w=5.0, no_touch_pen=-0.5, own_goal_pen=-3.0)),
]

# Module-level per-worker episode counter, updated by PowerShotMutator.apply().
# PowerShotReward reads this to pick the matching reward config.
_worker_episode_count = 0


def _get_stage(episode_count):
    for threshold, spawn_cfg, reward_cfg in _CURRICULUM:
        if threshold is None or episode_count < threshold:
            return spawn_cfg, reward_cfg
    return _CURRICULUM[-1][1], _CURRICULUM[-1][2]


# ---------------------------------------------------------------------------
# State Mutator
# ---------------------------------------------------------------------------
class PowerShotMutator:
    def __init__(self):
        self.episode_count = 0

    def apply(self, state, shared_info: dict) -> None:
        global _worker_episode_count
        self.episode_count += 1
        _worker_episode_count = self.episode_count

        spawn, _ = _get_stage(self.episode_count)

        # Car spawn
        car_x = random.uniform(-CAR_SPAWN_X_RANGE, CAR_SPAWN_X_RANGE)
        car_y = random.uniform(CAR_SPAWN_Y_MIN, CAR_SPAWN_Y_MAX)
        car_z = 17.0

        yaw_noise = random.uniform(-math.radians(15), math.radians(15))
        car_yaw = math.pi / 2 + yaw_noise  # facing orange goal (+y)

        agent_id = list(state.cars.keys())[0]
        car = state.cars[agent_id]
        car.physics.position = np.array([car_x, car_y, car_z], dtype=np.float32)
        car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.euler_angles = np.array([0.0, car_yaw, 0.0], dtype=np.float32)
        car.boost_amount = random.uniform(50, 100) / 100.0

        # Ball spawn
        ball_angle = car_yaw + random.uniform(-spawn["angle"], spawn["angle"])
        ball_dist = random.uniform(spawn["d_min"], spawn["d_max"])
        ball_x = float(np.clip(car_x + math.cos(ball_angle) * ball_dist, -SIDE_WALL_X + 200, SIDE_WALL_X - 200))
        ball_y = float(np.clip(car_y + math.sin(ball_angle) * ball_dist, -BACK_WALL_Y + 200, BACK_WALL_Y - 200))
        ball_z = float(random.uniform(spawn["h_min"], spawn["h_max"]))

        dx = car_x - ball_x
        dy = car_y - ball_y
        horiz_dist = math.sqrt(dx * dx + dy * dy) + 1e-6
        speed = random.uniform(spawn["s_min"], spawn["s_max"])

        if speed < 5.0:
            # Essentially stationary — tiny nudge so it's not a physics edge case
            vx, vy, vz = dx / horiz_dist * 5.0, dy / horiz_dist * 5.0, 0.0
        elif spawn["h_min"] == spawn["h_max"] == 93:
            # Ground roll: purely horizontal
            vx = speed * dx / horiz_dist
            vy = speed * dy / horiz_dist
            vz = 0.0
        else:
            # Arcing pass with gravity compensation
            dz = car_z - ball_z
            horiz_speed = speed * math.cos(math.atan2(abs(dz), horiz_dist))
            vx = horiz_speed * dx / horiz_dist
            vy = horiz_speed * dy / horiz_dist
            time_of_flight = horiz_dist / (horiz_speed + 1e-6)
            vz = dz / (time_of_flight + 1e-6) + 0.5 * GRAVITY * time_of_flight

        vel = np.array([vx, vy, vz], dtype=np.float32)
        mag = float(np.linalg.norm(vel))
        if mag > BALL_MAX_SPEED:
            vel = vel * (BALL_MAX_SPEED / mag)

        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)
        state.ball.linear_velocity = vel
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        shared_info["goal_pos"] = np.array([0.0, ORANGE_GOAL_Y, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------
class PowerShotReward:
    def __init__(self):
        self._has_touched: Dict[Any, bool] = {}
        self._prev_dist: Dict[Any, float] = {}

    def reset(self, agents, initial_state, shared_info: dict) -> None:
        ball_pos = initial_state.ball.position
        for agent in agents:
            self._has_touched[agent] = False
            car_pos = initial_state.cars[agent].physics.position
            self._prev_dist[agent] = float(np.linalg.norm(car_pos - ball_pos))

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info: dict) -> Dict[Any, float]:
        _, rcfg = _get_stage(_worker_episode_count)

        goal_pos = shared_info.get("goal_pos", np.array([0.0, ORANGE_GOAL_Y, 0.0]))
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity

        goal_dir = goal_pos - ball_pos
        goal_dist = float(np.linalg.norm(goal_dir)) + 1e-6
        goal_dir_unit = goal_dir / goal_dist
        ball_speed_toward_goal = float(np.dot(ball_vel, goal_dir_unit))

        rewards = {}
        for agent in agents:
            r = 0.0
            car_pos = state.cars[agent].physics.position
            dist_to_ball = float(np.linalg.norm(car_pos - ball_pos))

            # 1. Ball velocity toward goal (dense, always active)
            r += rcfg["ball_vel_w"] * (ball_speed_toward_goal / BALL_MAX_SPEED)

            # 2. Touch detection (proximity-based)
            touched_now = dist_to_ball < 200.0
            if touched_now:
                if not self._has_touched.get(agent, False):
                    r += rcfg["first_touch"]
                    self._has_touched[agent] = True
                else:
                    r += rcfg["subseq_touch"]

            # 3. Approach reward — before first touch, reward closing distance
            if not self._has_touched.get(agent, False) and rcfg["approach_w"] > 0:
                prev = self._prev_dist.get(agent, dist_to_ball)
                closing = (prev - dist_to_ball) / (BALL_MAX_SPEED / (120.0 / POWER_SHOT_TICK_SKIP))
                r += rcfg["approach_w"] * max(0.0, closing)
                # Also small gradient toward ball
                r += rcfg["approach_w"] * 0.1 * max(0.0, 1.0 - dist_to_ball / 5000.0)

            self._prev_dist[agent] = dist_to_ball

            # 4 & 5. Goal scored / own goal (terminal)
            if is_terminated.get(agent, False) and getattr(state, "goal_scored", False):
                if getattr(state, "scoring_team", -1) == 0:
                    ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
                    r += rcfg["goal_base"] + rcfg["shot_speed_w"] * (ball_speed / BALL_MAX_SPEED)
                else:
                    r += rcfg["own_goal_pen"]

            # 6. No-touch timeout penalty
            if is_truncated.get(agent, False) and not self._has_touched.get(agent, False):
                r += rcfg["no_touch_pen"]

            rewards[agent] = r

        return rewards


# ---------------------------------------------------------------------------
# Done Condition
# ---------------------------------------------------------------------------
class PowerShotTerminationCondition:
    def __init__(self):
        self._has_touched: Dict[Any, bool] = {}

    def reset(self, agents, initial_state, shared_info: dict) -> None:
        for agent in agents:
            self._has_touched[agent] = False

    def is_done(self, agents, state, shared_info: dict) -> Dict[Any, bool]:
        result = {}
        ball_pos = state.ball.position
        ball_speed = float(np.linalg.norm(state.ball.linear_velocity))

        for agent in agents:
            car_pos = state.cars[agent].physics.position
            if float(np.linalg.norm(ball_pos - car_pos)) < 200.0:
                self._has_touched[agent] = True

            done = False
            if getattr(state, "goal_scored", False):
                done = True
            elif self._has_touched.get(agent, False) and ball_speed < 50.0 and ball_pos[2] < 150.0:
                done = True
            result[agent] = done

        return result


# ---------------------------------------------------------------------------
# Observation Builder
# ---------------------------------------------------------------------------
class PowerShotObs:
    """
    DefaultObs (92 features with zero_padding=1) + 3 extra:
      goal_dir_x, goal_dir_y, ball_vel_toward_goal/6000
    Total: 95 features
    """

    def __init__(self, zero_padding=1, pos_coef=None, ang_coef=None, lin_vel_coef=None, ang_vel_coef=None):
        from rlgym.rocket_league.obs_builders import DefaultObs

        if pos_coef is None:
            pos_coef = np.ones(3, dtype=np.float32)
        if ang_coef is None:
            ang_coef = np.ones(1, dtype=np.float32)
        if lin_vel_coef is None:
            lin_vel_coef = np.ones(1, dtype=np.float32)
        if ang_vel_coef is None:
            ang_vel_coef = np.ones(1, dtype=np.float32)

        self._base_obs = DefaultObs(
            zero_padding=zero_padding,
            pos_coef=pos_coef,
            ang_coef=ang_coef,
            lin_vel_coef=lin_vel_coef,
            ang_vel_coef=ang_vel_coef,
        )
        self._goal_pos = np.array([0.0, ORANGE_GOAL_Y, 0.0], dtype=np.float32)

    def reset(self, agents, initial_state, shared_info: dict) -> None:
        self._base_obs.reset(agents, initial_state, shared_info)

    def get_obs_space(self, agent):
        obs_type, size = self._base_obs.get_obs_space(agent)
        return obs_type, size + 3

    def build_obs(self, agents, state, shared_info: dict):
        base_obs = self._base_obs.build_obs(agents, state, shared_info)
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity

        goal_dir = self._goal_pos - ball_pos
        goal_dist = float(np.linalg.norm(goal_dir)) + 1e-6
        goal_dir_unit = goal_dir / goal_dist
        ball_vel_toward_goal = float(np.dot(ball_vel, goal_dir_unit)) / BALL_MAX_SPEED

        extra = np.array([goal_dir_unit[0], goal_dir_unit[1], ball_vel_toward_goal], dtype=np.float32)
        for agent in agents:
            base_obs[agent] = np.concatenate([base_obs[agent], extra])
        return base_obs
