"""
Power Shot scenario: train a bot to shoot balls in front of it into the orange goal.

4-stage curriculum gated by per-stage rolling goal rate:
  Stage 0: Stationary ground ball 300–500 uu ahead, ±5°, car close to goal.
           Advance at 30% goal rate.
  Stage 1: Ground ball 400–800 uu ahead, ±15°, moderate distance.
           Advance at 40% goal rate.
  Stage 2: Ground or low-bounce ball 500–1200 uu ahead, ±25°.
           Advance at 50% goal rate.
  Stage 3: Ground or bounce ball 600–2000 uu ahead, ±40°, full field. No cap.

Reward design:
  - Small inverse-distance approach reward pre-touch (can't be hacked)
  - One-time first-touch bonus
  - One-time directional hit bonus at moment of touch: dot(ball_vel, goal_dir) — rewards hard, on-target hits
  - Large terminal goal + shot-speed bonus
  - Second-touch penalty if the car contacts the ball again after the first hit
  - No-touch and own-goal penalties
  Episode ends 1.5 seconds after first touch — barely enough time for ball to reach goal, no time to dribble.
"""

import math
import random
from collections import deque
from typing import Any, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POWER_SHOT_EPISODE_SECONDS = 8
POWER_SHOT_TICK_SKIP = 8
POWER_SHOT_NO_TOUCH_TIMEOUT = 6
POWER_SHOT_POST_TOUCH_SECONDS = 1.5

CAR_SPAWN_X_RANGE = 2500

BACK_WALL_Y = 5120
GOAL_CENTER_TO_POST = 892.755
SIDE_WALL_X = 4096
BALL_MAX_SPEED = 6000
ORANGE_GOAL_Y = BACK_WALL_Y
GRAVITY = 650.0

_DPS = 120.0 / POWER_SHOT_TICK_SKIP
_POST_TOUCH_STEPS = int(POWER_SHOT_POST_TOUCH_SECONDS * _DPS)  # 45 steps = 3s

# ---------------------------------------------------------------------------
# Stage progression
# ---------------------------------------------------------------------------
STAGE_ADVANCE_WINDOW = 100  # rolling window in episodes

# Per-worker module-level state (each worker process has its own copy)
_worker_goal_history: deque = deque(maxlen=STAGE_ADVANCE_WINDOW)
_worker_current_stage: int = 0


def _record_episode_outcome(scored: bool) -> None:
    """Called at end of each episode. Advances stage when rolling goal rate clears threshold."""
    global _worker_current_stage
    _worker_goal_history.append(1 if scored else 0)
    if (
        len(_worker_goal_history) >= STAGE_ADVANCE_WINDOW
        and _worker_current_stage < len(_CURRICULUM) - 1
    ):
        goal_rate = sum(_worker_goal_history) / len(_worker_goal_history)
        advance_rate = _CURRICULUM[_worker_current_stage]["advance_rate"]
        if goal_rate >= advance_rate:
            _worker_current_stage += 1
            _worker_goal_history.clear()


def _get_stage():
    return _CURRICULUM[_worker_current_stage]


# ---------------------------------------------------------------------------
# Curriculum
#
# spawn keys:
#   angle         — half-arc from car→goal direction (radians)
#   d_min/d_max   — ball distance ahead of car (uu)
#   h_min/h_max   — ball start height (uu); 93 = ground; >93 = dropped from height
#   ground_frac   — fraction of episodes where ball starts on the ground (z=93)
#   car_y_min/max — car spawn Y range
#   car_yaw_noise — max yaw deviation from facing goal (radians)
#
# reward keys:
#   approach_w        — inverse-distance pre-touch weight (small; anti-hack)
#   first_touch       — one-time bonus on first contact
#   hit_direction_w   — one-time bonus at touch: w × max(0, dot(ball_vel, goal_dir)/6000)
#                       rewards hitting hard AND toward goal in one shot
#   goal_base         — flat terminal bonus on scoring in orange goal
#   shot_speed_w      — terminal × ball_speed/6000 bonus on goal
#   second_touch_pen  — penalty if car contacts ball again after first hit
#   no_touch_pen      — penalty when episode ends without any touch
#   own_goal_pen      — penalty when bot scores in own goal
#
# advance_rate: rolling goal rate needed to advance to the next stage
# ---------------------------------------------------------------------------
_CURRICULUM = [
    # Stage 0: Point-blank stationary ground ball.
    # Car close to goal, ball straight ahead — a random forward drive can score.
    {
        "spawn": dict(
            angle=math.radians(5),
            d_min=300, d_max=500,
            h_min=93, h_max=93,   # ground only
            ground_frac=1.0,
            car_y_min=2500, car_y_max=3500,
            car_yaw_noise=math.radians(5),
        ),
        "reward": dict(
            approach_w=0.05, first_touch=2.0, hit_direction_w=8.0,
            goal_base=10.0, shot_speed_w=3.0,
            second_touch_pen=-2.0, no_touch_pen=-2.0, own_goal_pen=-1.0,
        ),
        "advance_rate": 0.30,
    },

    # Stage 1: Medium-range ground ball, wider angle variety.
    {
        "spawn": dict(
            angle=math.radians(15),
            d_min=400, d_max=800,
            h_min=93, h_max=93,
            ground_frac=1.0,
            car_y_min=1500, car_y_max=3000,
            car_yaw_noise=math.radians(10),
        ),
        "reward": dict(
            approach_w=0.05, first_touch=1.5, hit_direction_w=8.0,
            goal_base=10.0, shot_speed_w=5.0,
            second_touch_pen=-2.0, no_touch_pen=-2.0, own_goal_pen=-2.0,
        ),
        "advance_rate": 0.40,
    },

    # Stage 2: Ground or low-bounce ball, wider spawns.
    {
        "spawn": dict(
            angle=math.radians(25),
            d_min=500, d_max=1200,
            h_min=93, h_max=200,
            ground_frac=0.5,
            car_y_min=500, car_y_max=2500,
            car_yaw_noise=math.radians(15),
        ),
        "reward": dict(
            approach_w=0.05, first_touch=1.0, hit_direction_w=6.0,
            goal_base=8.0, shot_speed_w=8.0,
            second_touch_pen=-3.0, no_touch_pen=-2.0, own_goal_pen=-3.0,
        ),
        "advance_rate": 0.50,
    },

    # Stage 3: Full field, wider angles, harder shots. No advancement cap.
    {
        "spawn": dict(
            angle=math.radians(40),
            d_min=600, d_max=2000,
            h_min=93, h_max=300,
            ground_frac=0.4,
            car_y_min=-1000, car_y_max=2500,
            car_yaw_noise=math.radians(15),
        ),
        "reward": dict(
            approach_w=0.05, first_touch=1.0, hit_direction_w=5.0,
            goal_base=5.0, shot_speed_w=10.0,
            second_touch_pen=-3.0, no_touch_pen=-2.0, own_goal_pen=-3.0,
        ),
        "advance_rate": 1.01,  # never advances (final stage)
    },
]


# ---------------------------------------------------------------------------
# State Mutator
# ---------------------------------------------------------------------------
class PowerShotMutator:
    def apply(self, state, shared_info: dict) -> None:
        stage = _get_stage()
        spawn = stage["spawn"]

        # -- Car spawn --
        car_x = random.uniform(-CAR_SPAWN_X_RANGE, CAR_SPAWN_X_RANGE)
        car_y = random.uniform(spawn["car_y_min"], spawn["car_y_max"])
        car_z = 17.0

        yaw_noise = random.uniform(-spawn["car_yaw_noise"], spawn["car_yaw_noise"])
        car_yaw = math.pi / 2 + yaw_noise  # facing orange goal (+y)

        agent_id = list(state.cars.keys())[0]
        car = state.cars[agent_id]
        car.physics.position = np.array([car_x, car_y, car_z], dtype=np.float32)
        car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.euler_angles = np.array([0.0, car_yaw, 0.0], dtype=np.float32)
        car.boost_amount = random.uniform(50, 100) / 100.0

        # -- Ball spawn --
        ball_angle = car_yaw + random.uniform(-spawn["angle"], spawn["angle"])
        ball_dist = random.uniform(spawn["d_min"], spawn["d_max"])
        ball_x = float(np.clip(
            car_x + math.cos(ball_angle) * ball_dist,
            -SIDE_WALL_X + 200, SIDE_WALL_X - 200,
        ))
        ball_y = float(np.clip(
            car_y + math.sin(ball_angle) * ball_dist,
            -BACK_WALL_Y + 200, BACK_WALL_Y - 200,
        ))

        # Ground or dropped from height based on ground_frac
        if random.random() < spawn["ground_frac"]:
            ball_z = 93.0  # resting on ground
        else:
            ball_z = float(random.uniform(spawn["h_min"], spawn["h_max"]))

        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)  # stationary
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        shared_info["goal_pos"] = np.array([0.0, ORANGE_GOAL_Y, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------
class PowerShotReward:
    def __init__(self):
        self._has_touched: Dict[Any, bool] = {}
        self._hit_reward_given: Dict[Any, bool] = {}
        self._episode_scored: Dict[Any, bool] = {}

    def reset(self, agents, initial_state, shared_info: dict) -> None:
        for agent in agents:
            self._has_touched[agent] = False
            self._hit_reward_given[agent] = False
            self._episode_scored[agent] = False

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info: dict) -> Dict[Any, float]:
        rcfg = _get_stage()["reward"]

        goal_pos = shared_info.get("goal_pos", np.array([0.0, ORANGE_GOAL_Y, 0.0]))
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity

        goal_dir = goal_pos - ball_pos
        goal_dist = float(np.linalg.norm(goal_dir)) + 1e-6
        goal_dir_unit = goal_dir / goal_dist

        rewards = {}
        for agent in agents:
            r = 0.0
            car_pos = state.cars[agent].physics.position
            dist_to_ball = float(np.linalg.norm(car_pos - ball_pos))

            touched = self._has_touched.get(agent, False)

            # 1. Pre-touch: small inverse-distance nudge toward ball
            if not touched:
                r += rcfg["approach_w"] * max(0.0, 1.0 - dist_to_ball / 3000.0)

            # 2. Touch detection (proximity-based)
            touching_now = dist_to_ball < 200.0

            if touching_now and not touched:
                # First touch — fire one-time directional hit bonus
                self._has_touched[agent] = True
                touched = True
                r += rcfg["first_touch"]

                if not self._hit_reward_given.get(agent, False):
                    # Reward = how hard AND how on-target the hit is, in one shot.
                    # max(0,...) means sideways/backward gives nothing — no punishment for bad aim,
                    # but a clean goalward hit earns the full bonus.
                    ball_speed_toward_goal = float(np.dot(ball_vel, goal_dir_unit))
                    r += rcfg["hit_direction_w"] * max(0.0, ball_speed_toward_goal / BALL_MAX_SPEED)
                    self._hit_reward_given[agent] = True

            elif touching_now and touched and self._hit_reward_given.get(agent, False):
                # Second (or later) touch — penalise to discourage dribbling/follow-up hits
                r += rcfg["second_touch_pen"]

            # 3. Goal / own goal (terminal)
            if is_terminated.get(agent, False) and getattr(state, "goal_scored", False):
                if getattr(state, "scoring_team", -1) == 0:
                    ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
                    r += rcfg["goal_base"] + rcfg["shot_speed_w"] * (ball_speed / BALL_MAX_SPEED)
                    self._episode_scored[agent] = True
                else:
                    r += rcfg["own_goal_pen"]

            # 4. No-touch penalty
            episode_done = is_terminated.get(agent, False) or is_truncated.get(agent, False)
            if episode_done and not self._has_touched.get(agent, False):
                r += rcfg["no_touch_pen"]

            # 5. Record outcome for stage progression
            if episode_done:
                _record_episode_outcome(self._episode_scored.get(agent, False))

            rewards[agent] = r

        return rewards


# ---------------------------------------------------------------------------
# Done Condition
# ---------------------------------------------------------------------------
class PowerShotTerminationCondition:
    def __init__(self):
        self._has_touched: Dict[Any, bool] = {}
        self._post_touch_steps: Dict[Any, int] = {}

    def reset(self, agents, initial_state, shared_info: dict) -> None:
        for agent in agents:
            self._has_touched[agent] = False
            self._post_touch_steps[agent] = 0

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
            elif self._has_touched.get(agent, False):
                self._post_touch_steps[agent] = self._post_touch_steps.get(agent, 0) + 1
                if self._post_touch_steps[agent] >= _POST_TOUCH_STEPS:
                    done = True
                elif ball_speed < 50.0 and ball_pos[2] < 150.0:
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
