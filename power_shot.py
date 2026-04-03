"""
Power Shot scenario: bot receives varied passes and is judged on shot speed into the opponent's goal.
"""

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POWER_SHOT_EPISODE_SECONDS = 12
POWER_SHOT_TICK_SKIP = 8
POWER_SHOT_NO_TOUCH_TIMEOUT = 6

# Car spawn params
CAR_SPAWN_X_RANGE = 2500      # uu
CAR_SPAWN_Y_MIN = -2000       # uu (blue side)
CAR_SPAWN_Y_MAX = 2000        # uu

# Field constants
BACK_WALL_Y = 5120
GOAL_CENTER_TO_POST = 892.755
SIDE_WALL_X = 4096
BALL_MAX_SPEED = 6000
ORANGE_GOAL_Y = BACK_WALL_Y   # orange goal at +y
GRAVITY = 650.0               # uu/s^2

# ---------------------------------------------------------------------------
# Curriculum
# Each stage: (episodes_per_worker_threshold, spawn_params_dict)
# With 4 workers, multiply by 4 to get global episode count.
# Stage 0: 0–25k per worker (~100k global) — ground rolls, straight ahead
# Stage 1: 25k–75k per worker (~100k–300k global) — widening arc, low bounces
# Stage 2: 75k+ per worker (~300k+ global) — full chaos
# ---------------------------------------------------------------------------
_CURRICULUM = [
    # threshold, angle_half_arc, height_min, height_max, speed_min, speed_max, dist_min, dist_max
    (25_000,  math.radians(40),  93,  93,   200,  700,  600,  1800),
    (75_000,  math.radians(90),  93, 300,   300, 1500,  800,  2800),
    (None,    math.pi,           93, 800,   500, 3000, 1500,  4000),
]


def _get_curriculum_params(episode_count):
    for threshold, *params in _CURRICULUM:
        if threshold is None or episode_count < threshold:
            return params
    return _CURRICULUM[-1][1:]


def _yaw_to_facing(yaw_rad):
    """Returns (x, y) unit vector for a given yaw angle."""
    return math.cos(yaw_rad), math.sin(yaw_rad)


# ---------------------------------------------------------------------------
# State Mutator
# ---------------------------------------------------------------------------
class PowerShotMutator:
    """
    Spawns a single blue car facing the orange goal (+y direction).
    Ball spawn difficulty scales with episode count via a 3-stage curriculum:
      Stage 0 (~0-100k global): ground rolls in front of car
      Stage 1 (~100k-300k global): low bouncing passes, wider arc
      Stage 2 (300k+): full chaos — any direction, height, speed
    """

    def __init__(self):
        self.episode_count = 0

    def apply(self, state, shared_info: dict) -> None:
        from rlgym.rocket_league import common_values

        self.episode_count += 1
        angle_half_arc, height_min, height_max, speed_min, speed_max, dist_min, dist_max = (
            _get_curriculum_params(self.episode_count)
        )

        # --- Car spawn ---
        car_x = random.uniform(-CAR_SPAWN_X_RANGE, CAR_SPAWN_X_RANGE)
        car_y = random.uniform(CAR_SPAWN_Y_MIN, CAR_SPAWN_Y_MAX)
        car_z = 17.0  # standard car height on ground

        # Face approximately toward orange goal (+y) with small yaw noise
        yaw_noise = random.uniform(-math.radians(15), math.radians(15))
        car_yaw = math.pi / 2 + yaw_noise  # pi/2 = facing +y

        agent_id = list(state.cars.keys())[0]
        car = state.cars[agent_id]
        car.physics.position = np.array([car_x, car_y, car_z], dtype=np.float32)
        car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.euler_angles = np.array([0.0, car_yaw, 0.0], dtype=np.float32)
        car.boost_amount = random.uniform(50, 100) / 100.0

        # --- Ball spawn ---
        # Angle sampled within arc in front of the car (relative to car yaw)
        ball_angle = car_yaw + random.uniform(-angle_half_arc, angle_half_arc)
        ball_dist = random.uniform(dist_min, dist_max)
        ball_x = np.clip(car_x + math.cos(ball_angle) * ball_dist, -SIDE_WALL_X + 200, SIDE_WALL_X - 200)
        ball_y = np.clip(car_y + math.sin(ball_angle) * ball_dist, -BACK_WALL_Y + 200, BACK_WALL_Y - 200)
        ball_z = float(random.uniform(height_min, height_max))

        ball_pos = np.array([ball_x, ball_y, ball_z], dtype=np.float32)

        # Direction from ball toward car
        dx = car_x - ball_x
        dy = car_y - ball_y
        horiz_dist = math.sqrt(dx * dx + dy * dy) + 1e-6
        speed = random.uniform(speed_min, speed_max)

        if height_min == height_max == 93:
            # Ground roll: purely horizontal, no arc
            vx = speed * dx / horiz_dist
            vy = speed * dy / horiz_dist
            vz = 0.0
        else:
            # Arcing pass: gravity-compensated so ball reaches car position
            dz = car_z - ball_z
            horiz_speed = speed * math.cos(math.atan2(abs(dz), horiz_dist))
            vx = horiz_speed * dx / horiz_dist
            vy = horiz_speed * dy / horiz_dist
            time_of_flight = horiz_dist / (horiz_speed + 1e-6)
            vz = dz / (time_of_flight + 1e-6) + 0.5 * GRAVITY * time_of_flight

        vel = np.array([vx, vy, vz], dtype=np.float32)
        vel_magnitude = np.linalg.norm(vel)
        if vel_magnitude > BALL_MAX_SPEED:
            vel = vel * (BALL_MAX_SPEED / vel_magnitude)

        state.ball.position = ball_pos
        state.ball.linear_velocity = vel
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        shared_info["goal_pos"] = np.array([0.0, ORANGE_GOAL_Y, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------
class PowerShotReward:
    """
    6-component reward:
    1. Ball velocity toward orange goal (dense directional signal)
    2. Touch bonus (0.5 per touch, 1.0 for first touch)
    3. Goal scored: shot speed bonus = 5.0 * speed / 6000
    4. Own goal penalty: -3.0 when orange scores (ball went into blue goal at -y)
    5. Timeout/truncation penalty: -0.5 when no touch in time
    6. Approach reward: 0.2 * (1 - dist/5000) while ball not yet touched
    """

    def __init__(self):
        self._prev_ball_vel_toward_goal: Dict[Any, float] = {}
        self._has_touched: Dict[Any, bool] = {}

    def reset(self, agents, initial_state, shared_info: dict) -> None:
        for agent in agents:
            self._prev_ball_vel_toward_goal[agent] = 0.0
            self._has_touched[agent] = False

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info: dict) -> Dict[Any, float]:
        goal_pos = shared_info.get("goal_pos", np.array([0.0, ORANGE_GOAL_Y, 0.0]))
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity

        # Goal direction unit vector (from ball to orange goal)
        goal_dir = goal_pos - ball_pos
        goal_dist = np.linalg.norm(goal_dir) + 1e-6
        goal_dir_unit = goal_dir / goal_dist
        ball_speed_toward_goal = float(np.dot(ball_vel, goal_dir_unit))

        rewards = {}
        for agent in agents:
            r = 0.0

            # 1. Ball velocity toward goal (normalized to [-1, 1])
            r += ball_speed_toward_goal / BALL_MAX_SPEED

            car = state.cars[agent]
            car_pos = car.physics.position

            # 2. Touch detection
            ball_to_car = np.linalg.norm(ball_pos - car_pos)
            just_touched = ball_to_car < 200.0  # rough touch detection
            if just_touched:
                if not self._has_touched.get(agent, False):
                    r += 1.0  # first touch bonus
                    self._has_touched[agent] = True
                else:
                    r += 0.5  # subsequent touch bonus

            # 3. Goal scored reward (shot speed)
            if is_terminated.get(agent, False) and state.goal_scored:
                if state.scoring_team == 0:  # blue scored in orange goal
                    ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
                    r += 5.0 * ball_speed / BALL_MAX_SPEED

            # 4. Own goal penalty (orange scored in blue goal, scoring_team == 1)
            if is_terminated.get(agent, False) and state.goal_scored:
                if state.scoring_team == 1:
                    r += -3.0

            # 5. Timeout/truncation penalty
            if is_truncated.get(agent, False) and not self._has_touched.get(agent, False):
                r += -0.5

            # 6. Approach reward (before first touch)
            if not self._has_touched.get(agent, False):
                dist_to_ball = np.linalg.norm(car_pos - ball_pos)
                approach = 0.2 * max(0.0, 1.0 - dist_to_ball / 5000.0)
                r += approach

            self._prev_ball_vel_toward_goal[agent] = ball_speed_toward_goal
            rewards[agent] = r

        return rewards


# ---------------------------------------------------------------------------
# Done Condition (termination)
# ---------------------------------------------------------------------------
class PowerShotTerminationCondition:
    """
    Terminates when:
    - A goal is scored (state.goal_scored)
    - Ball is dead: speed < 50 uu/s and z < 150 uu, after a touch
    """

    def __init__(self):
        self._has_touched: Dict[Any, bool] = {}

    def reset(self, agents, initial_state, shared_info: dict) -> None:
        for agent in agents:
            self._has_touched[agent] = False

    def is_done(self, agents, state, shared_info: dict) -> Dict[Any, bool]:
        result = {}
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity
        ball_speed = float(np.linalg.norm(ball_vel))

        for agent in agents:
            car_pos = state.cars[agent].physics.position
            dist_to_ball = np.linalg.norm(ball_pos - car_pos)
            if dist_to_ball < 200.0:
                self._has_touched[agent] = True

            done = False
            if state.goal_scored:
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
    1. goal_dir_x — unit direction from ball to orange goal (x component)
    2. goal_dir_y — unit direction from ball to orange goal (y component)
    3. ball_vel_toward_goal — dot(ball_vel, goal_dir) / BALL_MAX_SPEED
    Total: 95 features
    """

    def __init__(self, zero_padding=1, pos_coef=None, ang_coef=None, lin_vel_coef=None, ang_vel_coef=None):
        from rlgym.rocket_league.obs_builders import DefaultObs
        import numpy as np

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
        goal_dist = np.linalg.norm(goal_dir) + 1e-6
        goal_dir_unit = goal_dir / goal_dist
        ball_vel_toward_goal = float(np.dot(ball_vel, goal_dir_unit)) / BALL_MAX_SPEED

        extra = np.array([
            goal_dir_unit[0],
            goal_dir_unit[1],
            ball_vel_toward_goal,
        ], dtype=np.float32)

        for agent in agents:
            base_obs[agent] = np.concatenate([base_obs[agent], extra])
        return base_obs
