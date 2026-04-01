from typing import Any, Dict, List, Optional

import numpy as np

from rlgym.api import AgentID, RewardFunction, StateMutator
from rlgym.api.config.done_condition import DoneCondition
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.obs_builders import DefaultObs

DRIBBLE_EPISODE_SECONDS = 30
DRIBBLE_TICK_SKIP = 8
DECISIONS_PER_SECOND = common_values.TICKS_PER_SECOND / DRIBBLE_TICK_SKIP

SAFE_FIELD_X = common_values.SIDE_WALL_X - 900.0
SAFE_FIELD_Y = common_values.BACK_WALL_Y - 1300.0
LOOP_SAFE_X = common_values.SIDE_WALL_X - 1550.0
LOOP_SAFE_Y = common_values.BACK_WALL_Y - 2100.0
LOOP_WARNING_X = common_values.SIDE_WALL_X - 1950.0
LOOP_WARNING_Y = common_values.BACK_WALL_Y - 2550.0

EASY_CURRICULUM_EPISODES = 2500
MIXED_CURRICULUM_EPISODES = 8000

BALL_GROUND_THRESHOLD = 100.0
BREADCRUMB_REACHED_DISTANCE = 500.0
BREADCRUMB_SUCCESS_BONUS = 1.4
BREADCRUMB_PROGRESS_REWARD_SCALE = 1 / 260.0
WALL_PENALTY_SCALE = 0.035
CORNER_PENALTY_SCALE = 0.06
WALL_APPROACH_PENALTY_SCALE = 0.09
SPEED_COMFORT_CENTER = 450.0   # uu/s — ideal dribble speed; peak of the comfort reward
SPEED_COMFORT_SIGMA = 350.0    # Gaussian half-width; ~700 uu/s yields ~75% of peak
SPEED_COMFORT_SCALE = 0.10     # per-step reward at peak (same weight as carry quality)
CARRY_THRESHOLD = 0.25

BOOST_CONSERVATION_SCALE = 0.03  # per-step reward at full boost while carrying
BOOST_PICKUP_REWARD = 0.10       # one-time bonus when a pad is collected
# Threshold to detect a pad pickup: boost increase > expected max consumption per step.
# At boost_consumption=1.0: 33.3 * (8/120) ≈ 2.2 boost/step, so 4.0 is a safe floor.
BOOST_PICKUP_DETECT_THRESHOLD = 4.0

GOAL_MOUTH_X_LIMIT = common_values.GOAL_CENTER_TO_POST + 140.0
GOAL_MOUTH_Y_LIMIT = common_values.BACK_WALL_Y - 120.0
BALL_FORWARD_OFFSET = 35.0
BALL_UP_OFFSET = 141.0

# Phase A: first N breadcrumbs follow the ellipse loop; after that, random field targets
LOOP_BREADCRUMB_COUNT = 5

# Phase B: random field waypoints (crumbs 6+)
RANDOM_FIELD_SAFE_X = common_values.SIDE_WALL_X - 1000.0
RANDOM_FIELD_SAFE_Y = common_values.BACK_WALL_Y - 1400.0
RANDOM_TARGET_MIN_DISTANCE = 1500.0
RANDOM_TARGET_MAX_DISTANCE = 4000.0
RANDOM_TARGET_MAX_ATTEMPTS = 30

# Breadcrumb types and their arc parameters: (arc_step_min, arc_step_max, arc_step_mid, min_distance, max_distance)
# Loop-phase types cover ~2π total: first(0.06) + moderate(0.45) + hard_turn(1.48) + hard_turn(1.48) + closing(2.59) ≈ 6.06
_BREADCRUMB_ARC_PARAMS = {
    "first":     (0.01, 0.08, 0.04,  350.0,  600.0),   # crumb 1: close, nearly straight ahead
    "moderate":  (0.30, 0.60, 0.45,  800.0, 1600.0),   # crumb 2: close along the circle after entering
    "hard_turn": (1.20, 1.75, 1.48, 2200.0, 3800.0),   # crumbs 3-4: sharp turns
    "closing":   (2.09, 3.09, 2.59, 2000.0, 4500.0),   # crumb 5: large arc to close the loop
    "gentle":    (0.12, 0.25, 0.19,  800.0, 1600.0),   # legacy
    "sharp":     (0.50, 0.85, 0.68, 1200.0, 2800.0),   # legacy
    "straight":  (0.02, 0.08, 0.05,  600.0, 1400.0),   # legacy
    "flip":      (0.12, 0.25, 0.19,  800.0, 1600.0),   # legacy
}

CURRENT_TURN_TARGET_XY = None
CURRENT_LOOP_STATE = None
REACHED_BREADCRUMB_POSITIONS: List[np.ndarray] = []


class DribbleStartMutator(StateMutator[GameState]):
    """
    Spawn a single car on an interior loop lane with the ball balanced on its hood.
    """

    def __init__(self):
        self.rng = np.random.default_rng()
        self.episode_counter = 0

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        self.episode_counter += 1

        reset_reached_breadcrumb_positions()
        difficulty = sample_waypoint_difficulty(self.episode_counter, self.rng)
        lane_scale = sample_lane_scale(difficulty, self.rng)
        turn_direction = 1.0 if self.episode_counter % 2 == 1 else -1.0
        profile = waypoint_profile(difficulty, lane_scale)

        spawn_angle = float(self.rng.uniform(-np.pi, np.pi))

        # First crumb is the tangent point on the ellipse.
        # Car spawns outside the ellipse, approaching along the tangent line.
        tangent_point = ellipse_point(spawn_angle, profile["loop_x"], profile["loop_y"])
        tangent_dir = ellipse_tangent(spawn_angle, profile["loop_x"], profile["loop_y"], turn_direction)
        tangent_angle = float(np.arctan2(float(tangent_dir[1]), float(tangent_dir[0])))

        # Cap spawn_distance so the car stays on the tangent line without needing to clamp position.
        max_d = _max_safe_spawn_distance(tangent_point, tangent_dir, SAFE_FIELD_X, SAFE_FIELD_Y)
        spawn_distance = min(float(self.rng.uniform(500.0, 1500.0)), max_d - 50.0)
        spawn_distance = max(spawn_distance, 200.0)
        spawn_xy = (tangent_point - tangent_dir * spawn_distance).astype(np.float32)

        # Velocity is exactly along the tangent so the car travels straight toward the first waypoint.
        # Facing has small yaw noise so the car starts slightly angled — adds variety without breaking the line.
        forward = np.array([np.cos(tangent_angle), np.sin(tangent_angle), 0.0], dtype=np.float32)
        right = np.array([-np.sin(tangent_angle), np.cos(tangent_angle), 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        yaw_noise = float(self.rng.uniform(-profile["yaw_noise"], profile["yaw_noise"]))
        facing_angle = tangent_angle + yaw_noise
        car_speed = float(self.rng.uniform(profile["speed_min"], profile["speed_max"]))

        target_xy = tangent_point.copy()
        target_angle = spawn_angle

        state.config.boost_consumption = 1.0  # full drain; model must actively manage boost

        for index, car in enumerate(state.cars.values()):
            car.physics.position = np.array([spawn_xy[0], spawn_xy[1], 17.0], dtype=np.float32)
            car.physics.linear_velocity = forward * car_speed  # along exact tangent
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0.0, facing_angle, 0.0], dtype=np.float32)
            car.boost_amount = 100.0

            if index != 0:
                continue

            shared_info["dribble_turn_target_xy"] = target_xy
            shared_info["dribble_turn_direction"] = turn_direction
            shared_info["dribble_waypoint_difficulty"] = difficulty
            shared_info["dribble_waypoint_index"] = 0
            shared_info["dribble_loop_lane_scale"] = lane_scale
            shared_info["dribble_loop_profile"] = profile
            shared_info["dribble_loop_target_angle"] = target_angle
            shared_info["dribble_breadcrumbs_reached"] = 0
            shared_info["dribble_direction_flips"] = 0
            shared_info["dribble_episode_reward_sum"] = 0.0

            set_current_loop_state(
                {
                    "turn_direction": turn_direction,
                    "difficulty": difficulty,
                    "lane_scale": lane_scale,
                    "loop_x": profile["loop_x"],
                    "loop_y": profile["loop_y"],
                    "safe_x": profile["safe_x"],
                    "safe_y": profile["safe_y"],
                    "target_angle": target_angle,
                    "target_xy": target_xy,
                    "breadcrumb_index": 0,
                    "breadcrumbs_reached": 0,
                    "direction_flips": 0,
                    "episode_reward_sum": 0.0,
                }
            )

            forward_noise = float(self.rng.uniform(-5.0, 5.0))
            lateral_noise = float(self.rng.uniform(-4.0, 4.0))
            vertical_noise = float(self.rng.uniform(-4.0, 4.0))
            ball_velocity_noise = self.rng.uniform(-15.0, 15.0, size=3).astype(np.float32)

            state.ball.position = (
                car.physics.position
                + forward * (BALL_FORWARD_OFFSET + forward_noise)
                + right * lateral_noise
                + up * (BALL_UP_OFFSET + vertical_noise)
            ).astype(np.float32)
            state.ball.linear_velocity = (
                car.physics.linear_velocity
                + ball_velocity_noise
                + np.array([0.0, 0.0, -15.0], dtype=np.float32)
            ).astype(np.float32)
            state.ball.angular_velocity = self.rng.uniform(-0.3, 0.3, size=3).astype(np.float32)


class BallDroppedCondition(DoneCondition[AgentID, GameState]):
    """
    End the episode when the ball touches the ground or enters a goal.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        ball_pos = state.ball.position
        done = (
            ball_pos[2] < BALL_GROUND_THRESHOLD
            or is_in_goal_mouth(ball_pos)
            or (bool(agents) and is_in_goal_mouth(state.cars[agents[0]].physics.position))
        )
        return {agent: done for agent in agents}


N_WAYPOINT_OBS = 4  # dx_norm, dy_norm, bearing_cos, bearing_sin
N_EXTRA_OBS = 5    # waypoint (4) + carry_quality (1)


class DribbleObs(DefaultObs):
    """
    DefaultObs extended with 5 extra features appended at the end:
      - dx_norm, dy_norm: waypoint position relative to car, normalised by BACK_NET_Y
      - bearing_cos, bearing_sin: how directly ahead the waypoint is (dot/cross with car forward)
      - carry_quality: pre-computed hood-alignment score in [0, 1]
    All waypoint features are zero when no waypoint is active.
    """

    def get_obs_space(self, agent):
        obs_type, size = super().get_obs_space(agent)
        return obs_type, size + N_EXTRA_OBS

    def build_obs(self, agents, state, shared_info):
        base_obs = super().build_obs(agents, state, shared_info)
        turn_target = shared_info.get("dribble_turn_target_xy")
        ball_pos = state.ball.position
        for agent in agents:
            car = state.cars[agent]
            if turn_target is not None:
                target_xy = np.asarray(turn_target, dtype=np.float32)
                diff = (target_xy - car.physics.position[:2]).astype(np.float32)
                dist = float(np.linalg.norm(diff))
                dx_norm = float(diff[0]) / common_values.BACK_NET_Y
                dy_norm = float(diff[1]) / common_values.BACK_NET_Y
                dir_xy = (diff / dist) if dist > 1e-6 else car.physics.forward[:2]
                fwd = car.physics.forward[:2]
                bearing_cos = float(np.dot(fwd, dir_xy))
                bearing_sin = float(fwd[0] * float(dir_xy[1]) - fwd[1] * float(dir_xy[0]))
            else:
                dx_norm = dy_norm = bearing_cos = bearing_sin = 0.0
            _, _, _, carry_quality = get_dribble_alignment(car, ball_pos)
            extra_feats = np.array(
                [dx_norm, dy_norm, bearing_cos, bearing_sin, carry_quality], dtype=np.float32
            )
            base_obs[agent] = np.concatenate([base_obs[agent], extra_feats])
        return base_obs


class DribbleCarryReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward ball control and breadcrumb completion. Ball hitting the ground ends
    the episode with a penalty. Breadcrumbs provide the primary positive signal.
    """

    def __init__(self):
        self.prev_positions: Dict[AgentID, np.ndarray] = {}
        self.prev_target_distances: Dict[AgentID, float] = {}
        self.prev_boost: Dict[AgentID, float] = {}
        self.rng = np.random.default_rng()

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_positions = {}
        self.prev_target_distances = {}
        self.prev_boost = {agent: initial_state.cars[agent].boost_amount for agent in agents}

        turn_target = shared_info.get("dribble_turn_target_xy")
        for agent in agents:
            car = initial_state.cars[agent]
            self.prev_positions[agent] = car.physics.position[:2].copy()
            if turn_target is not None:
                self.prev_target_distances[agent] = float(
                    np.linalg.norm(np.asarray(turn_target) - car.physics.position[:2])
                )

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position

        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position[:2].copy()
            _, _, _, carry_quality = get_dribble_alignment(car, ball_pos)

            # 1. Carry quality: small dense signal so the bot learns to keep the ball up
            reward = 0.10 * carry_quality

            carrying = carry_quality > CARRY_THRESHOLD
            if carrying:
                # 2. Breadcrumb approach + success (handles direction flips internally)
                reward += self._breadcrumb_rewards(agent, car, car_pos, carry_quality, shared_info)

                # 3. Speed comfort zone: Gaussian bell centered at SPEED_COMFORT_CENTER.
                # Rewards moderate dribbling speed; decays toward 0 at high speed so the
                # agent learns to slow down for turns instead of always sprinting.
                speed_xy = float(np.linalg.norm(car.physics.linear_velocity[:2]))
                speed_factor = float(np.exp(-((speed_xy - SPEED_COMFORT_CENTER) / SPEED_COMFORT_SIGMA) ** 2))
                reward += SPEED_COMFORT_SCALE * carry_quality * speed_factor

                # 4. Wall penalties
                wall_pressure = wall_pressure_score(car_pos)
                corner_pressure = corner_pressure_score(car_pos)
                wall_approach = wall_approach_score(car_pos, car.physics.forward[:2], car.physics.linear_velocity[:2])
                reward -= WALL_PENALTY_SCALE * carry_quality * wall_pressure
                reward -= CORNER_PENALTY_SCALE * carry_quality * corner_pressure
                reward -= WALL_APPROACH_PENALTY_SCALE * carry_quality * wall_approach

                # 5. Boost conservation: reward for keeping boost high while carrying
                reward += BOOST_CONSERVATION_SCALE * carry_quality * (car.boost_amount / 100.0)
            else:
                self.prev_target_distances.pop(agent, None)

            # 6. Boost pickup: one-time bonus when a pad is collected
            boost_delta = car.boost_amount - self.prev_boost.get(agent, car.boost_amount)
            if boost_delta > BOOST_PICKUP_DETECT_THRESHOLD:
                reward += BOOST_PICKUP_REWARD

            self.prev_boost[agent] = car.boost_amount

            # 7. Terminal penalty
            if is_terminated.get(agent, False) or is_truncated.get(agent, False):
                reward -= 1.0

            shared_info["dribble_episode_reward_sum"] = (
                shared_info.get("dribble_episode_reward_sum", 0.0) + reward
            )
            rewards[agent] = float(reward)

        # Mirror reward sum into loop state for metrics
        loop_state = get_current_loop_state()
        if loop_state is not None:
            loop_state["episode_reward_sum"] = shared_info.get("dribble_episode_reward_sum", 0.0)
            set_current_loop_state(loop_state)

        self.prev_positions = {a: state.cars[a].physics.position[:2].copy() for a in agents}
        return rewards

    def _breadcrumb_rewards(
        self,
        agent: AgentID,
        car,
        car_pos: np.ndarray,
        carry_quality: float,
        shared_info: Dict[str, Any],
    ) -> float:
        turn_target = shared_info.get("dribble_turn_target_xy")
        if turn_target is None:
            return 0.0

        difficulty = shared_info.get("dribble_waypoint_difficulty", "medium")
        turn_direction = float(shared_info.get("dribble_turn_direction", 1.0))
        lane_scale = float(shared_info.get("dribble_loop_lane_scale", 1.0))
        breadcrumbs_reached = int(shared_info.get("dribble_breadcrumbs_reached", 0))

        target_xy = np.asarray(turn_target, dtype=np.float32)
        target_distance = float(np.linalg.norm(target_xy - car_pos))
        prev_target_distance = self.prev_target_distances.get(agent, target_distance)
        progress = max(prev_target_distance - target_distance, 0.0)
        self.prev_target_distances[agent] = target_distance

        # Breadcrumb approach reward
        reward = carry_quality * min(
            progress * BREADCRUMB_PROGRESS_REWARD_SCALE * breadcrumb_progress_scale(difficulty), 1.0
        )

        if target_distance <= BREADCRUMB_REACHED_DISTANCE:
            reward += breadcrumb_success_bonus(difficulty)
            append_reached_breadcrumb_position(car_pos.copy())
            breadcrumbs_reached += 1
            direction_flips = int(shared_info.get("dribble_direction_flips", 0))
            next_waypoint_index = int(shared_info.get("dribble_waypoint_index", 0)) + 1

            if breadcrumbs_reached < LOOP_BREADCRUMB_COUNT:
                # Phase A: still on the ellipse loop
                next_type = breadcrumb_type_for_index(breadcrumbs_reached)
                next_target_xy, next_target_angle, next_profile = build_turn_target_xy(
                    car_pos,
                    car.physics.forward[:2],
                    car.physics.right[:2],
                    turn_direction,
                    self.rng,
                    difficulty=difficulty,
                    lane_scale=lane_scale,
                    current_target_angle=float(shared_info.get("dribble_loop_target_angle", 0.0)),
                    breadcrumb_type=next_type,
                )
                shared_info["dribble_turn_target_xy"] = next_target_xy
                shared_info["dribble_waypoint_index"] = next_waypoint_index
                shared_info["dribble_loop_profile"] = next_profile
                shared_info["dribble_loop_target_angle"] = next_target_angle
                shared_info["dribble_breadcrumbs_reached"] = breadcrumbs_reached
                set_current_loop_state({
                    "turn_direction": turn_direction,
                    "difficulty": difficulty,
                    "lane_scale": lane_scale,
                    "loop_x": next_profile["loop_x"],
                    "loop_y": next_profile["loop_y"],
                    "safe_x": next_profile["safe_x"],
                    "safe_y": next_profile["safe_y"],
                    "target_angle": next_target_angle,
                    "target_xy": next_target_xy,
                    "breadcrumb_index": next_waypoint_index,
                    "breadcrumbs_reached": breadcrumbs_reached,
                    "direction_flips": direction_flips,
                    "episode_reward_sum": shared_info.get("dribble_episode_reward_sum", 0.0),
                })
            else:
                # Phase B: random field waypoints, biased toward car's forward arc
                next_target_xy = build_random_field_target(
                    car_pos, self.rng, car_forward_xy=car.physics.forward[:2]
                )
                shared_info["dribble_turn_target_xy"] = next_target_xy
                shared_info["dribble_waypoint_index"] = next_waypoint_index
                shared_info["dribble_breadcrumbs_reached"] = breadcrumbs_reached
                set_current_loop_state({
                    "turn_direction": turn_direction,
                    "difficulty": difficulty,
                    "lane_scale": lane_scale,
                    "loop_x": 0.0,
                    "loop_y": 0.0,
                    "safe_x": RANDOM_FIELD_SAFE_X,
                    "safe_y": RANDOM_FIELD_SAFE_Y,
                    "target_angle": 0.0,
                    "target_xy": next_target_xy,
                    "breadcrumb_index": next_waypoint_index,
                    "breadcrumbs_reached": breadcrumbs_reached,
                    "direction_flips": direction_flips,
                    "episode_reward_sum": shared_info.get("dribble_episode_reward_sum", 0.0),
                })

            self.prev_target_distances[agent] = float(np.linalg.norm(next_target_xy - car_pos))

        return reward


def get_dribble_alignment(car, ball_pos):
    relative = ball_pos - car.physics.position
    forward_offset = float(np.dot(relative, car.physics.forward))
    lateral_offset = float(np.dot(relative, car.physics.right))
    vertical_offset = float(np.dot(relative, car.physics.up))

    hood_alignment = np.exp(-((forward_offset - 35.0) / 45.0) ** 2)
    lateral_alignment = np.exp(-(lateral_offset / 50.0) ** 2)
    height_alignment = np.exp(-((vertical_offset - 140.0) / 28.0) ** 2)
    carry_quality = float(hood_alignment * lateral_alignment * height_alignment)
    return forward_offset, lateral_offset, vertical_offset, carry_quality


def build_spawn_position(spawn_angle: float, profile: Dict[str, float], rng) -> np.ndarray:
    base_xy = ellipse_point(spawn_angle, profile["loop_x"], profile["loop_y"])
    normal = safe_normalize(np.array(
        [base_xy[0] / max(profile["loop_x"], 1.0), base_xy[1] / max(profile["loop_y"], 1.0)],
        dtype=np.float32,
    ))
    tangent = ellipse_tangent(spawn_angle, profile["loop_x"], profile["loop_y"], 1.0)
    spawn_xy = base_xy.copy()
    spawn_xy += normal * float(rng.uniform(-profile["spawn_normal_noise"], profile["spawn_normal_noise"]))
    spawn_xy += tangent * float(rng.uniform(-profile["spawn_tangent_noise"], profile["spawn_tangent_noise"]))
    spawn_xy[0] = float(np.clip(spawn_xy[0], -profile["safe_x"], profile["safe_x"]))
    spawn_xy[1] = float(np.clip(spawn_xy[1], -profile["safe_y"], profile["safe_y"]))
    return spawn_xy.astype(np.float32)


def build_turn_target_xy(
    car_pos_xy,
    forward_xy,
    right_xy,
    turn_direction,
    rng,
    difficulty="medium",
    lane_scale=1.0,
    current_target_angle=None,
    breadcrumb_type="gentle",
):
    car_pos_xy = np.asarray(car_pos_xy, dtype=np.float32)
    forward_xy = safe_normalize(forward_xy)
    _ = right_xy
    profile = waypoint_profile(difficulty, lane_scale)
    arc_min, arc_max, arc_mid, dist_min, dist_max = breadcrumb_type_arc_params(breadcrumb_type)

    base_angle = ellipse_angle(car_pos_xy, profile["loop_x"], profile["loop_y"])
    if current_target_angle is not None:
        base_angle = float(current_target_angle)

    best_target: Optional[np.ndarray] = None
    best_angle: Optional[float] = None
    best_score = -np.inf
    for delta in np.linspace(arc_min, arc_max, 6):
        target_angle = wrap_angle(base_angle + float(turn_direction) * float(delta))
        target_xy = ellipse_point(target_angle, profile["loop_x"], profile["loop_y"])
        if not breadcrumb_segment_is_safe(car_pos_xy, target_xy, profile["safe_x"], profile["safe_y"]):
            continue

        to_target = target_xy - car_pos_xy
        distance = float(np.linalg.norm(to_target))
        if distance < dist_min or distance > dist_max:
            continue

        target_dir = safe_normalize(to_target)
        tangent_dir = ellipse_tangent(target_angle, profile["loop_x"], profile["loop_y"], turn_direction)
        forward_score = float(np.dot(target_dir, forward_xy))
        tangent_score = float(np.dot(target_dir, tangent_dir))
        center_score = 1.0 - 0.5 * (
            abs(float(target_xy[0])) / max(profile["safe_x"], 1.0)
            + abs(float(target_xy[1])) / max(profile["safe_y"], 1.0)
        )
        if breadcrumb_type == "first":
            score = 2.8 * forward_score + 0.4 * tangent_score + 0.4 * center_score
        elif breadcrumb_type == "closing":
            score = 0.5 * forward_score + 1.5 * tangent_score + 0.8 * center_score
        else:
            score = 1.3 * forward_score + 0.9 * tangent_score + 0.25 * center_score
        if score > best_score:
            best_score = score
            best_target = target_xy
            best_angle = target_angle

    if best_target is None or best_angle is None:
        fallback_angle = wrap_angle(base_angle + float(turn_direction) * arc_mid)
        best_target = ellipse_point(fallback_angle, profile["loop_x"], profile["loop_y"])
        best_target[0] = float(np.clip(best_target[0], -profile["safe_x"], profile["safe_x"]))
        best_target[1] = float(np.clip(best_target[1], -profile["safe_y"], profile["safe_y"]))
        best_angle = fallback_angle

    jitter = rng.uniform(-profile["target_jitter"], profile["target_jitter"], size=2).astype(np.float32)
    target_xy = best_target + jitter
    target_xy[0] = float(np.clip(target_xy[0], -profile["safe_x"], profile["safe_x"]))
    target_xy[1] = float(np.clip(target_xy[1], -profile["safe_y"], profile["safe_y"]))

    if not breadcrumb_segment_is_safe(car_pos_xy, target_xy, profile["safe_x"], profile["safe_y"]):
        target_xy = best_target.astype(np.float32)

    return target_xy.astype(np.float32), float(best_angle), profile


def sample_breadcrumb_type(difficulty: str, breadcrumbs_reached: int, rng) -> str:
    """Sample a breadcrumb type for the next waypoint.

    Early in the episode (low breadcrumbs_reached) weights are biased toward
    gentle/straight regardless of curriculum difficulty, then interpolate to
    the full difficulty weights over the first ~7 breadcrumbs.
    Flips are gated to breadcrumbs_reached >= 3.
    """
    # [gentle, sharp, straight, flip]
    early_weights = np.array([0.55, 0.02, 0.43, 0.0], dtype=np.float64)
    if difficulty == "easy":
        full_weights = np.array([0.55, 0.05, 0.35, 0.05], dtype=np.float64)
    elif difficulty == "hard":
        full_weights = np.array([0.20, 0.30, 0.15, 0.35], dtype=np.float64)
    else:  # medium
        full_weights = np.array([0.30, 0.20, 0.20, 0.30], dtype=np.float64)

    # t=0 at breadcrumb 1, t=1 at breadcrumb 7+
    t = float(np.clip((breadcrumbs_reached - 1) / 6.0, 0.0, 1.0))
    weights = (1.0 - t) * early_weights + t * full_weights

    flip_min = 3
    if breadcrumbs_reached < flip_min:
        weights[0] += weights[3]
        weights[3] = 0.0

    weights /= weights.sum()
    return str(rng.choice(["gentle", "sharp", "straight", "flip"], p=weights))


def breadcrumb_type_arc_params(breadcrumb_type: str):
    """Return (arc_step_min, arc_step_max, arc_step_mid, min_distance, max_distance)."""
    return _BREADCRUMB_ARC_PARAMS.get(breadcrumb_type, _BREADCRUMB_ARC_PARAMS["gentle"])


def breadcrumb_type_for_index(breadcrumb_index: int) -> str:
    """Deterministic breadcrumb type for the loop phase (crumbs 1-5).

    breadcrumb_index is breadcrumbs_reached at the time the next crumb is placed,
    so 0 = placing the 1st crumb at spawn, 1 = placing crumb 2 after 1 is reached, etc.
    """
    _SCHEDULE = {0: "first", 1: "moderate", 2: "hard_turn", 3: "hard_turn", 4: "closing"}
    return _SCHEDULE.get(breadcrumb_index, "moderate")


# Minimum forward-bearing dot product for Phase B targets.
# dot(to_target, car_forward) > threshold filters out waypoints behind/beside the car.
# 0.0 = within 90° (hemisphere ahead); -0.5 = within 120°; -1.0 = unrestricted.
RANDOM_TARGET_MIN_FORWARD_DOT = -0.3  # ~107° forward arc — avoids U-turns, allows wide curves


def build_random_field_target(
    car_pos_xy: np.ndarray,
    rng,
    car_forward_xy: Optional[np.ndarray] = None,
    min_distance: float = RANDOM_TARGET_MIN_DISTANCE,
    max_distance: float = RANDOM_TARGET_MAX_DISTANCE,
    safe_x: float = RANDOM_FIELD_SAFE_X,
    safe_y: float = RANDOM_FIELD_SAFE_Y,
) -> np.ndarray:
    """Pick a random field point within safe margins at a reasonable distance from the car.

    When car_forward_xy is provided, only targets within a forward arc
    (dot product > RANDOM_TARGET_MIN_FORWARD_DOT) are accepted, preventing
    waypoints that require a U-turn while dribbling.

    Uses rejection sampling with up to RANDOM_TARGET_MAX_ATTEMPTS tries, then falls back
    to a heading-biased directional point.
    """
    car_pos_xy = np.asarray(car_pos_xy, dtype=np.float32)
    forward = safe_normalize(np.asarray(car_forward_xy, dtype=np.float32)) if car_forward_xy is not None else None

    for _ in range(RANDOM_TARGET_MAX_ATTEMPTS):
        x = float(rng.uniform(-safe_x, safe_x))
        y = float(rng.uniform(-safe_y, safe_y))
        target = np.array([x, y], dtype=np.float32)
        diff = target - car_pos_xy
        distance = float(np.linalg.norm(diff))
        if distance < min_distance or distance > max_distance:
            continue
        if not breadcrumb_segment_is_safe(car_pos_xy, target, safe_x, safe_y):
            continue
        if forward is not None:
            bearing = float(np.dot(safe_normalize(diff), forward))
            if bearing < RANDOM_TARGET_MIN_FORWARD_DOT:
                continue
        return target

    # Fallback: pick an angle biased toward the car's forward direction
    if forward is not None:
        base_angle = float(np.arctan2(float(forward[1]), float(forward[0])))
        angle = float(rng.uniform(base_angle - np.pi * 0.6, base_angle + np.pi * 0.6))
    else:
        angle = float(rng.uniform(-np.pi, np.pi))
    fallback = car_pos_xy + min_distance * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    fallback[0] = float(np.clip(fallback[0], -safe_x, safe_x))
    fallback[1] = float(np.clip(fallback[1], -safe_y, safe_y))
    return fallback.astype(np.float32)


def sample_waypoint_difficulty(episode_counter, rng):
    if episode_counter <= EASY_CURRICULUM_EPISODES:
        weights = (0.85, 0.15, 0.0)
    elif episode_counter <= MIXED_CURRICULUM_EPISODES:
        weights = (0.55, 0.35, 0.10)
    else:
        weights = (0.20, 0.50, 0.30)
    return str(rng.choice(("easy", "medium", "hard"), p=np.asarray(weights, dtype=np.float64)))


def sample_lane_scale(difficulty: str, rng) -> float:
    if difficulty == "easy":
        return float(rng.uniform(0.88, 0.95))
    if difficulty == "hard":
        return float(rng.uniform(0.98, 1.05))
    return float(rng.uniform(0.93, 1.0))


def waypoint_profile(difficulty, lane_scale=1.0):
    """Return ellipse geometry, safe bounds, spawn noise and speed params for the given difficulty."""
    difficulty = str(difficulty)
    if difficulty == "easy":
        base_safe_x = LOOP_SAFE_X - 260.0
        base_safe_y = LOOP_SAFE_Y - 320.0
        return {
            "safe_x": base_safe_x * lane_scale,
            "safe_y": base_safe_y * lane_scale,
            "loop_x": (base_safe_x - 170.0) * lane_scale,
            "loop_y": (base_safe_y - 190.0) * lane_scale,
            "target_jitter": 50.0,
            "spawn_normal_noise": 70.0,
            "spawn_tangent_noise": 110.0,
            "speed_min": 150.0,
            "speed_max": 350.0,
            "yaw_noise": 0.10,
        }
    if difficulty == "hard":
        base_safe_x = LOOP_SAFE_X
        base_safe_y = LOOP_SAFE_Y
        return {
            "safe_x": base_safe_x * min(lane_scale, 1.02),
            "safe_y": base_safe_y * min(lane_scale, 1.02),
            "loop_x": (base_safe_x - 150.0) * min(lane_scale, 1.02),
            "loop_y": (base_safe_y - 170.0) * min(lane_scale, 1.02),
            "target_jitter": 85.0,
            "spawn_normal_noise": 95.0,
            "spawn_tangent_noise": 140.0,
            "speed_min": 300.0,
            "speed_max": 550.0,
            "yaw_noise": 0.16,
        }
    # medium
    base_safe_x = LOOP_SAFE_X - 120.0
    base_safe_y = LOOP_SAFE_Y - 160.0
    return {
        "safe_x": base_safe_x * lane_scale,
        "safe_y": base_safe_y * lane_scale,
        "loop_x": (base_safe_x - 160.0) * lane_scale,
        "loop_y": (base_safe_y - 180.0) * lane_scale,
        "target_jitter": 65.0,
        "spawn_normal_noise": 80.0,
        "spawn_tangent_noise": 120.0,
        "speed_min": 200.0,
        "speed_max": 400.0,
        "yaw_noise": 0.13,
    }


def breadcrumb_progress_scale(difficulty):
    if difficulty == "easy":
        return 1.35
    if difficulty == "hard":
        return 1.0
    return 1.15


def breadcrumb_success_bonus(difficulty):
    if difficulty == "easy":
        return BREADCRUMB_SUCCESS_BONUS * 1.2
    if difficulty == "hard":
        return BREADCRUMB_SUCCESS_BONUS * 0.9
    return BREADCRUMB_SUCCESS_BONUS


def heading_angle(forward_xy) -> float:
    return float(np.arctan2(float(forward_xy[1]), float(forward_xy[0])))


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def ellipse_point(angle: float, loop_x: float, loop_y: float) -> np.ndarray:
    return np.array(
        [loop_x * np.cos(angle), loop_y * np.sin(angle)],
        dtype=np.float32,
    )


def ellipse_angle(position_xy, loop_x: float, loop_y: float) -> float:
    pos = np.asarray(position_xy, dtype=np.float32)
    if np.linalg.norm(pos) < 1e-6:
        return 0.0
    return float(np.arctan2(pos[1] / max(loop_y, 1.0), pos[0] / max(loop_x, 1.0)))


def ellipse_tangent(angle: float, loop_x: float, loop_y: float, turn_direction: float) -> np.ndarray:
    tangent = np.array(
        [-loop_x * np.sin(angle), loop_y * np.cos(angle)],
        dtype=np.float32,
    )
    if turn_direction < 0.0:
        tangent *= -1.0
    return safe_normalize(tangent)


def _max_safe_spawn_distance(tangent_point: np.ndarray, tangent_dir: np.ndarray, safe_x: float, safe_y: float) -> float:
    """Largest d >= 0 such that (tangent_point - tangent_dir * d) stays within safe bounds."""
    d_max = float("inf")
    for dim, bound in ((0, safe_x), (1, safe_y)):
        td = float(tangent_dir[dim])
        if abs(td) < 1e-9:
            continue
        tp = float(tangent_point[dim])
        # Need: -bound <= tp - td*d <= bound
        if td > 0:
            d_max = min(d_max, (tp + bound) / td)
        else:
            d_max = min(d_max, (tp - bound) / td)
    return max(0.0, d_max)


def breadcrumb_segment_is_safe(car_pos_xy, target_xy, safe_x: float, safe_y: float) -> bool:
    car_pos_xy = np.asarray(car_pos_xy, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32)
    if abs(float(target_xy[0])) > safe_x or abs(float(target_xy[1])) > safe_y:
        return False

    samples = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    segment = car_pos_xy[None, :] * (1.0 - samples[:, None]) + target_xy[None, :] * samples[:, None]
    if np.max(np.abs(segment[:, 0])) > safe_x or np.max(np.abs(segment[:, 1])) > safe_y:
        return False

    target_dir = safe_normalize(target_xy - car_pos_xy)
    outward = np.zeros(2, dtype=np.float32)
    if abs(float(car_pos_xy[0])) > safe_x - 220.0:
        outward += np.array([np.sign(car_pos_xy[0]), 0.0], dtype=np.float32)
    if abs(float(car_pos_xy[1])) > safe_y - 260.0:
        outward += np.array([0.0, np.sign(car_pos_xy[1])], dtype=np.float32)
    if np.linalg.norm(outward) < 1e-6:
        return True
    outward = safe_normalize(outward)
    return float(np.dot(target_dir, outward)) < 0.55


def reset_reached_breadcrumb_positions():
    global REACHED_BREADCRUMB_POSITIONS
    REACHED_BREADCRUMB_POSITIONS = []


def append_reached_breadcrumb_position(pos_xy: np.ndarray):
    REACHED_BREADCRUMB_POSITIONS.append(np.asarray(pos_xy, dtype=np.float32).copy())


def get_reached_breadcrumb_positions() -> List[np.ndarray]:
    return [p.copy() for p in REACHED_BREADCRUMB_POSITIONS]


def set_current_turn_target_xy(turn_target_xy):
    global CURRENT_TURN_TARGET_XY
    CURRENT_TURN_TARGET_XY = (
        None if turn_target_xy is None
        else np.asarray(turn_target_xy, dtype=np.float32).copy()
    )


def get_current_turn_target_xy():
    if CURRENT_TURN_TARGET_XY is None:
        return None
    return CURRENT_TURN_TARGET_XY.copy()


def set_current_loop_state(loop_state: Optional[Dict[str, Any]]):
    global CURRENT_LOOP_STATE
    if loop_state is None:
        CURRENT_LOOP_STATE = None
        set_current_turn_target_xy(None)
        return

    state = dict(loop_state)
    if state.get("target_xy") is not None:
        state["target_xy"] = np.asarray(state["target_xy"], dtype=np.float32).copy()
        set_current_turn_target_xy(state["target_xy"])
    else:
        set_current_turn_target_xy(None)
    CURRENT_LOOP_STATE = state


def get_current_loop_state():
    if CURRENT_LOOP_STATE is None:
        return None
    state = dict(CURRENT_LOOP_STATE)
    if state.get("target_xy") is not None:
        state["target_xy"] = np.asarray(state["target_xy"], dtype=np.float32).copy()
    return state


def safe_normalize(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def wall_pressure_score(car_pos_xy):
    x_pressure = max(0.0, (abs(float(car_pos_xy[0])) - (common_values.SIDE_WALL_X - 850.0)) / 850.0)
    y_pressure = max(0.0, (abs(float(car_pos_xy[1])) - (common_values.BACK_WALL_Y - 1100.0)) / 1100.0)
    return min(1.0, max(x_pressure, y_pressure))


def corner_pressure_score(car_pos_xy):
    x_pressure = max(0.0, (abs(float(car_pos_xy[0])) - (common_values.SIDE_WALL_X - 1100.0)) / 1100.0)
    y_pressure = max(0.0, (abs(float(car_pos_xy[1])) - (common_values.BACK_WALL_Y - 1500.0)) / 1500.0)
    return min(1.0, x_pressure * y_pressure)


def near_wall_warning_score(car_pos_xy) -> float:
    x_warning = max(0.0, (abs(float(car_pos_xy[0])) - LOOP_WARNING_X) / max(common_values.SIDE_WALL_X - LOOP_WARNING_X, 1.0))
    y_warning = max(0.0, (abs(float(car_pos_xy[1])) - LOOP_WARNING_Y) / max(common_values.BACK_WALL_Y - LOOP_WARNING_Y, 1.0))
    return float(min(1.0, max(x_warning, y_warning)))


def wall_approach_score(car_pos_xy, forward_xy, velocity_xy) -> float:
    velocity_mag = float(np.linalg.norm(velocity_xy))
    travel_dir = safe_normalize(velocity_xy if velocity_mag > 60.0 else forward_xy)
    x_warning = max(0.0, (abs(float(car_pos_xy[0])) - LOOP_WARNING_X) / max(common_values.SIDE_WALL_X - LOOP_WARNING_X, 1.0))
    y_warning = max(0.0, (abs(float(car_pos_xy[1])) - LOOP_WARNING_Y) / max(common_values.BACK_WALL_Y - LOOP_WARNING_Y, 1.0))

    score = 0.0
    if x_warning > 0.0:
        outward = np.array([np.sign(car_pos_xy[0]), 0.0], dtype=np.float32)
        score = max(score, x_warning * max(float(np.dot(travel_dir, outward)), 0.0))
    if y_warning > 0.0:
        outward = np.array([0.0, np.sign(car_pos_xy[1])], dtype=np.float32)
        score = max(score, y_warning * max(float(np.dot(travel_dir, outward)), 0.0))
    return float(min(1.0, score))


def is_in_goal_mouth(position):
    return abs(float(position[0])) <= GOAL_MOUTH_X_LIMIT and abs(float(position[1])) >= GOAL_MOUTH_Y_LIMIT
