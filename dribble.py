from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rlgym.api import AgentID, RewardFunction, StateMutator
from rlgym.api.config.done_condition import DoneCondition
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState

DRIBBLE_EPISODE_SECONDS = 20
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
BREADCRUMB_REACHED_DISTANCE = 500.0
BREADCRUMB_SUCCESS_BONUS = 1.4

STALL_RADIUS = 550.0
STALL_GRACE_STEPS = int(5 * DECISIONS_PER_SECOND)
DISTANCE_REWARD_SCALE = 1 / 150.0
STALL_PENALTY = 0.03
LOOP_PROGRESS_REWARD_SCALE = 1 / 260.0
BREADCRUMB_PROGRESS_REWARD_SCALE = 1 / 260.0
YAW_PROGRESS_REWARD_SCALE = 0.42
LOOP_TANGENT_ALIGNMENT_REWARD_SCALE = 0.09
TURN_LATERAL_REWARD_SCALE = 0.06
WALL_PENALTY_SCALE = 0.035
CORNER_PENALTY_SCALE = 0.06
WALL_APPROACH_PENALTY_SCALE = 0.09
CARRY_THRESHOLD = 0.25

GOAL_MOUTH_X_LIMIT = common_values.GOAL_CENTER_TO_POST + 140.0
GOAL_MOUTH_Y_LIMIT = common_values.BACK_WALL_Y - 120.0
BALL_FORWARD_OFFSET = 35.0
BALL_UP_OFFSET = 141.0

CURRENT_TURN_TARGET_XY = None
CURRENT_LOOP_STATE = None


class DribbleStartMutator(StateMutator[GameState]):
    """
    Spawn a single car on an interior loop lane with the ball balanced on its hood.
    """

    def __init__(self):
        self.rng = np.random.default_rng()
        self.episode_counter = 0

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        self.episode_counter += 1

        difficulty = sample_waypoint_difficulty(self.episode_counter, self.rng)
        lane_scale = sample_lane_scale(difficulty, self.rng)
        turn_direction = 1.0 if self.episode_counter % 2 == 1 else -1.0
        base_profile = waypoint_profile(difficulty, 0, lane_scale)

        spawn_angle = float(self.rng.uniform(-np.pi, np.pi))
        spawn_xy = build_spawn_position(spawn_angle, base_profile, self.rng)
        tangent_dir = ellipse_tangent(spawn_angle, base_profile["loop_x"], base_profile["loop_y"], turn_direction)
        yaw_noise = float(self.rng.uniform(-base_profile["yaw_noise"], base_profile["yaw_noise"]))
        tangent_angle = float(np.arctan2(tangent_dir[1], tangent_dir[0])) + yaw_noise

        forward = np.array([np.cos(tangent_angle), np.sin(tangent_angle), 0.0], dtype=np.float32)
        right = np.array([-np.sin(tangent_angle), np.cos(tangent_angle), 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        car_speed = float(self.rng.uniform(base_profile["speed_min"], base_profile["speed_max"]))

        target_xy, target_angle, profile = build_turn_target_xy(
            spawn_xy,
            forward[:2],
            right[:2],
            turn_direction,
            self.rng,
            difficulty=difficulty,
            chain_index=0,
            lane_scale=lane_scale,
            current_target_angle=spawn_angle,
        )

        for index, car in enumerate(state.cars.values()):
            car.physics.position = np.array([spawn_xy[0], spawn_xy[1], 17.0], dtype=np.float32)
            car.physics.linear_velocity = forward * car_speed
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0.0, tangent_angle, 0.0], dtype=np.float32)
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
                }
            )

            forward_noise = float(self.rng.uniform(-14.0, 14.0))
            lateral_noise = float(self.rng.uniform(-10.0, 10.0))
            vertical_noise = float(self.rng.uniform(-10.0, 10.0))
            ball_velocity_noise = self.rng.uniform(-40.0, 40.0, size=3).astype(np.float32)

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
            state.ball.angular_velocity = self.rng.uniform(-0.8, 0.8, size=3).astype(np.float32)


class BallDroppedCondition(DoneCondition[AgentID, GameState]):
    """
    End the episode when the ball is no longer plausibly being carried.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        ball_pos = state.ball.position
        ball_too_low = ball_pos[2] < 115.0

        done = ball_too_low or is_in_goal_mouth(ball_pos)
        if not done and agents:
            car = state.cars[agents[0]]
            relative = ball_pos - car.physics.position
            forward_offset, lateral_offset, vertical_offset, _ = get_dribble_alignment(car, ball_pos)
            distance = float(np.linalg.norm(relative))
            car_in_goal = is_in_goal_mouth(car.physics.position)

            done = (
                car_in_goal
                or distance > 260.0
                or abs(lateral_offset) > 150.0
                or forward_offset < -45.0
                or forward_offset > 195.0
                or vertical_offset < 105.0
                or vertical_offset > 260.0
            )

        return {agent: done for agent in agents}


class DribbleCarryReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward keeping the ball controlled while carrying it through loop turns.
    """

    def __init__(self):
        self.prev_positions: Dict[AgentID, np.ndarray] = {}
        self.anchor_positions: Dict[AgentID, np.ndarray] = {}
        self.stall_steps: Dict[AgentID, int] = {}
        self.prev_target_distances: Dict[AgentID, float] = {}
        self.prev_headings: Dict[AgentID, float] = {}
        self.prev_loop_angles: Dict[AgentID, float] = {}
        self.rng = np.random.default_rng()

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_positions = {}
        self.anchor_positions = {}
        self.stall_steps = {}
        self.prev_target_distances = {}
        self.prev_headings = {}
        self.prev_loop_angles = {}

        turn_target = shared_info.get("dribble_turn_target_xy")
        profile = shared_info.get("dribble_loop_profile")
        for agent in agents:
            car = initial_state.cars[agent]
            car_pos = car.physics.position[:2].copy()
            self.prev_positions[agent] = car_pos
            self.anchor_positions[agent] = car_pos
            self.stall_steps[agent] = 0
            self.prev_headings[agent] = heading_angle(car.physics.forward[:2])
            if turn_target is not None:
                self.prev_target_distances[agent] = float(np.linalg.norm(np.asarray(turn_target) - car_pos))
            if profile is not None:
                self.prev_loop_angles[agent] = ellipse_angle(car_pos, profile["loop_x"], profile["loop_y"])

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
        ball_vel = state.ball.linear_velocity

        for agent in agents:
            car = state.cars[agent]
            car_pos = car.physics.position[:2].copy()
            forward_offset, lateral_offset, _, carry_quality = get_dribble_alignment(car, ball_pos)

            forward_speed = max(float(np.dot(car.physics.linear_velocity, car.physics.forward)), 0.0)
            forward_speed_norm = min(forward_speed / 1400.0, 1.0)
            velocity_match = np.exp(-(np.linalg.norm(ball_vel - car.physics.linear_velocity) / 300.0) ** 2)
            reward = 0.10 * carry_quality + 0.55 * carry_quality * forward_speed_norm * velocity_match

            prev_pos = self.prev_positions.get(agent, car_pos)
            step_distance = float(np.linalg.norm(car_pos - prev_pos))
            self.prev_positions[agent] = car_pos

            carrying = carry_quality > CARRY_THRESHOLD
            if carrying:
                reward += carry_quality * min(step_distance * DISTANCE_REWARD_SCALE, 1.0)

                anchor_pos = self.anchor_positions.get(agent, car_pos)
                anchor_distance = float(np.linalg.norm(car_pos - anchor_pos))
                if anchor_distance <= STALL_RADIUS:
                    self.stall_steps[agent] = self.stall_steps.get(agent, 0) + 1
                else:
                    self.anchor_positions[agent] = car_pos
                    self.stall_steps[agent] = 0

                if self.stall_steps[agent] > STALL_GRACE_STEPS:
                    reward -= STALL_PENALTY

                reward += self._loop_rewards(agent, car, car_pos, forward_offset, lateral_offset, carry_quality, forward_speed_norm, shared_info)

                wall_pressure = wall_pressure_score(car_pos)
                corner_pressure = corner_pressure_score(car_pos)
                wall_approach = wall_approach_score(car_pos, car.physics.forward[:2], car.physics.linear_velocity[:2])
                reward -= WALL_PENALTY_SCALE * carry_quality * wall_pressure
                reward -= CORNER_PENALTY_SCALE * carry_quality * corner_pressure
                reward -= WALL_APPROACH_PENALTY_SCALE * carry_quality * wall_approach
            else:
                self.anchor_positions[agent] = car_pos
                self.stall_steps[agent] = 0
                self.prev_target_distances.pop(agent, None)
                self.prev_headings[agent] = heading_angle(car.physics.forward[:2])
                profile = shared_info.get("dribble_loop_profile")
                if profile is not None:
                    self.prev_loop_angles[agent] = ellipse_angle(car_pos, profile["loop_x"], profile["loop_y"])

            if is_terminated.get(agent, False) or is_truncated.get(agent, False):
                reward -= 1.0

            rewards[agent] = float(reward)

        return rewards

    def _loop_rewards(
        self,
        agent: AgentID,
        car,
        car_pos: np.ndarray,
        forward_offset: float,
        lateral_offset: float,
        carry_quality: float,
        forward_speed_norm: float,
        shared_info: Dict[str, Any],
    ) -> float:
        profile = shared_info.get("dribble_loop_profile")
        turn_target = shared_info.get("dribble_turn_target_xy")
        turn_direction = float(shared_info.get("dribble_turn_direction", 1.0))
        difficulty = shared_info.get("dribble_waypoint_difficulty", "medium")
        lane_scale = float(shared_info.get("dribble_loop_lane_scale", 1.0))
        if profile is None or turn_target is None:
            return 0.0

        reward = 0.0
        current_heading = heading_angle(car.physics.forward[:2])
        prev_heading = self.prev_headings.get(agent, current_heading)
        yaw_delta = wrap_angle(current_heading - prev_heading)
        self.prev_headings[agent] = current_heading
        reward += YAW_PROGRESS_REWARD_SCALE * carry_quality * forward_speed_norm * max(turn_direction * yaw_delta, 0.0)

        current_loop_angle = ellipse_angle(car_pos, profile["loop_x"], profile["loop_y"])
        prev_loop_angle = self.prev_loop_angles.get(agent, current_loop_angle)
        loop_delta = wrap_angle(current_loop_angle - prev_loop_angle)
        self.prev_loop_angles[agent] = current_loop_angle
        loop_radius = 0.5 * (profile["loop_x"] + profile["loop_y"])
        loop_progress = max(turn_direction * loop_delta, 0.0) * loop_radius
        reward += carry_quality * min(loop_progress * LOOP_PROGRESS_REWARD_SCALE, 1.0)

        tangent_dir = ellipse_tangent(current_loop_angle, profile["loop_x"], profile["loop_y"], turn_direction)
        tangent_alignment = max(float(np.dot(car.physics.forward[:2], tangent_dir)), 0.0)
        reward += LOOP_TANGENT_ALIGNMENT_REWARD_SCALE * carry_quality * forward_speed_norm * tangent_alignment

        desired_lateral = inside_turn_lateral_target(difficulty, turn_direction)
        turn_lateral_alignment = np.exp(-((lateral_offset - desired_lateral) / 42.0) ** 2)
        forward_alignment = np.exp(-((forward_offset - 35.0) / 55.0) ** 2)
        reward += TURN_LATERAL_REWARD_SCALE * carry_quality * forward_speed_norm * turn_lateral_alignment * forward_alignment

        target_xy = np.asarray(turn_target, dtype=np.float32)
        to_target = target_xy - car_pos
        target_distance = float(np.linalg.norm(to_target))
        prev_target_distance = self.prev_target_distances.get(agent, target_distance)
        progress = max(prev_target_distance - target_distance, 0.0)
        self.prev_target_distances[agent] = target_distance
        reward += carry_quality * min(progress * BREADCRUMB_PROGRESS_REWARD_SCALE * breadcrumb_progress_scale(difficulty), 1.0)

        if target_distance <= BREADCRUMB_REACHED_DISTANCE:
            reward += breadcrumb_success_bonus(difficulty)
            next_waypoint_index = int(shared_info.get("dribble_waypoint_index", 0)) + 1
            next_target_xy, next_target_angle, next_profile = build_turn_target_xy(
                car_pos,
                car.physics.forward[:2],
                car.physics.right[:2],
                turn_direction,
                self.rng,
                difficulty=difficulty,
                chain_index=next_waypoint_index,
                lane_scale=lane_scale,
                current_target_angle=float(shared_info.get("dribble_loop_target_angle", current_loop_angle)),
            )
            breadcrumbs_reached = int(shared_info.get("dribble_breadcrumbs_reached", 0)) + 1
            shared_info["dribble_turn_target_xy"] = next_target_xy
            shared_info["dribble_waypoint_index"] = next_waypoint_index
            shared_info["dribble_loop_profile"] = next_profile
            shared_info["dribble_loop_target_angle"] = next_target_angle
            shared_info["dribble_breadcrumbs_reached"] = breadcrumbs_reached
            set_current_loop_state(
                {
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
                }
            )
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
    normal = safe_normalize(np.array([base_xy[0] / max(profile["loop_x"], 1.0), base_xy[1] / max(profile["loop_y"], 1.0)], dtype=np.float32))
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
    chain_index=0,
    lane_scale=1.0,
    current_target_angle=None,
):
    car_pos_xy = np.asarray(car_pos_xy, dtype=np.float32)
    forward_xy = safe_normalize(forward_xy)
    _ = right_xy
    profile = waypoint_profile(difficulty, chain_index, lane_scale)
    base_angle = ellipse_angle(car_pos_xy, profile["loop_x"], profile["loop_y"])
    if current_target_angle is not None:
        base_angle = float(current_target_angle)

    best_target: Optional[np.ndarray] = None
    best_angle: Optional[float] = None
    best_score = -np.inf
    for delta in np.linspace(profile["arc_step_min"], profile["arc_step_max"], 6):
        target_angle = wrap_angle(base_angle + float(turn_direction) * float(delta))
        target_xy = ellipse_point(target_angle, profile["loop_x"], profile["loop_y"])
        if not breadcrumb_segment_is_safe(car_pos_xy, target_xy, profile["safe_x"], profile["safe_y"]):
            continue

        to_target = target_xy - car_pos_xy
        distance = float(np.linalg.norm(to_target))
        if distance < profile["min_distance"] or distance > profile["max_distance"]:
            continue

        target_dir = safe_normalize(to_target)
        tangent_dir = ellipse_tangent(target_angle, profile["loop_x"], profile["loop_y"], turn_direction)
        forward_score = float(np.dot(target_dir, forward_xy))
        tangent_score = float(np.dot(target_dir, tangent_dir))
        center_score = 1.0 - 0.5 * (
            abs(float(target_xy[0])) / max(profile["safe_x"], 1.0)
            + abs(float(target_xy[1])) / max(profile["safe_y"], 1.0)
        )
        score = 1.3 * forward_score + 0.9 * tangent_score + 0.25 * center_score
        if score > best_score:
            best_score = score
            best_target = target_xy
            best_angle = target_angle

    if best_target is None or best_angle is None:
        fallback_angle = wrap_angle(base_angle + float(turn_direction) * profile["arc_step_mid"])
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


def waypoint_profile(difficulty, chain_index, lane_scale=1.0):
    difficulty = str(difficulty)
    if difficulty == "easy":
        base_safe_x = LOOP_SAFE_X - 260.0
        base_safe_y = LOOP_SAFE_Y - 320.0
        return {
            "safe_x": base_safe_x * lane_scale,
            "safe_y": base_safe_y * lane_scale,
            "loop_x": (base_safe_x - 170.0) * lane_scale,
            "loop_y": (base_safe_y - 190.0) * lane_scale,
            "arc_step_min": 0.16 + 0.03 * min(chain_index, 4),
            "arc_step_max": 0.32 + 0.04 * min(chain_index, 4),
            "arc_step_mid": 0.26 + 0.035 * min(chain_index, 4),
            "min_distance": 850.0,
            "max_distance": 1900.0,
            "target_jitter": 50.0,
            "spawn_normal_noise": 70.0,
            "spawn_tangent_noise": 110.0,
            "speed_min": 320.0,
            "speed_max": 520.0,
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
            "arc_step_min": 0.42 + 0.05 * min(chain_index, 4),
            "arc_step_max": 0.72 + 0.06 * min(chain_index, 4),
            "arc_step_mid": 0.57 + 0.055 * min(chain_index, 4),
            "min_distance": 1400.0,
            "max_distance": 3200.0,
            "target_jitter": 85.0,
            "spawn_normal_noise": 95.0,
            "spawn_tangent_noise": 140.0,
            "speed_min": 430.0,
            "speed_max": 700.0,
            "yaw_noise": 0.16,
        }
    base_safe_x = LOOP_SAFE_X - 120.0
    base_safe_y = LOOP_SAFE_Y - 160.0
    return {
        "safe_x": base_safe_x * lane_scale,
        "safe_y": base_safe_y * lane_scale,
        "loop_x": (base_safe_x - 160.0) * lane_scale,
        "loop_y": (base_safe_y - 180.0) * lane_scale,
        "arc_step_min": 0.28 + 0.04 * min(chain_index, 4),
        "arc_step_max": 0.50 + 0.05 * min(chain_index, 4),
        "arc_step_mid": 0.39 + 0.045 * min(chain_index, 4),
        "min_distance": 1100.0,
        "max_distance": 2500.0,
        "target_jitter": 65.0,
        "spawn_normal_noise": 80.0,
        "spawn_tangent_noise": 120.0,
        "speed_min": 360.0,
        "speed_max": 610.0,
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


def inside_turn_lateral_target(difficulty: str, turn_direction: float) -> float:
    if difficulty == "easy":
        magnitude = 12.0
    elif difficulty == "hard":
        magnitude = 20.0
    else:
        magnitude = 16.0
    return -float(turn_direction) * magnitude


def heading_angle(forward_xy) -> float:
    return float(np.arctan2(float(forward_xy[1]), float(forward_xy[0])))


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def ellipse_point(angle: float, loop_x: float, loop_y: float) -> np.ndarray:
    return np.array(
        [
            loop_x * np.cos(angle),
            loop_y * np.sin(angle),
        ],
        dtype=np.float32,
    )


def ellipse_angle(position_xy, loop_x: float, loop_y: float) -> float:
    pos = np.asarray(position_xy, dtype=np.float32)
    if np.linalg.norm(pos) < 1e-6:
        return 0.0
    return float(np.arctan2(pos[1] / max(loop_y, 1.0), pos[0] / max(loop_x, 1.0)))


def ellipse_tangent(angle: float, loop_x: float, loop_y: float, turn_direction: float) -> np.ndarray:
    tangent = np.array(
        [
            -loop_x * np.sin(angle),
            loop_y * np.cos(angle),
        ],
        dtype=np.float32,
    )
    if turn_direction < 0.0:
        tangent *= -1.0
    return safe_normalize(tangent)


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


def set_current_turn_target_xy(turn_target_xy):
    global CURRENT_TURN_TARGET_XY
    CURRENT_TURN_TARGET_XY = None if turn_target_xy is None else np.asarray(turn_target_xy, dtype=np.float32).copy()


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
    outward = np.zeros(2, dtype=np.float32)
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
