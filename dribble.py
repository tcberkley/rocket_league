from typing import Any, Dict, List

import numpy as np

from rlgym.api import AgentID, RewardFunction, StateMutator
from rlgym.api.config.done_condition import DoneCondition
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState

DRIBBLE_EPISODE_SECONDS = 20
DRIBBLE_TICK_SKIP = 8
DECISIONS_PER_SECOND = common_values.TICKS_PER_SECOND / DRIBBLE_TICK_SKIP
STALL_RADIUS = 550.0
STALL_GRACE_STEPS = int(5 * DECISIONS_PER_SECOND)
DISTANCE_REWARD_SCALE = 1 / 150.0
STALL_PENALTY = 0.03
CARRY_THRESHOLD = 0.25
GOAL_MOUTH_X_LIMIT = common_values.GOAL_CENTER_TO_POST + 140.0
GOAL_MOUTH_Y_LIMIT = common_values.BACK_WALL_Y - 120.0


class DribbleStartMutator(StateMutator[GameState]):
    """
    Spawn a single car with the ball balanced on its hood.
    """

    def __init__(self, car_speed: float = 300.0):
        self.car_speed = car_speed

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        state.ball.position = np.array([0.0, -2465.0, 158.0], dtype=np.float32)
        state.ball.linear_velocity = np.array([0.0, self.car_speed, -15.0], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        for index, car in enumerate(state.cars.values()):
            spawn_x = float(index * 800)
            car.physics.position = np.array([spawn_x, -2500.0, 17.0], dtype=np.float32)
            car.physics.linear_velocity = np.array([0.0, self.car_speed, 0.0], dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0.0, np.pi / 2, 0.0], dtype=np.float32)
            car.boost_amount = 100.0


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
                or
                distance > 260.0
                or abs(lateral_offset) > 140.0
                or forward_offset < -40.0
                or forward_offset > 190.0
                or vertical_offset < 105.0
                or vertical_offset > 260.0
            )

        return {agent: done for agent in agents}


class DribbleCarryReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward keeping the ball centered over the hood while carrying it around the field.
    """

    def __init__(self):
        self.prev_positions: Dict[AgentID, np.ndarray] = {}
        self.anchor_positions: Dict[AgentID, np.ndarray] = {}
        self.stall_steps: Dict[AgentID, int] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_positions = {}
        self.anchor_positions = {}
        self.stall_steps = {}
        for agent in agents:
            car_pos = initial_state.cars[agent].physics.position[:2].copy()
            self.prev_positions[agent] = car_pos
            self.anchor_positions[agent] = car_pos
            self.stall_steps[agent] = 0

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
            _, _, _, carry_quality = get_dribble_alignment(car, ball_pos)

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
            else:
                self.anchor_positions[agent] = car_pos
                self.stall_steps[agent] = 0

            if is_terminated.get(agent, False) or is_truncated.get(agent, False):
                reward -= 1.0

            rewards[agent] = float(reward)

        return rewards


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


def is_in_goal_mouth(position):
    return abs(float(position[0])) <= GOAL_MOUTH_X_LIMIT and abs(float(position[1])) >= GOAL_MOUTH_Y_LIMIT
