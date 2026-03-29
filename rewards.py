from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Reward proportional to the component of ball velocity directed toward the opponent's goal.
    Returns a value in [-1, 1] (normalized by max ball speed).
    """

    def __init__(self):
        self.orange_goal = np.asarray(common_values.ORANGE_GOAL_CENTER, dtype=np.float32)
        self.blue_goal = np.asarray(common_values.BLUE_GOAL_CENTER, dtype=np.float32)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_pos = state.ball.position
        ball_vel = state.ball.linear_velocity

        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Blue attacks orange goal, orange attacks blue goal
            target_goal = self.orange_goal if car.is_blue else self.blue_goal

            direction = target_goal - ball_pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            vel_toward_goal = np.dot(ball_vel, direction)
            rewards[agent] = float(vel_toward_goal / common_values.BALL_MAX_SPEED)

        return rewards
