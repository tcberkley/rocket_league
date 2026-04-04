import argparse
import time
from pathlib import Path

import numpy as np
import torch

from learner import (
    POLICY_LAYER_SIZES,
    ROOT_DIR,
    DEFAULT_N_PROC,
    DEFAULT_SAVE_EVERY_TS,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_TIMESTEP_LIMIT,
    DEFAULT_TRAIN_SEGMENT_SECONDS,
    migrate_dribble_carry_quality_obs,
    prepare_runtime_locale,
    resolve_checkpoint_folder,
    run_learner,
)


def get_models_dir(scenario: str) -> Path:
    if scenario == "standard":
        return ROOT_DIR / "models"
    if scenario == "dribble":
        return ROOT_DIR / "models-dribble"
    if scenario == "power_shot":
        return ROOT_DIR / "models-power-shot"
    if scenario == "catch_ball":
        return ROOT_DIR / "models-catch-ball"
    raise ValueError(f"Unknown scenario: {scenario}")


def build_standard_rlgym_v2_env(render=False):
    # All imports must stay inside this function because training workers pickle it.
    prepare_runtime_locale()
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
    from rlgym_ppo.util import RLGymV2GymWrapper

    from rewards import VelocityBallToGoalReward

    tick_skip = 8  # 120Hz physics / 8 = 15 decisions/sec
    renderer = None

    if render:
        import os
        import sys

        rlviser_binary = os.path.join(os.getcwd(), "rlviser")
        if not os.path.exists(rlviser_binary):
            raise RuntimeError(
                "Rendering requires the RLViser executable named `rlviser` in the repo root. "
                "The Python package is installed, but it launches the viewer binary from the current working directory."
            )

        if sys.platform == "darwin":
            with open(rlviser_binary, "rb") as handle:
                magic = handle.read(4)
            mach_o_magics = {
                b"\xcf\xfa\xed\xfe",
                b"\xfe\xed\xfa\xcf",
                b"\xca\xfe\xba\xbe",
                b"\xbe\xba\xfe\xca",
            }
            if magic not in mach_o_magics:
                raise RuntimeError(
                    "RLViser upstream does not ship a macOS viewer binary. "
                    "Use the default `sandbox` renderer on macOS, or build RLViser locally and place the executable here."
                )

        try:
            from rlgym.rocket_league.rlviser import RLViserRenderer
        except ImportError as exc:
            raise RuntimeError(
                "Rendering requires RLViser. Reinstall dependencies with `pip install -r requirements.txt`."
            ) from exc
        renderer = RLViserRenderer(tick_rate=120 / tick_skip)

    rlgym_env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator(),
        ),
        obs_builder=DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray(
                [
                    1 / common_values.SIDE_WALL_X,
                    1 / common_values.BACK_NET_Y,
                    1 / common_values.CEILING_Z,
                ]
            ),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        ),
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        reward_fn=CombinedReward(
            (VelocityBallToGoalReward(), 1.0),
            (TouchReward(), 0.5),
        ),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=20),
            TimeoutCondition(timeout_seconds=300),
        ),
        transition_engine=RocketSimEngine(),
        renderer=renderer,
    )

    return RLGymV2GymWrapper(rlgym_env)


def build_dribble_rlgym_v2_env(render=False):
    # All imports must stay inside this function because training workers pickle it.
    prepare_runtime_locale()
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import AnyCondition, TimeoutCondition
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence
    from rlgym_ppo.util import RLGymV2GymWrapper

    from dribble import BallDroppedCondition, DribbleCarryReward, DribbleObs, DribbleStartMutator, DRIBBLE_EPISODE_SECONDS, DRIBBLE_TICK_SKIP

    tick_skip = DRIBBLE_TICK_SKIP
    renderer = None

    if render:
        import os
        import sys

        rlviser_binary = os.path.join(os.getcwd(), "rlviser")
        if not os.path.exists(rlviser_binary):
            raise RuntimeError(
                "Rendering requires the RLViser executable named `rlviser` in the repo root. "
                "The Python package is installed, but it launches the viewer binary from the current working directory."
            )

        if sys.platform == "darwin":
            with open(rlviser_binary, "rb") as handle:
                magic = handle.read(4)
            mach_o_magics = {
                b"\xcf\xfa\xed\xfe",
                b"\xfe\xed\xfa\xcf",
                b"\xca\xfe\xba\xbe",
                b"\xbe\xba\xfe\xca",
            }
            if magic not in mach_o_magics:
                raise RuntimeError(
                    "RLViser upstream does not ship a macOS viewer binary. "
                    "Use the default `sandbox` renderer on macOS, or build RLViser locally and place the executable here."
                )

        try:
            from rlgym.rocket_league.rlviser import RLViserRenderer
        except ImportError as exc:
            raise RuntimeError(
                "Rendering requires RLViser. Reinstall dependencies with `pip install -r requirements.txt`."
            ) from exc
        renderer = RLViserRenderer(tick_rate=120 / tick_skip)

    rlgym_env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            DribbleStartMutator(),
        ),
        obs_builder=DribbleObs(
            zero_padding=1,
            pos_coef=np.asarray(
                [
                    1 / common_values.SIDE_WALL_X,
                    1 / common_values.BACK_NET_Y,
                    1 / common_values.CEILING_Z,
                ]
            ),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        ),
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        reward_fn=DribbleCarryReward(),
        termination_cond=BallDroppedCondition(),
        truncation_cond=AnyCondition(
            TimeoutCondition(timeout_seconds=DRIBBLE_EPISODE_SECONDS),
        ),
        transition_engine=RocketSimEngine(),
        renderer=renderer,
    )

    return RLGymV2GymWrapper(rlgym_env)


def build_power_shot_rlgym_v2_env(render=False):
    # All imports must stay inside this function because training workers pickle it.
    prepare_runtime_locale()
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence
    from rlgym_ppo.util import RLGymV2GymWrapper

    from power_shot import (
        POWER_SHOT_EPISODE_SECONDS,
        POWER_SHOT_NO_TOUCH_TIMEOUT,
        POWER_SHOT_TICK_SKIP,
        PowerShotMutator,
        PowerShotObs,
        PowerShotReward,
        PowerShotTerminationCondition,
    )

    tick_skip = POWER_SHOT_TICK_SKIP
    renderer = None

    if render:
        import os
        import sys

        rlviser_binary = os.path.join(os.getcwd(), "rlviser")
        if not os.path.exists(rlviser_binary):
            raise RuntimeError(
                "Rendering requires the RLViser executable named `rlviser` in the repo root."
            )

        if sys.platform == "darwin":
            with open(rlviser_binary, "rb") as handle:
                magic = handle.read(4)
            mach_o_magics = {
                b"\xcf\xfa\xed\xfe",
                b"\xfe\xed\xfa\xcf",
                b"\xca\xfe\xba\xbe",
                b"\xbe\xba\xfe\xca",
            }
            if magic not in mach_o_magics:
                raise RuntimeError(
                    "RLViser upstream does not ship a macOS viewer binary. "
                    "Use the default `sandbox` renderer on macOS, or build RLViser locally."
                )

        try:
            from rlgym.rocket_league.rlviser import RLViserRenderer
        except ImportError as exc:
            raise RuntimeError(
                "Rendering requires RLViser. Reinstall dependencies with `pip install -r requirements.txt`."
            ) from exc
        renderer = RLViserRenderer(tick_rate=120 / tick_skip)

    rlgym_env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=0),
            PowerShotMutator(),
        ),
        obs_builder=PowerShotObs(
            zero_padding=1,
            pos_coef=np.asarray(
                [
                    1 / common_values.SIDE_WALL_X,
                    1 / common_values.BACK_NET_Y,
                    1 / common_values.CEILING_Z,
                ]
            ),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        ),
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        reward_fn=PowerShotReward(),
        termination_cond=AnyCondition(
            GoalCondition(),
            PowerShotTerminationCondition(),
        ),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=POWER_SHOT_NO_TOUCH_TIMEOUT),
            TimeoutCondition(timeout_seconds=POWER_SHOT_EPISODE_SECONDS),
        ),
        transition_engine=RocketSimEngine(),
        renderer=renderer,
    )

    return RLGymV2GymWrapper(rlgym_env)


def build_training_env():
    return build_standard_rlgym_v2_env(render=False)


def build_dribble_training_env():
    return build_dribble_rlgym_v2_env(render=False)


def build_power_shot_training_env():
    return build_power_shot_rlgym_v2_env(render=False)


def get_env_builder(scenario):
    if scenario == "standard":
        return build_standard_rlgym_v2_env
    if scenario == "dribble":
        return build_dribble_rlgym_v2_env
    if scenario in ("power_shot", "catch_ball"):
        return build_power_shot_rlgym_v2_env
    raise ValueError(f"Unsupported scenario: {scenario}")


def get_training_env_builder(scenario):
    if scenario == "standard":
        return build_training_env
    if scenario == "dribble":
        return build_dribble_training_env
    if scenario in ("power_shot", "catch_ball"):
        return build_power_shot_training_env
    raise ValueError(f"Unsupported scenario: {scenario}")


def _select_actions(policy, observations):
    obs_batch = np.asarray(observations, dtype=np.float32).reshape(len(observations), -1)
    with torch.no_grad():
        action_probs = policy.get_output(obs_batch)
        actions = torch.argmax(action_probs, dim=-1).cpu().numpy()
    return actions.reshape(-1, 1)


def watch_checkpoint(
    checkpoint="latest",
    episodes=3,
    scenario="standard",
    renderer="sandbox",
    render_delay=1 / 15,
    max_steps=None,
):
    prepare_runtime_locale()
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    checkpoint_path = resolve_checkpoint_folder(checkpoint, models_dir=get_models_dir(scenario))
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found. Train first or pass --checkpoint to an existing save.")

    use_rlviser = renderer == "rlviser"
    env = get_env_builder(scenario)(render=use_rlviser)
    obs_size = int(np.prod(env.observation_space.shape))
    action_count = env.action_space.n

    policy = DiscreteFF(obs_size, action_count, POLICY_LAYER_SIZES, "cpu")
    policy.load_state_dict(torch.load(f"{checkpoint_path}/PPO_POLICY.pt", map_location="cpu"))
    policy.eval()
    viewer = None

    if renderer == "sandbox":
        from sandbox import SandboxViewer

        viewer = SandboxViewer(scenario=scenario)

    try:
        for episode in range(1, episodes + 1):
            observations = env.reset()
            episode_rewards = np.zeros(len(observations), dtype=np.float32)
            step_count = 0

            if viewer is not None and not viewer.update(env.rlgym_env.state, episode, step_count):
                return

            if use_rlviser:
                env.render()
                if render_delay > 0:
                    time.sleep(render_delay)

            while True:
                actions = _select_actions(policy, observations)
                observations, rewards, terminated, truncated, info = env.step(actions)
                episode_rewards += np.asarray(rewards, dtype=np.float32)
                step_count += 1

                if viewer is not None and not viewer.update(info["state"], episode, step_count):
                    return

                if use_rlviser:
                    env.render()
                    if render_delay > 0:
                        time.sleep(render_delay)

                reached_step_cap = max_steps is not None and step_count >= max_steps
                if terminated or truncated or reached_step_cap:
                    status = "step cap" if reached_step_cap else "episode end"
                    print(
                        f"Episode {episode} finished after {step_count} steps "
                        f"({status}, rewards={episode_rewards.tolist()})"
                    )
                    break
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train or watch a Rocket League self-play bot.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the PPO bot.")
    train_parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEP_LIMIT,
                              help="Timestep limit (0 = no limit, run indefinitely)")
    train_parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY_TS)
    train_parser.add_argument("--n-proc", type=int, default=DEFAULT_N_PROC)
    train_parser.add_argument("--checkpoint", default="latest")
    train_parser.add_argument("--fresh", action="store_true", help="Start a new run instead of loading the latest checkpoint.")
    train_parser.add_argument("--scenario", choices=("standard", "dribble", "power_shot", "catch_ball"), default="standard")
    train_parser.add_argument("--dashboard", action="store_true", help="Show a live local dashboard during dribble training.")
    train_parser.add_argument(
        "--dashboard-update-seconds",
        type=float,
        default=1.0,
        help="Redraw the dribble dashboard on this timer in seconds.",
    )
    train_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument(
        "--train-segment-hours",
        type=float,
        default=DEFAULT_TRAIN_SEGMENT_SECONDS / 3600,
        help="Train for this many hours before checkpointing and pausing to cool down.",
    )
    train_parser.add_argument(
        "--cooldown-minutes",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS / 60,
        help="How long to pause between training segments. Set to 0 to disable cooldown pauses.",
    )

    watch_parser = subparsers.add_parser("watch", help="Watch a saved checkpoint play in self-play.")
    watch_parser.add_argument("--checkpoint", default="latest")
    watch_parser.add_argument("--episodes", type=int, default=3)
    watch_parser.add_argument("--scenario", choices=("standard", "dribble", "power_shot", "catch_ball"), default="standard")
    watch_parser.add_argument("--renderer", choices=("sandbox", "rlviser", "headless"), default="sandbox")
    watch_parser.add_argument("--render-delay", type=float, default=1 / 15)
    watch_parser.add_argument("--max-steps", type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command in (None, "train"):
        checkpoint = None if getattr(args, "fresh", False) else args.checkpoint
        metrics_logger = None
        if args.scenario == "dribble" and args.dashboard:
            from dribble_metrics import DribbleMetricsLogger

            metrics_logger = DribbleMetricsLogger(dashboard_update_seconds=args.dashboard_update_seconds)
        if args.scenario == "power_shot" and args.dashboard:
            from power_shot_metrics import PowerShotMetricsLogger

            metrics_logger = PowerShotMetricsLogger(dashboard_update_seconds=args.dashboard_update_seconds)
        # 0 means no limit — use the rlgym_ppo Learner's own practical maximum.
        timestep_limit = 5_000_000_000 if args.timesteps == 0 else args.timesteps
        run_learner(
            get_training_env_builder(args.scenario),
            checkpoint_load_folder=checkpoint,
            timestep_limit=timestep_limit,
            save_every_ts=args.save_every,
            n_proc=args.n_proc,
            log_to_wandb=args.wandb,
            device=args.device,
            train_segment_seconds=args.train_segment_hours * 3600,
            cooldown_seconds=args.cooldown_minutes * 60,
            metrics_logger=metrics_logger,
            checkpoint_migrate_fn=migrate_dribble_carry_quality_obs if args.scenario == "dribble" else None,
            models_dir=get_models_dir(args.scenario),
        )
        return

    if args.command == "watch":
        watch_checkpoint(
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            scenario=args.scenario,
            renderer=args.renderer,
            render_delay=args.render_delay,
            max_steps=args.max_steps,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
