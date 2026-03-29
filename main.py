import argparse
import time

import numpy as np
import torch

from learner import (
    POLICY_LAYER_SIZES,
    DEFAULT_N_PROC,
    DEFAULT_SAVE_EVERY_TS,
    DEFAULT_TIMESTEP_LIMIT,
    prepare_runtime_locale,
    resolve_checkpoint_folder,
    run_learner,
)


def build_rlgym_v2_env(render=False):
    # All imports must stay inside this function because training workers pickle it.
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


def build_training_env():
    return build_rlgym_v2_env(render=False)


def _select_actions(policy, observations):
    obs_batch = np.asarray(observations, dtype=np.float32).reshape(len(observations), -1)
    with torch.no_grad():
        action_probs = policy.get_output(obs_batch)
        actions = torch.argmax(action_probs, dim=-1).cpu().numpy()
    return actions.reshape(-1, 1)


def watch_checkpoint(
    checkpoint="latest",
    episodes=3,
    renderer="sandbox",
    render_delay=1 / 15,
    max_steps=None,
):
    prepare_runtime_locale()
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    checkpoint_path = resolve_checkpoint_folder(checkpoint)
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found. Train first or pass --checkpoint to an existing save.")

    use_rlviser = renderer == "rlviser"
    env = build_rlgym_v2_env(render=use_rlviser)
    obs_size = int(np.prod(env.observation_space.shape))
    action_count = env.action_space.n

    policy = DiscreteFF(obs_size, action_count, POLICY_LAYER_SIZES, "cpu")
    policy.load_state_dict(torch.load(f"{checkpoint_path}/PPO_POLICY.pt", map_location="cpu"))
    policy.eval()
    viewer = None

    if renderer == "sandbox":
        from sandbox import SandboxViewer

        viewer = SandboxViewer()

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
    train_parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEP_LIMIT)
    train_parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY_TS)
    train_parser.add_argument("--n-proc", type=int, default=DEFAULT_N_PROC)
    train_parser.add_argument("--checkpoint", default="latest")
    train_parser.add_argument("--fresh", action="store_true", help="Start a new run instead of loading the latest checkpoint.")
    train_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    train_parser.add_argument("--device", default="auto")

    watch_parser = subparsers.add_parser("watch", help="Watch a saved checkpoint play in self-play.")
    watch_parser.add_argument("--checkpoint", default="latest")
    watch_parser.add_argument("--episodes", type=int, default=3)
    watch_parser.add_argument("--renderer", choices=("sandbox", "rlviser", "headless"), default="sandbox")
    watch_parser.add_argument("--render-delay", type=float, default=1 / 15)
    watch_parser.add_argument("--max-steps", type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command in (None, "train"):
        checkpoint = None if getattr(args, "fresh", False) else args.checkpoint
        run_learner(
            build_training_env,
            checkpoint_load_folder=checkpoint,
            timestep_limit=args.timesteps,
            save_every_ts=args.save_every,
            n_proc=args.n_proc,
            log_to_wandb=args.wandb,
            device=args.device,
        )
        return

    if args.command == "watch":
        watch_checkpoint(
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            renderer=args.renderer,
            render_delay=args.render_delay,
            max_steps=args.max_steps,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
