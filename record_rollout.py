import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from learner import POLICY_LAYER_SIZES, prepare_runtime_locale, resolve_checkpoint_folder
from main import get_env_builder

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
WIDTH = 1200
HEIGHT = 800
PADDING = 40


def _scale():
    from rlgym.rocket_league import common_values

    return min(
        (WIDTH - 2 * PADDING) / (2 * common_values.SIDE_WALL_X),
        (HEIGHT - 2 * PADDING) / (2 * common_values.BACK_WALL_Y),
    )


def _to_canvas(position):
    x = WIDTH / 2 + float(position[0]) * _scale()
    y = HEIGHT / 2 - float(position[1]) * _scale()
    return x, y


def _rotated_triangle(center_x, center_y, angle, length=34, width=22):
    points = [
        (length / 2, 0),
        (-length / 2, -width / 2),
        (-length / 2, width / 2),
    ]
    out = []
    for dx, dy in points:
        px = center_x + dx * np.cos(angle) - dy * np.sin(angle)
        py = center_y - (dx * np.sin(angle) + dy * np.cos(angle))
        out.append((px, py))
    return out


def draw_frame(state, title):
    from rlgym.rocket_league import common_values

    image = Image.new("RGB", (WIDTH, HEIGHT), "#135d36")
    draw = ImageDraw.Draw(image)

    left, top = _to_canvas((-common_values.SIDE_WALL_X, common_values.BACK_WALL_Y))
    right, bottom = _to_canvas((common_values.SIDE_WALL_X, -common_values.BACK_WALL_Y))
    draw.rectangle((left, top, right, bottom), outline="#ecf7ec", width=4)

    center_x = WIDTH / 2
    draw.line((center_x, top, center_x, bottom), fill="#ecf7ec", width=2)
    circle_radius = 600 * _scale()
    draw.ellipse(
        (
            center_x - circle_radius,
            HEIGHT / 2 - circle_radius,
            center_x + circle_radius,
            HEIGHT / 2 + circle_radius,
        ),
        outline="#ecf7ec",
        width=2,
    )

    ball_x, ball_y = _to_canvas(state.ball.position[:2])
    draw.ellipse((ball_x - 10, ball_y - 10, ball_x + 10, ball_y + 10), fill="#f8fafc")

    for idx, car in enumerate(state.cars.values(), start=1):
        x, y = _to_canvas(car.physics.position[:2])
        forward = car.physics.rotation_mtx[:, 0]
        angle = np.arctan2(float(forward[1]), float(forward[0]))
        color = "#55a4ff" if car.is_blue else "#ff9a3d"
        draw.polygon(_rotated_triangle(x, y, angle), fill=color, outline="#0f172a")
        draw.text((x + 10, y - 10), f"{'B' if car.is_blue else 'O'}{idx}", fill="#f8fafc")

    draw.text((16, 16), title, fill="#f8fafc")
    return image


def select_actions(policy, observations):
    obs_batch = np.asarray(observations, dtype=np.float32).reshape(len(observations), -1)
    with torch.no_grad():
        action_probs = policy.get_output(obs_batch)
        actions = torch.argmax(action_probs, dim=-1).cpu().numpy()
    return actions.reshape(-1, 1)


def record_rollout(scenario, checkpoint, max_steps, output_path):
    prepare_runtime_locale()
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    checkpoint_path = resolve_checkpoint_folder(checkpoint)
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found to record.")

    env = get_env_builder(scenario)(render=False)
    obs_size = int(np.prod(env.observation_space.shape))
    action_count = env.action_space.n

    policy = DiscreteFF(obs_size, action_count, POLICY_LAYER_SIZES, "cpu")
    policy.load_state_dict(torch.load(f"{checkpoint_path}/PPO_POLICY.pt", map_location="cpu"))
    policy.eval()

    frames = []
    observations = env.reset()
    total_reward = 0.0
    frames.append(draw_frame(env.rlgym_env.state, f"{scenario.title()} rollout | step 0 | reward 0.00"))

    try:
        for step in range(1, max_steps + 1):
            actions = select_actions(policy, observations)
            observations, rewards, terminated, truncated, info = env.step(actions)
            total_reward += float(np.sum(rewards))
            frames.append(
                draw_frame(
                    info["state"],
                    f"{scenario.title()} rollout | step {step} | total reward {total_reward:.2f}",
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // 15,
        loop=0,
    )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Record a rollout as a GIF.")
    parser.add_argument("--scenario", choices=("standard", "dribble"), default="dribble")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--output", default=str(ARTIFACTS_DIR / "dribble_attempt.gif"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path
    record_rollout(args.scenario, args.checkpoint, args.max_steps, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
