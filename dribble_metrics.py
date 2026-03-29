import csv
import os
import time
import tkinter as tk
from pathlib import Path

import numpy as np

from dribble import CARRY_THRESHOLD, DECISIONS_PER_SECOND, get_dribble_alignment

ROOT_DIR = Path(__file__).resolve().parent
METRICS_DIR = ROOT_DIR / "metrics"
METRICS_CSV = METRICS_DIR / "dribble_episode_metrics.csv"
PLOT_WIDTH = 1180
PLOT_HEIGHT = 720
PLOT_PADDING = 60
ROLLING_WINDOW = 50


def _rolling_average(values, window):
    if not values:
        return []

    arr = np.asarray(values, dtype=np.float32)
    out = []
    for index in range(len(arr)):
        start = max(0, index - window + 1)
        out.append(float(arr[start:index + 1].mean()))
    return out


def _lighten_color(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = min(255, int(r + (255 - r) * 0.45))
    g = min(255, int(g + (255 - g) * 0.45))
    b = min(255, int(b + (255 - b) * 0.45))
    return f"#{r:02x}{g:02x}{b:02x}"


class DribbleDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dribble Training Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        self.canvas = tk.Canvas(
            self.root,
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            bg="#f7f7f2",
            highlightthickness=0,
        )
        self.canvas.pack()
        self.update([], [], 0)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.root.destroy()

    def update(self, carry_seconds, distances, update_every):
        if self.closed:
            return

        self.canvas.delete("all")
        self._draw_plot(
            x0=PLOT_PADDING,
            y0=PLOT_PADDING,
            width=(PLOT_WIDTH - PLOT_PADDING * 3) / 2,
            height=PLOT_HEIGHT - PLOT_PADDING * 2,
            values=carry_seconds,
            title="Carry Time Per Episode (s)",
            point_color="#2563eb",
        )
        self._draw_plot(
            x0=(PLOT_WIDTH + PLOT_PADDING) / 2,
            y0=PLOT_PADDING,
            width=(PLOT_WIDTH - PLOT_PADDING * 3) / 2,
            height=PLOT_HEIGHT - PLOT_PADDING * 2,
            values=distances,
            title="Distance Traveled Per Episode (uu)",
            point_color="#ea580c",
        )
        self.canvas.create_text(
            PLOT_PADDING,
            20,
            anchor="w",
            text=(
                f"Episodes: {len(carry_seconds)}    "
                f"Dashboard refresh: every {update_every} episodes    "
                f"Updated: {time.strftime('%H:%M:%S')}"
            ),
            fill="#111827",
            font=("Helvetica", 14, "bold"),
        )

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def _draw_plot(self, x0, y0, width, height, values, title, point_color):
        x1 = x0 + width
        y1 = y0 + height
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#9ca3af", width=2)
        self.canvas.create_text(
            x0,
            y0 - 18,
            anchor="w",
            text=title,
            fill="#111827",
            font=("Helvetica", 13, "bold"),
        )

        series = values
        if not series:
            self.canvas.create_text(
                x0 + width / 2,
                y0 + height / 2,
                text="Waiting for completed episodes...",
                fill="#6b7280",
                font=("Helvetica", 12),
            )
            return

        rolling = _rolling_average(series, ROLLING_WINDOW)
        min_val = min(min(series), min(rolling))
        max_val = max(max(series), max(rolling))
        if max_val - min_val < 1e-6:
            max_val = min_val + 1.0

        for tick in range(5):
            frac = tick / 4
            y = y1 - frac * height
            label_val = min_val + frac * (max_val - min_val)
            self.canvas.create_line(x0, y, x1, y, fill="#e5e7eb")
            self.canvas.create_text(
                x0 - 10,
                y,
                anchor="e",
                text=f"{label_val:.1f}",
                fill="#6b7280",
                font=("Helvetica", 10),
            )

        rolling_points = []
        for index, value in enumerate(rolling):
            x = x0 + (index / max(len(rolling) - 1, 1)) * width
            y = y1 - ((value - min_val) / (max_val - min_val)) * height
            rolling_points.extend((x, y))

        if len(rolling_points) >= 4:
            self.canvas.create_line(*rolling_points, fill=point_color, width=3, smooth=True)

        if len(series) >= 2:
            regression_input_x = np.arange(len(series), dtype=np.float32)
            regression_input_y = np.asarray(series, dtype=np.float32)
            slope, intercept = np.polyfit(regression_input_x, regression_input_y, 1)
            start_y = intercept
            end_y = slope * (len(series) - 1) + intercept
            start_canvas_y = y1 - ((start_y - min_val) / (max_val - min_val)) * height
            end_canvas_y = y1 - ((end_y - min_val) / (max_val - min_val)) * height
            self.canvas.create_line(
                x0,
                start_canvas_y,
                x1,
                end_canvas_y,
                fill=_lighten_color(point_color),
                width=2,
                dash=(8, 6),
            )


class DribbleMetricsLogger:
    def __init__(self, dashboard_update_every=10):
        self.worker_pid = None
        self.episode_counter = 0
        self.process_state = {}
        self.carry_seconds = []
        self.distances = []
        self.dashboard_update_every = max(1, int(dashboard_update_every))
        self.last_dashboard_episode = 0
        self.dashboard = DribbleDashboard()

        METRICS_DIR.mkdir(exist_ok=True)
        if not METRICS_CSV.exists():
            with METRICS_CSV.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode", "carry_seconds", "distance_traveled_uu"])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["dashboard"] = None
        state["worker_pid"] = None
        state["process_state"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.dashboard is None:
            self.dashboard = None

    def collect_metrics(self, game_state) -> np.ndarray:
        metrics_arrays = self._collect_metrics(game_state)
        unraveled = []
        for arr in metrics_arrays:
            shape = np.shape(arr)
            unraveled.append(len(shape))
            unraveled += list(shape)
            unraveled += np.ravel(arr).tolist()
        return np.asarray(unraveled, dtype=np.float32)

    def _collect_metrics(self, game_state) -> np.ndarray:
        if self.worker_pid is None:
            self.worker_pid = os.getpid()

        if not game_state.cars:
            return [np.zeros(5, dtype=np.float32)]

        car = next(iter(game_state.cars.values()))
        _, _, _, carry_quality = get_dribble_alignment(car, game_state.ball.position)
        carrying = 1.0 if carry_quality > CARRY_THRESHOLD else 0.0
        metric = np.array(
            [
                float(self.worker_pid),
                float(game_state.tick_count),
                float(car.physics.position[0]),
                float(car.physics.position[1]),
                carrying,
            ],
            dtype=np.float32,
        )
        return [metric]

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        for serialized_metrics in collected_metrics:
            metrics_arrays = self._deserialize(serialized_metrics)
            if not metrics_arrays:
                continue
            metric = np.asarray(metrics_arrays[0], dtype=np.float32)
            if metric.size != 5:
                continue
            self._consume_metric(metric)

        if wandb_run is not None and self.carry_seconds:
            wandb_run.log(
                {
                    "Dribble/Latest Carry Seconds": self.carry_seconds[-1],
                    "Dribble/Latest Distance": self.distances[-1],
                    "Dribble/Mean Carry Seconds (Last 20)": float(np.mean(self.carry_seconds[-20:])),
                    "Dribble/Mean Distance (Last 20)": float(np.mean(self.distances[-20:])),
                    "Cumulative Timesteps": cumulative_timesteps,
                }
            )

        if self.episode_counter - self.last_dashboard_episode >= self.dashboard_update_every:
            self.dashboard.update(self.carry_seconds, self.distances, self.dashboard_update_every)
            self.last_dashboard_episode = self.episode_counter

    def _consume_metric(self, metric):
        pid = int(metric[0])
        tick_count = int(metric[1])
        position = metric[2:4].astype(np.float32)
        carrying = bool(metric[4] > 0.5)

        state = self.process_state.get(pid)
        if state is None:
            self.process_state[pid] = {
                "last_tick": tick_count,
                "last_pos": position,
                "carry_steps": 1 if carrying else 0,
                "distance": 0.0,
            }
            return

        if tick_count <= state["last_tick"]:
            self._finalize_episode(state)
            state["carry_steps"] = 1 if carrying else 0
            state["distance"] = 0.0
            state["last_pos"] = position
            state["last_tick"] = tick_count
            return

        state["distance"] += float(np.linalg.norm(position - state["last_pos"]))
        state["carry_steps"] += 1 if carrying else 0
        state["last_pos"] = position
        state["last_tick"] = tick_count

    def _finalize_episode(self, state):
        carry_seconds = state["carry_steps"] / DECISIONS_PER_SECOND
        distance = state["distance"]
        self.episode_counter += 1
        self.carry_seconds.append(carry_seconds)
        self.distances.append(distance)

        with METRICS_CSV.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([self.episode_counter, carry_seconds, distance])

    @staticmethod
    def _deserialize(serialized_metrics):
        metrics_arrays = []
        i = 0
        while i < len(serialized_metrics):
            n_shape = int(serialized_metrics[i])
            n_values_in_metric = 1
            shape = []
            i += 1
            for arg in serialized_metrics[i:i + n_shape]:
                n_values_in_metric *= arg
                shape.append(int(arg))
            n_values_in_metric = int(n_values_in_metric)
            metric = serialized_metrics[i + n_shape:i + n_shape + n_values_in_metric]
            metrics_arrays.append(metric)
            i = i + n_shape + n_values_in_metric
        return metrics_arrays
