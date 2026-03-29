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
MAX_RENDER_POINTS = 480


def _rolling_average(values, window):
    if not values:
        return []

    arr = np.asarray(values, dtype=np.float32)
    indices = np.arange(len(arr), dtype=np.int32)
    starts = np.maximum(indices - window + 1, 0)
    cumulative = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
    totals = cumulative[indices + 1] - cumulative[starts]
    counts = indices - starts + 1
    return (totals / counts).astype(np.float32).tolist()


def _compress_plot_series(values, max_points):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    if arr.size <= max_points:
        x_values = np.arange(arr.size, dtype=np.float32)
        return x_values, arr

    bin_edges = np.linspace(0, arr.size, max_points + 1, dtype=np.int32)
    x_values = []
    y_values = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end <= start:
            continue
        segment = arr[start:end]
        x_values.append((start + end - 1) / 2)
        y_values.append(float(segment.mean(dtype=np.float64)))
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def _load_metric_history():
    if not METRICS_CSV.exists():
        return [], []

    try:
        history = np.loadtxt(METRICS_CSV, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError):
        return [], []

    if history.size == 0 or history.shape[1] < 3:
        return [], []

    return history[:, 1].astype(np.float32).tolist(), history[:, 2].astype(np.float32).tolist()


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
        self.update([], [], 0, 0)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.root.destroy()

    def update(self, carry_seconds, distances, episode_count, update_seconds):
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
                f"Episodes: {episode_count}    "
                f"Display: up to {MAX_RENDER_POINTS} bins    "
                f"Dashboard refresh: every {update_seconds:.1f}s    "
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

        target_points = max(80, min(MAX_RENDER_POINTS, int(width)))
        plot_x, plot_y = _compress_plot_series(series, target_points)
        min_val = float(plot_y.min())
        max_val = float(plot_y.max())
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

        self.canvas.create_text(
            x0,
            y1 + 16,
            anchor="w",
            text="Episode 1",
            fill="#6b7280",
            font=("Helvetica", 10),
        )
        self.canvas.create_text(
            x1,
            y1 + 16,
            anchor="e",
            text=f"Episode {len(series)}",
            fill="#6b7280",
            font=("Helvetica", 10),
        )

        rolling_points = []
        for x_value, value in zip(plot_x, plot_y):
            x = x0 + (x_value / max(len(series) - 1, 1)) * width
            y = y1 - ((value - min_val) / (max_val - min_val)) * height
            rolling_points.extend((x, y))

        if len(rolling_points) >= 4:
            self.canvas.create_line(*rolling_points, fill=point_color, width=3, smooth=True)



class DribbleMetricsLogger:
    def __init__(self, dashboard_update_seconds=1.0):
        self.worker_pid = None
        self.process_state = {}
        self.dashboard_update_seconds = max(0.25, float(dashboard_update_seconds))
        self.last_dashboard_time = 0.0

        METRICS_DIR.mkdir(exist_ok=True)
        self.carry_seconds, self.distances = _load_metric_history()
        self.carry_rolling = _rolling_average(self.carry_seconds, ROLLING_WINDOW)
        self.distance_rolling = _rolling_average(self.distances, ROLLING_WINDOW)
        self.episode_counter = len(self.carry_seconds)

        if not METRICS_CSV.exists():
            with METRICS_CSV.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode", "carry_seconds", "distance_traveled_uu"])

        self.dashboard = DribbleDashboard()
        self.dashboard.update(
            self.carry_rolling,
            self.distance_rolling,
            self.episode_counter,
            self.dashboard_update_seconds,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["dashboard"] = None
        state["worker_pid"] = None
        state["process_state"] = {}
        state["carry_seconds"] = []
        state["distances"] = []
        state["carry_rolling"] = []
        state["distance_rolling"] = []
        state["episode_counter"] = 0
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

        now = time.monotonic()
        if now - self.last_dashboard_time >= self.dashboard_update_seconds:
            self.dashboard.update(
                self.carry_rolling,
                self.distance_rolling,
                self.episode_counter,
                self.dashboard_update_seconds,
            )
            self.last_dashboard_time = now

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
        self.carry_rolling.append(float(np.mean(self.carry_seconds[-ROLLING_WINDOW:])))
        self.distance_rolling.append(float(np.mean(self.distances[-ROLLING_WINDOW:])))

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
