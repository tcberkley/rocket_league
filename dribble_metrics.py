import csv
import json
import os
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from rlgym.rocket_league import common_values

from dribble import (
    CARRY_THRESHOLD,
    DECISIONS_PER_SECOND,
    WALL_APPROACH_PENALTY_SCALE,
    get_current_loop_state,
    get_dribble_alignment,
    get_reached_breadcrumb_positions,
    heading_angle,
    near_wall_warning_score,
    wall_approach_score,
    wrap_angle,
)

ROOT_DIR = Path(__file__).resolve().parent
METRICS_DIR = ROOT_DIR / "metrics"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
P95_GIFS_DIR = ARTIFACTS_DIR / "periodic_p95"
METRICS_CSV = METRICS_DIR / "dribble_episode_metrics.csv"
PHASE4_METRICS_CSV = METRICS_DIR / "dribble_phase4_metrics.csv"
METRICS_MARKERS_JSON = METRICS_DIR / "dribble_markers.json"

MAX_PLOT_WIDTH = 1200
MAX_PLOT_HEIGHT = 920
MIN_PLOT_WIDTH = 1000
MIN_PLOT_HEIGHT = 800
PLOT_PADDING = 50
ROLLING_WINDOW = 50
MAX_RENDER_POINTS = 420
HEADER_FONT = ("Helvetica", 13, "bold")
TITLE_FONT = ("Helvetica", 11, "bold")
AXIS_FONT = ("Helvetica", 9)
MARKER_FONT = ("Helvetica", 9, "bold")
LEGEND_FONT = ("Helvetica", 9, "bold")
EPISODE_GIF_WIDTH = 1200
EPISODE_GIF_HEIGHT = 800
EPISODE_GIF_PADDING = 40
EPISODE_GIF_FRAME_STRIDE = 2
EPISODE_GIF_FRAME_DURATION_MS = 1000 // 12
P95_GIF_EPISODE_INTERVAL = 10_000
MAX_TRACKED_CRUMBS = 10  # fixed-size crumb slot so shm_view is never resized


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


def _rolling_percentile(values, window, percentile):
    if not values:
        return []

    arr = np.asarray(values, dtype=np.float32)
    prefix = [
        float(np.percentile(arr[: index + 1], percentile))
        for index in range(min(window - 1, len(arr)))
    ]

    if len(arr) < window:
        return prefix

    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    full = np.percentile(windows, percentile, axis=1)
    return prefix + full.astype(np.float32).tolist()


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

    carry_seconds = []
    distances = []
    with METRICS_CSV.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                carry_seconds.append(float(row[1]))
                distances.append(float(row[2]))
            except ValueError:
                continue
    return carry_seconds, distances


def _load_phase4_history():
    history = {
        "episodes": [],
        "breadcrumbs_reached": [],
        "direction_flips": [],
        "avg_reward": [],
        "wall_approach_penalty": [],
        "near_wall_carry_seconds": [],
    }
    if not PHASE4_METRICS_CSV.exists():
        return history

    with PHASE4_METRICS_CSV.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or "breadcrumbs_reached" not in header:
            return history  # old format, skip
        for row in reader:
            if len(row) < 6:
                continue
            try:
                history["episodes"].append(int(row[0]))
                history["breadcrumbs_reached"].append(float(row[1]))
                history["direction_flips"].append(float(row[2]))
                history["avg_reward"].append(float(row[3]))
                history["wall_approach_penalty"].append(float(row[4]))
                history["near_wall_carry_seconds"].append(float(row[5]))
            except ValueError:
                continue
    return history


def _load_markers():
    if not METRICS_MARKERS_JSON.exists():
        return []

    try:
        markers = json.loads(METRICS_MARKERS_JSON.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return []

    if not isinstance(markers, list):
        return []

    out = []
    for marker in markers:
        if not isinstance(marker, dict):
            continue
        try:
            episode = int(marker["episode"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(
            {
                "episode": episode,
                "label": str(marker.get("label", f"Phase @ {episode}")),
                "color": str(marker.get("color", "#b91c1c")),
            }
        )
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


def _append_rolling_stats(series, rolling_mean, rolling_p05, rolling_p95, value):
    series.append(value)
    window_values = series[-ROLLING_WINDOW:]
    rolling_mean.append(float(np.mean(window_values)))
    rolling_p05.append(float(np.percentile(window_values, 5)))
    rolling_p95.append(float(np.percentile(window_values, 95)))


def _project_markers_to_tracked(markers, tracked_global_episodes):
    projected = []
    if not tracked_global_episodes:
        return projected

    for marker in markers:
        for index, global_episode in enumerate(tracked_global_episodes, start=1):
            if global_episode >= marker["episode"]:
                projected.append(
                    {
                        "episode": index,
                        "label": marker.get("label", f"Phase @ {marker['episode']}"),
                        "color": marker.get("color", "#b91c1c"),
                    }
                )
                break
    return projected


def _field_scale(canvas_width, canvas_height, padding):
    return min(
        (canvas_width - 2 * padding) / (2 * common_values.SIDE_WALL_X),
        (canvas_height - 2 * padding) / (2 * common_values.BACK_WALL_Y),
    )


def _field_to_canvas(position_xy, canvas_width, canvas_height, scale):
    x = canvas_width / 2 + float(position_xy[0]) * scale
    y = canvas_height / 2 - float(position_xy[1]) * scale
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


_BALL_RADIUS_UU = 91.25  # game-unit ball radius


def _draw_episode_frame(frame, title, canvas_width=EPISODE_GIF_WIDTH, canvas_height=EPISODE_GIF_HEIGHT, padding=EPISODE_GIF_PADDING, trail=None):
    image = Image.new("RGB", (canvas_width, canvas_height), "#135d36")
    draw = ImageDraw.Draw(image)
    scale = _field_scale(canvas_width, canvas_height, padding)
    ball_r = max(8, int(_BALL_RADIUS_UU * scale))

    def to_canvas(pos):
        return _field_to_canvas(pos, canvas_width, canvas_height, scale)

    left, top = to_canvas((-common_values.SIDE_WALL_X, common_values.BACK_WALL_Y))
    right, bottom = to_canvas((common_values.SIDE_WALL_X, -common_values.BACK_WALL_Y))
    draw.rectangle((left, top, right, bottom), outline="#ecf7ec", width=4)

    cx = canvas_width / 2
    draw.line((cx, top, cx, bottom), fill="#ecf7ec", width=2)
    circle_radius = 600 * scale
    draw.ellipse(
        (
            cx - circle_radius,
            canvas_height / 2 - circle_radius,
            cx + circle_radius,
            canvas_height / 2 + circle_radius,
        ),
        outline="#ecf7ec",
        width=2,
    )

    loop_x = float(frame.get("loop_x", 0.0))
    loop_y = float(frame.get("loop_y", 0.0))
    if loop_x > 0.0 and loop_y > 0.0:
        lane_left, lane_top = to_canvas((-loop_x, loop_y))
        lane_right, lane_bottom = to_canvas((loop_x, -loop_y))
        draw.ellipse((lane_left, lane_top, lane_right, lane_bottom), outline="#f59e0b", width=3)

    if trail and len(trail) >= 2:
        trail_points = [to_canvas((f["car_x"], f["car_y"])) for f in trail]
        draw.line(trail_points, fill="#93c5fd", width=2)

    reached_crumbs = frame.get("reached_crumbs", [])
    for crumb_xy in reached_crumbs:
        cx, cy = to_canvas((crumb_xy[0], crumb_xy[1]))
        r = ball_r
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#22c55e", outline="#14532d", width=2)
        draw.text((cx, cy), "✓", fill="#14532d", anchor="mm")

    crumb_count = len(reached_crumbs)
    draw.text((canvas_width // 2, 16), f"Breadcrumbs: {crumb_count}", fill="#22c55e", anchor="mt")

    target_x = float(frame.get("target_x", 0.0))
    target_y = float(frame.get("target_y", 0.0))
    if frame.get("has_target", False):
        wp_x, wp_y = to_canvas((target_x, target_y))
        draw.ellipse((wp_x - ball_r, wp_y - ball_r, wp_x + ball_r, wp_y + ball_r), outline="#ff4fd8", width=3)
        xhair = ball_r + 4
        draw.line((wp_x - xhair, wp_y, wp_x + xhair, wp_y), fill="#ff4fd8", width=2)
        draw.line((wp_x, wp_y - xhair, wp_x, wp_y + xhair), fill="#ff4fd8", width=2)
        draw.text((wp_x + ball_r + 2, wp_y - ball_r - 2), "WP", fill="#ff4fd8")

    ball_x, ball_y = to_canvas((frame["ball_x"], frame["ball_y"]))
    draw.ellipse((ball_x - ball_r, ball_y - ball_r, ball_x + ball_r, ball_y + ball_r), fill="#f8fafc")

    car_x, car_y = to_canvas((frame["car_x"], frame["car_y"]))
    draw.polygon(_rotated_triangle(car_x, car_y, frame["heading"]), fill="#55a4ff", outline="#0f172a")
    draw.text((car_x + 10, car_y - 10), "B1", fill="#f8fafc")

    draw.text((16, 16), title, fill="#f8fafc")
    return image


class DribbleDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dribble Training Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        screen_width = max(int(self.root.winfo_screenwidth()), MIN_PLOT_WIDTH)
        screen_height = max(int(self.root.winfo_screenheight()), MIN_PLOT_HEIGHT)
        self.plot_width = max(MIN_PLOT_WIDTH, min(MAX_PLOT_WIDTH, screen_width - 120))
        self.plot_height = max(MIN_PLOT_HEIGHT, min(MAX_PLOT_HEIGHT, screen_height - 140))
        self.root.geometry(f"{self.plot_width}x{self.plot_height}+40+40")
        self.canvas = tk.Canvas(
            self.root,
            width=self.plot_width,
            height=self.plot_height,
            bg="#f7f7f2",
            highlightthickness=0,
        )
        self.canvas.pack()
        self._minimap_photo = None
        self.update({}, 0, 0, 0)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.root.destroy()

    def update(self, plot_data, episode_count, update_seconds, cumulative_timesteps, minimap_frames=None):
        if self.closed:
            return

        self.canvas.delete("all")

        ts_display = f"{cumulative_timesteps / 1000:.0f}k" if cumulative_timesteps >= 1000 else str(cumulative_timesteps)
        self.canvas.create_text(
            PLOT_PADDING,
            22,
            anchor="w",
            text=(
                f"Episodes: {episode_count}    "
                f"Timesteps: {ts_display}    "
                f"Refresh: every {update_seconds:.1f}s    "
                f"Updated: {time.strftime('%H:%M:%S')}"
            ),
            fill="#111827",
            font=HEADER_FONT,
        )

        minimap_width = int(self.plot_width * 0.32)
        charts_width = self.plot_width - minimap_width - PLOT_PADDING
        top_y = 60
        chart_gap_x = PLOT_PADDING
        chart_gap_y = 60
        chart_w = (charts_width - chart_gap_x) / 2
        available_height = self.plot_height - top_y - PLOT_PADDING
        chart_h = max(180, (available_height - chart_gap_y) / 2)

        chart_specs = [
            ("carry_time", "Carry Time Per Episode (s)", "#2563eb", "Episode",
             PLOT_PADDING, top_y),
            ("breadcrumbs_reached", "Breadcrumbs Reached Per Episode", "#b45309", "Tracked Episode",
             PLOT_PADDING + chart_w + chart_gap_x, top_y),
            ("avg_speed", "Avg Car Speed Per Episode (uu/s)", "#7c3aed", "Tracked Episode",
             PLOT_PADDING, top_y + chart_h + chart_gap_y),
            ("wall_approach_penalty", "Wall Approach Penalty Per Episode", "#0891b2", "Tracked Episode",
             PLOT_PADDING + chart_w + chart_gap_x, top_y + chart_h + chart_gap_y),
        ]
        for key, title, color, x_prefix, x0, y0 in chart_specs:
            self._draw_plot(
                x0=x0, y0=y0, width=chart_w, height=chart_h,
                spec=plot_data.get(key, {}),
                title=title, point_color=color, x_label_prefix=x_prefix,
            )

        minimap_x = self.plot_width - minimap_width
        minimap_y = top_y
        minimap_h = available_height
        self._render_minimap(minimap_frames, minimap_x, minimap_y, minimap_width, minimap_h)

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def _render_minimap(self, frames, x0, y0, width, height):
        if not frames:
            self.canvas.create_rectangle(x0, y0, x0 + width, y0 + height, outline="#9ca3af", width=2)
            self.canvas.create_text(
                x0 + width / 2, y0 + height / 2,
                text="Waiting for\nepisode data...",
                fill="#6b7280", font=("Helvetica", 11), justify="center",
            )
            return

        last_frame = frames[-1]
        padding = 20
        image = _draw_episode_frame(
            last_frame, "", canvas_width=width, canvas_height=height, padding=padding, trail=frames,
        )
        self._minimap_photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(x0, y0, anchor="nw", image=self._minimap_photo)

    def _draw_plot(self, x0, y0, width, height, spec, title, point_color, x_label_prefix):
        x1 = x0 + width
        y1 = y0 + height
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#9ca3af", width=2)
        self.canvas.create_text(
            x0,
            y0 - 16,
            anchor="w",
            text=title,
            fill="#111827",
            font=TITLE_FONT,
        )

        mean_values = spec.get("mean", [])
        low_values = spec.get("p05", [])
        high_values = spec.get("p95", [])
        markers = spec.get("markers", [])
        x_end_label = spec.get("x_end_label")

        if not mean_values:
            self.canvas.create_text(
                x0 + width / 2,
                y0 + height / 2,
                text="Waiting for data...",
                fill="#6b7280",
                font=("Helvetica", 11),
            )
            return

        target_points = max(60, min(MAX_RENDER_POINTS, int(width)))
        plot_x, plot_mean = _compress_plot_series(mean_values, target_points)
        _, plot_low = _compress_plot_series(low_values, target_points)
        _, plot_high = _compress_plot_series(high_values, target_points)
        min_val = float(min(plot_low.min(), plot_mean.min(), plot_high.min()))
        max_val = float(max(plot_low.max(), plot_mean.max(), plot_high.max()))
        max_series_values = spec.get("max_series", [])
        plot_max_x, plot_max_s = (np.empty(0), np.empty(0))
        if max_series_values:
            plot_max_x, plot_max_s = _compress_plot_series(max_series_values, target_points)
            if plot_max_s.size > 0:
                max_val = float(max(max_val, float(plot_max_s.max())))
        if max_val - min_val < 1e-6:
            max_val = min_val + 1.0

        for tick in range(5):
            frac = tick / 4
            y = y1 - frac * height
            label_val = min_val + frac * (max_val - min_val)
            self.canvas.create_line(x0, y, x1, y, fill="#e5e7eb")
            self.canvas.create_text(
                x0 - 8,
                y,
                anchor="e",
                text=f"{label_val:.1f}",
                fill="#6b7280",
                font=AXIS_FONT,
            )

        for marker_index, marker in enumerate(markers):
            if marker["episode"] < 1 or marker["episode"] > len(mean_values):
                continue
            marker_x = x0 + ((marker["episode"] - 1) / max(len(mean_values) - 1, 1)) * width
            marker_color = marker.get("color", "#b91c1c")
            self.canvas.create_line(marker_x, y0, marker_x, y1, fill=marker_color, width=2, dash=(10, 6))

            if marker_x > x1 - 85:
                label_x = marker_x - 6
                anchor = "ne"
            else:
                label_x = marker_x + 6
                anchor = "nw"
            label_y = y0 + 8 + 18 * (marker_index % 3)
            text_id = self.canvas.create_text(
                label_x,
                label_y,
                anchor=anchor,
                text=marker.get("label", "Phase"),
                fill=marker_color,
                font=MARKER_FONT,
            )
            bbox = self.canvas.bbox(text_id)
            if bbox is not None:
                pad = 3
                rect_id = self.canvas.create_rectangle(
                    bbox[0] - pad,
                    bbox[1] - pad,
                    bbox[2] + pad,
                    bbox[3] + pad,
                    fill="#f7f7f2",
                    outline="",
                )
                self.canvas.tag_lower(rect_id, text_id)

        self.canvas.create_text(
            x0,
            y1 + 14,
            anchor="w",
            text=f"{x_label_prefix} 1",
            fill="#6b7280",
            font=AXIS_FONT,
        )
        end_label = len(mean_values) if x_end_label is None else x_end_label
        self.canvas.create_text(
            x1,
            y1 + 14,
            anchor="e",
            text=f"{x_label_prefix} {end_label}",
            fill="#6b7280",
            font=AXIS_FONT,
        )

        legend_color = _lighten_color(point_color)
        self._draw_legend(x1, y0 - 16, point_color, legend_color, has_max_series=bool(max_series_values))

        low_points = []
        high_points = []
        mean_points = []
        for x_value, low_value, mean_value, high_value in zip(plot_x, plot_low, plot_mean, plot_high):
            x = x0 + (x_value / max(len(mean_values) - 1, 1)) * width
            low_y = y1 - ((low_value - min_val) / (max_val - min_val)) * height
            high_y = y1 - ((high_value - min_val) / (max_val - min_val)) * height
            mean_y = y1 - ((mean_value - min_val) / (max_val - min_val)) * height
            low_points.extend((x, low_y))
            high_points.extend((x, high_y))
            mean_points.extend((x, mean_y))

        if len(low_points) >= 4:
            self.canvas.create_line(*low_points, fill=legend_color, width=2, dash=(6, 6), smooth=True)
        if len(high_points) >= 4:
            self.canvas.create_line(*high_points, fill=legend_color, width=2, dash=(6, 6), smooth=True)
        if len(mean_points) >= 4:
            self.canvas.create_line(*mean_points, fill=point_color, width=3, smooth=True)

        if plot_max_s.size >= 2:
            max_points = []
            n_ref = max(len(mean_values) - 1, 1)
            for x_value, max_value in zip(plot_max_x, plot_max_s):
                x = x0 + (float(x_value) / n_ref) * width
                my = y1 - ((float(max_value) - min_val) / (max_val - min_val)) * height
                max_points.extend((x, my))
            if len(max_points) >= 4:
                self.canvas.create_line(*max_points, fill="#f59e0b", width=2, smooth=False)

    def _draw_legend(self, x1, title_y, point_color, legend_color, has_max_series=False):
        base_y = title_y + 2

        if has_max_series:
            # Shift items left to make room for the "best" entry on the right.
            self.canvas.create_line(x1 - 215, base_y, x1 - 195, base_y, fill=point_color, width=3)
            self.canvas.create_text(x1 - 191, base_y, anchor="w", text="mean", fill=point_color, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 153, base_y, x1 - 133, base_y, fill=legend_color, width=2, dash=(6, 6))
            self.canvas.create_text(x1 - 129, base_y, anchor="w", text="5th", fill=legend_color, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 97, base_y, x1 - 77, base_y, fill=legend_color, width=2, dash=(6, 6))
            self.canvas.create_text(x1 - 73, base_y, anchor="w", text="95th", fill=legend_color, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 41, base_y, x1 - 21, base_y, fill="#f59e0b", width=2)
            self.canvas.create_text(x1 - 17, base_y, anchor="w", text="best", fill="#f59e0b", font=LEGEND_FONT)
        else:
            self.canvas.create_line(x1 - 165, base_y, x1 - 145, base_y, fill=point_color, width=3)
            self.canvas.create_text(x1 - 141, base_y, anchor="w", text="mean", fill=point_color, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 103, base_y, x1 - 83, base_y, fill=legend_color, width=2, dash=(6, 6))
            self.canvas.create_text(x1 - 79, base_y, anchor="w", text="5th", fill=legend_color, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 47, base_y, x1 - 27, base_y, fill=legend_color, width=2, dash=(6, 6))
            self.canvas.create_text(x1 - 23, base_y, anchor="w", text="95th", fill=legend_color, font=LEGEND_FONT)


class DribbleMetricsLogger:
    def __init__(self, dashboard_update_seconds=1.0):
        self.worker_pid = None
        self.process_state = {}
        self.dashboard_update_seconds = max(0.25, float(dashboard_update_seconds))
        self.last_dashboard_time = 0.0

        METRICS_DIR.mkdir(exist_ok=True)
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        P95_GIFS_DIR.mkdir(exist_ok=True)

        self.carry_seconds, self.distances = _load_metric_history()
        self.carry_rolling = _rolling_average(self.carry_seconds, ROLLING_WINDOW)
        self.carry_p05 = _rolling_percentile(self.carry_seconds, ROLLING_WINDOW, 5)
        self.carry_p95 = _rolling_percentile(self.carry_seconds, ROLLING_WINDOW, 95)

        phase4 = _load_phase4_history()
        self.phase4_episodes = phase4["episodes"]
        self.breadcrumbs_reached = phase4["breadcrumbs_reached"]
        self.direction_flips = phase4["direction_flips"]
        self.avg_reward = phase4["avg_reward"]
        self.wall_approach_penalty = phase4["wall_approach_penalty"]
        self.near_wall_carry_seconds = phase4["near_wall_carry_seconds"]

        self.breadcrumbs_reached_rolling = _rolling_average(self.breadcrumbs_reached, ROLLING_WINDOW)
        self.breadcrumbs_reached_p05 = _rolling_percentile(self.breadcrumbs_reached, ROLLING_WINDOW, 5)
        self.breadcrumbs_reached_p95 = _rolling_percentile(self.breadcrumbs_reached, ROLLING_WINDOW, 95)
        # Running max of breadcrumbs_reached (for "best so far" line)
        self.breadcrumbs_max_series = []
        running_max = 0.0
        for val in self.breadcrumbs_reached:
            running_max = max(running_max, val)
            self.breadcrumbs_max_series.append(running_max)
        self.avg_reward_rolling = _rolling_average(self.avg_reward, ROLLING_WINDOW)
        # Carry rate (fraction of steps ball is on car) — in-memory only, not persisted
        self.carry_rate = []
        self.carry_rate_rolling = []
        self.carry_rate_p05 = []
        self.carry_rate_p95 = []
        # Avg car speed per episode (uu/s) — in-memory only, not persisted
        self.avg_speed = []
        self.avg_speed_rolling = []
        self.avg_speed_p05 = []
        self.avg_speed_p95 = []
        self.avg_reward_p05 = _rolling_percentile(self.avg_reward, ROLLING_WINDOW, 5)
        self.avg_reward_p95 = _rolling_percentile(self.avg_reward, ROLLING_WINDOW, 95)
        self.wall_approach_penalty_rolling = _rolling_average(self.wall_approach_penalty, ROLLING_WINDOW)
        self.wall_approach_penalty_p05 = _rolling_percentile(self.wall_approach_penalty, ROLLING_WINDOW, 5)
        self.wall_approach_penalty_p95 = _rolling_percentile(self.wall_approach_penalty, ROLLING_WINDOW, 95)

        self.markers = _load_markers()
        self.episode_counter = len(self.carry_seconds)

        self.episode_frame_buffer = deque(maxlen=ROLLING_WINDOW)
        self.last_episode_frames = []
        self.cumulative_timesteps = 0

        if not METRICS_CSV.exists():
            with METRICS_CSV.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode", "carry_seconds", "distance_traveled_uu"])

        if not PHASE4_METRICS_CSV.exists():
            with PHASE4_METRICS_CSV.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "episode",
                        "breadcrumbs_reached",
                        "direction_flips",
                        "avg_reward",
                        "wall_approach_penalty",
                        "near_wall_carry_seconds",
                    ]
                )

        self.dashboard = DribbleDashboard()
        self.dashboard.update(self._build_plot_data(), self.episode_counter, self.dashboard_update_seconds, 0)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["dashboard"] = None
        state["worker_pid"] = None
        state["process_state"] = {}
        for key in (
            "carry_seconds",
            "distances",
            "carry_rolling",
            "carry_p05",
            "carry_p95",
            "phase4_episodes",
            "breadcrumbs_reached",
            "direction_flips",
            "avg_reward",
            "wall_approach_penalty",
            "near_wall_carry_seconds",
            "breadcrumbs_reached_rolling",
            "breadcrumbs_reached_p05",
            "breadcrumbs_reached_p95",
            "avg_reward_rolling",
            "carry_rate",
            "carry_rate_rolling",
            "carry_rate_p05",
            "carry_rate_p95",
            "avg_reward_p05",
            "avg_reward_p95",
            "wall_approach_penalty_rolling",
            "wall_approach_penalty_p05",
            "wall_approach_penalty_p95",
            "markers",
        ):
            state[key] = []
        state["episode_counter"] = 0
        state["episode_frame_buffer"] = deque(maxlen=ROLLING_WINDOW)
        state["last_episode_frames"] = []
        state["cumulative_timesteps"] = 0
        state["breadcrumbs_max_series"] = []
        state["carry_rate"] = []
        state["carry_rate_rolling"] = []
        state["carry_rate_p05"] = []
        state["carry_rate_p95"] = []
        state["avg_speed"] = []
        state["avg_speed_rolling"] = []
        state["avg_speed_p05"] = []
        state["avg_speed_p95"] = []
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
            return [np.zeros(19, dtype=np.float32)]

        car = next(iter(game_state.cars.values()))
        _, _, _, carry_quality = get_dribble_alignment(car, game_state.ball.position)
        carrying = 1.0 if carry_quality > CARRY_THRESHOLD else 0.0
        current_heading = heading_angle(car.physics.forward[:2])

        loop_state = get_current_loop_state()
        turn_direction = 0.0
        breadcrumbs_reached = 0.0
        direction_flips = 0.0
        target_x = 0.0
        target_y = 0.0
        has_target = 0.0
        loop_x = 0.0
        loop_y = 0.0
        episode_reward_sum = 0.0
        if loop_state is not None:
            loop_x = float(loop_state.get("loop_x", 0.0))
            loop_y = float(loop_state.get("loop_y", 0.0))
            turn_direction = float(loop_state.get("turn_direction", 0.0))
            breadcrumbs_reached = float(loop_state.get("breadcrumbs_reached", 0.0))
            direction_flips = float(loop_state.get("direction_flips", 0.0))
            episode_reward_sum = float(loop_state.get("episode_reward_sum", 0.0))
            target_xy = loop_state.get("target_xy")
            if target_xy is not None:
                target_x = float(target_xy[0])
                target_y = float(target_xy[1])
                has_target = 1.0

        wall_approach_penalty = WALL_APPROACH_PENALTY_SCALE * carry_quality * wall_approach_score(
            car.physics.position[:2],
            car.physics.forward[:2],
            car.physics.linear_velocity[:2],
        )
        near_wall_flag = 1.0 if carrying > 0.5 and near_wall_warning_score(car.physics.position[:2]) > 0.0 else 0.0

        car_speed_xy = float(np.linalg.norm(car.physics.linear_velocity[:2]))

        # Metric array layout (20 elements):
        # 0: worker_pid, 1: tick_count, 2: car_x, 3: car_y, 4: heading,
        # 5: ball_x, 6: ball_y, 7: carrying, 8: turn_direction,
        # 9: breadcrumbs_reached, 10: direction_flips, 11: wall_approach_penalty,
        # 12: near_wall_flag, 13: target_x, 14: target_y, 15: has_target,
        # 16: loop_x, 17: loop_y, 18: episode_reward_sum, 19: car_speed_xy
        metric = np.array(
            [
                float(self.worker_pid),
                float(game_state.tick_count),
                float(car.physics.position[0]),
                float(car.physics.position[1]),
                current_heading,
                float(game_state.ball.position[0]),
                float(game_state.ball.position[1]),
                carrying,
                turn_direction,
                breadcrumbs_reached,
                direction_flips,
                wall_approach_penalty,
                near_wall_flag,
                target_x,
                target_y,
                has_target,
                loop_x,
                loop_y,
                episode_reward_sum,
                car_speed_xy,
            ],
            dtype=np.float32,
        )
        # Always emit a fixed-size crumb buffer so shm_view is never resized.
        # Layout: [n_valid, x0, y0, x1, y1, ..., <zeros>] with MAX_TRACKED_CRUMBS slots.
        reached_positions = get_reached_breadcrumb_positions()
        crumb_array = np.zeros(1 + MAX_TRACKED_CRUMBS * 2, dtype=np.float32)
        n_valid = min(len(reached_positions), MAX_TRACKED_CRUMBS)
        crumb_array[0] = float(n_valid)
        for i, pos in enumerate(reached_positions[:n_valid]):
            crumb_array[1 + i * 2] = float(pos[0])
            crumb_array[1 + i * 2 + 1] = float(pos[1])
        return [metric, crumb_array]

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        self.cumulative_timesteps = cumulative_timesteps

        for serialized_metrics in collected_metrics:
            metrics_arrays = self._deserialize(serialized_metrics)
            if not metrics_arrays:
                continue
            metric = np.asarray(metrics_arrays[0], dtype=np.float32)
            if metric.size != 20:
                continue
            reached_crumbs = []
            if len(metrics_arrays) > 1:
                raw = np.asarray(metrics_arrays[1], dtype=np.float32)
                if raw.size >= 1:
                    n_valid = int(raw[0])
                    pairs = raw[1:1 + n_valid * 2]
                    if pairs.size == n_valid * 2 and n_valid > 0:
                        reached_crumbs = pairs.reshape(-1, 2).tolist()
            self._consume_metric(metric, reached_crumbs)

        if wandb_run is not None and self.carry_seconds:
            log_data = {
                "Dribble/Latest Carry Seconds": self.carry_seconds[-1],
                "Dribble/Mean Carry Seconds (Last 20)": float(np.mean(self.carry_seconds[-20:])),
                "Cumulative Timesteps": cumulative_timesteps,
            }
            if self.breadcrumbs_reached:
                log_data.update(
                    {
                        "Dribble/Latest Breadcrumbs Reached": self.breadcrumbs_reached[-1],
                        "Dribble/Latest Direction Flips": self.direction_flips[-1],
                        "Dribble/Latest Avg Reward": self.avg_reward[-1],
                        "Dribble/Latest Wall Approach Penalty": self.wall_approach_penalty[-1],
                        "Dribble/Latest Near-Wall Carry Seconds": self.near_wall_carry_seconds[-1],
                    }
                )
            wandb_run.log(log_data)

        now = time.monotonic()
        if now - self.last_dashboard_time >= self.dashboard_update_seconds:
            self.dashboard.update(
                self._build_plot_data(), self.episode_counter,
                self.dashboard_update_seconds, cumulative_timesteps,
                minimap_frames=self.last_episode_frames,
            )
            self.last_dashboard_time = now

    def _build_plot_data(self):
        tracked_count = len(self.breadcrumbs_reached_rolling)
        tracked_markers = _project_markers_to_tracked(self.markers, self.phase4_episodes)
        return {
            "carry_time": {
                "mean": self.carry_rolling,
                "p05": self.carry_p05,
                "p95": self.carry_p95,
                "markers": self.markers,
                "x_end_label": self.episode_counter,
            },
            "breadcrumbs_reached": {
                "mean": self.breadcrumbs_reached_rolling,
                "p05": self.breadcrumbs_reached_p05,
                "p95": self.breadcrumbs_reached_p95,
                "max_series": self.breadcrumbs_max_series,
                "markers": tracked_markers,
                "x_end_label": tracked_count,
            },
            "avg_speed": {
                "mean": self.avg_speed_rolling,
                "p05": self.avg_speed_p05,
                "p95": self.avg_speed_p95,
                "markers": tracked_markers,
                "x_end_label": tracked_count,
            },
            "wall_approach_penalty": {
                "mean": self.wall_approach_penalty_rolling,
                "p05": self.wall_approach_penalty_p05,
                "p95": self.wall_approach_penalty_p95,
                "markers": tracked_markers,
                "x_end_label": tracked_count,
            },
        }

    def _consume_metric(self, metric, reached_crumbs=()):
        pid = int(metric[0])
        tick_count = int(metric[1])
        position = metric[2:4].astype(np.float32)
        current_heading = float(metric[4])
        ball_position = metric[5:7].astype(np.float32)
        carrying = bool(metric[7] > 0.5)
        turn_direction = float(metric[8])
        breadcrumbs_reached = float(metric[9])
        direction_flips = float(metric[10])
        wall_approach_penalty = float(metric[11])
        near_wall_flag = bool(metric[12] > 0.5)
        target_position = metric[13:15].astype(np.float32)
        has_target = bool(metric[15] > 0.5)
        loop_x = float(metric[16])
        loop_y = float(metric[17])
        episode_reward_sum = float(metric[18])
        car_speed_xy = float(metric[19])

        state = self.process_state.get(pid)
        if state is None:
            self.process_state[pid] = {
                "last_tick": tick_count,
                "last_pos": position,
                "carry_steps": 1 if carrying else 0,
                "total_steps": 1,
                "distance": 0.0,
                "speed_sum": car_speed_xy,
                "breadcrumbs_reached": breadcrumbs_reached,
                "direction_flips": direction_flips,
                "episode_reward_sum": episode_reward_sum,
                "wall_approach_penalty": wall_approach_penalty,
                "near_wall_carry_steps": 1 if carrying and near_wall_flag else 0,
                "crumb_positions": [],
                "frames": [self._build_episode_frame(
                    position, current_heading, ball_position, target_position, has_target, loop_x, loop_y,
                    reached_crumbs=[],
                )],
            }
            return

        if tick_count <= state["last_tick"]:
            self._finalize_episode(state)
            state["last_tick"] = tick_count
            state["last_pos"] = position
            state["carry_steps"] = 1 if carrying else 0
            state["total_steps"] = 1
            state["distance"] = 0.0
            state["breadcrumbs_reached"] = breadcrumbs_reached
            state["direction_flips"] = direction_flips
            state["episode_reward_sum"] = episode_reward_sum
            state["wall_approach_penalty"] = wall_approach_penalty
            state["near_wall_carry_steps"] = 1 if carrying and near_wall_flag else 0
            state["speed_sum"] = car_speed_xy
            state["crumb_positions"] = []
            state["frames"] = [self._build_episode_frame(
                position, current_heading, ball_position, target_position, has_target, loop_x, loop_y,
                reached_crumbs=[],
            )]
            return

        # Detect when a new crumb is reached and record car position.
        # This avoids relying on REACHED_BREADCRUMB_POSITIONS globals, which are cleared
        # by env.reset() before collect_metrics is called on the terminal step.
        if breadcrumbs_reached > state["breadcrumbs_reached"]:
            state["crumb_positions"].append([float(position[0]), float(position[1])])

        step_distance = float(np.linalg.norm(position - state["last_pos"]))
        state["distance"] += step_distance
        state["carry_steps"] += 1 if carrying else 0
        state["total_steps"] = state.get("total_steps", 0) + 1
        state["speed_sum"] = state.get("speed_sum", 0.0) + car_speed_xy
        if carrying and near_wall_flag:
            state["near_wall_carry_steps"] += 1
        state["breadcrumbs_reached"] = max(state["breadcrumbs_reached"], breadcrumbs_reached)
        state["direction_flips"] = max(state["direction_flips"], direction_flips)
        # episode_reward_sum is reset to 0 when env.reset() runs before collect_metrics.
        # Only overwrite when the new value is non-zero or we're still at zero (first steps).
        if episode_reward_sum != 0.0 or state["episode_reward_sum"] == 0.0:
            state["episode_reward_sum"] = episode_reward_sum
        state["wall_approach_penalty"] += wall_approach_penalty
        state["last_pos"] = position
        state["last_tick"] = tick_count
        state["frames"].append(self._build_episode_frame(
            position, current_heading, ball_position, target_position, has_target, loop_x, loop_y,
            reached_crumbs=list(state["crumb_positions"]),
        ))

    def _finalize_episode(self, state):
        carry_seconds = state["carry_steps"] / DECISIONS_PER_SECOND
        distance = state["distance"]
        breadcrumbs_reached = state["breadcrumbs_reached"]
        direction_flips = state["direction_flips"]
        total_steps = max(state.get("total_steps", 1), 1)
        avg_reward = state["episode_reward_sum"] / total_steps
        carry_rate = state["carry_steps"] / total_steps
        avg_speed = state.get("speed_sum", 0.0) / total_steps
        wall_approach_penalty = state["wall_approach_penalty"]
        near_wall_carry_seconds = state["near_wall_carry_steps"] / DECISIONS_PER_SECOND
        self.episode_counter += 1

        _append_rolling_stats(self.carry_seconds, self.carry_rolling, self.carry_p05, self.carry_p95, carry_seconds)
        self.distances.append(distance)

        self.phase4_episodes.append(self.episode_counter)
        _append_rolling_stats(
            self.breadcrumbs_reached,
            self.breadcrumbs_reached_rolling,
            self.breadcrumbs_reached_p05,
            self.breadcrumbs_reached_p95,
            breadcrumbs_reached,
        )
        self.direction_flips.append(direction_flips)
        _append_rolling_stats(
            self.avg_reward,
            self.avg_reward_rolling,
            self.avg_reward_p05,
            self.avg_reward_p95,
            avg_reward,
        )
        _append_rolling_stats(
            self.carry_rate,
            self.carry_rate_rolling,
            self.carry_rate_p05,
            self.carry_rate_p95,
            carry_rate,
        )
        _append_rolling_stats(
            self.avg_speed,
            self.avg_speed_rolling,
            self.avg_speed_p05,
            self.avg_speed_p95,
            avg_speed,
        )
        _append_rolling_stats(
            self.wall_approach_penalty,
            self.wall_approach_penalty_rolling,
            self.wall_approach_penalty_p05,
            self.wall_approach_penalty_p95,
            wall_approach_penalty,
        )
        self.near_wall_carry_seconds.append(near_wall_carry_seconds)

        with METRICS_CSV.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([self.episode_counter, carry_seconds, distance])

        with PHASE4_METRICS_CSV.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    self.episode_counter,
                    breadcrumbs_reached,
                    direction_flips,
                    avg_reward,
                    wall_approach_penalty,
                    near_wall_carry_seconds,
                ]
            )

        episode_frames = list(state.get("frames", []))
        # The last frame was collected AFTER env.reset() cleared globals (loop_x=0, crumbs=[]).
        # Drop it so GIFs and the minimap show only in-episode data.
        if len(episode_frames) > 1:
            episode_frames = episode_frames[:-1]

        # Track running max and save a GIF only when a new crumb record is set.
        prev_max = self.breadcrumbs_max_series[-1] if self.breadcrumbs_max_series else 0.0
        new_max = max(prev_max, breadcrumbs_reached)
        self.breadcrumbs_max_series.append(new_max)
        if new_max > prev_max and episode_frames:
            self._save_record_gif(breadcrumbs_reached, carry_seconds, episode_frames)

        self.episode_frame_buffer.append((breadcrumbs_reached, carry_seconds, episode_frames))
        self.last_episode_frames = episode_frames

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

    @staticmethod
    def _build_episode_frame(position, current_heading, ball_position, target_position, has_target, loop_x, loop_y, reached_crumbs=()):
        return {
            "car_x": float(position[0]),
            "car_y": float(position[1]),
            "heading": float(current_heading),
            "ball_x": float(ball_position[0]),
            "ball_y": float(ball_position[1]),
            "target_x": float(target_position[0]),
            "target_y": float(target_position[1]),
            "has_target": bool(has_target),
            "loop_x": float(loop_x),
            "loop_y": float(loop_y),
            "reached_crumbs": [list(c) for c in reached_crumbs],
        }

    def _save_record_gif(self, breadcrumbs_reached, carry_seconds, episode_frames):
        sampled_frames = episode_frames[::EPISODE_GIF_FRAME_STRIDE]
        if sampled_frames[-1] is not episode_frames[-1]:
            sampled_frames.append(episode_frames[-1])

        images = []
        for index, frame in enumerate(sampled_frames):
            images.append(
                _draw_episode_frame(
                    frame,
                    (
                        f"NEW RECORD | ep {self.episode_counter} | "
                        f"{breadcrumbs_reached:.0f} crumbs | "
                        f"carry {carry_seconds:.2f}s | "
                        f"frame {index + 1}/{len(sampled_frames)}"
                    ),
                    trail=episode_frames,
                )
            )

        filename = f"record_ep{self.episode_counter:07d}_{breadcrumbs_reached:.0f}crumbs_{carry_seconds:.2f}s.gif"
        output_path = P95_GIFS_DIR / filename
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=EPISODE_GIF_FRAME_DURATION_MS,
            loop=0,
            optimize=False,
        )
        print(f"New crumb record GIF: {output_path}")
