"""
Dashboard and metrics logger for the Power Shot scenario.
Similar structure to dribble_metrics.py: collect_metrics / report_metrics hooks
for rlgym_ppo, plus a Tkinter live dashboard.
"""

import csv
import math
import os
import time
import tkinter as tk
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from rlgym.rocket_league import common_values

ROOT_DIR = Path(__file__).resolve().parent
METRICS_DIR = ROOT_DIR / "metrics"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
POWER_SHOT_METRICS_CSV = METRICS_DIR / "power_shot_episode_metrics.csv"

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
LEGEND_FONT = ("Helvetica", 9, "bold")

BACK_WALL_Y = 5120.0
SIDE_WALL_X = 4096.0
BALL_MAX_SPEED = 6000.0
ORANGE_GOAL_Y = BACK_WALL_Y
GOAL_HALF_WIDTH = 892.755

# Metric array layout (15 elements):
# 0: worker_pid, 1: tick_count, 2: car_x, 3: car_y, 4: heading,
# 5: ball_x, 6: ball_y, 7: goal_scored, 8: scoring_team,
# 9: ball_speed, 10: episode_reward_sum, 11: car_speed, 12: boost_amount,
# 13: car_vel_x, 14: car_vel_y
METRIC_SIZE = 15
DECISIONS_PER_SECOND = 120.0 / 8  # tick_rate / tick_skip


# ---------------------------------------------------------------------------
# Helper functions (shared with dribble_metrics)
# ---------------------------------------------------------------------------
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
        float(np.percentile(arr[:i + 1], percentile))
        for i in range(min(window - 1, len(arr)))
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
        return np.arange(arr.size, dtype=np.float32), arr
    bin_edges = np.linspace(0, arr.size, max_points + 1, dtype=np.int32)
    x_values, y_values = [], []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end <= start:
            continue
        segment = arr[start:end]
        x_values.append((start + end - 1) / 2)
        y_values.append(float(segment.mean(dtype=np.float64)))
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def _lighten_color(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
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


def _load_csv_history():
    """Load persisted episode metrics. Returns dict of lists."""
    history = {
        "episode_seconds": [],
        "shot_speed": [],        # ball speed at goal, 0 if no goal
        "goal_scored": [],       # 1.0 if blue goal, 0.0 otherwise
        "avg_reward": [],
        "boost_used": [],
        "direction_flips": [],
    }
    if not POWER_SHOT_METRICS_CSV.exists():
        return history
    with POWER_SHOT_METRICS_CSV.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return history
        for row in reader:
            if len(row) < 4:
                continue
            try:
                history["episode_seconds"].append(float(row[1]))
                history["shot_speed"].append(float(row[2]))
                history["goal_scored"].append(float(row[3]))
                history["avg_reward"].append(float(row[4]) if len(row) > 4 else 0.0)
                history["boost_used"].append(float(row[5]) if len(row) > 5 else 0.0)
                history["direction_flips"].append(float(row[6]) if len(row) > 6 else 0.0)
            except ValueError:
                continue
    return history


# ---------------------------------------------------------------------------
# Field drawing helpers
# ---------------------------------------------------------------------------
def _field_scale(w, h, padding):
    return min((w - 2 * padding) / (2 * SIDE_WALL_X), (h - 2 * padding) / (2 * BACK_WALL_Y))


def _field_to_canvas(xy, w, h, scale):
    x = w / 2 + float(xy[0]) * scale
    y = h / 2 - float(xy[1]) * scale
    return x, y


def _rotated_triangle(cx, cy, angle, length=30, width=20):
    pts_local = [(length / 2, 0), (-length / 2, -width / 2), (-length / 2, width / 2)]
    out = []
    for dx, dy in pts_local:
        px = cx + dx * math.cos(angle) - dy * math.sin(angle)
        py = cy - (dx * math.sin(angle) + dy * math.cos(angle))
        out.append((px, py))
    return out


def _draw_power_shot_frame(frame, title, canvas_width=800, canvas_height=600, padding=30, trail=None):
    image = Image.new("RGB", (canvas_width, canvas_height), "#1e1b4b")
    draw = ImageDraw.Draw(image)
    scale = _field_scale(canvas_width, canvas_height, padding)
    ball_r = max(6, int(91.25 * scale))

    def to_canvas(pos):
        return _field_to_canvas(pos, canvas_width, canvas_height, scale)

    # Field boundary
    left, top = to_canvas((-SIDE_WALL_X, BACK_WALL_Y))
    right, bottom = to_canvas((SIDE_WALL_X, -BACK_WALL_Y))
    draw.rectangle((left, top, right, bottom), outline="#6366f1", width=3)

    # Midfield line
    mid_left, mid_y = to_canvas((-SIDE_WALL_X, 0))
    mid_right, _ = to_canvas((SIDE_WALL_X, 0))
    draw.line((mid_left, mid_y, mid_right, mid_y), fill="#6366f1", width=1)

    # Center circle
    cx_canvas = canvas_width / 2
    cy_canvas = canvas_height / 2
    cr = int(600 * scale)
    draw.ellipse((cx_canvas - cr, cy_canvas - cr, cx_canvas + cr, cy_canvas + cr), outline="#6366f1", width=1)

    # Orange goal (top of field, +y)
    goal_lx, goal_ty = to_canvas((-GOAL_HALF_WIDTH, BACK_WALL_Y))
    goal_rx, goal_by = to_canvas((GOAL_HALF_WIDTH, BACK_WALL_Y - 200))
    draw.rectangle((goal_lx, goal_ty, goal_rx, goal_by), fill="#f97316", outline="#ea580c", width=2)
    draw.text((cx_canvas, goal_ty + 10), "ORANGE GOAL", fill="#fed7aa", anchor="mt")

    # Blue goal (bottom of field, -y)
    blue_lx, blue_ty = to_canvas((-GOAL_HALF_WIDTH, -BACK_WALL_Y + 200))
    blue_rx, blue_by = to_canvas((GOAL_HALF_WIDTH, -BACK_WALL_Y))
    draw.rectangle((blue_lx, blue_ty, blue_rx, blue_by), fill="#3b82f6", outline="#1d4ed8", width=2)

    # Car trail
    if trail and len(trail) >= 2:
        trail_pts = [to_canvas((f["car_x"], f["car_y"])) for f in trail]
        draw.line(trail_pts, fill="#93c5fd", width=2)

    # Ball
    ball_pos = to_canvas((frame["ball_x"], frame["ball_y"]))
    draw.ellipse((ball_pos[0] - ball_r, ball_pos[1] - ball_r,
                  ball_pos[0] + ball_r, ball_pos[1] + ball_r), fill="#f8fafc")

    # Car
    car_pos = to_canvas((frame["car_x"], frame["car_y"]))
    draw.polygon(_rotated_triangle(car_pos[0], car_pos[1], frame["heading"]), fill="#55a4ff", outline="#0f172a")

    draw.text((16, 14), title, fill="#f8fafc")
    return image


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class PowerShotDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Power Shot Training Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        screen_w = max(int(self.root.winfo_screenwidth()), MIN_PLOT_WIDTH)
        screen_h = max(int(self.root.winfo_screenheight()), MIN_PLOT_HEIGHT)
        self.plot_width = max(MIN_PLOT_WIDTH, min(MAX_PLOT_WIDTH, screen_w - 120))
        self.plot_height = max(MIN_PLOT_HEIGHT, min(MAX_PLOT_HEIGHT, screen_h - 140))
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
            PLOT_PADDING, 22,
            anchor="w",
            text=(
                f"Episodes: {episode_count}    "
                f"Timesteps: {ts_display}    "
                f"Refresh: every {update_seconds:.1f}s    "
                f"Updated: {time.strftime('%H:%M:%S')}"
            ),
            fill="#111827", font=HEADER_FONT,
        )

        minimap_width = int(self.plot_width * 0.32)
        charts_width = self.plot_width - minimap_width - PLOT_PADDING
        top_y = 60
        chart_gap_x = PLOT_PADDING
        chart_gap_y = 60
        chart_w = (charts_width - chart_gap_x) / 2
        available_height = self.plot_height - top_y - PLOT_PADDING
        chart_h = max(180, (available_height - chart_gap_y) / 2)

        # Top-left: Episode Duration
        self._draw_plot(
            x0=PLOT_PADDING, y0=top_y,
            width=chart_w, height=chart_h,
            spec=plot_data.get("episode_seconds", {}),
            title="Episode Duration (s)", point_color="#2563eb", x_label_prefix="Episode",
        )

        # Top-right: Shot Speed on Goal (uu/s)
        self._draw_plot(
            x0=PLOT_PADDING + chart_w + chart_gap_x, y0=top_y,
            width=chart_w, height=chart_h,
            spec=plot_data.get("shot_speed", {}),
            title="Shot Speed on Goal (uu/s)", point_color="#dc2626", x_label_prefix="Episode",
        )

        # Bottom-left: Goals Scored Rate (rolling %)
        self._draw_plot(
            x0=PLOT_PADDING, y0=top_y + chart_h + chart_gap_y,
            width=chart_w, height=chart_h,
            spec=plot_data.get("goal_rate", {}),
            title="Goal Rate (rolling %)", point_color="#16a34a", x_label_prefix="Episode",
        )

        # Bottom-right: Boost Used (left/orange) + Direction Flips (right/teal)
        self._draw_dual_plot(
            x0=PLOT_PADDING + chart_w + chart_gap_x, y0=top_y + chart_h + chart_gap_y,
            width=chart_w, height=chart_h,
            spec1=plot_data.get("boost_used", {}),
            spec2=plot_data.get("direction_flips", {}),
            title="Boost & Flips Per Episode",
            color1="#f97316",
            color2="#0891b2",
            label1="boost",
            label2="flips",
        )

        # Right panel: minimap + shot speed histogram + top shots leaderboard
        minimap_x = self.plot_width - minimap_width
        minimap_y = top_y
        panel_h = 175
        minimap_h = available_height - panel_h - 10
        self._render_minimap(minimap_frames, minimap_x, minimap_y, minimap_width, minimap_h)
        panel_y = minimap_y + minimap_h + 10
        hist_w = int(minimap_width * 0.58)
        lb_w = minimap_width - hist_w - 4
        self._render_shot_speed_hist(
            plot_data.get("shot_speed_hist", {}),
            minimap_x, panel_y, hist_w, panel_h,
        )
        self._render_top_shots(
            plot_data.get("top_shots", []),
            minimap_x + hist_w + 4, panel_y, lb_w, panel_h,
        )

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
        image = _draw_power_shot_frame(
            last_frame, "", canvas_width=width, canvas_height=height, padding=20, trail=frames,
        )
        self._minimap_photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(x0, y0, anchor="nw", image=self._minimap_photo)

    def _render_shot_speed_hist(self, speed_hist, x0, y0, width, height):
        x1 = x0 + width
        y1 = y0 + height
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#9ca3af", width=2)
        self.canvas.create_text(
            x0 + width / 2, y0 + 14,
            anchor="center", text="Shot Speed Dist (last 100k)", fill="#111827", font=TITLE_FONT,
        )
        if not speed_hist:
            self.canvas.create_text(
                x0 + width / 2, y0 + height / 2,
                text="No goals yet", fill="#6b7280", font=AXIS_FONT,
            )
            return

        pad_l, pad_r, pad_top, pad_bot = 28, 12, 28, 22
        cx0 = x0 + pad_l
        cx1 = x1 - pad_r
        cy0 = y0 + pad_top
        cy1 = y1 - pad_bot
        cw = cx1 - cx0
        ch = cy1 - cy0

        buckets = sorted(speed_hist.items())
        n = len(buckets)
        if n == 0:
            return
        max_count = max(c for _, c in buckets) if buckets else 1
        bar_w = max(2, cw / n - 1)
        gap = (cw - bar_w * n) / max(n - 1, 1)

        for i, (bucket_label, count) in enumerate(buckets):
            bx = cx0 + i * (bar_w + gap)
            bar_h = int(ch * count / max_count) if max_count > 0 else 0
            by0 = cy1 - bar_h
            # Color by speed bucket: green = fast, amber = medium, grey = slow
            speed_val = bucket_label * 500  # bucket_label is speed // 500
            color = "#22c55e" if speed_val >= 4000 else ("#f59e0b" if speed_val >= 2000 else "#6b7280")
            if bar_h > 0:
                self.canvas.create_rectangle(bx, by0, bx + bar_w, cy1, fill=color, outline="")
            if i % 2 == 0:
                self.canvas.create_text(
                    bx + bar_w / 2, cy1 + 4, anchor="n",
                    text=f"{speed_val // 1000}k" if speed_val >= 1000 else str(speed_val),
                    fill="#6b7280", font=AXIS_FONT,
                )

        self.canvas.create_text(
            x0 + pad_l - 4, cy0, anchor="e",
            text=f"{max_count:,}", fill="#6b7280", font=AXIS_FONT,
        )
        self.canvas.create_line(cx0, cy0, cx0, cy1, fill="#e5e7eb")
        self.canvas.create_line(cx0, cy1, cx1, cy1, fill="#e5e7eb")

    def _render_top_shots(self, top_shots, x0, y0, width, height):
        x1 = x0 + width
        y1 = y0 + height
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#9ca3af", width=2)
        self.canvas.create_text(
            x0 + width / 2, y0 + 14,
            anchor="center", text="Top 5 Shots", fill="#111827", font=TITLE_FONT,
        )
        if not top_shots:
            self.canvas.create_text(
                x0 + width / 2, y0 + height / 2,
                text="No goals yet", fill="#6b7280", font=AXIS_FONT,
            )
            return

        row_h = (height - 30) / 5
        medals = ["#f59e0b", "#9ca3af", "#b45309", "#6b7280", "#6b7280"]
        for i, (speed, episode) in enumerate(top_shots):
            cy = y0 + 28 + i * row_h
            color = medals[i]
            self.canvas.create_text(
                x0 + 10, cy, anchor="w",
                text=f"#{i+1}", fill=color, font=("Helvetica", 9, "bold"),
            )
            self.canvas.create_text(
                x0 + 30, cy, anchor="w",
                text=f"{speed:.0f} uu/s", fill="#111827", font=AXIS_FONT,
            )
            self.canvas.create_text(
                x0 + width - 8, cy, anchor="e",
                text=f"ep{episode}", fill="#6b7280", font=AXIS_FONT,
            )

    def _draw_plot(self, x0, y0, width, height, spec, title, point_color, x_label_prefix):
        x1 = x0 + width
        y1 = y0 + height
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#9ca3af", width=2)
        self.canvas.create_text(x0, y0 - 16, anchor="w", text=title, fill="#111827", font=TITLE_FONT)

        mean_values = spec.get("mean", [])
        low_values = spec.get("p05", [])
        high_values = spec.get("p95", [])
        max_series = spec.get("max_series", [])

        if not mean_values:
            self.canvas.create_text(
                x0 + width / 2, y0 + height / 2,
                text="Waiting for data...", fill="#6b7280", font=("Helvetica", 11),
            )
            return

        target_points = max(60, min(MAX_RENDER_POINTS, int(width)))
        plot_x, plot_mean = _compress_plot_series(mean_values, target_points)
        _, plot_low = _compress_plot_series(low_values, target_points)
        _, plot_high = _compress_plot_series(high_values, target_points)
        min_val = float(min(plot_low.min(), plot_mean.min(), plot_high.min()))
        max_val = float(max(plot_low.max(), plot_mean.max(), plot_high.max()))
        if max_series:
            plot_mx, plot_ms = _compress_plot_series(max_series, target_points)
            if plot_ms.size > 0:
                max_val = float(max(max_val, float(plot_ms.max())))
        if max_val - min_val < 1e-6:
            max_val = min_val + 1.0

        for tick in range(5):
            frac = tick / 4
            y = y1 - frac * height
            label_val = min_val + frac * (max_val - min_val)
            self.canvas.create_line(x0, y, x1, y, fill="#e5e7eb")
            self.canvas.create_text(x0 - 8, y, anchor="e", text=f"{label_val:.1f}", fill="#6b7280", font=AXIS_FONT)

        self.canvas.create_text(x0, y1 + 14, anchor="w", text=f"{x_label_prefix} 1", fill="#6b7280", font=AXIS_FONT)
        end_label = spec.get("x_end_label", len(mean_values))
        self.canvas.create_text(x1, y1 + 14, anchor="e", text=f"{x_label_prefix} {end_label}", fill="#6b7280", font=AXIS_FONT)

        # Legend
        light = _lighten_color(point_color)
        has_max = bool(max_series)
        base_y = y0 - 14
        if has_max:
            self.canvas.create_line(x1 - 215, base_y, x1 - 195, base_y, fill=point_color, width=3)
            self.canvas.create_text(x1 - 191, base_y, anchor="w", text="mean", fill=point_color, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 153, base_y, x1 - 133, base_y, fill=light, width=2, dash=(6, 6))
            self.canvas.create_text(x1 - 129, base_y, anchor="w", text="5th", fill=light, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 97, base_y, x1 - 77, base_y, fill=light, width=2, dash=(6, 6))
            self.canvas.create_text(x1 - 73, base_y, anchor="w", text="95th", fill=light, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 41, base_y, x1 - 21, base_y, fill="#f59e0b", width=2)
            self.canvas.create_text(x1 - 17, base_y, anchor="w", text="best", fill="#f59e0b", font=LEGEND_FONT)
        else:
            self.canvas.create_line(x1 - 165, base_y, x1 - 145, base_y, fill=point_color, width=3)
            self.canvas.create_text(x1 - 141, base_y, anchor="w", text="mean", fill=point_color, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 103, base_y, x1 - 83, base_y, fill=light, width=2, dash=(6, 6))
            self.canvas.create_text(x1 - 79, base_y, anchor="w", text="5th", fill=light, font=LEGEND_FONT)
            self.canvas.create_line(x1 - 47, base_y, x1 - 27, base_y, fill=light, width=2, dash=(6, 6))
            self.canvas.create_text(x1 - 23, base_y, anchor="w", text="95th", fill=light, font=LEGEND_FONT)

        # Bands
        legend_color = light
        low_pts, high_pts, mean_pts = [], [], []
        n_ref = max(len(mean_values) - 1, 1)
        for xv, lo, mn, hi in zip(plot_x, plot_low, plot_mean, plot_high):
            x = x0 + (float(xv) / n_ref) * width
            low_pts.extend([x, y1 - ((float(lo) - min_val) / (max_val - min_val)) * height])
            high_pts.extend([x, y1 - ((float(hi) - min_val) / (max_val - min_val)) * height])
            mean_pts.extend([x, y1 - ((float(mn) - min_val) / (max_val - min_val)) * height])

        if len(low_pts) >= 4:
            self.canvas.create_line(*low_pts, fill=legend_color, width=2, dash=(6, 6), smooth=True)
        if len(high_pts) >= 4:
            self.canvas.create_line(*high_pts, fill=legend_color, width=2, dash=(6, 6), smooth=True)
        if len(mean_pts) >= 4:
            self.canvas.create_line(*mean_pts, fill=point_color, width=3, smooth=True)

        if has_max and plot_ms.size >= 2:
            max_pts = []
            for xv, mv in zip(plot_mx, plot_ms):
                x = x0 + (float(xv) / n_ref) * width
                max_pts.extend([x, y1 - ((float(mv) - min_val) / (max_val - min_val)) * height])
            if len(max_pts) >= 4:
                self.canvas.create_line(*max_pts, fill="#f59e0b", width=2, smooth=False)

    def _draw_dual_plot(self, x0, y0, width, height, spec1, spec2, title, color1, color2, label1, label2):
        """Dual-axis chart: spec1 (mean+bands) on left axis, spec2 (mean only) on right axis."""
        x1 = x0 + width
        y1 = y0 + height
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#9ca3af", width=2)
        self.canvas.create_text(x0, y0 - 16, anchor="w", text=title, fill="#111827", font=TITLE_FONT)

        mean1 = spec1.get("mean", [])
        p05_1 = spec1.get("p05", [])
        p95_1 = spec1.get("p95", [])
        mean2 = spec2.get("mean", [])
        x_end_label = spec1.get("x_end_label") or spec2.get("x_end_label")

        if not mean1 and not mean2:
            self.canvas.create_text(
                x0 + width / 2, y0 + height / 2,
                text="Waiting for data...", fill="#6b7280", font=("Helvetica", 11),
            )
            return

        target_points = max(60, min(MAX_RENDER_POINTS, int(width)))

        if mean1:
            plot_x1, plot_m1 = _compress_plot_series(mean1, target_points)
            _, plot_l1 = _compress_plot_series(p05_1, target_points) if p05_1 else (plot_x1, plot_m1.copy())
            _, plot_h1 = _compress_plot_series(p95_1, target_points) if p95_1 else (plot_x1, plot_m1.copy())
            min1 = float(min(float(plot_l1.min()), float(plot_m1.min()), float(plot_h1.min())))
            max1 = float(max(float(plot_l1.max()), float(plot_m1.max()), float(plot_h1.max())))
            if max1 - min1 < 1e-6:
                max1 = min1 + 1.0
            n_ref1 = max(len(mean1) - 1, 1)
        else:
            plot_x1 = plot_m1 = plot_l1 = plot_h1 = np.empty(0)
            min1, max1, n_ref1 = 0.0, 1.0, 1

        if mean2:
            plot_x2, plot_m2 = _compress_plot_series(mean2, target_points)
            min2 = float(plot_m2.min())
            max2 = float(plot_m2.max())
            if max2 - min2 < 1e-6:
                max2 = min2 + 1.0
            n_ref2 = max(len(mean2) - 1, 1)
        else:
            plot_x2 = plot_m2 = np.empty(0)
            min2, max2, n_ref2 = 0.0, 1.0, 1

        for tick in range(5):
            frac = tick / 4
            y = y1 - frac * height
            self.canvas.create_line(x0, y, x1, y, fill="#e5e7eb")
            self.canvas.create_text(
                x0 - 8, y, anchor="e",
                text=f"{min1 + frac * (max1 - min1):.0f}",
                fill=color1, font=AXIS_FONT,
            )
            if mean2:
                self.canvas.create_text(
                    x1 + 8, y, anchor="w",
                    text=f"{min2 + frac * (max2 - min2):.1f}",
                    fill=color2, font=AXIS_FONT,
                )

        self.canvas.create_text(x0, y1 + 14, anchor="w", text="Episode 1", fill="#6b7280", font=AXIS_FONT)
        end_val = len(mean1 or mean2) if x_end_label is None else x_end_label
        self.canvas.create_text(x1, y1 + 14, anchor="e", text=f"Episode {end_val}", fill="#6b7280", font=AXIS_FONT)

        light1 = _lighten_color(color1)
        base_y = y0 - 14
        self.canvas.create_line(x1 - 215, base_y, x1 - 195, base_y, fill=color1, width=3)
        self.canvas.create_text(x1 - 191, base_y, anchor="w", text=label1, fill=color1, font=LEGEND_FONT)
        self.canvas.create_line(x1 - 153, base_y, x1 - 133, base_y, fill=light1, width=2, dash=(6, 6))
        self.canvas.create_text(x1 - 129, base_y, anchor="w", text="5/95th", fill=light1, font=LEGEND_FONT)
        self.canvas.create_line(x1 - 70, base_y, x1 - 50, base_y, fill=color2, width=3)
        self.canvas.create_text(x1 - 46, base_y, anchor="w", text=label2, fill=color2, font=LEGEND_FONT)

        if plot_m1.size >= 2:
            if plot_l1.size >= 2:
                pts = []
                for xv, v in zip(plot_x1, plot_l1):
                    pts.extend([x0 + (float(xv) / n_ref1) * width,
                                 y1 - ((float(v) - min1) / (max1 - min1)) * height])
                self.canvas.create_line(*pts, fill=light1, width=2, dash=(6, 6), smooth=True)
            if plot_h1.size >= 2:
                pts = []
                for xv, v in zip(plot_x1, plot_h1):
                    pts.extend([x0 + (float(xv) / n_ref1) * width,
                                 y1 - ((float(v) - min1) / (max1 - min1)) * height])
                self.canvas.create_line(*pts, fill=light1, width=2, dash=(6, 6), smooth=True)
            pts = []
            for xv, v in zip(plot_x1, plot_m1):
                pts.extend([x0 + (float(xv) / n_ref1) * width,
                             y1 - ((float(v) - min1) / (max1 - min1)) * height])
            self.canvas.create_line(*pts, fill=color1, width=3, smooth=True)

        if plot_m2.size >= 2:
            pts = []
            for xv, v in zip(plot_x2, plot_m2):
                pts.extend([x0 + (float(xv) / n_ref2) * width,
                             y1 - ((float(v) - min2) / (max2 - min2)) * height])
            self.canvas.create_line(*pts, fill=color2, width=3, smooth=True)


# ---------------------------------------------------------------------------
# Metrics Logger (plugs into rlgym_ppo MetricsLogger interface)
# ---------------------------------------------------------------------------
class PowerShotMetricsLogger:
    def __init__(self, dashboard_update_seconds=1.0):
        self.worker_pid = None
        self.process_state = {}
        self.dashboard_update_seconds = max(0.25, float(dashboard_update_seconds))
        self.last_dashboard_time = 0.0

        METRICS_DIR.mkdir(exist_ok=True)

        history = _load_csv_history()
        self.episode_seconds = history["episode_seconds"]
        self.shot_speeds = history["shot_speed"]
        self.goal_scored = history["goal_scored"]
        self.avg_reward = history["avg_reward"]

        # Rolling stats
        self.ep_sec_rolling = _rolling_average(self.episode_seconds, ROLLING_WINDOW)
        self.ep_sec_p05 = _rolling_percentile(self.episode_seconds, ROLLING_WINDOW, 5)
        self.ep_sec_p95 = _rolling_percentile(self.episode_seconds, ROLLING_WINDOW, 95)

        self.shot_speed_rolling = _rolling_average(self.shot_speeds, ROLLING_WINDOW)
        self.shot_speed_p05 = _rolling_percentile(self.shot_speeds, ROLLING_WINDOW, 5)
        self.shot_speed_p95 = _rolling_percentile(self.shot_speeds, ROLLING_WINDOW, 95)
        self.shot_speed_max_series = []
        running_max = 0.0
        for v in self.shot_speeds:
            running_max = max(running_max, v)
            self.shot_speed_max_series.append(running_max)

        self.goal_rate_rolling = _rolling_average(self.goal_scored, ROLLING_WINDOW)
        self.goal_rate_p05 = _rolling_percentile(self.goal_scored, ROLLING_WINDOW, 5)
        self.goal_rate_p95 = _rolling_percentile(self.goal_scored, ROLLING_WINDOW, 95)

        self.avg_reward_rolling = _rolling_average(self.avg_reward, ROLLING_WINDOW)
        self.avg_reward_p05 = _rolling_percentile(self.avg_reward, ROLLING_WINDOW, 5)
        self.avg_reward_p95 = _rolling_percentile(self.avg_reward, ROLLING_WINDOW, 95)

        self.boost_used = history["boost_used"]
        self.boost_used_rolling = _rolling_average(self.boost_used, ROLLING_WINDOW)
        self.boost_used_p05 = _rolling_percentile(self.boost_used, ROLLING_WINDOW, 5)
        self.boost_used_p95 = _rolling_percentile(self.boost_used, ROLLING_WINDOW, 95)

        self.direction_flips = history["direction_flips"]
        self.direction_flips_rolling = _rolling_average(self.direction_flips, ROLLING_WINDOW)

        # Top-5 shots by speed: list of (speed, episode)
        self.top_shots = sorted(
            [(float(s), i + 1) for i, s in enumerate(self.shot_speeds) if s > 0],
            key=lambda x: x[0], reverse=True,
        )[:5]

        self.episode_counter = len(self.episode_seconds)
        self.last_episode_frames = []
        self.cumulative_timesteps = 0

        if not POWER_SHOT_METRICS_CSV.exists():
            with POWER_SHOT_METRICS_CSV.open("w", newline="") as handle:
                csv.writer(handle).writerow(["episode", "episode_seconds", "shot_speed", "goal_scored", "avg_reward"])

        self.dashboard = PowerShotDashboard()
        self.dashboard.update(self._build_plot_data(), self.episode_counter, self.dashboard_update_seconds, 0)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["dashboard"] = None
        state["worker_pid"] = None
        state["process_state"] = {}
        for key in (
            "episode_seconds", "shot_speeds", "goal_scored", "avg_reward",
            "boost_used", "direction_flips",
            "ep_sec_rolling", "ep_sec_p05", "ep_sec_p95",
            "shot_speed_rolling", "shot_speed_p05", "shot_speed_p95", "shot_speed_max_series",
            "goal_rate_rolling", "goal_rate_p05", "goal_rate_p95",
            "avg_reward_rolling", "avg_reward_p05", "avg_reward_p95",
            "boost_used_rolling", "boost_used_p05", "boost_used_p95",
            "direction_flips_rolling",
            "top_shots",
        ):
            state[key] = []
        state["episode_counter"] = 0
        state["last_episode_frames"] = []
        state["cumulative_timesteps"] = 0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def collect_metrics(self, game_state) -> np.ndarray:
        metrics_arrays = self._collect_metrics(game_state)
        unraveled = []
        for arr in metrics_arrays:
            shape = np.shape(arr)
            unraveled.append(len(shape))
            unraveled += list(shape)
            unraveled += np.ravel(arr).tolist()
        return np.asarray(unraveled, dtype=np.float32)

    def _collect_metrics(self, game_state):
        if self.worker_pid is None:
            self.worker_pid = os.getpid()

        if not game_state.cars:
            return [np.zeros(METRIC_SIZE, dtype=np.float32)]

        car = next(iter(game_state.cars.values()))
        car_pos = car.physics.position
        car_vel = car.physics.linear_velocity
        car_speed = float(np.linalg.norm(car_vel[:2]))
        heading = float(math.atan2(float(car.physics.forward[1]), float(car.physics.forward[0])))

        ball_pos = game_state.ball.position
        ball_vel = game_state.ball.linear_velocity
        ball_speed = float(np.linalg.norm(ball_vel))

        goal_scored = 1.0 if getattr(game_state, "goal_scored", False) else 0.0
        scoring_team = float(getattr(game_state, "scoring_team", -1) if goal_scored else -1)
        boost_amount = float(car.boost_amount)

        metric = np.array([
            float(self.worker_pid),
            float(game_state.tick_count),
            float(car_pos[0]),
            float(car_pos[1]),
            heading,
            float(ball_pos[0]),
            float(ball_pos[1]),
            goal_scored,
            scoring_team,
            ball_speed,
            0.0,  # episode_reward_sum — unused slot
            car_speed,
            boost_amount,
            float(car_vel[0]),
            float(car_vel[1]),
        ], dtype=np.float32)
        return [metric]

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        self.cumulative_timesteps = cumulative_timesteps

        for serialized in collected_metrics:
            arrays = self._deserialize(serialized)
            if not arrays:
                continue
            metric = np.asarray(arrays[0], dtype=np.float32)
            if metric.size != METRIC_SIZE:
                continue
            self._consume_metric(metric)

        now = time.monotonic()
        if now - self.last_dashboard_time >= self.dashboard_update_seconds:
            self.dashboard.update(
                self._build_plot_data(),
                self.episode_counter,
                self.dashboard_update_seconds,
                cumulative_timesteps,
                minimap_frames=self.last_episode_frames,
            )
            self.last_dashboard_time = now

    def _build_plot_data(self):
        ep_count = len(self.shot_speed_rolling)
        # Shot speed histogram: bucket by 500 uu/s intervals, only goal episodes
        shot_speed_hist = {}
        for s in self.shot_speeds[-100_000:]:
            if s > 0:
                bucket = int(s) // 500
                shot_speed_hist[bucket] = shot_speed_hist.get(bucket, 0) + 1

        return {
            "episode_seconds": {
                "mean": self.ep_sec_rolling,
                "p05": self.ep_sec_p05,
                "p95": self.ep_sec_p95,
                "x_end_label": self.episode_counter,
            },
            "shot_speed": {
                "mean": self.shot_speed_rolling,
                "p05": self.shot_speed_p05,
                "p95": self.shot_speed_p95,
                "max_series": self.shot_speed_max_series,
                "x_end_label": ep_count,
            },
            "goal_rate": {
                "mean": [v * 100 for v in self.goal_rate_rolling],
                "p05": [v * 100 for v in self.goal_rate_p05],
                "p95": [v * 100 for v in self.goal_rate_p95],
                "x_end_label": self.episode_counter,
            },
            "boost_used": {
                "mean": self.boost_used_rolling,
                "p05": self.boost_used_p05,
                "p95": self.boost_used_p95,
                "x_end_label": self.episode_counter,
            },
            "direction_flips": {
                "mean": self.direction_flips_rolling,
                "x_end_label": self.episode_counter,
            },
            "shot_speed_hist": shot_speed_hist,
            "top_shots": list(self.top_shots),
        }

    def _consume_metric(self, metric):
        pid = int(metric[0])
        tick_count = int(metric[1])
        position = metric[2:4].astype(np.float32)
        heading = float(metric[4])
        ball_position = metric[5:7].astype(np.float32)
        goal_scored = bool(metric[7] > 0.5)
        scoring_team = int(metric[8])
        ball_speed = float(metric[9])
        car_speed = float(metric[11])
        boost_amount = float(metric[12])
        car_vel_x = float(metric[13])
        car_vel_y = float(metric[14])

        # Forward-dot-velocity: positive = moving forward, negative = reversing
        fwd_x = math.cos(heading)
        fwd_y = math.sin(heading)
        fwd_dot_vel = fwd_x * car_vel_x + fwd_y * car_vel_y

        def _fresh_state():
            return {
                "last_tick": tick_count,
                "total_steps": 1,
                "goal_scored": False,
                "scoring_team": -1,
                "final_ball_speed": ball_speed,
                "boost_consumed": 0.0,
                "last_boost": boost_amount,
                "direction_flips": 0,
                "prev_fwd_dot_vel": fwd_dot_vel,
                "frames": [self._make_frame(position, heading, ball_position)],
            }

        state = self.process_state.get(pid)
        if state is None:
            self.process_state[pid] = _fresh_state()
            return

        if tick_count <= state["last_tick"]:
            self._finalize_episode(state)
            self.process_state[pid] = _fresh_state()
            return

        state["total_steps"] += 1
        state["last_tick"] = tick_count
        state["final_ball_speed"] = ball_speed
        if goal_scored:
            state["goal_scored"] = True
            state["scoring_team"] = scoring_team

        boost_step = max(0.0, state["last_boost"] - boost_amount)
        state["boost_consumed"] = state.get("boost_consumed", 0.0) + boost_step
        state["last_boost"] = boost_amount

        prev_dot = state.get("prev_fwd_dot_vel", fwd_dot_vel)
        if prev_dot * fwd_dot_vel < 0 and abs(fwd_dot_vel) > 50:
            state["direction_flips"] = state.get("direction_flips", 0) + 1
        state["prev_fwd_dot_vel"] = fwd_dot_vel

        state["frames"].append(self._make_frame(position, heading, ball_position))

    def _finalize_episode(self, state):
        total_steps = max(state.get("total_steps", 1), 1)
        ep_seconds = total_steps / DECISIONS_PER_SECOND
        goal = state["goal_scored"]
        scoring_team = state["scoring_team"]
        shot_speed = state["final_ball_speed"] if (goal and scoring_team == 0) else 0.0
        goal_flag = 1.0 if (goal and scoring_team == 0) else 0.0
        avg_reward = 0.0  # reward not tracked in metric array
        boost_used = state.get("boost_consumed", 0.0)
        direction_flips = float(state.get("direction_flips", 0))

        self.episode_counter += 1

        _append_rolling_stats(self.episode_seconds, self.ep_sec_rolling, self.ep_sec_p05, self.ep_sec_p95, ep_seconds)
        _append_rolling_stats(self.shot_speeds, self.shot_speed_rolling, self.shot_speed_p05, self.shot_speed_p95, shot_speed)
        _append_rolling_stats(self.goal_scored, self.goal_rate_rolling, self.goal_rate_p05, self.goal_rate_p95, goal_flag)
        _append_rolling_stats(self.boost_used, self.boost_used_rolling, self.boost_used_p05, self.boost_used_p95, boost_used)
        self.direction_flips.append(direction_flips)
        window_flips = self.direction_flips[-ROLLING_WINDOW:]
        self.direction_flips_rolling.append(float(np.mean(window_flips)))

        prev_max = self.shot_speed_max_series[-1] if self.shot_speed_max_series else 0.0
        self.shot_speed_max_series.append(max(prev_max, shot_speed))

        if shot_speed > 0:
            self.top_shots = sorted(
                self.top_shots + [(shot_speed, self.episode_counter)],
                key=lambda x: x[0], reverse=True,
            )[:5]

        frames = list(state.get("frames", []))
        if len(frames) > 1:
            frames = frames[:-1]
        self.last_episode_frames = frames

        with POWER_SHOT_METRICS_CSV.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                self.episode_counter, ep_seconds, shot_speed, goal_flag, avg_reward,
                boost_used, direction_flips,
            ])

    @staticmethod
    def _make_frame(position, heading, ball_position):
        return {
            "car_x": float(position[0]),
            "car_y": float(position[1]),
            "heading": float(heading),
            "ball_x": float(ball_position[0]),
            "ball_y": float(ball_position[1]),
        }

    @staticmethod
    def _deserialize(serialized_metrics):
        metrics_arrays = []
        i = 0
        while i < len(serialized_metrics):
            n_shape = int(serialized_metrics[i])
            n_values = 1
            shape = []
            i += 1
            for arg in serialized_metrics[i:i + n_shape]:
                n_values *= int(arg)
                shape.append(int(arg))
            metric = serialized_metrics[i + n_shape:i + n_shape + n_values]
            metrics_arrays.append(metric)
            i = i + n_shape + n_values
        return metrics_arrays
