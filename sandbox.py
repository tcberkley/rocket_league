import math
import tkinter as tk

from rlgym.rocket_league import common_values

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 800
PADDING = 40
FIELD_HALF_WIDTH = common_values.SIDE_WALL_X
FIELD_HALF_LENGTH = common_values.BACK_WALL_Y
BALL_RADIUS = 10
CAR_LENGTH = 36
CAR_WIDTH = 20


class SandboxViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Rocket League Sandbox")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.is_open = True

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#135d36",
            highlightthickness=0,
        )
        self.canvas.pack()

        self._draw_field()
        self.ball = self.canvas.create_oval(0, 0, 0, 0, fill="#f2f4f8", outline="")
        self.status_text = self.canvas.create_text(
            16,
            16,
            anchor="nw",
            fill="#f7f7f7",
            font=("Helvetica", 14, "bold"),
            text="",
        )
        self.car_items = {}

    def close(self):
        if not self.is_open:
            return

        self.is_open = False
        self.root.destroy()

    def update(self, state, episode, step_count):
        if not self.is_open:
            return False

        self._draw_ball(state.ball.position)
        active_ids = set()
        for index, (agent_id, car) in enumerate(state.cars.items(), start=1):
            active_ids.add(agent_id)
            self._draw_car(agent_id, car, index)

        for agent_id in list(self.car_items):
            if agent_id not in active_ids:
                for item_id in self.car_items.pop(agent_id).values():
                    self.canvas.delete(item_id)

        self.canvas.itemconfigure(
            self.status_text,
            text=f"Episode {episode}  Step {step_count}",
        )

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.is_open = False

        return self.is_open

    def _draw_field(self):
        left, top = self._to_canvas((-FIELD_HALF_WIDTH, FIELD_HALF_LENGTH))
        right, bottom = self._to_canvas((FIELD_HALF_WIDTH, -FIELD_HALF_LENGTH))
        self.canvas.create_rectangle(left, top, right, bottom, outline="#ecf7ec", width=3)

        center_left, center_top = self._to_canvas((-1, 1))
        center_right, center_bottom = self._to_canvas((1, -1))
        center_x = (center_left + center_right) / 2
        self.canvas.create_line(center_x, top, center_x, bottom, fill="#ecf7ec", width=2)

        circle_radius = 600 * self._scale()
        self.canvas.create_oval(
            center_x - circle_radius,
            (top + bottom) / 2 - circle_radius,
            center_x + circle_radius,
            (top + bottom) / 2 + circle_radius,
            outline="#ecf7ec",
            width=2,
        )

        goal_half_width = 900 * self._scale()
        goal_depth = 140
        self.canvas.create_rectangle(
            center_x - goal_half_width,
            top - goal_depth,
            center_x + goal_half_width,
            top,
            outline="#7cb4ff",
            width=2,
        )
        self.canvas.create_rectangle(
            center_x - goal_half_width,
            bottom,
            center_x + goal_half_width,
            bottom + goal_depth,
            outline="#ffb16d",
            width=2,
        )

    def _draw_ball(self, position):
        x, y = self._to_canvas(position[:2])
        self.canvas.coords(
            self.ball,
            x - BALL_RADIUS,
            y - BALL_RADIUS,
            x + BALL_RADIUS,
            y + BALL_RADIUS,
        )

    def _draw_car(self, agent_id, car, label_index):
        if agent_id not in self.car_items:
            color = "#55a4ff" if car.is_blue else "#ff9a3d"
            self.car_items[agent_id] = {
                "body": self.canvas.create_polygon(0, 0, 0, 0, 0, 0, fill=color, outline="#0e1726", width=2),
                "label": self.canvas.create_text(0, 0, fill="#f7f7f7", font=("Helvetica", 11, "bold")),
            }

        x, y = self._to_canvas(car.physics.position[:2])
        forward = car.physics.rotation_mtx[:, 0]
        angle = math.atan2(float(forward[1]), float(forward[0]))
        polygon = self._rotated_box(x, y, CAR_LENGTH, CAR_WIDTH, angle)

        label = f"{'B' if car.is_blue else 'O'}{label_index}"
        self.canvas.coords(self.car_items[agent_id]["body"], *polygon)
        self.canvas.coords(self.car_items[agent_id]["label"], x, y)
        self.canvas.itemconfigure(self.car_items[agent_id]["label"], text=label)

    def _rotated_box(self, center_x, center_y, length, width, angle):
        half_length = length / 2
        half_width = width / 2
        corners = [
            (half_length, 0),
            (-half_length, -half_width),
            (-half_length, half_width),
        ]

        rotated = []
        for dx, dy in corners:
            px = center_x + dx * math.cos(angle) - dy * math.sin(angle)
            py = center_y - (dx * math.sin(angle) + dy * math.cos(angle))
            rotated.extend((px, py))
        return rotated

    def _scale(self):
        return min(
            (CANVAS_WIDTH - 2 * PADDING) / (2 * FIELD_HALF_WIDTH),
            (CANVAS_HEIGHT - 2 * PADDING) / (2 * FIELD_HALF_LENGTH),
        )

    def _to_canvas(self, position):
        scale = self._scale()
        x = CANVAS_WIDTH / 2 + float(position[0]) * scale
        y = CANVAS_HEIGHT / 2 - float(position[1]) * scale
        return x, y
