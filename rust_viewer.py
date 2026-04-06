"""
RustViewer: Python bridge to the rl-viewer Rust 3D renderer.

Launches the binary as a subprocess and sends game state over a Unix domain
socket as newline-delimited JSON. Interface mirrors SandboxViewer.update().
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

SOCKET_PATH = "/tmp/rl-viewer.sock"

_REPO_ROOT = Path(__file__).parent
_BINARY_RELEASE = _REPO_ROOT / "rl-viewer" / "target" / "release" / "rl-viewer"
_BINARY_DEBUG   = _REPO_ROOT / "rl-viewer" / "target" / "debug"   / "rl-viewer"


def _find_binary() -> Path:
    if _BINARY_RELEASE.exists():
        return _BINARY_RELEASE
    if _BINARY_DEBUG.exists():
        return _BINARY_DEBUG
    raise FileNotFoundError(
        "rl-viewer binary not found. Build it with:\n"
        "  cd rl-viewer && cargo build --release"
    )


def _rot_to_list(rot_mtx) -> list:
    """Convert 3x3 numpy rotation matrix to list-of-rows for JSON."""
    import numpy as np
    arr = np.asarray(rot_mtx, dtype=float)
    return arr.tolist()


def _vec3(arr) -> dict:
    return {"x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2])}


def _serialize_frame(state, tick: int) -> bytes:
    cars = []
    for agent_id, car in state.cars.items():
        phys = car.physics
        rot = getattr(phys, "rotation_mtx", None)
        if rot is None:
            rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            rot = _rot_to_list(rot)

        vel = getattr(phys, "linear_velocity", None)
        if vel is None:
            vel = [0.0, 0.0, 0.0]

        cars.append({
            "pos": _vec3(phys.position),
            "vel": _vec3(vel),
            "rot": rot,
            "boost": float(getattr(car, "boost_amount", 0.0)),
            "team": 0 if getattr(car, "is_blue", True) else 1,
            "on_ground": bool(getattr(car, "on_ground", True)),
        })

    ball_vel = getattr(state.ball, "linear_velocity", None)
    if ball_vel is None:
        ball_vel = [0.0, 0.0, 0.0]

    ball_rot = getattr(state.ball, "rotation_mtx", None)
    if ball_rot is None:
        ball_rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        ball_rot = _rot_to_list(ball_rot)

    boost_pads = []
    raw_pads = getattr(state, "boost_pad_timers", None)
    if raw_pads is not None:
        boost_pads = [float(t) for t in raw_pads]

    msg = {
        "type": "frame",
        "tick": tick,
        "ball": {
            "pos": _vec3(state.ball.position),
            "vel": _vec3(ball_vel),
            "rot": ball_rot,
        },
        "cars": cars,
        "boost_pads": boost_pads,
    }
    return (json.dumps(msg) + "\n").encode()


class RustViewer:
    def __init__(self, record: str | None = None):
        """
        record: output video path (e.g. "replay.mp4"). Requires ffmpeg on PATH.
                If None, no recording.
        """
        binary = _find_binary()

        # Remove stale socket before starting (binary also does this, but
        # race-proof by removing it here first so connect() waits for fresh one)
        try:
            os.remove(SOCKET_PATH)
        except FileNotFoundError:
            pass

        cmd = [str(binary)]
        if record:
            cmd += ["--record", str(record)]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=None,  # let ffmpeg/record logs show in terminal
        )
        self._record_path = record

        self._sock = self._connect_with_retry(timeout=10.0)
        self._tick = 0
        self._open = True

    def _connect_with_retry(self, timeout: float) -> socket.socket:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(SOCKET_PATH):
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.connect(SOCKET_PATH)
                    sock.setblocking(False)
                    return sock
                except (ConnectionRefusedError, OSError):
                    sock.close()
            time.sleep(0.05)
        raise RuntimeError(
            f"rl-viewer socket {SOCKET_PATH} did not appear within {timeout}s"
        )

    def update(self, state, episode: int, step_count: int) -> bool:
        if not self._open:
            return False
        if self._proc.poll() is not None:
            self._open = False
            return False

        self._tick += 1
        try:
            data = _serialize_frame(state, self._tick)
            self._sock.sendall(data)
        except (BrokenPipeError, OSError):
            self._open = False
            return False

        return True

    def reset(self):
        """Send reset signal (new episode)."""
        if not self._open:
            return
        try:
            self._sock.sendall(b'{"type":"reset"}\n')
        except OSError:
            pass

    def close(self):
        if not self._open:
            return
        self._open = False
        try:
            self._sock.sendall(b'{"type":"close"}\n')
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
