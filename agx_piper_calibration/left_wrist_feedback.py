"""Continuous passive feedback for the wrist capture window; no CAN writer."""
from collections import deque
import socket
import threading
import time

import numpy as np


def valid_left(states):
    state = states.get("left", {})
    q = np.asarray(state.get("joint_deg", []), dtype=float)
    return bool(
        state.get("complete") and not state.get("stale")
        and state.get("error_code") == 0 and state.get("arm_status") == 0
        and len(state.get("enabled_drive_bits", [])) == 6
        and not any(state["enabled_drive_bits"])
        and q.shape == (6,) and np.all(np.isfinite(q))
    )


class FeedbackStream:
    """Reuse the configured CanBus decoder, keeping 3 s of time-tagged snapshots."""

    def __init__(self, config, period=0.05, bus_factory=None):
        if config.get("expected_hostname") not in (None, socket.gethostname()):
            raise ValueError("Selected site configuration does not match this host")
        self.config, self.period, self.bus_factory = config, period, bus_factory
        self.history = deque(maxlen=60)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = "waiting for joints"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)

    def snapshot(self):
        with self.lock:
            return list(self.history), self.error

    def _run(self):
        from agilex_control.can_bus import CanBus

        while not self.stop_event.is_set():
            try:
                with (self.bus_factory or CanBus)({"left": self.config["arms"]["left"]}) as bus:
                    while not self.stop_event.is_set():
                        bus.pump(self.period)
                        states = bus.state()
                        epoch, mono = time.time(), time.monotonic()
                        joint_times = [
                            epoch - (mono - bus.stamps.get("left", {}).get(cid, -float("inf")))
                            for cid in (0x2A5, 0x2A6, 0x2A7)
                        ]
                        row = {"epoch": epoch, "states": states, "joint_epochs": joint_times}
                        with self.lock:
                            self.history.append(row)
                            self.error = ""
                        # CanBus keeps an event ledger; this subscriber has no writer.
                        # Drain the ledger so long manual sessions use bounded memory.
                        bus.events.clear()
            except Exception as exc:
                with self.lock:
                    self.history.clear()
                    self.error = str(exc)
                self.stop_event.wait(0.5)


def frame_feedback(history, stamp, now, since=0.0):
    """Check freshness and motion, then bracket a frame with measured feedback.

    Sampling uses a measured pose, never interpolation. Both snapshots surrounding
    the image must be healthy and agree within the existing 0.02 degree limit.
    """
    if not history or not np.isfinite(stamp) or not -0.05 <= now - stamp <= 0.25:
        return None, float("inf"), None
    latest = history[-1]
    if not 0 <= now - latest["epoch"] <= 0.15:
        return None, float("inf"), None
    # Include the snapshot preceding the last GUI check so a movement between
    # rendered frames cannot disappear behind a subsequent stationary snapshot.
    start = max(0, next((i for i, row in enumerate(history) if row["epoch"] >= since), len(history)) - 1)
    recent = history[start:]
    drift = float("inf")
    continuous = all(0 <= b["epoch"] - a["epoch"] <= 0.15 for a, b in zip(recent, recent[1:]))
    if len(recent) >= 2 and continuous and all(valid_left(row["states"]) for row in recent):
        joints = np.asarray([row["states"]["left"]["joint_deg"] for row in recent])
        drift = float(np.max(np.ptp(joints, axis=0)))
    before = next((row for row in reversed(history) if row["epoch"] <= stamp), None)
    after = next((row for row in history if row["epoch"] >= stamp), None)
    pair = None
    if before is not None and after is not None:
        rows = (before, after)
        if all(
            abs(row["epoch"] - stamp) <= 0.12 and valid_left(row["states"])
            and all(abs(t - stamp) <= 0.12 for t in row["joint_epochs"])
            for row in rows
        ):
            q = [np.asarray(row["states"]["left"]["joint_deg"]) for row in rows]
            if np.max(np.abs(q[1] - q[0])) <= 0.02:
                pair = {
                    "states": after["states"],
                    "feedback_before_epoch": before["epoch"],
                    "feedback_after_epoch": after["epoch"],
                    "joint_feedback_epochs": [row["joint_epochs"] for row in rows],
                    "joint_drift_deg": float(np.max(np.abs(q[1] - q[0]))),
                }
    return latest, drift, pair
