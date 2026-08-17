"""Shared buffered capture and software timestamp pairing helpers."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class TimedFrame:
    index: int
    timestamp: float
    frame_bgr: np.ndarray
    device_timestamp_ms: Optional[float] = None
    system_timestamp_us: Optional[int] = None


FrameReadResult = (
    np.ndarray
    | tuple[np.ndarray, Optional[float], Optional[int]]
)


class FrameWorker:
    """Continuously read one camera and keep a timestamped frame buffer."""

    def __init__(
        self,
        name: str,
        read_fn: Callable[[], FrameReadResult],
        *,
        buffer_size: int,
        stop_timeout_s: float = 2.0,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least one")
        self.name = name
        self.read_fn = read_fn
        self.frames: deque[TimedFrame] = deque(maxlen=int(buffer_size))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.last_error: Optional[str] = None
        self.stop_timeout_s = float(stop_timeout_s)
        self._index = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.stop_timeout_s)

    def snapshot(self) -> list[TimedFrame]:
        with self.lock:
            return list(self.frames)

    @staticmethod
    def _normalize_result(
        result: FrameReadResult,
    ) -> tuple[np.ndarray, Optional[float], Optional[int]]:
        if isinstance(result, tuple):
            if len(result) != 3:
                raise ValueError("camera read tuple must contain three values")
            frame, device_timestamp_ms, system_timestamp_us = result
            return frame, device_timestamp_ms, system_timestamp_us
        return result, None, None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                started = time.monotonic()
                frame, device_timestamp_ms, system_timestamp_us = (
                    self._normalize_result(self.read_fn())
                )
                finished = time.monotonic()
                if frame is None or frame.size == 0:
                    raise RuntimeError("camera returned an empty frame")
                item = TimedFrame(
                    index=self._index,
                    timestamp=0.5 * (started + finished),
                    frame_bgr=frame,
                    device_timestamp_ms=device_timestamp_ms,
                    system_timestamp_us=system_timestamp_us,
                )
                self._index += 1
                with self.lock:
                    self.frames.append(item)
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.02)


def select_synchronized_pair(
    first_frames: list[TimedFrame],
    second_frames: list[TimedFrame],
    last_pair: Optional[tuple[int, int]],
    max_skew_s: float,
) -> Optional[tuple[TimedFrame, TimedFrame, float]]:
    """Choose the newest unused software-timestamped frame pair."""
    if not first_frames or not second_frames:
        return None
    last_first, last_second = (
        last_pair if last_pair is not None else (-1, -1)
    )
    candidates: list[tuple[TimedFrame, TimedFrame, float]] = []
    for first in first_frames:
        if first.index <= last_first:
            continue
        for second in second_frames:
            if second.index <= last_second:
                continue
            skew = abs(first.timestamp - second.timestamp)
            if skew <= max_skew_s:
                candidates.append((first, second, skew))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            min(item[0].timestamp, item[1].timestamp),
            -item[2],
        ),
    )


def put_lines(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Overlay readable status lines using the legacy scripts' styling."""
    output = image.copy()
    for row, line in enumerate(lines):
        y = 28 + row * 28
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
