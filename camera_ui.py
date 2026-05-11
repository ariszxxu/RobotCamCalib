from __future__ import annotations

import os
import time
from typing import Callable, Optional

import cv2
import numpy as np


class InteractiveCamera:
    """Interactive OpenCV capture UI for any camera object with start/stop/read."""

    def __init__(
        self,
        camera,
        window_name: str = "InteractiveCamera",
        display_scale: Optional[float] = None,
        show_fps: bool = True,
        undistort_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        buffer_max: Optional[int] = None,
        checkerboard: Optional[tuple[int, int]] = None,
    ) -> None:
        self.cam = camera
        self.window_name = window_name
        self.display_scale = display_scale
        self.show_fps = show_fps
        self.undistort_fn = undistort_fn
        self.buffer_max = buffer_max

        self.checkerboard = checkerboard
        self.detection_enabled = checkerboard is not None
        self.last_detection_ok = False
        self.last_corners: Optional[np.ndarray] = None

        self.buffer: list[np.ndarray] = []
        self.paused = False
        self.show_help = True
        self.last_time = time.time()
        self.fps = 0.0

    def get_buffer_array(self) -> Optional[np.ndarray]:
        if not self.buffer:
            return None
        return np.stack(self.buffer, axis=0)

    def clear_buffer(self) -> None:
        self.buffer.clear()

    def save_buffer(
        self,
        save_dir: str,
        prefix: str = "frame",
        start_idx: int = 0,
        ext: str = "png",
    ) -> list[str]:
        os.makedirs(save_dir, exist_ok=True)
        paths: list[str] = []

        for idx, img_rgb in enumerate(self.buffer, start=int(start_idx)):
            path = os.path.join(save_dir, f"{prefix}_{idx:06d}.{ext.lower()}")
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(path, bgr):
                raise RuntimeError(f"Failed to write: {path}")
            paths.append(path)

        return paths

    def run(self) -> Optional[np.ndarray]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

        last_frame_rgb: Optional[np.ndarray] = None
        with self.cam as cam:
            while True:
                if not self.paused or last_frame_rgb is None:
                    frame_rgb = cam.read()
                    if self.undistort_fn is not None:
                        try:
                            frame_rgb = self.undistort_fn(frame_rgb)
                        except Exception:
                            pass
                    last_frame_rgb = frame_rgb
                    self._update_fps()

                disp_bgr = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR)

                if self.detection_enabled:
                    gray = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2GRAY)
                    ok, corners = self._detect_corners(gray, self.checkerboard)
                    self.last_detection_ok = ok
                    self.last_corners = corners
                    disp_bgr = self._overlay_detections(disp_bgr, ok, corners)

                if self.display_scale is not None and self.display_scale > 0:
                    disp_bgr = cv2.resize(
                        disp_bgr,
                        None,
                        fx=self.display_scale,
                        fy=self.display_scale,
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imshow(self.window_name, self._draw_hud(disp_bgr))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("h"):
                    self.show_help = not self.show_help
                elif key == ord("p"):
                    self.paused = not self.paused
                elif key == ord("c"):
                    self.clear_buffer()
                elif key == ord("s") and last_frame_rgb is not None:
                    self._store(last_frame_rgb)

        cv2.destroyWindow(self.window_name)
        return self.get_buffer_array()

    def _detect_corners(
        self,
        gray: np.ndarray,
        pattern_size: tuple[int, int],
    ) -> tuple[bool, Optional[np.ndarray]]:
        cols, rows = pattern_size

        if hasattr(cv2, "findChessboardCornersSB"):
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
            if ok:
                corners = corners.reshape(-1, 1, 2).astype(np.float32)
            return ok, corners

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if ok:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        return ok, corners

    def _overlay_detections(
        self,
        bgr: np.ndarray,
        ok: bool,
        corners: Optional[np.ndarray],
    ) -> np.ndarray:
        if ok and corners is not None:
            cv2.drawChessboardCorners(bgr, self.checkerboard, corners, ok)
            label = "CHESSBOARD DETECTED"
            color = (40, 220, 40)
        else:
            label = "no chessboard"
            color = (40, 40, 220)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        margin = 10
        x2 = bgr.shape[1] - margin
        y1 = margin
        x1 = x2 - text_width - 12
        y2 = y1 + text_height + 14

        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(
            bgr,
            label,
            (x1 + 6, y2 - 6),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return bgr

    def _update_fps(self) -> None:
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps = 1.0 / dt

    def _draw_hud(self, bgr: np.ndarray, msg: str = "") -> np.ndarray:
        height, width = bgr.shape[:2]
        y = 28

        if self.detection_enabled:
            status_color = (40, 220, 40) if self.last_detection_ok else (40, 40, 220)
            status_text = "Detected" if self.last_detection_ok else "Not detected"
            cv2.putText(
                bgr,
                f"Chessboard: {status_text}",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
                cv2.LINE_AA,
            )
            y += 26

        if self.show_help:
            lines = [
                "[s] store    [p] pause    [h] help on/off    [c] clear buffer    [q] quit",
                f"buffer: {len(self.buffer)}    paused: {self.paused}",
            ]
            for line in lines:
                cv2.putText(
                    bgr,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                y += 26

        if self.show_fps:
            cv2.putText(
                bgr,
                f"{self.fps:5.1f} FPS",
                (width - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 220, 30),
                2,
                cv2.LINE_AA,
            )

        if msg:
            cv2.putText(
                bgr,
                msg,
                (12, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 180, 255),
                2,
                cv2.LINE_AA,
            )

        return bgr

    def _store(self, frame_rgb: np.ndarray) -> None:
        if self.buffer_max is not None and len(self.buffer) >= self.buffer_max:
            self.buffer.pop(0)
        self.buffer.append(frame_rgb.copy())
