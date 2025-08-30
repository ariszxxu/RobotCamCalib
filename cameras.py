import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable
import numpy as np
import cv2


# ---------------------------- Base API ---------------------------- #

class Camera(ABC):
    """
    Unified camera interface that yields RGB uint8 frames.

    Methods:
      - start(): open the device
      - stop(): close the device
      - read(): return a single RGB frame (H,W,3) uint8
    """
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return a single RGB frame (H,W,3) uint8. Must raise RuntimeError on failure."""
        pass


# ---------------------------- OpenCV Camera ---------------------------- #

class CV2Camera(Camera):
    """
    OpenCV-backed camera (USB/webcam or video file).

    Args:
      src: device index (int) or path to video file (str)
      width, height: requested resolution (OpenCV will try to set; may not be guaranteed)
      fps: requested frame rate (best-effort)
      convert_to_rgb: if True, convert BGR->RGB (default True)
    """
    def __init__(
        self,
        src: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        convert_to_rgb: bool = True,
        api_pref: Optional[int] = None,  # e.g., cv2.CAP_V4L2 on Linux if you want
    ):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.convert_to_rgb = convert_to_rgb
        self.api_pref = api_pref

        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        if self.api_pref is None:
            self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src, self.api_pref)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {self.src}")

        # Best-effort property set
        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        # Small warmup for USB cams
        time.sleep(0.1)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() or use context manager.")

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("Failed to read frame from CV2Camera.")

        # OpenCV returns BGR; convert to RGB if requested
        if self.convert_to_rgb:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Ensure output is RGB as per interface; if not converting, assume already RGB
            frame_rgb = frame_bgr  # only safe if your source is already RGB
        return frame_rgb


# ---------------------------- RealSense Camera ---------------------------- #

class RealsenseCamera(Camera):
    """
    Intel RealSense color camera via pyrealsense2.

    Args:
      serial: optional device serial to select a specific camera
      width, height, fps: color stream settings
      format: 'bgr8' or 'rgb8' (we default 'bgr8' and then convert to RGB)
      auto_exposure: enable/disable AE (best-effort)
      exposure_us: manual exposure in microseconds if AE disabled
      white_balance: optional manual white balance (Kelvin), best-effort
    """
    def __init__(
        self,
        serial: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        format: str = "bgr8",
        auto_exposure: bool = True,
        exposure_us: Optional[float] = None,
        white_balance: Optional[float] = None,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.format = format.lower()
        self.auto_exposure = auto_exposure
        self.exposure_us = exposure_us
        self.white_balance = white_balance

        self.rs = None              # module handle
        self.pipeline = None
        self.config = None
        self.color_sensor = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise ImportError(
                "pyrealsense2 is required for RealsenseCamera. Install Intel RealSense SDK."
            ) from e

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial:
            self.config.enable_device(self.serial)

        # Choose stream format
        if self.format == "rgb8":
            fmt = rs.format.rgb8
            convert_needed = False
        else:
            # default bgr8; we'll convert to RGB
            fmt = rs.format.bgr8
            convert_needed = True

        self._convert_bgr_to_rgb = convert_needed

        self.config.enable_stream(rs.stream.color, self.width, self.height, fmt, self.fps)
        profile = self.pipeline.start(self.config)

        # Try to get color sensor for options
        try:
            dev = profile.get_device()
            sensors = dev.query_sensors()
            self.color_sensor = None
            for s in sensors:
                if s.get_info(rs.camera_info.name).lower().find("rgb") >= 0:
                    self.color_sensor = s
                    break
            # Auto exposure / manual exposure
            if self.color_sensor is not None:
                if self.auto_exposure:
                    self.color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
                else:
                    self.color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
                    if self.exposure_us is not None:
                        self.color_sensor.set_option(rs.option.exposure, float(self.exposure_us))
                if self.white_balance is not None and self.color_sensor.supports(rs.option.white_balance):
                    # If AE is on, manual WB might be overridden; still best-effort
                    self.color_sensor.set_option(rs.option.white_balance, float(self.white_balance))
        except Exception:
            # Non-fatal: keep streaming even if options fail
            pass

        # Small warmup
        time.sleep(0.2)

    def stop(self) -> None:
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        finally:
            self.pipeline = None
            self.config = None
            self.color_sensor = None
            self.rs = None

    def read(self) -> np.ndarray:
        if self.pipeline is None or self.rs is None:
            raise RuntimeError("Camera not started. Call start() or use context manager.")

        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame from RealSense.")

        img = np.asanyarray(color.get_data())
        # img is either BGR or RGB depending on stream format; ensure output is RGB
        if self._convert_bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img



class InteractiveCamera:
    """
    Minimal interactive capture UI for any `Camera` that returns RGB frames.

    Controls:
      s : store current frame in memory buffer
      p : pause / resume
      h : toggle help overlay
      c : clear buffer
      q/ESC : quit

    Args:
      camera: instance of your Camera (e.g., CV2Camera(...))
      window_name: cv2 window title
      display_scale: optional scale factor for on-screen display only
      show_fps: overlay FPS on screen
      undistort_fn: optional callable(img_rgb) -> img_rgb (preview + stored)
      buffer_max: optional maximum buffer size (oldest frames dropped when exceeded)
    """
    def __init__(
        self,
        camera,
        window_name: str = "InteractiveCamera",
        display_scale: Optional[float] = None,
        show_fps: bool = True,
        undistort_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        buffer_max: Optional[int] = None,
    ):
        self.cam = camera
        self.window_name = window_name
        self.display_scale = display_scale
        self.show_fps = show_fps
        self.undistort_fn = undistort_fn
        self.buffer_max = buffer_max

        self.buffer: List[np.ndarray] = []
        self.paused = False
        self.show_help = True
        self.last_time = time.time()
        self.fps = 0.0

    # ------------------------ public buffer helpers ------------------------ #
    def get_buffer_array(self) -> Optional[np.ndarray]:
        """Return stacked buffer as (n, H, W, 3) RGB uint8, or None if empty."""
        if not self.buffer:
            return None
        return np.stack(self.buffer, axis=0)

    def clear_buffer(self) -> None:
        """Clear the in-memory buffer."""
        self.buffer.clear()

    def save_buffer(self, save_dir: str, prefix: str = "frame",
                    start_idx: int = 0, ext: str = "png") -> List[str]:
        """Optional: write all buffered frames to disk (RGB -> BGR for OpenCV)."""
        os.makedirs(save_dir, exist_ok=True)
        paths: List[str] = []
        idx = int(start_idx)
        for img_rgb in self.buffer:
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            name = f"{prefix}_{idx:06d}.{ext.lower()}"
            path = os.path.join(save_dir, name)
            if not cv2.imwrite(path, bgr):
                raise RuntimeError(f"Failed to write: {path}")
            paths.append(path)
            idx += 1
        return paths

    # ------------------------ internal UI helpers ------------------------- #
    def _update_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps = 1.0 / dt

    def _draw_hud(self, bgr: np.ndarray, msg: str = "") -> np.ndarray:
        h, w = bgr.shape[:2]
        overlay = bgr.copy()
        y = 28
        if self.show_help:
            lines = [
                "[s] store    [p] pause    [h] help on/off    [c] clear buffer    [q] quit",
                f"buffer: {len(self.buffer)}    paused: {self.paused}",
            ]
            for line in lines:
                cv2.putText(overlay, line, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
                y += 26
        if self.show_fps:
            cv2.putText(overlay, f"{self.fps:5.1f} FPS", (w - 130, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)
        if msg:
            cv2.putText(overlay, msg, (12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 180, 255), 2, cv2.LINE_AA)
        return overlay

    def _store(self, frame_rgb: np.ndarray) -> None:
        if self.buffer_max is not None and len(self.buffer) >= self.buffer_max:
            # Drop oldest to maintain max size
            self.buffer.pop(0)
        self.buffer.append(frame_rgb.copy())

    # ------------------------------ main loop ----------------------------- #
    def run(self) -> Optional[np.ndarray]:
        """
        Start the UI loop. Returns the stacked buffer (n, H, W, 3) on exit, or None if empty.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

        last_frame_rgb = None
        with self.cam as cam:
            while True:
                # Acquire a frame unless paused
                if not self.paused or last_frame_rgb is None:
                    frame_rgb = cam.read()
                    if self.undistort_fn is not None:
                        try:
                            frame_rgb = self.undistort_fn(frame_rgb)
                        except Exception:
                            pass
                    last_frame_rgb = frame_rgb
                    self._update_fps()

                # Prepare display (RGB -> BGR for OpenCV)
                disp_bgr = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2BGR)

                # Optional downscale for display only
                if self.display_scale is not None and self.display_scale > 0:
                    disp_bgr = cv2.resize(
                        disp_bgr, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA
                    )

                disp_bgr = self._draw_hud(disp_bgr)
                cv2.imshow(self.window_name, disp_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('h'):
                    self.show_help = not self.show_help
                elif key == ord('p'):
                    self.paused = not self.paused
                elif key == ord('c'):
                    self.clear_buffer()
                elif key == ord('s'):
                    self._store(last_frame_rgb)

        cv2.destroyWindow(self.window_name)
        return self.get_buffer_array()
    
