import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable
import numpy as np
import cv2
import av
import cv2
import time
import pyudev
import threading
from typing import Optional, Dict, List, Tuple
from termcolor import cprint
import numpy as np
from recorder_av_cam import _AVStreamWorker


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
        format: str = "bgra8",
        auto_exposure: bool = True,
        exposure_us: Optional[float] = None,
        white_balance: Optional[float] = None,
    ):
        if serial is None:
            serial = self.get_realsense_serial_numbers()
            if serial is None:
                raise RuntimeError("No RealSense device found; cannot proceed.")
        # assert current_rgb_profile in all_profiles[0], all_profiles
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

    def get_realsense_serial_numbers(self):
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise ImportError(
                "pyrealsense2 is required for RealsenseCamera. Install Intel RealSense SDK."
            ) from e
        # Create a context object. This object owns the handles to all connected devices.
        context = rs.context()

        # Get connected devices
        connected_devices = context.query_devices()
        serial = None

        # Check if any devices are connected
        if len(connected_devices) == 0:
            print("No RealSense devices found.")
        else:
            print(f"{len(connected_devices)} RealSense device(s) found:")
            for i, dev in enumerate(connected_devices):
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                product_line = dev.get_info(rs.camera_info.product_line)
                firmware = dev.get_info(rs.camera_info.firmware_version)
                print(f"\nDevice {i+1}:")
                print(f"  Name         : {name}")
                print(f"  Serial Number: {serial}")
                print(f"  Product Line : {product_line}")
                print(f"  Firmware     : {firmware}")
        return serial

    def get_profiles(self, verbose=False):
        # ----------------------------------------------------------------------------
        # -                        Open3D: www.open3d.org                            -
        # ----------------------------------------------------------------------------
        # Copyright (c) 2018-2023 www.open3d.org
        # SPDX-License-Identifier: MIT
        # ----------------------------------------------------------------------------

        # examples/python/reconstruction_system/sensors/realsense_helper.py

        # pyrealsense2 is required.
        # Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise ImportError(
                "pyrealsense2 is required for RealsenseCamera. Install Intel RealSense SDK."
            ) from e
        ctx = rs.context()
        devices = ctx.query_devices()

        color_profiles = []
        depth_profiles = []
        for device in devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            if verbose:
                print("Sensor: {}, {}".format(name, serial))
                print("Supported video formats:")
            for sensor in device.query_sensors():
                for stream_profile in sensor.get_stream_profiles():
                    stream_type = str(stream_profile.stream_type())

                    if stream_type in ["stream.color", "stream.depth"]:
                        v_profile = stream_profile.as_video_stream_profile()
                        fmt = stream_profile.format()
                        w, h = v_profile.width(), v_profile.height()
                        fps = v_profile.fps()

                        video_type = stream_type.split(".")[-1]
                        if verbose:
                            print(
                                "  {}: width={}, height={}, fps={}, fmt={}".format(
                                    video_type, w, h, fps, fmt
                                )
                            )
                        if video_type == "color":
                            color_profiles.append((w, h, fps, fmt))
                        else:
                            depth_profiles.append((w, h, fps, fmt))

        return color_profiles, depth_profiles

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
        if self.format == "rgba8":
            fmt = rs.format.rgba8
            convert_needed = False
        else:
            # default bgr8; we'll convert to RGB
            fmt = rs.format.bgra8
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
      checkerboard: optional tuple (cols, rows) for chessboard corner detection
    """
    def __init__(
        self,
        camera,
        window_name: str = "InteractiveCamera",
        display_scale: Optional[float] = None,
        show_fps: bool = True,
        undistort_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        buffer_max: Optional[int] = None,
        checkerboard: Optional[Tuple[int, int]] = None,  # 新增：棋盘格参数
    ):
        self.cam = camera
        self.window_name = window_name
        self.display_scale = display_scale
        self.show_fps = show_fps
        self.undistort_fn = undistort_fn
        self.buffer_max = buffer_max
        
        # 新增：棋盘格检测相关
        self.checkerboard = checkerboard
        self.detection_enabled = checkerboard is not None
        self.last_detection_ok = False
        self.last_corners = None

        self.buffer: List[np.ndarray] = []
        self.paused = False
        self.show_help = True
        self.last_time = time.time()
        self.fps = 0.0

    # ------------------------ 棋盘格检测相关方法 ------------------------ #
    def _detect_corners(self, gray: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
        """检测棋盘格角点"""
        cols, rows = pattern_size
        if hasattr(cv2, "findChessboardCornersSB"):
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
            if ok:
                corners = corners.reshape(-1, 1, 2).astype(np.float32)
            return ok, corners
        else:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            if ok:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return ok, corners

    def _overlay_detections(self, bgr: np.ndarray, ok: bool, corners: Optional[np.ndarray]) -> np.ndarray:
        """绘制棋盘格角点和检测状态"""
        if ok and corners is not None:
            # 绘制角点连线
            cv2.drawChessboardCorners(bgr, self.checkerboard, corners, ok)
            label = "CHESSBOARD DETECTED"
            color = (40, 220, 40)  # 绿色
        else:
            label = "no chessboard"
            color = (40, 40, 220)  # 红色

        # 在右上角添加状态标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.6, 2
        (tw, th_pix), _ = cv2.getTextSize(label, font, fs, th)

        margin = 10
        x2 = bgr.shape[1] - margin        # 右边缘
        y1 = margin                       # 上边缘
        x1 = x2 - (tw + 12)               # 框宽度 = 文本 + 内边距
        y2 = y1 + th_pix + 14

        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(bgr, label, (x1 + 6, y2 - 6), font, fs, color, th, cv2.LINE_AA)

        return bgr

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
        
        if self.detection_enabled:
            status_color = (40, 220, 40) if self.last_detection_ok else (40, 40, 220)
            status_text = "Detected" if self.last_detection_ok else "Not detected"
            cv2.putText(overlay, f"Chessboard: {status_text}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
            y += 26

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

                # 新增：棋盘格检测和绘制
                if self.detection_enabled:
                    gray = cv2.cvtColor(last_frame_rgb, cv2.COLOR_RGB2GRAY)
                    ok, corners = self._detect_corners(gray, self.checkerboard)
                    self.last_detection_ok = ok
                    self.last_corners = corners
                    disp_bgr = self._overlay_detections(disp_bgr, ok, corners)

                # Optional downscale for display only
                if self.display_scale is not None and self.display_scale > 0:
                    disp_bgr = cv2.resize(
                        disp_bgr, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA
                    )

                disp_bgr = self._draw_hud(disp_bgr)
                disp_bgr = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)

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
    
class AVCameraManager(Camera):
    """
    AV camera manager that yields RGB uint8 frames.

    Args:
      src: device index (int) or path to video file (str)
      width, height: requested resolution (OpenCV will try to set; may not be guaranteed)
      fps: requested frame rate (best-effort)
      convert_to_rgb: if True, convert BGR->RGB (default True)
    """
    def __init__(
        self,
        camera_to_port: Dict[str, str],
        camera_left_right_order,
        default_options: Optional[Dict[str, str]] = None,
        per_camera_options: Optional[Dict[str, Dict[str, str]]] = None,
        stream_index: int = 0,
    ):
        """
        Args:
            camera_to_port: {name: "/dev/videoX" or other v4l2 device path}
            default_options: applied to all cameras unless overridden
                Common useful keys:
                    - "input_format": "mjpeg" | "yuyv422" | ...
                    - "video_size": "1280x480"
                    - "framerate": "30"
            per_camera_options: {name: {option_key: value}} to override defaults
            stream_index: video stream index, usually 0
        """
        self.camera_to_port = dict(camera_to_port)
        self.camera_left_right_order = dict(camera_left_right_order)

        self.default_options = dict(default_options or {})
        self.per_camera_options = dict(per_camera_options or {})
        self.stream_index = stream_index

        self._workers: Dict[str, _AVStreamWorker] = {}
        self._active: Dict[str, bool] = {}

    def _merged_options_for(self, name: str) -> Dict[str, str]:
        merged = dict(self.default_options)
        if name in self.per_camera_options:
            merged.update(self.per_camera_options[name])
        return merged

    def open_camera_by_name(self, camera_name: str) -> bool:
        device = self.camera_to_port.get(camera_name)
        if not device:
            cprint(f"Unknown camera: {camera_name}", "red")
            return False
        if camera_name in self._workers:
            cprint(f"{camera_name} already opened.", "yellow")
            return True

        options = self._merged_options_for(camera_name)
        worker = _AVStreamWorker(
            name=camera_name,
            device=device,
            options=options,
            stream_index=self.stream_index,
        )
        worker.start()
        # Wait briefly to confirm open succeeded (non-blocking overall)
        time_limit = time.time() + 1.5
        while time.time() < time_limit and not worker.is_open and worker.last_error is None:
            time.sleep(0.02)

        if worker.is_open:
            self._workers[camera_name] = worker
            self._active[camera_name] = True
            return True
        else:
            # If open failed fast, join and report
            worker.stop()
            worker.join(timeout=0.5)
            err = worker.last_error or "Unknown error while opening"
            cprint(f"Failed to open {camera_name}: {err}", "red")
            return False

    def open_all_cameras(self) -> int:
        count = 0
        for name in self.camera_to_port.keys():
            if self.open_camera_by_name(name):
                count += 1
        return count

    def release_camera(self, identifier: str):
        name = self._resolve_camera_name(identifier)
        if name and name in self._workers:
            w = self._workers.pop(name)
            self._active.pop(name, None)
            w.stop()
            w.join(timeout=1.0)
            cprint(f"Released {name}", "cyan")

    def release_all(self):
        for name in list(self._workers.keys()):
            self.release_camera(name)
        cprint("Released all cameras", "cyan")

    def stereo_to_mono_frame_dict(self, stereo_frames):
        mono_frames = {}
        for cam_name, frame in stereo_frames.items():
            if cam_name in self.camera_left_right_order:
                left_name, right_name = self.camera_left_right_order[cam_name]
                h, w, _ = frame.shape
                assert w % 2 == 0, f"Expected even width for stereo frame from {cam_name}"
                mid = w // 2
                mono_frames[left_name] = frame[:, :mid, :]
                mono_frames[right_name] = frame[:, mid:, :]
            else:
                mono_frames[cam_name] = frame
        return mono_frames
    
    # ---------- Read / Get Frames ----------

    def get_frames(
        self,
        camera_names: Optional[List[str]] = None,
        img_size: Optional[Tuple[int, int]] = None,  # (W, H)
    ):
        """
        Get the *latest* frames for the requested cameras (non-blocking).
        Returns a dict {name: BGR np.ndarray}. If return_timestamps=True, also returns {name: ts}.
        """
        if camera_names is None:
            camera_names = list(self._workers.keys())
        if not camera_names:
            cprint("No active cameras to read from.", "red")
            return {}

        frames = {}
        for name in camera_names:
            w = self._workers.get(name)
            if w is None:
                cprint(f"Camera '{name}' is not opened.", "red")
                continue
            frame, ts = w.get_latest()
            if frame is None:
                # Not yet decoded a frame; skip silently or warn
                continue
            if img_size is not None:
                stereo_img_size = (img_size[0]*2, img_size[1])
                frame = cv2.resize(frame, stereo_img_size)
            frames[name] = frame

        mono_frames = self.stereo_to_mono_frame_dict(frames)

        return mono_frames


    def _resolve_camera_name(self, identifier: str) -> Optional[str]:
        # Identifier can be a camera name or a device path
        if identifier in self.camera_to_port:
            return identifier
        for name, dev in self.camera_to_port.items():
            if dev == identifier:
                return name
        return None

    def set_camera_options(self, camera_name: str, options: Dict[str, str]) -> None:
        """
        Update/override PyAV open options for one camera.
        You must re-open the camera for changes to take effect.
        """
        cur = self.per_camera_options.get(camera_name, {})
        cur.update(options)
        self.per_camera_options[camera_name] = cur
        cprint(f"[{camera_name}] Updated options: {self.per_camera_options[camera_name]}", "blue")

    def start(self) -> None:
        self.open_all_cameras()

    def stop(self) -> None:
        self.release_all()

    def read(self) -> np.ndarray:
        """Return a single RGB frame (H,W,3) uint8. Must raise RuntimeError on failure."""
        return self.get_frames()["I-tip"]
