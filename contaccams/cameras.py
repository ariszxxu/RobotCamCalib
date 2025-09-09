import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable, Dict, Tuple
import numpy as np
import cv2
import pyudev
from termcolor import cprint


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

def print_all_ports_in_use_for_USB_cameras() -> List[Tuple[str, str]]:
    """
    Print all USB cameras currently connected and their device paths.

    Returns:
        A list of (usb_port, device_path) tuples.
    """
    context = pyudev.Context()
    cameras = []

    for device in context.list_devices(subsystem='video4linux'):
        if device.parent and device.parent.subsystem == 'usb':
            usb_port = device.parent.get('DEVPATH', '').split('/')[-1]
            device_path = device.device_node
            cameras.append((usb_port, device_path))
            print(f"USB Port: {usb_port} -> Device: {device_path}")

    if not cameras:
        print("No USB cameras detected.")

    return cameras

def find_camera_by_usb_port(usb_port: str) -> Optional[str]:
    """
    Find the /dev/video* device associated with a specific USB port
    """
    context = pyudev.Context()
    
    for device in context.list_devices(subsystem='video4linux'):
        if device.parent and 'usb' in device.parent.subsystem:
            device_usb_port = device.parent.get('DEVPATH', '').split('/')[-1]
            if usb_port in device_usb_port:
                return device.device_node
    
    return None

def open_camera_by_usb_port(usb_port: str, api_preference: int = cv2.CAP_ANY) -> Optional[cv2.VideoCapture]:
    """
    Open a camera by its USB port identifier
    
    Args:
        usb_port: USB port identifier (e.g., "3-10.4")
        api_preference: OpenCV API preference (default: cv2.CAP_ANY)
    
    Returns:
        OpenCV VideoCapture object or None if failed
    """
    device_path = find_camera_by_usb_port(usb_port)
    
    if device_path is None:
        print(f"No camera found on USB port {usb_port}")
        return None
    
    print(f"Opening camera on USB port {usb_port} at {device_path}")
    cap = cv2.VideoCapture(device_path, api_preference)
    
    if not cap.isOpened():
        cprint(f"Failed to open camera on USB port {usb_port}", "red")
        return None
    else: 
        cprint(f"Finish opening camera on USB port {usb_port}", "green")
        return cap

class CameraManager:
    def __init__(self, camera_to_port):
        self.camera_to_port = camera_to_port
        self.port_to_camera = {v: k for k, v in self.camera_to_port.items()}
        self.all_port_names = list(self.port_to_camera.keys())
        self.all_camera_names = list(self.camera_to_port.keys())
        self.active_cameras = {}
    
    def open_camera_by_name(self, camera_name: str):
        """
        Open a camera by its name
        
        Args:
            camera_name: Name of the camera to open
        
        Returns:
            VideoCapture object or None if failed
        """
        port = self.camera_to_port.get(camera_name)
        if not port:
            print(f"Unknown camera: {camera_name}")
            return None
        
        cap = open_camera_by_usb_port(port)
        if cap:
            self.active_cameras[camera_name] = cap
            print(f"Opened {camera_name} on port {port}")
        return cap
    
    def open_camera_by_port(self, port: str):
        """
        Open a camera by its USB port
        
        Args:
            port: USB port of the camera to open
        
        Returns:
            VideoCapture object or None if failed
        """
        camera_name = self.port_to_camera.get(port)
        if not camera_name:
            print(f"Unknown port: {port}")
            return None
        
        cap = open_camera_by_usb_port(port)
        if cap:
            self.active_cameras[camera_name] = cap
            print(f"Opened {camera_name} on port {port}")
        return cap
    
    def open_all_cameras(self):
        """
        Open all cameras in the mapping
        
        Returns:
            Number of successfully opened cameras
        """
        success_count = 0
        for camera_name, port in self.camera_to_port.items():
            if self.open_camera_by_name(camera_name):
                success_count += 1
        return success_count
    
    def read_image(self, identifier: str, convert_rgb: bool = False) -> Optional["np.ndarray"]:
        """
        Read a single frame from a specific camera (by name or port).

        Args:
            identifier: Camera name or USB port string (e.g., "front_cam" or "3-10.4").
            convert_rgb: If True, convert BGR -> RGB before returning.

        Returns:
            The image frame (numpy array) or None if read failed / camera not open.
        """
        cap = self.get_camera(identifier)
        if cap is None:
            print(f"[read_image] Camera '{identifier}' is not opened.")
            return None

        ok, frame = cap.read()
        if not ok:
            print(f"[read_image] Failed to read from camera '{identifier}'.")
            return None

        if convert_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def read_all_images(self, convert_rgb: bool = False) -> Dict[str, Optional["np.ndarray"]]:
        """
        Read one frame from all currently active/opened cameras.

        Args:
            convert_rgb: If True, convert BGR -> RGB for every frame.

        Returns:
            Dict mapping {camera_name: frame or None if read failed}.
        """
        results: Dict[str, Optional["np.ndarray"]] = {}

        # If you want tighter temporal alignment, you can first .grab() all, then .retrieve() all.
        # Here we keep it simple with .read() per camera.
        for camera_name, cap in self.active_cameras.items():
            ok, frame = cap.read()
            if not ok:
                print(f"[read_all_images] Failed to read from '{camera_name}'.")
                results[camera_name] = None
                continue
            if convert_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results[camera_name] = frame

        return results
    
    def get_camera(self, identifier: str):
        """
        Get camera by name or port
        
        Args:
            identifier: Either camera name or USB port
        
        Returns:
            VideoCapture object or None if not found
        """
        # Check if identifier is a camera name
        if identifier in self.camera_to_port:
            return self.active_cameras.get(identifier)
        
        # Check if identifier is a port
        if identifier in self.port_to_camera:
            camera_name = self.port_to_camera[identifier]
            return self.active_cameras.get(camera_name)
        
        return None
    
    
    def release_camera(self, identifier: str):
        """
        Release a camera by name or port
        """
        camera_name = None
        
        if identifier in self.camera_to_port:
            camera_name = identifier
        elif identifier in self.port_to_camera:
            camera_name = self.port_to_camera[identifier]
        
        if camera_name and camera_name in self.active_cameras:
            self.active_cameras[camera_name].release()
            del self.active_cameras[camera_name]
            print(f"Released {camera_name}")
    
    def release_all(self):
        """Release all cameras"""
        for camera_name, cap in list(self.active_cameras.items()):
            cap.release()
            del self.active_cameras[camera_name]
        print("Released all cameras")

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
    


# ---------- small drawing helpers ----------
def _fit_size_preserve_aspect(src_w, src_h, dst_w, dst_h) -> Tuple[int, int]:
    if src_w == 0 or src_h == 0:
        return dst_w, dst_h
    scale = min(dst_w / src_w, dst_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))

def _put_label(img, text, org=(12, 28), fg=(255, 255, 255), bg=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6, 2
    (tw, th_pix), _ = cv2.getTextSize(text, font, fs, th)
    x1, y1 = org[0] - 8, org[1] - th_pix - 10
    x2, y2 = org[0] + tw + 8, org[1] + 6
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, thickness=-1)
    cv2.putText(img, text, org, font, fs, fg, th, cv2.LINE_AA)

def _draw_border(img, color=(60, 200, 60), thickness=3):
    h, w = img.shape[:2]
    cv2.rectangle(img, (1, 1), (w - 2, h - 2), color, thickness)

# ---------- Interactive manager ----------
class InteractiveCameraManager:
    """
    Interactive multi-camera capture UI for a `CameraManager`.

    - Shows a tiled preview of all *opened* cameras (uses manager.read_all_images()).
    - Maintains per-camera RGB buffers (list of frames).
    - Optional undistort_fn(img_rgb) applied to preview and stored frames.

    Controls:
      s : store current frame of  ALL cameras
      p : pause / resume
      h : toggle help overlay
      c : clear all buffers
      q / ESC : quit
    """
    def __init__(
        self,
        manager,                               # your CameraManager
        window_name: str = "InteractiveCameraManager",
        window_wh: Tuple[int, int] = (1600, 900),
        display_scale: Optional[float] = None, # if set, scale composite window
        show_fps: bool = True,
        undistort_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        buffer_max: Optional[int] = None,
        rgb_input: bool = False,               # if manager returns BGR frames, set False (default)
    ):
        self.mgr = manager
        self.window_name = window_name
        self.window_wh = window_wh
        self.display_scale = display_scale
        self.show_fps = show_fps
        self.undistort_fn = undistort_fn
        self.buffer_max = buffer_max
        self.rgb_input = rgb_input

        # buffers per camera name
        self.buffers: Dict[str, List[np.ndarray]] = {name: [] for name in self.mgr.all_camera_names}

        self.paused = False
        self.show_help = True
        self.last_time = time.time()
        self.fps = 0.0

        # keep last good frames even when paused or a read returns None
        self.last_frames: Dict[str, Optional[np.ndarray]] = {name: None for name in self.mgr.all_camera_names}

        self.names: List[str] = list(self.mgr.all_camera_names)

    # ------------------------ public buffer helpers ------------------------ #
    def get_buffer_array(self, camera_name: str) -> Optional[np.ndarray]:
        buf = self.buffers.get(camera_name, [])
        if not buf:
            return None
        return np.stack(buf, axis=0)

    def get_all_buffers(self) -> Dict[str, Optional[np.ndarray]]:
        out: Dict[str, Optional[np.ndarray]] = {}
        for k in self.names:
            arr = self.get_buffer_array(k)
            out[k] = arr
        return out

    def clear_buffer(self, camera_name: str) -> None:
        if camera_name in self.buffers:
            self.buffers[camera_name].clear()

    def clear_all_buffers(self) -> None:
        for k in self.buffers:
            self.buffers[k].clear()

    def save_buffers(self, save_root: str, prefix: str = "frame", ext: str = "png") -> Dict[str, List[str]]:
        """
        Save all per-camera buffers under:
          save_root/<camera_name>/<prefix>_000000.png ...
        Returns: dict of camera_name -> list of saved paths
        """
        os.makedirs(save_root, exist_ok=True)
        saved: Dict[str, List[str]] = {}
        for cam, frames in self.buffers.items():
            cam_dir = os.path.join(save_root, cam)
            os.makedirs(cam_dir, exist_ok=True)
            paths = []
            for idx, img_rgb in enumerate(frames):
                bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                path = os.path.join(cam_dir, f"{prefix}_{idx:06d}.{ext.lower()}")
                if not cv2.imwrite(path, bgr):
                    raise RuntimeError(f"Failed to write: {path}")
                paths.append(path)
            saved[cam] = paths
        return saved

    # ------------------------ internal helpers ---------------------------- #
    def _update_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps = 1.0 / dt

    def _store_one(self, cam_name: str, frame_rgb: np.ndarray):
        if self.buffer_max is not None and len(self.buffers[cam_name]) >= self.buffer_max:
            self.buffers[cam_name].pop(0)
        self.buffers[cam_name].append(frame_rgb.copy())

    def _store_all(self):
        """Store current frames from ALL cameras into their buffers."""
        for name, frame in self.last_frames.items():
            if frame is None:
                continue
            if self.buffer_max is not None and len(self.buffers[name]) >= self.buffer_max:
                self.buffers[name].pop(0)
            self.buffers[name].append(frame.copy())

    def _read_all(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Read frames from manager; apply undistort if provided.
        Returns RGB frames (uint8) or None.
        """
        # Ask manager for frames. If it returns BGR (default), convert to RGB.
        frames = self.mgr.read_all_images(convert_rgb=self.rgb_input)
        # convert to RGB if needed
        if not self.rgb_input:
            frames = {k: (cv2.cvtColor(v, cv2.COLOR_BGR2RGB) if v is not None else None)
                      for k, v in frames.items()}

        # undistort if provided
        if self.undistort_fn is not None:
            for k, f in frames.items():
                if f is None:
                    continue
                try:
                    frames[k] = self.undistort_fn(f)
                except Exception:
                    pass
        return frames

    def _read_all_synced(self) -> dict[str, np.ndarray | None]:
        """Return latest RGB frames for all active cameras using grab->retrieve."""
        frames = {}
        # 1) grab all
        for name, cap in self.mgr.active_cameras.items():
            if cap is not None:
                cap.grab()
        # 2) retrieve all
        for name, cap in self.mgr.active_cameras.items():
            if cap is None:
                frames[name] = None
                continue
            ok, bgr = cap.retrieve()
            if not ok or bgr is None:
                frames[name] = None
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # optional undistort hook
            if self.undistort_fn is not None:
                try:
                    rgb = self.undistort_fn(rgb)
                except Exception:
                    pass
            frames[name] = rgb
        return frames

    def _tile_frames(self, name_to_rgb: Dict[str, Optional[np.ndarray]],
                     window_wh: Tuple[int, int],
                     padding: int = 8,
                     bg_color=(30, 30, 30)) -> np.ndarray:
        W, H = window_wh
        names = [n for n in self.names if name_to_rgb.get(n) is not None]
        frames = [name_to_rgb[n] for n in names]
        n = len(frames)

        canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)
        if n == 0:
            _put_label(canvas, "No frames available")
            return canvas

        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        total_pad_x = padding * (cols + 1)
        total_pad_y = padding * (rows + 1)
        tile_w = max(1, (W - total_pad_x) // cols)
        tile_h = max(1, (H - total_pad_y) // rows)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                frame_rgb = frames[idx]
                name = names[idx]
                h, w = frame_rgb.shape[:2]
                new_w, new_h = _fit_size_preserve_aspect(w, h, tile_w, tile_h)
                disp = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

                # annotate
                buf_len = len(self.buffers.get(name, []))
                label = f"[{idx}] {name}  buf:{buf_len}"
                _put_label(disp_bgr, label)

                # paste centered
                x0 = padding + c * (tile_w + padding) + (tile_w - new_w) // 2
                y0 = padding + r * (tile_h + padding) + (tile_h - new_h) // 2
                canvas[y0:y0 + new_h, x0:x0 + new_w] = disp_bgr
                idx += 1

        # HUD (global)
        if self.show_fps:
            _put_label(canvas, f"{self.fps:5.1f} FPS", org=(W - 160, 32), fg=(30, 220, 30), bg=(0, 0, 0))
        if self.show_help:
            help_lines = [
                "[s] store ALL   [p] pause   [h] help   [c] clear all   [q]/[ESC] quit",
            ]
            y = H - 56
            for line in help_lines:
                _put_label(canvas, line, org=(12, y), fg=(240, 240, 240), bg=(0, 0, 0))
                y += 28
        return canvas

    # ------------------------------ main loop ----------------------------- #
    def run(self):
        """
        Start the UI loop. Returns dict(name -> stacked buffer ndarray) on exit.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_wh[0], self.window_wh[1])

        # main loop
        while True:
            # read or hold
            if not self.paused:
                frames = self._read_all_synced()
                # update last frames where new frames exist
                for k, f in frames.items():
                    if f is not None:
                        self.last_frames[k] = f
                self._update_fps()

            # build a dict of display frames, falling back to last known frames
            disp_frames = {k: (self.last_frames[k]) for k in self.names if self.last_frames[k] is not None}

            # composite
            canvas = self._tile_frames(disp_frames, self.window_wh)

            # optional global scaling of canvas for display only
            if self.display_scale and self.display_scale > 0:
                W, H = self.window_wh
                sw, sh = int(W * self.display_scale), int(H * self.display_scale)
                canvas = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)

            cv2.imshow(self.window_name, canvas)

            # key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q/ESC
                break
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('p'):
                self.paused = not self.paused
            elif key == ord('c'):
                self.clear_all_buffers()
            elif key == ord('s'):
                # store ALL cameras into buffers
                self._store_all()

            # exit if window closed
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyWindow(self.window_name)
        return self.get_all_buffers()

