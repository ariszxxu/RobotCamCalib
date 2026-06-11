from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
try:
    from termcolor import cprint
except Exception:
    def cprint(message, *args, **kwargs):
        print(message)


class Camera(ABC):
    """Common interface for cameras that return RGB uint8 frames."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    @abstractmethod
    def start(self) -> None:
        """Open the camera."""

    @abstractmethod
    def stop(self) -> None:
        """Close the camera."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return one RGB frame as an (H, W, 3) uint8 array."""


def start_cv2_camera_capture(
    src: int | str = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    api_pref: Optional[int] = None,
    buffer_size: Optional[int] = 1,
    warmup_s: float = 0.1,
) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture and apply best-effort startup properties."""
    cap = cv2.VideoCapture(src) if api_pref is None else cv2.VideoCapture(src, api_pref)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {src}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    if buffer_size is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))

    if warmup_s > 0:
        time.sleep(warmup_s)

    return cap


def resolve_cv2_camera_source(src: int | str) -> int | str:
    """
    Resolve a CV2 source into something cv2.VideoCapture can open.

    Supports integer indexes, /dev/videoX paths, and USB port identifiers such as
    "4-8.1:1.0".
    """
    if isinstance(src, int):
        return src
    if src.startswith("/dev/video"):
        return src

    resolved = find_camera_by_usb_port(src)
    return resolved if resolved is not None else src


class CV2Camera(Camera):
    """OpenCV-backed camera for USB cameras, webcams, and video files."""

    def __init__(
        self,
        src: int | str = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        convert_to_rgb: bool = True,
        api_pref: Optional[int] = None,
        buffer_size: Optional[int] = 1,
    ) -> None:
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.convert_to_rgb = convert_to_rgb
        self.api_pref = api_pref
        self.buffer_size = buffer_size
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        self.cap = start_cv2_camera_capture(
            src=resolve_cv2_camera_source(self.src),
            width=self.width,
            height=self.height,
            fps=self.fps,
            api_pref=self.api_pref,
            buffer_size=self.buffer_size,
        )

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() or use context manager.")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from CV2Camera.")

        if self.convert_to_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame


def find_camera_by_usb_port(usb_port: str) -> Optional[str]:
    """Resolve a Linux USB port id such as ``4-9.4.4.1:1.0`` to ``/dev/videoX``."""
    try:
        import pyudev
    except Exception as exc:
        raise ImportError("pyudev is required to resolve cameras by USB port id.") from exc

    context = pyudev.Context()
    for device in context.list_devices(subsystem="video4linux"):
        if device.parent is None or "usb" not in device.parent.subsystem:
            continue
        device_usb_port = device.parent.get("DEVPATH", "").split("/")[-1]
        if usb_port in device_usb_port:
            return device.device_node
    return None


class _AVStreamWorker(threading.Thread):
    """Background PyAV reader for one V4L2 camera, keeping only the latest frame."""

    def __init__(
        self,
        name: str,
        device: str,
        options: dict[str, str],
        stream_index: int = 0,
    ) -> None:
        super().__init__(daemon=True)
        self.name = name
        if device.startswith("/dev/video"):
            self.device = device
        else:
            resolved_device = find_camera_by_usb_port(device)
            if resolved_device is None:
                raise RuntimeError(f"Could not find a video device on USB port {device}.")
            self.device = resolved_device

        self.options = dict(options or {})
        self.stream_index = stream_index
        self._container: Any | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: Optional[float] = None
        self._opened = False
        self._open_error: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def last_error(self) -> Optional[str]:
        return self._open_error

    def get_latest(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_ts

    def stop(self) -> None:
        self._stop_event.set()

    def close(self) -> None:
        try:
            if self._container is not None:
                self._container.close()
        except Exception:
            pass
        self._container = None

    def run(self) -> None:
        try:
            import av
        except Exception as exc:
            self._open_error = "PyAV is required for AVCameraManager. Install package `av`."
            cprint(f"[{self.name}] {self._open_error}", "red")
            return

        try:
            self._container = av.open(self.device, format="v4l2", options=self.options)
            self._opened = True
            cprint(f"[{self.name}] Opened via PyAV at {self.device} with options={self.options}", "green")
        except Exception as exc:
            self._open_error = str(exc)
            cprint(f"[{self.name}] Failed to open: {exc}", "red")
            return

        try:
            video_stream = self._container.streams.video[self.stream_index]
            video_stream.thread_type = "AUTO"
        except Exception as exc:
            self._open_error = f"Failed to get video stream {self.stream_index}: {exc}"
            cprint(f"[{self.name}] {self._open_error}", "red")
            self.close()
            return

        while not self._stop_event.is_set():
            try:
                for frame in self._container.decode(video=video_stream.index):
                    if self._stop_event.is_set():
                        break
                    img = frame.to_ndarray(format="bgr24")
                    ts = time.perf_counter()
                    with self._lock:
                        self._latest_frame = img
                        self._latest_ts = ts
            except Exception as exc:
                av_error = getattr(av, "AVError", None)
                if isinstance(av_error, type) and isinstance(exc, av_error):
                    cprint(f"[{self.name}] AVError while decoding: {exc}", "yellow")
                    time.sleep(0.02)
                else:
                    cprint(f"[{self.name}] Unexpected error: {exc}", "red")
                    time.sleep(0.05)

        self.close()
        cprint(f"[{self.name}] Reader stopped and closed.", "cyan")


class AVCameraManager(Camera):
    """PyAV-backed multi-camera manager."""

    def __init__(
        self,
        camera_to_port: dict[str, str],
        camera_left_right_order: Optional[dict[str, tuple[str, str] | list[str]]] = None,
        default_options: Optional[dict[str, str]] = None,
        per_camera_options: Optional[dict[str, dict[str, str]]] = None,
        stream_index: int = 0,
        read_camera_name: Optional[str] = None,
    ) -> None:
        self.camera_to_port = dict(camera_to_port)
        self.camera_left_right_order = dict(camera_left_right_order or {})
        self.default_options = dict(default_options or {})
        self.per_camera_options = dict(per_camera_options or {})
        self.stream_index = stream_index
        self.read_camera_name = read_camera_name

        self._workers: dict[str, _AVStreamWorker] = {}
        self._active: dict[str, bool] = {}

    def _merged_options_for(self, name: str) -> dict[str, str]:
        merged = dict(self.default_options)
        merged.update(self.per_camera_options.get(name, {}))
        return merged

    def open_camera_by_name(self, camera_name: str) -> bool:
        device = self.camera_to_port.get(camera_name)
        if not device:
            cprint(f"Unknown camera: {camera_name}", "red")
            return False

        if camera_name in self._workers:
            cprint(f"{camera_name} already opened.", "yellow")
            return True

        worker = _AVStreamWorker(
            name=camera_name,
            device=device,
            options=self._merged_options_for(camera_name),
            stream_index=self.stream_index,
        )
        worker.start()

        deadline = time.time() + 1.5
        while time.time() < deadline and not worker.is_open and worker.last_error is None:
            time.sleep(0.02)

        if worker.is_open:
            self._workers[camera_name] = worker
            self._active[camera_name] = True
            return True

        worker.stop()
        worker.join(timeout=0.5)
        error = worker.last_error or "Unknown error while opening"
        cprint(f"Failed to open {camera_name}: {error}", "red")
        return False

    def open_all_cameras(self) -> int:
        return sum(1 for name in self.camera_to_port if self.open_camera_by_name(name))

    def release_camera(self, identifier: str) -> None:
        name = self._resolve_camera_name(identifier)
        if name is None or name not in self._workers:
            return

        worker = self._workers.pop(name)
        self._active.pop(name, None)
        worker.stop()
        worker.join(timeout=1.0)
        cprint(f"Released {name}", "cyan")

    def release_all(self) -> None:
        for name in list(self._workers):
            self.release_camera(name)
        cprint("Released all cameras", "cyan")

    def stereo_to_mono_frame_dict(
        self,
        stereo_frames: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        mono_frames: dict[str, np.ndarray] = {}

        for camera_name, frame in stereo_frames.items():
            if camera_name not in self.camera_left_right_order:
                mono_frames[camera_name] = frame
                continue

            left_name, right_name = self.camera_left_right_order[camera_name]
            _height, width, _channels = frame.shape
            if width % 2 != 0:
                raise RuntimeError(f"Expected even width for stereo frame from {camera_name}.")

            middle = width // 2
            mono_frames[left_name] = frame[:, :middle, :]
            mono_frames[right_name] = frame[:, middle:, :]

        return mono_frames

    def get_frames(
        self,
        camera_names: Optional[list[str]] = None,
        img_size: Optional[tuple[int, int]] = None,
    ) -> dict[str, np.ndarray]:
        """Return latest frames as BGR arrays, keyed by camera name."""
        if camera_names is None:
            camera_names = list(self._workers)
        if not camera_names:
            cprint("No active cameras to read from.", "red")
            return {}

        frames: dict[str, np.ndarray] = {}
        for name in camera_names:
            worker = self._workers.get(name)
            if worker is None:
                cprint(f"Camera '{name}' is not opened.", "red")
                continue

            frame, _timestamp = worker.get_latest()
            if frame is None:
                continue

            if img_size is not None:
                output_size = img_size
                if name in self.camera_left_right_order:
                    output_size = (img_size[0] * 2, img_size[1])
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)

            frames[name] = frame

        return self.stereo_to_mono_frame_dict(frames)

    def set_camera_options(self, camera_name: str, options: dict[str, str]) -> None:
        """Update PyAV open options for one camera. Re-open for changes to take effect."""
        current = self.per_camera_options.get(camera_name, {})
        current.update(options)
        self.per_camera_options[camera_name] = current
        cprint(f"[{camera_name}] Updated options: {self.per_camera_options[camera_name]}", "blue")

    def start(self) -> None:
        opened = self.open_all_cameras()
        if opened <= 0:
            raise RuntimeError("Failed to open any AV camera.")

    def stop(self) -> None:
        self.release_all()

    def read(self) -> np.ndarray:
        frames = self.get_frames()
        if not frames:
            raise RuntimeError("No AV camera frame is available yet.")

        camera_name = self.read_camera_name or next(iter(frames))
        frame_bgr = frames.get(camera_name)
        if frame_bgr is None:
            available = ", ".join(frames)
            raise RuntimeError(f"Camera '{camera_name}' has no frame. Available: {available}")

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _resolve_camera_name(self, identifier: str) -> Optional[str]:
        if identifier in self.camera_to_port:
            return identifier

        for name, device in self.camera_to_port.items():
            if device == identifier:
                return name

        return None


class RealsenseCamera(Camera):
    """Intel RealSense color camera via pyrealsense2."""

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
    ) -> None:
        if serial is None:
            serial = self.get_realsense_serial_numbers()
            if serial is None:
                raise RuntimeError("No RealSense device found; cannot proceed.")

        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.format = format.lower()
        self.auto_exposure = auto_exposure
        self.exposure_us = exposure_us
        self.white_balance = white_balance

        self.rs: Any | None = None
        self.pipeline: Any | None = None
        self.config: Any | None = None
        self.color_sensor: Any | None = None

    @staticmethod
    def _import_realsense():
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise ImportError(
                "pyrealsense2 is required for RealsenseCamera. Install Intel RealSense SDK."
            ) from exc

        return rs

    def get_realsense_serial_numbers(self) -> Optional[str]:
        rs = self._import_realsense()
        devices = rs.context().query_devices()
        serial: Optional[str] = None

        if len(devices) == 0:
            print("No RealSense devices found.")
            return None

        print(f"{len(devices)} RealSense device(s) found:")
        for index, device in enumerate(devices):
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            product_line = device.get_info(rs.camera_info.product_line)
            firmware = device.get_info(rs.camera_info.firmware_version)
            print(f"\nDevice {index + 1}:")
            print(f"  Name         : {name}")
            print(f"  Serial Number: {serial}")
            print(f"  Product Line : {product_line}")
            print(f"  Firmware     : {firmware}")

        return serial

    def get_profiles(self, verbose: bool = False):
        rs = self._import_realsense()
        devices = rs.context().query_devices()

        color_profiles = []
        depth_profiles = []
        for device in devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            if verbose:
                print(f"Sensor: {name}, {serial}")
                print("Supported video formats:")

            for sensor in device.query_sensors():
                for stream_profile in sensor.get_stream_profiles():
                    stream_type = str(stream_profile.stream_type())
                    if stream_type not in ["stream.color", "stream.depth"]:
                        continue

                    video_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    width = video_profile.width()
                    height = video_profile.height()
                    fps = video_profile.fps()
                    video_type = stream_type.split(".")[-1]

                    if verbose:
                        print(
                            f"  {video_type}: width={width}, height={height}, "
                            f"fps={fps}, fmt={fmt}"
                        )

                    if video_type == "color":
                        color_profiles.append((width, height, fps, fmt))
                    else:
                        depth_profiles.append((width, height, fps, fmt))

        return color_profiles, depth_profiles

    def start(self) -> None:
        rs = self._import_realsense()
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial:
            self.config.enable_device(self.serial)

        format_map = {
            "rgb8": rs.format.rgb8,
            "rgba8": rs.format.rgba8,
            "bgr8": rs.format.bgr8,
            "bgra8": rs.format.bgra8,
        }
        if self.format not in format_map:
            raise ValueError(f"Unsupported RealSense color format: {self.format}")

        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            format_map[self.format],
            self.fps,
        )
        profile = self.pipeline.start(self.config)

        try:
            self._configure_color_sensor(profile)
        except Exception:
            pass

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
        if self.pipeline is None:
            raise RuntimeError("Camera not started. Call start() or use context manager.")

        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame from RealSense.")

        image = np.asanyarray(color.get_data())
        if self.format == "rgb8":
            return image
        if self.format == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if self.format == "bgr8":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.format == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        raise RuntimeError(f"Unsupported RealSense color format: {self.format}")

    def _configure_color_sensor(self, profile: Any) -> None:
        rs = self.rs
        if rs is None:
            return

        sensors = profile.get_device().query_sensors()
        self.color_sensor = None
        for sensor in sensors:
            if "rgb" in sensor.get_info(rs.camera_info.name).lower():
                self.color_sensor = sensor
                break

        if self.color_sensor is None:
            return

        if self.auto_exposure:
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        else:
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if self.exposure_us is not None:
                self.color_sensor.set_option(rs.option.exposure, float(self.exposure_us))

        if (
            self.white_balance is not None
            and self.color_sensor.supports(rs.option.white_balance)
        ):
            self.color_sensor.set_option(rs.option.white_balance, float(self.white_balance))


def _import_depthai():
    try:
        import depthai as dai
    except Exception as exc:
        raise ImportError("depthai is required for OAK cameras.") from exc

    return dai


def list_oak_devices() -> list[dict[str, Any]]:
    """List currently available OAK / DepthAI devices."""
    dai = _import_depthai()
    devices = dai.Device.getAllAvailableDevices()
    infos: list[dict[str, Any]] = []

    print(f"[INFO] Found {len(devices)} OAK / DepthAI device(s):")
    for index, device in enumerate(devices):
        info = {
            "index": index,
            "name": getattr(device, "name", "unknown"),
            "mxid": getattr(device, "mxid", "unknown"),
            "state": getattr(device, "state", "unknown"),
        }
        infos.append(info)
        print(
            f"  [{index}] name={info['name']}, "
            f"mxid={info['mxid']}, state={info['state']}"
        )

    return infos


def check_unique_devices(camera_to_device: dict[str, str]) -> None:
    """Ensure no two logical cameras use the same physical OAK device name."""
    seen: dict[str, str] = {}

    for camera_name, device_name in camera_to_device.items():
        if not device_name:
            continue
        if device_name in seen:
            raise ValueError(
                f"Duplicated OAK device name: {device_name}. "
                f"Used by both '{seen[device_name]}' and '{camera_name}'."
            )
        seen[device_name] = camera_name


@dataclass(frozen=True)
class OAKCameraConfig:
    """Configuration for one logical OAK camera."""

    name: str
    device_name: str
    isp_scale: tuple[int, int]
    fps: int
    rotate_180: bool
    queue_size: int
    queue_blocking: bool


class OAK1WStreamWorker:
    """Background reader for one OAK-1-W camera."""

    def __init__(self, config: OAKCameraConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._device: Any | None = None
        self._pipeline: Any | None = None
        self._queue: Any | None = None

        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float | None = None
        self._opened = False
        self._open_error: str | None = None
        self._read_error_count = 0

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def last_error(self) -> str | None:
        return self._open_error

    def open(self) -> bool:
        dai = _import_depthai()
        cfg = self.config
        print(f"[DEBUG] Opening {cfg.name} with cfg.device_name={cfg.device_name}")

        try:
            if not cfg.device_name:
                raise ValueError(
                    f"[{cfg.name}] device_name is empty. "
                    "Provide explicit USB path or MXID."
                )

            device_info = dai.DeviceInfo(cfg.device_name)
            device = dai.Device(device_info)
            pipeline = dai.Pipeline(device)

            cam = pipeline.create(dai.node.ColorCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
            cam.setIspScale(cfg.isp_scale[0], cfg.isp_scale[1])
            cam.setInterleaved(False)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

            if hasattr(cam, "setFps"):
                cam.setFps(cfg.fps)

            queue = cam.isp.createOutputQueue(
                maxSize=cfg.queue_size,
                blocking=cfg.queue_blocking,
            )

            pipeline.start()
            self._device = device
            self._pipeline = pipeline
            self._queue = queue
            self._opened = True
            self._open_error = None

            print(
                f"[INFO] [{cfg.name}] opened: device_name={cfg.device_name}, "
                f"mxid={self._safe_get_mxid(device)}, "
                f"isp_scale={cfg.isp_scale[0]}/{cfg.isp_scale[1]}, fps={cfg.fps}"
            )
            return True

        except (RuntimeError, ValueError) as exc:
            self._opened = False
            self._open_error = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] [{cfg.name}] failed to open: {self._open_error}")
            self.close()
            return False

    def start(self) -> None:
        if not self._opened:
            raise RuntimeError(f"[{self.config.name}] cannot start before open().")
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def close(self) -> None:
        self.stop()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

        pipeline = self._pipeline
        device = self._device

        self._thread = None
        self._pipeline = None
        self._device = None
        self._queue = None
        self._opened = False

        if pipeline is not None and hasattr(pipeline, "stop"):
            try:
                pipeline.stop()
            except RuntimeError as exc:
                print(f"[WARNING] [{self.config.name}] pipeline.stop() failed: {exc}")

        if device is not None and hasattr(device, "close"):
            try:
                device.close()
            except RuntimeError as exc:
                print(f"[WARNING] [{self.config.name}] device.close() failed: {exc}")

    def get_latest(self) -> tuple[np.ndarray | None, float | None]:
        with self._lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_ts

    def _read_loop(self) -> None:
        cfg = self.config

        while not self._stop_event.is_set():
            queue = self._queue
            if queue is None:
                time.sleep(0.01)
                continue

            try:
                msg = queue.get()
                frame = msg.getCvFrame()

                if cfg.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = time.perf_counter()

            except RuntimeError as exc:
                self._read_error_count += 1
                print(
                    f"[WARNING] [{cfg.name}] read error "
                    f"#{self._read_error_count}: {type(exc).__name__}: {exc}"
                )
                time.sleep(0.02)

        print(f"[INFO] [{cfg.name}] reader stopped.")

    @staticmethod
    def _safe_get_mxid(device: Any) -> str:
        if not hasattr(device, "getMxId"):
            return "unknown"

        try:
            return str(device.getMxId())
        except RuntimeError:
            return "unknown"


class OAK1WCameraManager:
    """Multi-OAK-1-W camera manager."""

    def __init__(
        self,
        camera_to_device: dict[str, str],
        isp_scale: tuple[int, int] = (1, 3),
        fps: int = 25,
        rotate_180_names: Optional[set[str]] = None,
        queue_size: int = 4,
        queue_blocking: bool = False,
        output_size: Optional[tuple[int, int]] = None,
    ) -> None:
        del output_size
        check_unique_devices(camera_to_device)

        rotate_names = rotate_180_names or set()
        self.configs: dict[str, OAKCameraConfig] = {
            name: OAKCameraConfig(
                name=name,
                device_name=device_name,
                isp_scale=isp_scale,
                fps=fps,
                rotate_180=name in rotate_names,
                queue_size=queue_size,
                queue_blocking=queue_blocking,
            )
            for name, device_name in camera_to_device.items()
        }
        self._workers: dict[str, OAK1WStreamWorker] = {}

    def open_camera_by_name(self, camera_name: str) -> bool:
        if camera_name not in self.configs:
            print(f"[ERROR] Unknown OAK camera: {camera_name}")
            return False

        if camera_name in self._workers:
            print(f"[WARNING] {camera_name} already opened.")
            return True

        worker = OAK1WStreamWorker(self.configs[camera_name])
        if not worker.open():
            error = worker.last_error or "unknown open error"
            print(f"[ERROR] Failed to open {camera_name}: {error}")
            return False

        worker.start()
        self._workers[camera_name] = worker
        return True

    def open_all_cameras(self) -> int:
        opened = sum(1 for camera_name in self.configs if self.open_camera_by_name(camera_name))

        print(f"[INFO] Requested OAK cameras: {list(self.configs)}")
        print(f"[INFO] Opened OAK cameras: {list(self._workers)}")

        missing = [name for name in self.configs if name not in self._workers]
        if missing:
            print(f"[WARNING] Failed or missing OAK cameras: {missing}")

        return opened

    def get_frames(
        self,
        camera_names: Optional[list[str]] = None,
        img_size: Optional[tuple[int, int]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Return resized BGR frames and original ISP BGR frames.

        Use img_size=(width, height) when host-side resizing is needed.
        """
        if camera_names is None:
            camera_names = list(self._workers)

        frames: dict[str, np.ndarray] = {}
        origin_frames: dict[str, np.ndarray] = {}

        for name in camera_names:
            worker = self._workers.get(name)
            if worker is None:
                print(f"[WARNING] OAK camera is not opened: {name}")
                continue

            frame, _timestamp = worker.get_latest()
            if frame is None:
                continue

            origin_frames[name] = frame
            if img_size is None:
                frames[name] = frame
            else:
                frames[name] = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)

        return frames, origin_frames

    def get_timestamps(
        self,
        camera_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        if camera_names is None:
            camera_names = list(self._workers)

        timestamps: dict[str, float] = {}
        for name in camera_names:
            worker = self._workers.get(name)
            if worker is None:
                continue

            _frame, timestamp = worker.get_latest()
            if timestamp is not None:
                timestamps[name] = timestamp

        return timestamps

    def wait_for_first_frames(
        self,
        camera_names: Optional[list[str]] = None,
        timeout_s: float = 5.0,
    ) -> bool:
        if camera_names is None:
            camera_names = list(self._workers)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            frames, _origin_frames = self.get_frames(camera_names=camera_names)
            if all(name in frames for name in camera_names):
                return True
            time.sleep(0.02)

        frames, _origin_frames = self.get_frames(camera_names=camera_names)
        missing = [name for name in camera_names if name not in frames]
        print(f"[WARNING] Timeout waiting for first OAK frames. Missing: {missing}")
        return False

    def release_camera(self, camera_name: str) -> None:
        worker = self._workers.pop(camera_name, None)
        if worker is None:
            return

        worker.close()
        print(f"[INFO] Released {camera_name}")

    def release_all(self) -> None:
        for camera_name in list(self._workers):
            self.release_camera(camera_name)
        print("[INFO] Released all OAK cameras.")

    def is_open(self, camera_name: str) -> bool:
        worker = self._workers.get(camera_name)
        return bool(worker is not None and worker.is_open)

    def get_active_camera_names(self) -> list[str]:
        return list(self._workers)
