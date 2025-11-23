import av
import cv2
import time
import pyudev
import threading
from typing import Optional, Dict, List, Tuple
from termcolor import cprint
import numpy as np

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

class _AVStreamWorker(threading.Thread):
    """
    Background reader for a single camera using PyAV (FFmpeg).
    Keeps the *latest* frame (numpy BGR) and last_decode_time.
    """
    def __init__(
        self,
        name: str,
        device: str,
        options: Dict[str, str],
        stream_index: int = 0,
    ):
        super().__init__(daemon=True)
        self.name = name
        if device.startswith("/dev/video"):
            self.device = device
        else: 
            self.device = find_camera_by_usb_port(device)
            cprint(self.device, "blue")
            assert self.device is not None, f"Couldn't find the device on port {device}"
        self.options = options or {}
        self.stream_index = stream_index

        self._container: Optional[av.container.input.InputContainer] = None
        self._stop_event = threading.Event()   # <--- renamed
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

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest_frame is None:
                return None, None
            # Return a copy to avoid race conditions when caller manipulates frames
            return self._latest_frame.copy(), self._latest_ts

    def stop(self):
        self._stop_event.set()

    def close(self):
        try:
            if self._container is not None:
                self._container.close()
        except Exception:
            pass
        self._container = None

    def run(self):
        """
        Open device with PyAV and continuously decode frames.
        Stores only the most recent frame and timestamp.
        """
        try:
            # Use v4l2 input explicitly; works for /dev/videoX devices on Linux
            self._container = av.open(self.device, format="v4l2", options=self.options)
            self._opened = True
            cprint(f"[{self.name}] Opened via PyAV at {self.device} with options={self.options}", "green")
        except Exception as e:
            self._open_error = str(e)
            cprint(f"[{self.name}] Failed to open: {e}", "red")
            return

        # Choose a stream (usually 0)
        try:
            video_stream = self._container.streams.video[self.stream_index]
            video_stream.thread_type = "AUTO"
        except Exception as e:
            self._open_error = f"Failed to get video stream {self.stream_index}: {e}"
            cprint(f"[{self.name}] {self._open_error}", "red")
            self.close()
            return

        # Main decode loop
        while not self._stop_event.is_set():
            try:
                # decode() yields AVFrames; using specific stream is more robust
                for frame in self._container.decode(video=video_stream.index):
                    if self._stop_event.is_set():
                        break
                    img = frame.to_ndarray(format="bgr24")  # numpy BGR
                    ts = time.perf_counter()
                    with self._lock:
                        self._latest_frame = img
                        self._latest_ts = ts
            except av.AVError as e:
                # Transient decode errors can happen (e.g., short reads). Log and continue.
                cprint(f"[{self.name}] AVError while decoding: {e}", "yellow")
                time.sleep(0.02)
            except Exception as e:
                cprint(f"[{self.name}] Unexpected error: {e}", "red")
                time.sleep(0.05)

        self.close()
        cprint(f"[{self.name}] Reader stopped and closed.", "cyan")


class AVCameraManager:
    """
    PyAV-backed multi-camera manager.
    - Input: dict {camera_name: device_path} (e.g., {"root":" /dev/video0", "tip":"/dev/video4"})
    - Options: per-camera PyAV options (e.g., {"input_format":"mjpeg", "video_size":"1280x480", "framerate":"25"})
    - Methods mirror your OpenCV manager where sensible.
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

    # ---------- Open/Close ----------

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

    # ---------- Helpers ----------

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


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Map camera names to v4l2 device nodes
    camera_to_port = {
        # "root": "/dev/video0",
        # "tip": "/dev/video2",
        "root": "3-9",
        "tip": "3-10",
    }

    # Defaults applied to all cameras (override per-camera below if needed)
    default_opts = {
        # Force MJPEG when devices otherwise fail in raw modes
        "input_format": "mjpeg",
        "video_size": "1280x480",  # or "640x480"
        "framerate": "25",         # or "60"
    }

    # Per-camera overrides (optional)
    per_cam_opts = {
        # "tip": {"video_size": "640x480", "framerate": "30"},
    }

    mgr = AVCameraManager(
        camera_to_port=camera_to_port,
        default_options=default_opts,
        per_camera_options=per_cam_opts,
        stream_index=0,
    )
    opened = mgr.open_all_cameras()
    print(f"Opened {opened} cameras")

    try:
        t_last = time.time()
        frames_cnt = 0
        while True:
            frames, ts = mgr.get_frames(return_timestamps=True, img_size=None)
            # Simple FPS and preview for one camera
            if "root" in frames:
                cv2.imshow("root", frames["root"])
            if "tip" in frames:
                cv2.imshow("tip", frames["tip"])

            frames_cnt += 1
            if time.time() - t_last >= 1.0:
                print(f"Pull FPS (not device FPS): {frames_cnt:.1f}")
                frames_cnt = 0
                t_last = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.04)

    finally:
        mgr.release_all()
        cv2.destroyAllWindows()
