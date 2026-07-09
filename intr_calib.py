import numpy as np
import cv2
import yaml
import os
import re
from datetime import datetime
from typing import Any, Dict, Tuple, List, Optional

# Fixed checkerboard you print from PDF (edit if needed)
CHECKERBOARD: Tuple[int, int] = (8, 6)  # (cols, rows) of inner corners
SQUARE_SIZE: float = 24.0                 # square length in your chosen unit
MIN_SAMPLES: int = 12                     # minimum valid detections recommended

# ---------------------------- Script camera macros ---------------------------- #
# Put every calibration camera used by this script here. Do not depend on external
# personal/lab parameter tables for the actual calibration resolution.
SUPPORTED_CAMERA_TYPES: Tuple[str, ...] = ("realsense", "cv2", "cv2_fisheye", "av", "oak1w")
DEFAULT_CAMERA_TYPE: str = "realsense"
DEFAULT_INTRINSICS_OUTPUT_TEMPLATE: str = "outputs/intrinsics_{camera_name}_{width}x{height}.yaml"
AUTO_STORE_VALID_IMAGES: bool = True
AUTO_STORE_COOLDOWN_S: float = 0.8
INTRINSICS_SAMPLE_IMAGE_ROOT: str = "outputs/intrinsics_checkerboard_samples"
INTRINSICS_SAMPLE_IMAGE_EXT: str = "png"

# RealSense camera macros.
REALSENSE_CAMERA_NAME: str = "realsense"
REALSENSE_SERIAL: Optional[str] = "244222070135"
REALSENSE_RESOLUTION: Tuple[int, int] = (1280, 720)  # (width, height)
REALSENSE_FPS: int = 30
REALSENSE_FORMAT: str = "bgra8"
REALSENSE_AUTO_EXPOSURE: bool = True
REALSENSE_EXPOSURE_US: Optional[float] = None
REALSENSE_WHITE_BALANCE: Optional[float] = None

# CV2/USB camera macros.
DEFAULT_CV2_CAMERA_NAME: str = "cam0"
CV2_CAMERA_CONFIGS: dict[str, dict[str, Any]] = {
    "cam0": {
        "port": "3-10.1:1.0",
        "resolution": (2592, 1944),  # (width, height), highest advertised mode
        "fps": 50,
        "fourcc": "MJPG",
    },
    # Add real calibration cameras here when needed:
    # "cam1": {"port": "4-8.2:1.0", "resolution": (1920, 1080), "fps": 30},
    # "cam2": {"port": "4-8.3:1.0", "resolution": (1920, 1080), "fps": 30},
}

CV2_CAMERA_COUNT: int = len(CV2_CAMERA_CONFIGS)
CAMERA_TO_PORT: dict[str, str] = {
    name: str(config["port"]) for name, config in CV2_CAMERA_CONFIGS.items()
}

# AV camera example macros.
AV_CAMERA_TO_PORT: dict[str, str] = {
    "I": "3-5.4.4.3:1.0",
}
AV_CAMERA_LEFT_RIGHT_ORDER: dict[str, list[str]] = {
    "I": ["I-root", "I-tip"],
}
AV_VIDEO_SIZE: Tuple[int, int] = (640, 240)
AV_FRAMERATE: int = 25

# OAK example macros. Fill this in here if you calibrate OAK cameras with this
# script; the script intentionally does not import CAMERA_TO_DEVICE from cameras.py.
OAK_CAMERA_TO_DEVICE: dict[str, str] = {}
OAK_DETECT_IMG_SIZE: Tuple[int, int] = (1280, 960)  # (width, height)
OAK_ISP_SCALE: Tuple[int, int] = (1, 3)
OAK_FPS: int = 25
OAK_QUEUE_SIZE: int = 4
OAK_QUEUE_BLOCKING: bool = False
OAK_ROTATE_180_NAMES: set[str] = set()


def get_cv2_camera_config(camera_name: str) -> dict[str, Any]:
    if camera_name not in CV2_CAMERA_CONFIGS:
        available = ", ".join(CV2_CAMERA_CONFIGS.keys()) or "<empty>"
        raise ValueError(f"Unknown camera_name '{camera_name}'. Available: {available}")
    return CV2_CAMERA_CONFIGS[camera_name]


def default_intrinsics_output_path(camera_name: str, width: int, height: int) -> str:
    return DEFAULT_INTRINSICS_OUTPUT_TEMPLATE.format(
        camera_name=camera_name,
        width=width,
        height=height,
    )


def append_timestamp_to_yaml_path(path: str) -> str:
    """Append a MMDD_HHMMSS timestamp before the .yaml suffix."""
    root, ext = os.path.splitext(path)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    if ext.lower() == ".yaml":
        return f"{root}_{timestamp}{ext}"
    return f"{path}_{timestamp}"


def create_intrinsics_sample_image_dir(
    camera_name: str,
    width: int,
    height: int,
) -> str:
    """Create a timestamped directory for automatically stored calibration images."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(camera_name)).strip("_") or "camera"
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = os.path.join(
        INTRINSICS_SAMPLE_IMAGE_ROOT,
        f"{safe_name}_{int(width)}x{int(height)}_{timestamp}",
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


class SimpleIntrinsicsCalibrator:
    """
    Minimal, camera-agnostic intrinsics calibrator:
      - calibrate(frames_rgb): input (n, h, w, 3) RGB array; returns K, dist, etc.
      - undistort(image_rgb): undistort a single RGB image using the result
      - save_yaml(path): save intrinsics to YAML (essential fields only)
    """
    def __init__(self):
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.rms: Optional[float] = None
        self.mean_reproj_error: Optional[float] = None
        self.image_size: Optional[Tuple[int, int]] = None  # (w, h)

    @staticmethod
    def _detect_chessboard(gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        cols, rows = CHECKERBOARD
        # Prefer the SB detector when available (OpenCV >= 4.5); fallback otherwise
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

    @staticmethod
    def _make_object_points(n_views: int) -> List[np.ndarray]:
        cols, rows = CHECKERBOARD
        objp = np.zeros((rows * cols, 3), np.float32)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp[:, :2] = grid
        objp *= float(SQUARE_SIZE)
        return [objp.copy() for _ in range(n_views)]

    def _compute_mean_reproj_error(self, K, dist, rvecs, tvecs, objpoints, imgpoints) -> float:
        total_err = 0.0
        total_pts = 0
        for i in range(len(objpoints)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
            total_err += err * err
            total_pts += len(objpoints[i])
        return float(np.sqrt(total_err / max(total_pts, 1)))

    def calibrate(self, frames_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args:
            frames_rgb: (n, h, w, 3) RGB array (uint8, or float in [0,1]/[0,255])
        Returns:
            dict with keys: K, dist, rvecs, tvecs, rms, mean_err, used_indices, image_size, fx, fy, cx, cy
        """
        assert frames_rgb.ndim == 4 and frames_rgb.shape[-1] == 3, \
            "frames_rgb must be (n, h, w, 3) RGB array"
        n, h, w, _ = frames_rgb.shape
        self.image_size = (w, h)

        # Normalize to uint8 for OpenCV
        if frames_rgb.dtype != np.uint8:
            arr = frames_rgb.astype(np.float32)
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            frames_u8 = arr.astype(np.uint8)
        else:
            frames_u8 = frames_rgb

        imgpoints: List[np.ndarray] = []
        used_idx: List[int] = []

        # Detect chessboard corners per frame
        for i in range(n):
            gray = cv2.cvtColor(frames_u8[i], cv2.COLOR_RGB2GRAY)
            ok, corners = self._detect_chessboard(gray)
            if ok:
                imgpoints.append(corners)
                used_idx.append(i)

        if len(imgpoints) < MIN_SAMPLES:
            raise RuntimeError(f"Not enough valid chessboard samples: {len(imgpoints)}; need >= {MIN_SAMPLES}")

        objpoints = self._make_object_points(len(imgpoints))

        # Pinhole + standard distortion calibration
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        if not ret:
            raise RuntimeError("cv2.calibrateCamera failed; check samples/visibility.")

        mean_err = self._compute_mean_reproj_error(K, dist, rvecs, tvecs, objpoints, imgpoints)

        # Store results
        self.K, self.dist = K, dist
        self.rvecs, self.tvecs = rvecs, tvecs
        self.rms = float(ret)
        self.mean_reproj_error = float(mean_err)

        return {
            "K": K, "dist": dist, "rvecs": rvecs, "tvecs": tvecs,
            "rms": float(ret), "mean_err": float(mean_err),
            "used_indices": np.array(used_idx, dtype=int),
            "image_size": (w, h),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        }

    def undistort(self, image_rgb: np.ndarray) -> np.ndarray:
        """Undistort a single RGB image using the current intrinsics."""
        if self.K is None or self.dist is None or self.image_size is None:
            raise RuntimeError("Run calibrate() first.")
        newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, self.image_size, 0)
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        und_bgr = cv2.undistort(bgr, self.K, self.dist, None, newK)
        return cv2.cvtColor(und_bgr, cv2.COLOR_BGR2RGB)

    def save_yaml(self, path: str = "intrinsics.yaml") -> None:
        """Save essential intrinsics to YAML (no extra fields)."""
        if self.K is None or self.dist is None or self.image_size is None:
            raise RuntimeError("Nothing to save; run calibrate() first.")
        data = {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "K": self.K.tolist(),
            "dist": self.dist.reshape(-1).tolist(),
            "fx": float(self.K[0, 0]),
            "fy": float(self.K[1, 1]),
            "cx": float(self.K[0, 2]),
            "cy": float(self.K[1, 2]),
            "rms": float(self.rms) if self.rms is not None else None,
            "mean_reproj_error": float(self.mean_reproj_error) if self.mean_reproj_error is not None else None,
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


class SimpleFisheyeIntrinsicsCalibrator:
    """
    OpenCV fisheye intrinsics calibrator.

    This uses cv2.fisheye.calibrate and saves a 4-parameter fisheye D vector.
    Do not mix this YAML with the standard pinhole distortion model.
    """
    def __init__(self):
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.rms: Optional[float] = None
        self.mean_reproj_error: Optional[float] = None
        self.image_size: Optional[Tuple[int, int]] = None  # (w, h)

    @staticmethod
    def _detect_chessboard(gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        return SimpleIntrinsicsCalibrator._detect_chessboard(gray)

    @staticmethod
    def _make_object_points(n_views: int) -> List[np.ndarray]:
        cols, rows = CHECKERBOARD
        objp = np.zeros((1, rows * cols, 3), np.float64)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp[0, :, :2] = grid
        objp *= float(SQUARE_SIZE)
        return [objp.copy() for _ in range(n_views)]

    @staticmethod
    def _initial_camera_matrix(objpoints, imgpoints, image_size) -> np.ndarray:
        objpoints_pinhole = [obj.reshape(-1, 3).astype(np.float32) for obj in objpoints]
        imgpoints_pinhole = [img.reshape(-1, 1, 2).astype(np.float32) for img in imgpoints]
        ok, K, _dist, _rvecs, _tvecs = cv2.calibrateCamera(
            objpoints_pinhole,
            imgpoints_pinhole,
            image_size,
            None,
            None,
        )
        if not ok:
            raise RuntimeError("Initial pinhole calibration failed; cannot seed fisheye calibration.")
        return K.astype(np.float64)

    def _compute_mean_reproj_error(self, K, D, rvecs, tvecs, objpoints, imgpoints) -> float:
        total_err = 0.0
        total_pts = 0
        for i in range(len(objpoints)):
            proj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
            err = cv2.norm(imgpoints[i].reshape(-1, 2), proj.reshape(-1, 2), cv2.NORM_L2)
            total_err += err * err
            total_pts += objpoints[i].reshape(-1, 3).shape[0]
        return float(np.sqrt(total_err / max(total_pts, 1)))

    def calibrate(self, frames_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args:
            frames_rgb: (n, h, w, 3) RGB array (uint8, or float in [0,1]/[0,255])
        Returns:
            dict with keys: K, D, rvecs, tvecs, rms, mean_err, used_indices, image_size, fx, fy, cx, cy
        """
        assert frames_rgb.ndim == 4 and frames_rgb.shape[-1] == 3, \
            "frames_rgb must be (n, h, w, 3) RGB array"
        n, h, w, _ = frames_rgb.shape
        self.image_size = (w, h)

        if frames_rgb.dtype != np.uint8:
            arr = frames_rgb.astype(np.float32)
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            frames_u8 = arr.astype(np.uint8)
        else:
            frames_u8 = frames_rgb

        imgpoints: List[np.ndarray] = []
        used_idx: List[int] = []

        for i in range(n):
            gray = cv2.cvtColor(frames_u8[i], cv2.COLOR_RGB2GRAY)
            ok, corners = self._detect_chessboard(gray)
            if ok and corners is not None:
                imgpoints.append(corners.astype(np.float64).reshape(1, -1, 2))
                used_idx.append(i)

        if len(imgpoints) < MIN_SAMPLES:
            raise RuntimeError(f"Not enough valid chessboard samples: {len(imgpoints)}; need >= {MIN_SAMPLES}")

        objpoints = self._make_object_points(len(imgpoints))
        K = self._initial_camera_matrix(objpoints, imgpoints, self.image_size)
        D = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6,
        )
        flags = (
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_CHECK_COND
            | cv2.fisheye.CALIB_FIX_SKEW
        )
        print(f"[INFO] Fisheye initial K from pinhole calibration:\n{K}")

        active_indices = list(range(len(objpoints)))
        rejected_indices: List[int] = []
        last_error: Optional[Exception] = None
        while len(active_indices) >= MIN_SAMPLES:
            sub_objpoints = [objpoints[i] for i in active_indices]
            sub_imgpoints = [imgpoints[i] for i in active_indices]
            sub_used_idx = [used_idx[i] for i in active_indices]
            try:
                K_try = self._initial_camera_matrix(sub_objpoints, sub_imgpoints, self.image_size)
                D_try = np.zeros((4, 1), dtype=np.float64)
                rvecs_try = [np.zeros((1, 1, 3), dtype=np.float64) for _ in sub_objpoints]
                tvecs_try = [np.zeros((1, 1, 3), dtype=np.float64) for _ in sub_objpoints]
                rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    sub_objpoints,
                    sub_imgpoints,
                    self.image_size,
                    K_try,
                    D_try,
                    rvecs_try,
                    tvecs_try,
                    flags=flags,
                    criteria=criteria,
                )
                objpoints = sub_objpoints
                imgpoints = sub_imgpoints
                used_idx = sub_used_idx
                break
            except cv2.error as exc:
                last_error = exc
                match = re.search(r"input array (\d+)", str(exc))
                if match is None:
                    raise RuntimeError(
                        "cv2.fisheye.calibrate failed without identifying a bad input view. "
                        "Usually this means the captured checkerboard set is too degenerate "
                        "for fisheye calibration; retake more diverse samples."
                    ) from exc
                bad_local_idx = int(match.group(1))
                if bad_local_idx < 0 or bad_local_idx >= len(active_indices):
                    raise RuntimeError(
                        f"cv2.fisheye.calibrate reported invalid bad view index {bad_local_idx}."
                    ) from exc
                bad_global_idx = active_indices.pop(bad_local_idx)
                rejected_indices.append(used_idx[bad_global_idx])
                print(
                    "[WARN] Rejecting ill-conditioned fisheye sample "
                    f"captured_index={used_idx[bad_global_idx]} "
                    f"(valid_view_index={bad_local_idx}); remaining={len(active_indices)}"
                )
        else:
            raise RuntimeError(
                "cv2.fisheye.calibrate rejected too many views. "
                f"Rejected captured frame indices: {rejected_indices}. "
                f"Detected valid frame indices: {used_idx}"
            ) from last_error

        if rejected_indices:
            print(f"[INFO] Fisheye rejected captured frame indices: {rejected_indices}")
            print(f"[INFO] Fisheye used captured frame indices: {used_idx}")

        mean_err = self._compute_mean_reproj_error(K, D, rvecs, tvecs, objpoints, imgpoints)

        self.K, self.D = K, D
        self.rvecs, self.tvecs = rvecs, tvecs
        self.rms = float(rms)
        self.mean_reproj_error = float(mean_err)

        return {
            "K": K,
            "D": D,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "rms": float(rms),
            "mean_err": float(mean_err),
            "used_indices": np.array(used_idx, dtype=int),
            "image_size": (w, h),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }

    def undistort(self, image_rgb: np.ndarray, balance: float = 1.0) -> np.ndarray:
        """Undistort a single RGB fisheye image using the current intrinsics."""
        if self.K is None or self.D is None or self.image_size is None:
            raise RuntimeError("Run calibrate() first.")
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K,
            self.D,
            self.image_size,
            np.eye(3),
            balance=float(balance),
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K,
            self.D,
            np.eye(3),
            newK,
            self.image_size,
            cv2.CV_16SC2,
        )
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        und_bgr = cv2.remap(bgr, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return cv2.cvtColor(und_bgr, cv2.COLOR_BGR2RGB)

    def save_yaml(self, path: str = "intrinsics_fisheye.yaml") -> None:
        """Save essential fisheye intrinsics to YAML."""
        if self.K is None or self.D is None or self.image_size is None:
            raise RuntimeError("Nothing to save; run calibrate() first.")
        data = {
            "camera_model": "fisheye",
            "distortion_model": "opencv_fisheye",
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "K": self.K.tolist(),
            "D": self.D.reshape(-1).tolist(),
            "dist": self.D.reshape(-1).tolist(),
            "fx": float(self.K[0, 0]),
            "fy": float(self.K[1, 1]),
            "cx": float(self.K[0, 2]),
            "cy": float(self.K[1, 2]),
            "rms": float(self.rms) if self.rms is not None else None,
            "mean_reproj_error": float(self.mean_reproj_error) if self.mean_reproj_error is not None else None,
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


def realsense_intrinsics_calibration_example():
    """
    Example usage with a Realsense camera and interactive frame capture.
    Requires RealsenseCamera and InteractiveCamera.
    """
    from camera_ui import InteractiveCamera
    from cameras import RealsenseCamera
    cam = RealsenseCamera(
        serial=REALSENSE_SERIAL,
        width=REALSENSE_RESOLUTION[0],
        height=REALSENSE_RESOLUTION[1],
        fps=REALSENSE_FPS,
        format=REALSENSE_FORMAT,
        auto_exposure=REALSENSE_AUTO_EXPOSURE,
        exposure_us=REALSENSE_EXPOSURE_US,
        white_balance=REALSENSE_WHITE_BALANCE,
    )
    sample_image_dir = create_intrinsics_sample_image_dir(
        REALSENSE_CAMERA_NAME,
        REALSENSE_RESOLUTION[0],
        REALSENSE_RESOLUTION[1],
    )
    cam_ui = InteractiveCamera(
        camera=cam,
        checkerboard=CHECKERBOARD,
        window_name=f"RealSense Intrinsics - {REALSENSE_RESOLUTION[0]}x{REALSENSE_RESOLUTION[1]}",
        auto_store_valid=AUTO_STORE_VALID_IMAGES,
        auto_store_cooldown_s=AUTO_STORE_COOLDOWN_S,
        auto_store_dir=sample_image_dir,
        auto_store_ext=INTRINSICS_SAMPLE_IMAGE_EXT,
    )
    print(f"[INFO] Supported camera types = {', '.join(SUPPORTED_CAMERA_TYPES)}")
    print(f"[INFO] RealSense configured resolution = {REALSENSE_RESOLUTION[0]}x{REALSENSE_RESOLUTION[1]} @ {REALSENSE_FPS} fps")
    print(f"[INFO] Auto-store valid images: {AUTO_STORE_VALID_IMAGES}, cooldown={AUTO_STORE_COOLDOWN_S}s")
    print(f"[INFO] Sample image dir = {sample_image_dir}")
    rgb_frames = cam_ui.run()
    if rgb_frames is None or len(rgb_frames) == 0:
        raise RuntimeError("No frames captured for calibration.")
    captured_width, captured_height = int(rgb_frames.shape[2]), int(rgb_frames.shape[1])
    print(f"[INFO] RealSense calibration image size = {captured_width}x{captured_height}")

    intr_calib = SimpleIntrinsicsCalibrator()
    calib_results = intr_calib.calibrate(rgb_frames)
    print("Calibration results:", calib_results)
    output_path = default_intrinsics_output_path(
        REALSENSE_CAMERA_NAME,
        captured_width,
        captured_height,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    intr_calib.save_yaml(output_path)
    print(f"Saved intrinsics to {output_path}")

def usbcam_intrinsics_calibration_example():
    """
    Example usage with a generic USB camera and interactive frame capture.
    Prefer cv2cam_intrinsics_calibration() for configured calibration cameras.
    """
    from camera_ui import InteractiveCamera
    from cameras import CV2Camera
    cam = CV2Camera(src=0)
    cam_ui = InteractiveCamera(camera=cam)
    rgb_frames = cam_ui.run()

    intr_calib = SimpleIntrinsicsCalibrator()
    calib_results = intr_calib.calibrate(rgb_frames)
    print("Calibration results:", calib_results)
    intr_calib.save_yaml("outputs/intrinsics.yaml")
    print("Saved intrinsics to outputs/intrinsics.yaml")

def cv2camera_intrinsics_calibration_example():
    """
    Example usage with a CV2camera and interactive frame capture.
    Requires CV2Camera and InteractiveCamera.
    """
    from camera_ui import InteractiveCamera
    from cameras import CV2Camera
    cam = CV2Camera(src=3)
    cam_ui = InteractiveCamera(camera=cam)

def cv2cam_intrinsics_calibration(
    src: int | str = 0,
    camera_name: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    fourcc: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """
    Example usage with a CV2 camera and interactive frame capture.
    Requires CV2Camera and InteractiveCamera.
    """
    from camera_ui import InteractiveCamera
    from cameras import CV2Camera, resolve_cv2_camera_source

    configured_resolution: Optional[Tuple[int, int]] = None
    if camera_name is not None:
        config = get_cv2_camera_config(camera_name)
        src = str(config["port"])
        configured_resolution = tuple(config["resolution"])
        width = int(width if width is not None else configured_resolution[0])
        height = int(height if height is not None else configured_resolution[1])
        fps = int(fps if fps is not None else config["fps"])
        fourcc = str(fourcc if fourcc is not None else config.get("fourcc", "")) or None

    resolved_src = resolve_cv2_camera_source(src)

    cam = CV2Camera(
        src=resolved_src,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
    )
    sample_image_dir = create_intrinsics_sample_image_dir(
        camera_name or "cv2",
        width or 0,
        height or 0,
    )
    cam_ui = InteractiveCamera(
        camera=cam,
        checkerboard=CHECKERBOARD,
        terminal_fps_interval_s=1.0,
        window_name=f"CV2 Intrinsics - {camera_name or resolved_src}",
        auto_store_valid=AUTO_STORE_VALID_IMAGES,
        auto_store_cooldown_s=AUTO_STORE_COOLDOWN_S,
        auto_store_dir=sample_image_dir,
        auto_store_ext=INTRINSICS_SAMPLE_IMAGE_EXT,
    )
    print(f"[INFO] CV2 configured cameras = {CV2_CAMERA_COUNT}: {', '.join(CV2_CAMERA_CONFIGS.keys())}")
    print(f"[INFO] Supported camera types = {', '.join(SUPPORTED_CAMERA_TYPES)}")
    print(f"[INFO] CV2 requested source = {src}")
    print(f"[INFO] CV2 resolved source = {resolved_src}")
    if configured_resolution is not None:
        print(f"[INFO] CV2 configured resolution = {configured_resolution[0]}x{configured_resolution[1]} @ {fps} fps")
    if fourcc is not None:
        print(f"[INFO] CV2 requested FOURCC = {fourcc}")
    print(f"[INFO] Auto-store valid images: {AUTO_STORE_VALID_IMAGES}, cooldown={AUTO_STORE_COOLDOWN_S}s")
    print(f"[INFO] Sample image dir = {sample_image_dir}")
    rgb_frames = cam_ui.run()
    if rgb_frames is None or len(rgb_frames) == 0:
        raise RuntimeError("No frames captured for calibration.")
    captured_width, captured_height = int(rgb_frames.shape[2]), int(rgb_frames.shape[1])
    print(f"[INFO] CV2 calibration image size = {captured_width}x{captured_height}")

    intr_calib = SimpleIntrinsicsCalibrator()
    calib_results = intr_calib.calibrate(rgb_frames)
    print("Calibration results:", calib_results)

    if output_path is None:
        output_path = default_intrinsics_output_path(
            camera_name or "cv2",
            captured_width,
            captured_height,
        )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    intr_calib.save_yaml(output_path)
    print(f"Saved intrinsics to {output_path}")


def cv2fisheye_intrinsics_calibration(
    src: int | str = 0,
    camera_name: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    fourcc: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """
    CV2 fisheye camera intrinsics calibration at the requested capture resolution.
    Uses cv2.fisheye.calibrate instead of cv2.calibrateCamera.
    """
    from camera_ui import InteractiveCamera
    from cameras import CV2Camera, resolve_cv2_camera_source

    configured_resolution: Optional[Tuple[int, int]] = None
    if camera_name is not None:
        config = get_cv2_camera_config(camera_name)
        src = str(config["port"])
        configured_resolution = tuple(config["resolution"])
        width = int(width if width is not None else configured_resolution[0])
        height = int(height if height is not None else configured_resolution[1])
        fps = int(fps if fps is not None else config["fps"])
        fourcc = str(fourcc if fourcc is not None else config.get("fourcc", "")) or None

    resolved_src = resolve_cv2_camera_source(src)

    cam = CV2Camera(
        src=resolved_src,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
    )
    sample_image_dir = create_intrinsics_sample_image_dir(
        f"{camera_name or 'cv2'}_fisheye",
        width or 0,
        height or 0,
    )
    cam_ui = InteractiveCamera(
        camera=cam,
        checkerboard=CHECKERBOARD,
        corner_detector=SimpleFisheyeIntrinsicsCalibrator._detect_chessboard,
        terminal_fps_interval_s=1.0,
        window_name=f"CV2 Fisheye Intrinsics - {camera_name or resolved_src}",
        auto_store_valid=AUTO_STORE_VALID_IMAGES,
        auto_store_cooldown_s=AUTO_STORE_COOLDOWN_S,
        auto_store_dir=sample_image_dir,
        auto_store_ext=INTRINSICS_SAMPLE_IMAGE_EXT,
    )
    print(f"[INFO] CV2 configured cameras = {CV2_CAMERA_COUNT}: {', '.join(CV2_CAMERA_CONFIGS.keys())}")
    print(f"[INFO] Supported camera types = {', '.join(SUPPORTED_CAMERA_TYPES)}")
    print(f"[INFO] CV2 fisheye requested source = {src}")
    print(f"[INFO] CV2 fisheye resolved source = {resolved_src}")
    if configured_resolution is not None:
        print(f"[INFO] CV2 fisheye configured resolution = {configured_resolution[0]}x{configured_resolution[1]} @ {fps} fps")
    if fourcc is not None:
        print(f"[INFO] CV2 fisheye requested FOURCC = {fourcc}")
    print(f"[INFO] Auto-store valid images: {AUTO_STORE_VALID_IMAGES}, cooldown={AUTO_STORE_COOLDOWN_S}s")
    print(f"[INFO] Sample image dir = {sample_image_dir}")

    rgb_frames = cam_ui.run()
    if rgb_frames is None or len(rgb_frames) == 0:
        raise RuntimeError("No frames captured for fisheye calibration.")
    captured_width, captured_height = int(rgb_frames.shape[2]), int(rgb_frames.shape[1])
    print(f"[INFO] CV2 fisheye calibration image size = {captured_width}x{captured_height}")

    intr_calib = SimpleFisheyeIntrinsicsCalibrator()
    calib_results = intr_calib.calibrate(rgb_frames)
    print("Fisheye calibration results:", calib_results)

    if output_path is None:
        output_path = default_intrinsics_output_path(
            f"{camera_name or 'cv2'}_fisheye",
            captured_width,
            captured_height,
        )
    output_path = append_timestamp_to_yaml_path(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    intr_calib.save_yaml(output_path)
    print(f"Saved fisheye intrinsics to {output_path}")


def avcam_intrinsics_calibration():
    """
    Example usage with a AV camera and interactive frame capture.
    Requires AVCameraManager and InteractiveCamera.
    """
    from camera_ui import InteractiveCamera
    from cameras import AVCameraManager
    default_opts = {
        "input_format": "mjpeg",
        "video_size": f"{AV_VIDEO_SIZE[0]}x{AV_VIDEO_SIZE[1]}",
        "framerate": str(AV_FRAMERATE),
    }

    per_cam_opts = {
        # "tip": {"video_size": "640x480", "framerate": "30"},
    }

    cam = AVCameraManager(
        camera_to_port=AV_CAMERA_TO_PORT,
        camera_left_right_order=AV_CAMERA_LEFT_RIGHT_ORDER,
        default_options=default_opts,
        per_camera_options=per_cam_opts,
        stream_index=0,
        read_camera_name="I-tip",
    )
    sample_image_dir = create_intrinsics_sample_image_dir(
        "av",
        AV_VIDEO_SIZE[0],
        AV_VIDEO_SIZE[1],
    )
    cam_ui = InteractiveCamera(
        camera=cam,
        checkerboard=CHECKERBOARD,
        window_name=f"AV Intrinsics - {AV_VIDEO_SIZE[0]}x{AV_VIDEO_SIZE[1]}",
        auto_store_valid=AUTO_STORE_VALID_IMAGES,
        auto_store_cooldown_s=AUTO_STORE_COOLDOWN_S,
        auto_store_dir=sample_image_dir,
        auto_store_ext=INTRINSICS_SAMPLE_IMAGE_EXT,
    )
    print(f"[INFO] Supported camera types = {', '.join(SUPPORTED_CAMERA_TYPES)}")
    print(f"[INFO] AV configured cameras = {len(AV_CAMERA_TO_PORT)}: {', '.join(AV_CAMERA_TO_PORT.keys())}")
    print(f"[INFO] AV configured resolution = {AV_VIDEO_SIZE[0]}x{AV_VIDEO_SIZE[1]} @ {AV_FRAMERATE} fps")
    print(f"[INFO] Auto-store valid images: {AUTO_STORE_VALID_IMAGES}, cooldown={AUTO_STORE_COOLDOWN_S}s")
    print(f"[INFO] Sample image dir = {sample_image_dir}")
    rgb_frames = cam_ui.run()
    if rgb_frames is None or len(rgb_frames) == 0:
        raise RuntimeError("No frames captured for calibration.")
    captured_width, captured_height = int(rgb_frames.shape[2]), int(rgb_frames.shape[1])
    print(f"[INFO] AV calibration image size = {captured_width}x{captured_height}")

    intr_calib = SimpleIntrinsicsCalibrator()
    calib_results = intr_calib.calibrate(rgb_frames)
    print("Calibration results:", calib_results)
    output_path = default_intrinsics_output_path(
        "av",
        captured_width,
        captured_height,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    intr_calib.save_yaml(output_path)
    print(f"Saved intrinsics to {output_path}")


def oak1w_intrinsics_calibration(camera_name: Optional[str] = None):
    """
    Example usage with an OAK-1-W camera and interactive frame capture.
    Uses the OAK camera macros defined at the top of this script.
    """
    from camera_ui import InteractiveCamera
    from cameras import OAK1WCameraManager

    if not OAK_CAMERA_TO_DEVICE:
        raise RuntimeError("OAK_CAMERA_TO_DEVICE is empty in intr_calib.py.")

    if camera_name is None:
        camera_name = next(iter(OAK_CAMERA_TO_DEVICE))

    if camera_name not in OAK_CAMERA_TO_DEVICE:
        available = ", ".join(OAK_CAMERA_TO_DEVICE.keys())
        raise ValueError(f"Unknown OAK camera '{camera_name}'. Available: {available}")

    class OAK1WInteractiveAdapter:
        """Adapt OAK1WCameraManager to InteractiveCamera's start/stop/read interface."""

        def __init__(self, manager: OAK1WCameraManager, read_camera_name: str):
            self.manager = manager
            self.read_camera_name = read_camera_name

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.stop()

        def start(self) -> None:
            opened = self.manager.open_all_cameras()
            if opened <= 0:
                raise RuntimeError("Failed to open any OAK camera.")
            if not self.manager.wait_for_first_frames([self.read_camera_name], timeout_s=5.0):
                raise RuntimeError(f"Timed out waiting for frames from '{self.read_camera_name}'.")

        def stop(self) -> None:
            self.manager.release_all()

        def read(self) -> np.ndarray:
            frames, _origin_frames = self.manager.get_frames(
                camera_names=[self.read_camera_name],
                img_size=OAK_DETECT_IMG_SIZE,
            )
            frame_bgr = frames.get(self.read_camera_name)
            if frame_bgr is None:
                raise RuntimeError(f"No frame available from '{self.read_camera_name}'.")
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    manager = OAK1WCameraManager(
        camera_to_device={camera_name: OAK_CAMERA_TO_DEVICE[camera_name]},
        isp_scale=OAK_ISP_SCALE,
        fps=OAK_FPS,
        rotate_180_names=OAK_ROTATE_180_NAMES,
        queue_size=OAK_QUEUE_SIZE,
        queue_blocking=OAK_QUEUE_BLOCKING,
    )
    sample_image_dir = create_intrinsics_sample_image_dir(
        camera_name,
        OAK_DETECT_IMG_SIZE[0],
        OAK_DETECT_IMG_SIZE[1],
    )
    cam_ui = InteractiveCamera(
        camera=OAK1WInteractiveAdapter(manager=manager, read_camera_name=camera_name),
        checkerboard=(8, 6),
        window_name=f"OAK1W Intrinsics - {camera_name}",
        auto_store_valid=AUTO_STORE_VALID_IMAGES,
        auto_store_cooldown_s=AUTO_STORE_COOLDOWN_S,
        auto_store_dir=sample_image_dir,
        auto_store_ext=INTRINSICS_SAMPLE_IMAGE_EXT,
    )
    print(f"[INFO] Supported camera types = {', '.join(SUPPORTED_CAMERA_TYPES)}")
    print(f"[INFO] OAK configured cameras = {len(OAK_CAMERA_TO_DEVICE)}: {', '.join(OAK_CAMERA_TO_DEVICE.keys())}")
    print(f"[INFO] OAK configured image size = {OAK_DETECT_IMG_SIZE}")
    print(f"[INFO] Auto-store valid images: {AUTO_STORE_VALID_IMAGES}, cooldown={AUTO_STORE_COOLDOWN_S}s")
    print(f"[INFO] Sample image dir = {sample_image_dir}")
    rgb_frames = cam_ui.run()
    if rgb_frames is None or len(rgb_frames) == 0:
        raise RuntimeError("No frames captured for calibration.")
    captured_width, captured_height = int(rgb_frames.shape[2]), int(rgb_frames.shape[1])
    print(f"[INFO] OAK calibration image size = {captured_width}x{captured_height}")

    intr_calib = SimpleIntrinsicsCalibrator()
    calib_results = intr_calib.calibrate(rgb_frames)
    print("Calibration results:", calib_results)

    os.makedirs("outputs", exist_ok=True)
    output_path = default_intrinsics_output_path(
        camera_name,
        captured_width,
        captured_height,
    )
    intr_calib.save_yaml(output_path)
    print(f"Saved intrinsics to {output_path}")


def run_default_intrinsics_calibration() -> None:
    print(f"[INFO] Default camera type = {DEFAULT_CAMERA_TYPE}")
    if DEFAULT_CAMERA_TYPE == "realsense":
        realsense_intrinsics_calibration_example()
    elif DEFAULT_CAMERA_TYPE == "cv2":
        cv2cam_intrinsics_calibration(camera_name=DEFAULT_CV2_CAMERA_NAME)
    elif DEFAULT_CAMERA_TYPE == "cv2_fisheye":
        cv2fisheye_intrinsics_calibration(camera_name=DEFAULT_CV2_CAMERA_NAME)
    elif DEFAULT_CAMERA_TYPE == "av":
        avcam_intrinsics_calibration()
    elif DEFAULT_CAMERA_TYPE == "oak1w":
        oak1w_intrinsics_calibration()
    else:
        supported = ", ".join(SUPPORTED_CAMERA_TYPES)
        raise ValueError(f"Unknown DEFAULT_CAMERA_TYPE '{DEFAULT_CAMERA_TYPE}'. Supported: {supported}")


if __name__ == "__main__":
    run_default_intrinsics_calibration()
    # avcam_intrinsics_calibration()
