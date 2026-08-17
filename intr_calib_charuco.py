import argparse
import csv
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from robot_cam_calib.config import CalibrationCatalog


# ---------------------------- User macros ---------------------------- #
# supported: "charuco", "charuco_scale0p25", "apriltag_grid"
CALIBRATION_TARGET = "charuco"
CHARUCO_TARGET_MODES = frozenset({"charuco", "charuco_scale0p25"})

# Defaults from the provided ChArUco board note. Used when
# CALIBRATION_TARGET == "charuco".
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH = 0.04
CHARUCO_MARKER_LENGTH = 0.03
CHARUCO_DICTIONARY = "DICT_5X5_50"
CHARUCO_LEGACY_PATTERN = False

# Quarter-scale A4 board generated from the default ChArUco target. This is a
# separate calibration mode so its metric dimensions cannot be confused with
# the original 40 mm / 30 mm board.
CHARUCO_SCALE0P25_CONFIG = (
    Path(__file__).resolve().parent
    / "outputs/charuco_a4_scale0p25/"
    "charuco_7x5_scale0p25_square10mm_marker7p5mm_DICT_5X5_50_A4_landscape.yaml"
)

# Default AprilGrid retained for direct script compatibility. Configured tasks
# should provide ``--target-yaml`` instead.
APRILTAG_GRID_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/apriltag_grid_36h10_a4_near_8mm/apriltag_36h10_grid_20x29_ids_579_to_0_tag8mm_gap2mm_margin3mm_a4_near.yaml"
)

MIN_SAMPLES = 20
MIN_CORNERS_PER_SAMPLE = 12
CHARUCO_MIN_GRID_ROWS_PER_SAMPLE = 2
CHARUCO_MIN_GRID_COLS_PER_SAMPLE = 4
CHARUCO_MIN_BOARD_BBOX_FRACTION = 0.35

# Previous CV2 camera defaults. Kept here for reference.
DEFAULT_CV2_CAMERA_NAME: Optional[str] = None
DEFAULT_CV2_SOURCE: str = "0"
DEFAULT_CV2_PORT: Optional[str] = "5-4:1.0"
DEFAULT_CV2_WIDTH: Optional[int] = 2592
DEFAULT_CV2_HEIGHT: Optional[int] = 1944
DEFAULT_CV2_FPS: Optional[int] = 50
DEFAULT_CV2_FOURCC: Optional[str] = "MJPG"
DEFAULT_OUTPUT_NAME: Optional[str] = None
DEFAULT_DISPLAY_SCALE: Optional[float] = 0.4
DEFAULT_WINDOW_NAME: str = "ChArUco intrinsics"

# Intel RealSense D435 RGB defaults: 1920x1080 @ 30 FPS, YUYV via V4L2.
# DEFAULT_CV2_CAMERA_NAME: Optional[str] = None
# DEFAULT_CV2_SOURCE: str = "/dev/video4"
# DEFAULT_CV2_PORT: Optional[str] = None
# DEFAULT_CV2_WIDTH: Optional[int] = 1920
# DEFAULT_CV2_HEIGHT: Optional[int] = 1080
# DEFAULT_CV2_FPS: Optional[int] = 30
# DEFAULT_CV2_FOURCC: Optional[str] = "YUYV"
# DEFAULT_OUTPUT_NAME: Optional[str] = "d435_color"
# DEFAULT_DISPLAY_SCALE: Optional[float] = 0.4
# DEFAULT_WINDOW_NAME: str = "D435 RGB ChArUco intrinsics"

CAMERA_MODEL: str = "pinhole"  # supported: "pinhole", "fisheye"

AUTO_SAVE_VALID_IMAGES: bool = True
AUTO_SAVE_COOLDOWN_S: float = 0.8
SAMPLE_IMAGE_ROOT: Path = (
    Path("outputs/intrinsics_apriltag_grid_samples")
    if CALIBRATION_TARGET == "apriltag_grid"
    else (
        Path("outputs/intrinsics_charuco_scale0p25_samples")
        if CALIBRATION_TARGET == "charuco_scale0p25"
        else Path("outputs/intrinsics_charuco_samples")
    )
)

OPEN_TEST_NUM_FRAMES: int = 10
OPEN_TEST_SLEEP_S: float = 0.03

# Final calibration pipeline. Saved images are re-evaluated after capture so
# motion-blurred frames and redundant poses do not bias the intrinsics.
FINAL_FILTER_MOTION_BLUR: bool = True
FINAL_USE_GPU_FOR_SHARPNESS: bool = True
FINAL_BLUR_REJECT_FRACTION: float = 0.15
FINAL_MAX_CALIBRATION_VIEWS: int = 72
FINAL_REJECT_REPROJ_OUTLIERS: bool = True
FINAL_MAX_VIEW_ERROR_PX: float = 0.8
FINAL_MAX_REJECTION_ROUNDS: int = 5
FINAL_PROGRESS_BAR_WIDTH: int = 36
FINAL_PIPELINE_STAGE_COUNT: int = 6
FINAL_CROSS_VALIDATE: bool = True
FINAL_CONTACT_SHEET_MAX_IMAGES: int = 24


@dataclass
class AprilTagGridBoard:
    path: Path
    tag_family: str
    id_grid: list[list[int]]
    tag_object_points: dict[int, np.ndarray]
    rows: int
    cols: int
    tag_size_m: float
    tag_gap_m: float
    board_width_m: float
    board_height_m: float
    min_corners_per_sample: int = MIN_CORNERS_PER_SAMPLE


def is_charuco_target(target: Optional[str] = None) -> bool:
    return (CALIBRATION_TARGET if target is None else target) in CHARUCO_TARGET_MODES


def charuco_mode_defaults() -> dict:
    defaults = {
        "squares_x": CHARUCO_SQUARES_X,
        "squares_y": CHARUCO_SQUARES_Y,
        "square_length": CHARUCO_SQUARE_LENGTH,
        "marker_length": CHARUCO_MARKER_LENGTH,
        "dictionary": CHARUCO_DICTIONARY,
        "legacy_pattern": CHARUCO_LEGACY_PATTERN,
    }
    if CALIBRATION_TARGET != "charuco_scale0p25":
        return defaults

    config_path = CHARUCO_SCALE0P25_CONFIG.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"ChArUco scale-0.25 mode config does not exist: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or data.get("target_type") != "charuco":
        raise ValueError(f"Expected target_type=charuco in {config_path}")
    config = data.get("charuco")
    if not isinstance(config, dict):
        raise ValueError(f"Missing charuco mapping in {config_path}")

    required = (
        "squares_x",
        "squares_y",
        "square_length",
        "marker_length",
        "dictionary",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing ChArUco keys in {config_path}: {missing}")
    return {
        "squares_x": int(config["squares_x"]),
        "squares_y": int(config["squares_y"]),
        "square_length": float(config["square_length"]),
        "marker_length": float(config["marker_length"]),
        "dictionary": str(config["dictionary"]),
        "legacy_pattern": bool(config.get("legacy_pattern", False)),
    }


def append_timestamp_to_yaml_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    if ext.lower() == ".yaml":
        return f"{root}_{timestamp}{ext}"
    return f"{path}_{timestamp}"


def create_sample_image_dir() -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    path = SAMPLE_IMAGE_ROOT / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_sample_image(
    image_dir: Path,
    sample_index: int,
    frame_index: int,
    frame_bgr: np.ndarray,
    capture_mode: str,
) -> str:
    path = image_dir / f"sample_{sample_index:04d}_frame_{frame_index:06d}_{capture_mode}.png"
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Failed to save sample image: {path}")
    return str(path)


def store_sample(
    samples: list[dict],
    image_dir: Path,
    frame_index: int,
    frame_bgr: np.ndarray,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    detected_corners: int,
    detected_markers: int,
    capture_mode: str,
) -> dict:
    sample_index = len(samples)
    image_path = save_sample_image(
        image_dir,
        sample_index,
        frame_index,
        frame_bgr,
        capture_mode,
    )
    sample = {
        "index": int(frame_index),
        "sample_index": int(sample_index),
        "charuco_corners": charuco_corners.copy(),
        "charuco_ids": charuco_ids.copy(),
        "corner_count": int(detected_corners),
        "marker_count": int(detected_markers),
        "image_path": image_path,
        "capture_mode": str(capture_mode),
        "timestamp": float(time.time()),
    }
    samples.append(sample)
    return sample


def store_apriltag_grid_sample(
    samples: list[dict],
    image_dir: Path,
    frame_index: int,
    frame_bgr: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    used_tag_ids: list[int],
    detected_markers: int,
    capture_mode: str,
) -> dict:
    sample_index = len(samples)
    image_path = save_sample_image(
        image_dir,
        sample_index,
        frame_index,
        frame_bgr,
        capture_mode,
    )
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
    sample = {
        "index": int(frame_index),
        "sample_index": int(sample_index),
        "object_points": object_points.copy(),
        "image_points": image_points.copy(),
        "tag_ids": [int(v) for v in used_tag_ids],
        "corner_count": int(object_points.shape[0]),
        "marker_count": int(detected_markers),
        "image_path": image_path,
        "capture_mode": str(capture_mode),
        "timestamp": float(time.time()),
    }
    samples.append(sample)
    return sample


def parse_camera_source(src: str) -> int | str:
    if src.isdigit():
        return int(src)
    return src


def _video_node_sort_key(device_node: str) -> int:
    try:
        return int(str(device_node).replace("/dev/video", ""))
    except Exception:
        return 10**9


def find_camera_nodes_by_usb_port(usb_port: str) -> list[str]:
    """Return all /dev/video* candidates for one USB port."""
    try:
        import pyudev
    except Exception as exc:
        raise RuntimeError("pyudev is required when resolving cameras by USB port.") from exc

    context = pyudev.Context()
    matched_nodes: list[str] = []

    for device in context.list_devices(subsystem="video4linux"):
        parent = device.parent
        if parent is None or "usb" not in str(parent.subsystem):
            continue

        device_usb_port = parent.get("DEVPATH", "").split("/")[-1]
        if usb_port in device_usb_port and device.device_node is not None:
            matched_nodes.append(str(device.device_node))

    return sorted(set(matched_nodes), key=_video_node_sort_key)


def resolve_camera_candidates(src: int | str) -> list[int | str]:
    if isinstance(src, int):
        return [src]
    if src.startswith("/dev/video"):
        return [src]

    candidates = find_camera_nodes_by_usb_port(src)
    if candidates:
        return candidates

    return [src]


def configure_capture(
    cap: cv2.VideoCapture,
    width: Optional[int],
    height: Optional[int],
    fps: Optional[int],
    fourcc: Optional[str],
) -> None:
    if fourcc:
        if len(fourcc) != 4:
            raise ValueError(f"FOURCC must be 4 characters, got: {fourcc}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def test_capture_read(cap: cv2.VideoCapture) -> tuple[bool, Optional[np.ndarray]]:
    for _ in range(OPEN_TEST_NUM_FRAMES):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(OPEN_TEST_SLEEP_S)

    return False, None


def start_capture(
    src: int | str,
    width: Optional[int],
    height: Optional[int],
    fps: Optional[int],
    fourcc: Optional[str],
) -> tuple[cv2.VideoCapture, int | str]:
    candidates = resolve_camera_candidates(src)
    print(f"[INFO] CV2 source candidates for {src}: {candidates}")

    last_error = "No usable camera node found."
    for candidate in candidates:
        print(f"[INFO] Trying CV2 source {candidate} ...")
        cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
        if not cap.isOpened():
            last_error = f"cap.isOpened() is False for {candidate}"
            cap.release()
            print(f"[WARN] {last_error}")
            continue

        configure_capture(cap, width, height, fps, fourcc)
        ok, frame = test_capture_read(cap)
        if not ok or frame is None:
            last_error = f"opened but failed to read valid frame from {candidate}"
            cap.release()
            print(f"[WARN] {last_error}")
            continue

        actual_height, actual_width = frame.shape[:2]
        print(
            f"[INFO] Opened {candidate}: actual_frame={actual_width}x{actual_height}, "
            f"requested={width}x{height}, fps={fps}, fourcc={fourcc}"
        )
        return cap, candidate

    raise RuntimeError(
        f"Failed to open camera source: {src}. "
        f"Candidates tried: {candidates}. Last error: {last_error}"
    )


def get_cv2_config(camera_name: Optional[str]) -> dict:
    if camera_name is None:
        return {}

    catalog = CalibrationCatalog.load(
        Path(__file__).resolve().parent / "configs"
    )
    camera = catalog.cameras.get(camera_name)
    if not isinstance(camera, dict):
        available = ", ".join(catalog.cameras) or "<empty>"
        raise ValueError(
            f"Unknown camera_name '{camera_name}'. Available: {available}"
        )
    if camera.get("backend") != "opencv":
        raise ValueError(
            f"Camera '{camera_name}' uses backend={camera.get('backend')!r}; "
            "this entry point requires backend='opencv'."
        )
    connections = camera["connections"]
    profiles = camera["profiles"]
    profile = next(iter(profiles.values()))
    return {
        "port": next(iter(connections.values())),
        "resolution": tuple(profile["resolution"]),
        "fps": profile["fps"],
        "fourcc": profile.get("fourcc"),
    }


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dictionary_name: str,
    legacy_pattern: bool,
):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (int(squares_x), int(squares_y)),
            float(square_length),
            float(marker_length),
            dictionary,
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            int(squares_x),
            int(squares_y),
            float(square_length),
            float(marker_length),
            dictionary,
        )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(legacy_pattern))
    return board, dictionary


class CharucoDetectorAdapter:
    def __init__(self, board, dictionary):
        self.board = board
        self.dictionary = dictionary

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:
            self.detector_params = cv2.aruco.DetectorParameters_create()

        self.charuco_detector = None
        if hasattr(cv2.aruco, "CharucoDetector"):
            charuco_params = cv2.aruco.CharucoParameters()
            self.charuco_detector = cv2.aruco.CharucoDetector(
                board,
                charuco_params,
                self.detector_params,
            )

    def detect(self, gray: np.ndarray):
        if self.charuco_detector is not None:
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                self.charuco_detector.detectBoard(gray)
            )
            return charuco_corners, charuco_ids, marker_corners, marker_ids

        marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.detector_params,
        )
        if marker_ids is None or len(marker_ids) == 0:
            return None, None, marker_corners, marker_ids

        _count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            self.board,
        )
        return charuco_corners, charuco_ids, marker_corners, marker_ids


def load_apriltag_grid_board(path: Path) -> AprilTagGridBoard:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data.get("target_type") != "apriltag_grid":
        raise ValueError(f"Expected target_type=apriltag_grid in {resolved}")

    tag_object_points = {
        int(tag_id): np.asarray(points, dtype=np.float32).reshape(4, 3)
        for tag_id, points in data["tag_object_points"].items()
    }
    id_grid = [[int(v) for v in row] for row in data["id_grid"]]
    return AprilTagGridBoard(
        path=resolved,
        tag_family=str(data["tag_family"]),
        id_grid=id_grid,
        tag_object_points=tag_object_points,
        rows=int(data["rows"]),
        cols=int(data["cols"]),
        tag_size_m=float(data["tag_size_m"]),
        tag_gap_m=float(data["tag_gap_m"]),
        board_width_m=float(data["board_width_m"]),
        board_height_m=float(data["board_height_m"]),
        min_corners_per_sample=int(data.get("min_corners_per_sample", MIN_CORNERS_PER_SAMPLE)),
    )


class AprilTagGridDetectorAdapter:
    def __init__(self, board: AprilTagGridBoard):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco is missing. Install opencv-contrib-python.")
        if not hasattr(cv2.aruco, board.tag_family):
            raise ValueError(f"OpenCV does not provide AprilTag dictionary {board.tag_family}")

        self.board = board
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.tag_family))
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.params = cv2.aruco.DetectorParameters()
        else:
            self.params = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, gray: np.ndarray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)


def detect_apriltag_grid_points(
    gray: np.ndarray,
    detector: AprilTagGridDetectorAdapter,
    board: AprilTagGridBoard,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[int]]:
    marker_corners, marker_ids, _rejected = detector.detect(gray)
    object_points = []
    image_points = []
    used_ids: list[int] = []

    if marker_corners is not None and marker_ids is not None:
        for corners, marker_id_raw in zip(marker_corners, marker_ids.reshape(-1)):
            marker_id = int(marker_id_raw)
            if marker_id not in board.tag_object_points:
                continue
            object_points.append(board.tag_object_points[marker_id].reshape(4, 3))
            image_points.append(np.asarray(corners, dtype=np.float32).reshape(4, 2))
            used_ids.append(marker_id)

    if not object_points:
        return None, None, marker_corners, marker_ids, []

    return (
        np.vstack(object_points).astype(np.float32).reshape(-1, 3),
        np.vstack(image_points).astype(np.float32).reshape(-1, 1, 2),
        marker_corners,
        marker_ids,
        used_ids,
    )


def get_charuco_board_corners(board) -> np.ndarray:
    """Return ChArUco chessboard corners across OpenCV API versions."""
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        corners = board.chessboardCorners
    else:
        raise AttributeError(
            "Unsupported OpenCV ChArUco board API: expected "
            "getChessboardCorners() or chessboardCorners."
        )
    return np.asarray(corners, dtype=np.float32).reshape(-1, 3)


def charuco_to_calibration_points(board, charuco_corners, charuco_ids):
    if charuco_corners is None or charuco_ids is None:
        return None, None

    ids = charuco_ids.reshape(-1).astype(int)
    corners = charuco_corners.reshape(-1, 2).astype(np.float32)
    board_corners = get_charuco_board_corners(board)

    valid = (ids >= 0) & (ids < len(board_corners))
    if not np.all(valid):
        ids = ids[valid]
        corners = corners[valid]

    objpoints = board_corners[ids].reshape(-1, 3)
    imgpoints = corners.reshape(-1, 1, 2)
    return objpoints, imgpoints


def charuco_detection_quality(charuco_ids, min_corners: int) -> tuple[bool, str]:
    if charuco_ids is None:
        return False, f"corners 0 < {min_corners}"

    ids = np.asarray(charuco_ids, dtype=int).reshape(-1)
    corner_count = int(ids.size)
    if corner_count < min_corners:
        return False, f"corners {corner_count} < {min_corners}"

    inner_cols = int(CHARUCO_SQUARES_X) - 1
    inner_rows = int(CHARUCO_SQUARES_Y) - 1
    if inner_cols <= 0 or inner_rows <= 0:
        return False, "invalid ChArUco board dimensions"

    valid = (ids >= 0) & (ids < inner_cols * inner_rows)
    ids = ids[valid]
    if int(ids.size) < min_corners:
        return False, f"valid corners {int(ids.size)} < {min_corners}"

    rows = ids // inner_cols
    cols = ids % inner_cols
    row_count = int(np.unique(rows).size)
    col_count = int(np.unique(cols).size)
    if row_count < CHARUCO_MIN_GRID_ROWS_PER_SAMPLE:
        return False, f"grid rows {row_count} < {CHARUCO_MIN_GRID_ROWS_PER_SAMPLE}"
    if col_count < CHARUCO_MIN_GRID_COLS_PER_SAMPLE:
        return False, f"grid cols {col_count} < {CHARUCO_MIN_GRID_COLS_PER_SAMPLE}"

    bbox_cols = int(np.max(cols) - np.min(cols) + 1)
    bbox_rows = int(np.max(rows) - np.min(rows) + 1)
    bbox_fraction = float((bbox_cols * bbox_rows) / max(inner_cols * inner_rows, 1))
    if bbox_fraction < CHARUCO_MIN_BOARD_BBOX_FRACTION:
        return False, f"board bbox {bbox_fraction:.2f} < {CHARUCO_MIN_BOARD_BBOX_FRACTION:.2f}"

    return True, (
        f"corners={corner_count} rows={row_count} cols={col_count} "
        f"board_bbox={bbox_fraction:.2f}"
    )


def final_progress(
    stage: int,
    label: str,
    current: int,
    total: int,
    detail: str = "",
) -> None:
    """Render one terminal progress bar for the final calibration pipeline."""
    safe_total = max(int(total), 1)
    safe_current = min(max(int(current), 0), safe_total)
    fraction = safe_current / safe_total
    filled = int(round(FINAL_PROGRESS_BAR_WIDTH * fraction))
    bar = "#" * filled + "-" * (FINAL_PROGRESS_BAR_WIDTH - filled)
    suffix = f"  {detail}" if detail else ""
    line = (
        f"\r[FINAL {stage}/{FINAL_PIPELINE_STAGE_COUNT}] {label:<23} "
        f"|{bar}| {safe_current:>4}/{safe_total:<4}{suffix}"
    )
    print(line, end="\n" if safe_current >= safe_total else "", flush=True)


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, float(np.std(finite)) * 0.1, 1e-6)
    return median, scale


def configure_final_sharpness_backend() -> dict:
    """Select OpenCL/UMat GPU filtering when available, otherwise use CPU."""
    backend = {
        "requested_gpu": bool(FINAL_USE_GPU_FOR_SHARPNESS),
        "gradient_backend": "cpu",
        "opencl_device": {},
        "calibration_backend": "cpu_opencv",
    }
    if not FINAL_USE_GPU_FOR_SHARPNESS:
        cv2.ocl.setUseOpenCL(False)
        return backend

    try:
        if not cv2.ocl.haveOpenCL():
            print("[WARN] OpenCL is unavailable; sharpness analysis will use CPU.")
            return backend
        device = cv2.ocl.Device_getDefault()
        if not device.available():
            print("[WARN] No available OpenCL device; sharpness analysis will use CPU.")
            return backend
        cv2.ocl.setUseOpenCL(True)
        # Trigger OpenCL context creation before the progress loop.
        warmup = cv2.UMat(np.zeros((32, 32), dtype=np.uint8))
        cv2.Sobel(warmup, cv2.CV_32F, 1, 0, ksize=3).get()
        backend["gradient_backend"] = "opencl_umat"
        backend["opencl_device"] = {
            "name": str(device.name()),
            "vendor": str(device.vendorName()),
            "version": str(device.version()),
        }
    except Exception as exc:
        cv2.ocl.setUseOpenCL(False)
        print(
            f"[WARN] GPU sharpness initialization failed ({exc}); falling back to CPU."
        )
    return backend


def compute_sharpness_metrics(
    gray: np.ndarray,
    mask: Optional[np.ndarray],
    use_gpu: bool,
) -> tuple[float, float]:
    """Return Laplacian variance and mean squared Sobel gradient."""
    if use_gpu:
        gray_umat = cv2.UMat(gray)
        laplacian = cv2.Laplacian(gray_umat, cv2.CV_32F)
        grad_x = cv2.Sobel(gray_umat, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_umat, cv2.CV_32F, 0, 1, ksize=3)
        gradient_sq = cv2.add(
            cv2.multiply(grad_x, grad_x),
            cv2.multiply(grad_y, grad_y),
        )
        # OpenCV 4.5.x can fail on masked UMat statistical reductions. Keep
        # the expensive filters on the GPU and reduce their outputs on CPU.
        laplacian_array = laplacian.get()
        gradient_sq_array = gradient_sq.get()
    else:
        laplacian_array = cv2.Laplacian(gray, cv2.CV_32F)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_sq_array = grad_x * grad_x + grad_y * grad_y

    if mask is None:
        return (
            float(np.var(laplacian_array)),
            float(np.mean(gradient_sq_array)),
        )
    valid = np.asarray(mask) > 0
    if int(np.count_nonzero(valid)) < 100:
        return math.nan, math.nan
    return (
        float(np.var(laplacian_array[valid])),
        float(np.mean(gradient_sq_array[valid])),
    )


def estimate_charuco_pixels_per_square(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    squares_x: int,
) -> float:
    points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    by_id = {int(marker_id): point for marker_id, point in zip(ids, points)}
    inner_cols = int(squares_x) - 1
    distances: list[float] = []
    for marker_id, point in by_id.items():
        right_id = marker_id + 1
        if marker_id // inner_cols == right_id // inner_cols and right_id in by_id:
            distances.append(float(np.linalg.norm(by_id[right_id] - point)))
        down_id = marker_id + inner_cols
        if down_id in by_id:
            distances.append(float(np.linalg.norm(by_id[down_id] - point)))
    if distances:
        return float(np.median(distances))
    hull_area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
    return float(math.sqrt(max(hull_area, 1.0) / max(len(points), 1)))


def measure_charuco_sample_sharpness(
    sample: dict,
    squares_x: int,
    squares_y: int,
    use_gpu: bool,
) -> tuple[float, float, float]:
    image_path = Path(str(sample.get("image_path", ""))).expanduser()
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read saved sample image: {image_path}")

    corners = np.asarray(sample["charuco_corners"], dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(sample["charuco_ids"], dtype=np.int32).reshape(-1)
    canonical_pixels_per_square = 100.0
    canonical_points = np.column_stack(
        (
            (ids % (int(squares_x) - 1) + 1) * canonical_pixels_per_square,
            (ids // (int(squares_x) - 1) + 1) * canonical_pixels_per_square,
        )
    ).astype(np.float32)
    homography, _mask = cv2.findHomography(corners, canonical_points, method=0)
    if homography is None:
        raise RuntimeError("Could not estimate board homography for sharpness.")
    rectified = cv2.warpPerspective(
        gray,
        homography,
        (
            int(round(int(squares_x) * canonical_pixels_per_square)),
            int(round(int(squares_y) * canonical_pixels_per_square)),
        ),
        flags=cv2.INTER_LINEAR,
    )
    rectified_mask = np.zeros_like(rectified, dtype=np.uint8)
    cv2.fillConvexPoly(
        rectified_mask,
        cv2.convexHull(np.round(canonical_points).astype(np.int32)),
        255,
    )
    rectified_mask = cv2.erode(
        rectified_mask,
        np.ones((7, 7), dtype=np.uint8),
    )
    laplacian_var, tenengrad_mean = compute_sharpness_metrics(
        rectified,
        rectified_mask,
        use_gpu,
    )
    pixels_per_square = estimate_charuco_pixels_per_square(
        corners,
        ids,
        squares_x,
    )
    return laplacian_var, tenengrad_mean, pixels_per_square


def filter_motion_blur_samples(
    samples: list[dict],
    squares_x: int,
    squares_y: int,
    min_corners: int,
    backend: dict,
) -> tuple[list[dict], dict]:
    """Reject the lowest board-region sharpness tail at comparable scales."""
    valid_samples: list[dict] = []
    for sample in samples:
        quality_ok, reason = charuco_detection_quality(
            sample.get("charuco_ids"),
            min_corners,
        )
        sample["final_quality_reason"] = reason
        if quality_ok:
            valid_samples.append(sample)
        else:
            sample["final_filter_status"] = "rejected_detection"

    use_gpu = backend.get("gradient_backend") == "opencl_umat"
    scored_samples: list[dict] = []
    total = len(valid_samples)
    if total == 0:
        final_progress(2, "Sharpness analysis", 1, 1, "no valid samples")
        return [], {
            "valid_before_blur_filter": 0,
            "rejected_blur_count": 0,
        }

    for position, sample in enumerate(valid_samples, start=1):
        try:
            laplacian_var, tenengrad_mean, pixels_per_square = (
                measure_charuco_sample_sharpness(
                    sample,
                    squares_x,
                    squares_y,
                    use_gpu,
                )
            )
            sample["rectified_laplacian_var"] = float(laplacian_var)
            sample["rectified_tenengrad_mean"] = float(tenengrad_mean)
            sample["pixels_per_square"] = float(pixels_per_square)
            if np.isfinite(laplacian_var) and np.isfinite(tenengrad_mean):
                scored_samples.append(sample)
        except Exception as exc:
            sample["sharpness_error"] = str(exc)
            print(f"\n[WARN] Sharpness check skipped for {sample.get('image_path')}: {exc}")
        final_progress(
            2,
            "Sharpness analysis",
            position,
            total,
            f"GPU={use_gpu}",
        )

    if not FINAL_FILTER_MOTION_BLUR or len(scored_samples) < 4:
        for sample in valid_samples:
            sample["final_filter_status"] = "sharpness_kept"
        return valid_samples, {
            "valid_before_blur_filter": len(valid_samples),
            "scored_for_blur": len(scored_samples),
            "rejected_blur_count": 0,
        }

    pixels_per_square = np.asarray(
        [sample["pixels_per_square"] for sample in scored_samples],
        dtype=np.float64,
    )
    if len(scored_samples) >= 48:
        bin_count = 4
    elif len(scored_samples) >= 24:
        bin_count = 3
    elif len(scored_samples) >= 12:
        bin_count = 2
    else:
        bin_count = 1
    edges = np.unique(
        np.quantile(pixels_per_square, np.linspace(0.0, 1.0, bin_count + 1))
    )
    if edges.size < 2:
        edges = np.asarray(
            [float(np.min(pixels_per_square)) - 1.0, float(np.max(pixels_per_square)) + 1.0]
        )
    scores = np.zeros(len(scored_samples), dtype=np.float64)
    assigned = np.zeros(len(scored_samples), dtype=bool)
    for bin_index in range(len(edges) - 1):
        if bin_index == len(edges) - 2:
            mask = (
                (pixels_per_square >= edges[bin_index])
                & (pixels_per_square <= edges[bin_index + 1])
            )
        else:
            mask = (
                (pixels_per_square >= edges[bin_index])
                & (pixels_per_square < edges[bin_index + 1])
            )
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        log_laplacian = np.log1p(
            np.asarray(
                [
                    scored_samples[index]["rectified_laplacian_var"]
                    for index in indices
                ]
            )
        )
        log_tenengrad = np.log1p(
            np.asarray(
                [
                    scored_samples[index]["rectified_tenengrad_mean"]
                    for index in indices
                ]
            )
        )
        lap_median, lap_scale = robust_location_scale(log_laplacian)
        ten_median, ten_scale = robust_location_scale(log_tenengrad)
        scores[indices] = 0.5 * (
            (log_laplacian - lap_median) / lap_scale
            + (log_tenengrad - ten_median) / ten_scale
        )
        assigned[indices] = True
    scores[~assigned] = 0.0

    order = np.argsort(scores)
    percentiles = np.empty_like(scores)
    if len(scores) == 1:
        percentiles[:] = 1.0
    else:
        percentiles[order] = np.arange(len(scores)) / float(len(scores) - 1)
    for index, sample in enumerate(scored_samples):
        sample["sharpness_score"] = float(scores[index])
        sample["sharpness_percentile"] = float(percentiles[index])

    requested_reject_count = int(
        round(len(scored_samples) * float(FINAL_BLUR_REJECT_FRACTION))
    )
    max_reject_count = max(0, len(valid_samples) - MIN_SAMPLES)
    reject_count = min(requested_reject_count, max_reject_count)
    rejected_ids = {id(scored_samples[index]) for index in order[:reject_count]}
    kept_samples: list[dict] = []
    rejected_count = 0
    for sample in valid_samples:
        if id(sample) in rejected_ids:
            sample["final_filter_status"] = "rejected_blur"
            rejected_count += 1
        else:
            sample["final_filter_status"] = "sharpness_kept"
            kept_samples.append(sample)
    return kept_samples, {
        "valid_before_blur_filter": len(valid_samples),
        "scored_for_blur": len(scored_samples),
        "blur_reject_fraction": float(FINAL_BLUR_REJECT_FRACTION),
        "requested_blur_reject_count": requested_reject_count,
        "rejected_blur_count": rejected_count,
    }


def charuco_pose_features(
    sample: dict,
    image_size: tuple[int, int],
    board,
) -> np.ndarray:
    object_points, image_points = charuco_to_calibration_points(
        board,
        sample["charuco_corners"],
        sample["charuco_ids"],
    )
    if object_points is None or image_points is None:
        raise RuntimeError("Missing ChArUco calibration points.")
    object_xy = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)[:, :2]
    image_xy = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    width, height = image_size
    center = np.mean(image_xy, axis=0)
    hull_area = float(cv2.contourArea(cv2.convexHull(image_xy)))
    area_fraction = hull_area / max(float(width * height), 1.0)

    homography, _mask = cv2.findHomography(object_xy, image_xy, method=0)
    if homography is None:
        direction = image_xy[-1] - image_xy[0]
        horizontal_tilt = 0.0
        vertical_tilt = 0.0
    else:
        minimum = np.min(object_xy, axis=0)
        maximum = np.max(object_xy, axis=0)
        object_quad = np.asarray(
            [
                [minimum[0], minimum[1]],
                [maximum[0], minimum[1]],
                [maximum[0], maximum[1]],
                [minimum[0], maximum[1]],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(object_quad, homography).reshape(-1, 2)
        edges = np.asarray(
            [
                np.linalg.norm(quad[1] - quad[0]),
                np.linalg.norm(quad[2] - quad[1]),
                np.linalg.norm(quad[3] - quad[2]),
                np.linalg.norm(quad[0] - quad[3]),
            ],
            dtype=np.float64,
        )
        direction = quad[1] - quad[0]
        horizontal_tilt = math.log(max(edges[0], 1e-6) / max(edges[2], 1e-6))
        vertical_tilt = math.log(max(edges[1], 1e-6) / max(edges[3], 1e-6))
    angle = math.atan2(float(direction[1]), float(direction[0]))
    return np.asarray(
        [
            3.0 * center[0] / width,
            3.0 * center[1] / height,
            1.3 * math.log(max(area_fraction, 1e-8)),
            0.5 * math.cos(angle),
            0.5 * math.sin(angle),
            0.8 * horizontal_tilt,
            0.8 * vertical_tilt,
        ],
        dtype=np.float64,
    )


def select_pose_diverse_charuco_samples(
    samples: list[dict],
    image_size: tuple[int, int],
    board,
) -> tuple[list[dict], int]:
    target_count = min(len(samples), int(FINAL_MAX_CALIBRATION_VIEWS))
    if len(samples) <= target_count:
        final_progress(
            3,
            "Pose diversity",
            target_count,
            target_count,
            "all sharp views kept",
        )
        return list(samples), []

    features = np.vstack(
        [charuco_pose_features(sample, image_size, board) for sample in samples]
    )
    medians = np.median(features, axis=0)
    mad = np.median(np.abs(features - medians), axis=0)
    scale = np.maximum(1.4826 * mad, np.std(features, axis=0) * 0.25)
    scale = np.maximum(scale, 1e-6)
    features = (features - medians) / scale
    sharpness = np.asarray(
        [float(sample.get("sharpness_score", 0.0)) for sample in samples],
        dtype=np.float64,
    )
    sharp_median, sharp_scale = robust_location_scale(sharpness)
    sharp_z = np.clip((sharpness - sharp_median) / sharp_scale, -3.0, 3.0)

    selected_indices = [int(np.argmax(sharp_z))]
    minimum_distance = np.linalg.norm(features - features[selected_indices[0]], axis=1)
    minimum_distance[selected_indices[0]] = -np.inf
    final_progress(3, "Pose diversity", 1, target_count, "farthest-pose selection")
    while len(selected_indices) < target_count:
        score = minimum_distance + 0.12 * sharp_z
        next_index = int(np.argmax(score))
        selected_indices.append(next_index)
        distance = np.linalg.norm(features - features[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected_indices] = -np.inf
        final_progress(
            3,
            "Pose diversity",
            len(selected_indices),
            target_count,
            "farthest-pose selection",
        )

    selected_set = set(selected_indices)
    selected = [sample for index, sample in enumerate(samples) if index in selected_set]
    rejected_count = 0
    for index, sample in enumerate(samples):
        if index in selected_set:
            sample["final_filter_status"] = "pose_selected"
            continue
        sample["final_filter_status"] = "rejected_pose_redundancy"
        rejected_count += 1
    return selected, rejected_count


def robust_calibrate_final_samples(
    samples: list[dict],
    image_size: tuple[int, int],
    board,
    camera_model: str,
    min_corners: int,
) -> tuple[dict, dict]:
    active = list(samples)
    rejected_indices: list[int] = []
    rounds: list[dict] = []
    max_rounds = (
        int(FINAL_MAX_REJECTION_ROUNDS)
        if FINAL_REJECT_REPROJ_OUTLIERS and camera_model == "pinhole"
        else 0
    )
    results: dict = {}
    for round_index in range(max_rounds + 1):
        final_progress(
            4,
            "Robust calibration",
            round_index,
            max_rounds + 1,
            f"round={round_index + 1} views={len(active)} optimizing",
        )
        results = calibrate_target_samples(
            active,
            image_size,
            board,
            camera_model,
            min_corners,
        )
        errors = np.asarray(results["per_view_errors"], dtype=np.float64)
        median, robust_scale = robust_location_scale(errors)
        threshold = min(
            float(FINAL_MAX_VIEW_ERROR_PX),
            median + 3.0 * robust_scale,
        )
        threshold = max(threshold, median + 0.12)
        bad = np.flatnonzero(errors > threshold)
        max_remove = max(1, int(math.ceil(0.10 * len(active))))
        if bad.size > max_remove:
            bad = bad[np.argsort(errors[bad])[::-1][:max_remove]]
        if len(active) - int(bad.size) < MIN_SAMPLES:
            allowable = max(0, len(active) - MIN_SAMPLES)
            bad = bad[np.argsort(errors[bad])[::-1][:allowable]]

        rounds.append(
            {
                "round": round_index + 1,
                "num_views": len(active),
                "rms": float(results["rms"]),
                "median_view_error": median,
                "threshold": float(threshold),
                "num_rejected": int(bad.size),
            }
        )
        print(
            f"\n[INFO] Final calibration round {round_index + 1}: "
            f"views={len(active)}, rms={results['rms']:.6f}px, "
            f"median={median:.6f}px, threshold={threshold:.6f}px, "
            f"reject={bad.size}"
        )
        if bad.size == 0 or round_index == max_rounds:
            break

        bad_set = set(int(index) for index in bad)
        for index in sorted(bad_set):
            sample = active[index]
            sample["final_filter_status"] = "rejected_reprojection"
            sample["final_reproj_error"] = float(errors[index])
            rejected_indices.append(int(sample.get("index", -1)))
        active = [
            sample for index, sample in enumerate(active) if index not in bad_set
        ]

    for sample, metadata, error in zip(
        active,
        results.get("sample_metadata", []),
        results["per_view_errors"],
    ):
        sample["final_filter_status"] = "selected"
        sample["final_reproj_error"] = float(error)
        metadata["final_filter_status"] = "selected"
        metadata["final_reproj_error"] = float(error)
        for key in (
            "rectified_laplacian_var",
            "rectified_tenengrad_mean",
            "pixels_per_square",
            "sharpness_score",
            "sharpness_percentile",
        ):
            value = sample.get(key)
            if value is not None and np.isfinite(float(value)):
                metadata[key] = float(value)
    results["rejected_indices"] = [
        *[int(value) for value in results.get("rejected_indices", [])],
        *rejected_indices,
    ]
    final_progress(
        4,
        "Robust calibration",
        1,
        1,
        f"done views={len(active)} rms={results['rms']:.6f}px",
    )
    return results, {
        "calibration_rounds": rounds,
        "rejected_reprojection_count": len(rejected_indices),
        "final_selected_count": len(active),
    }


def independent_sample_pnp_error(
    sample: dict,
    board,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    object_points, image_points = charuco_to_calibration_points(
        board,
        sample.get("charuco_corners"),
        sample.get("charuco_ids"),
    )
    if object_points is None or image_points is None or len(object_points) < 4:
        return math.nan
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return math.nan
    projected, _jacobian = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist,
    )
    delta = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def cross_validate_final_samples(
    selected_samples: list[dict],
    image_size: tuple[int, int],
    board,
    camera_model: str,
    min_corners: int,
) -> dict:
    if not FINAL_CROSS_VALIDATE:
        final_progress(5, "Cross-validation", 1, 1, "disabled")
        return {"enabled": False, "folds": [], "reason": "disabled"}
    if camera_model != "pinhole":
        final_progress(5, "Cross-validation", 1, 1, "pinhole only")
        return {
            "enabled": False,
            "folds": [],
            "reason": "cross-validation currently supports pinhole only",
        }

    ordered = sorted(
        selected_samples,
        key=lambda sample: (
            int(sample.get("index", -1)),
            int(sample.get("sample_index", -1)),
        ),
    )
    partitions = [ordered[::2], ordered[1::2]]
    if min(len(partition) for partition in partitions) < MIN_SAMPLES:
        final_progress(
            5,
            "Cross-validation",
            1,
            1,
            f"skipped: need >= {MIN_SAMPLES * 2} selected views",
        )
        return {
            "enabled": False,
            "folds": [],
            "reason": (
                f"need at least {MIN_SAMPLES * 2} final views for two-fold "
                f"cross-validation; got {len(ordered)}"
            ),
        }

    fold_results: list[dict] = []
    for fold_index in range(2):
        train_samples = partitions[fold_index]
        holdout_samples = partitions[1 - fold_index]
        final_progress(
            5,
            "Cross-validation",
            fold_index,
            2,
            f"fold={fold_index + 1} train={len(train_samples)} "
            f"holdout={len(holdout_samples)}",
        )
        trained = calibrate_target_samples(
            train_samples,
            image_size,
            board,
            camera_model,
            min_corners,
        )
        holdout_errors = np.asarray(
            [
                independent_sample_pnp_error(
                    sample,
                    board,
                    trained["K"],
                    trained["dist"],
                )
                for sample in holdout_samples
            ],
            dtype=np.float64,
        )
        finite_errors = holdout_errors[np.isfinite(holdout_errors)]
        if finite_errors.size == 0:
            raise RuntimeError(
                f"Cross-validation fold {fold_index + 1} produced no valid PnP errors."
            )
        fold_results.append(
            {
                "fold": fold_index + 1,
                "train_views": len(train_samples),
                "holdout_views": len(holdout_samples),
                "train_rms": float(trained["rms"]),
                "holdout_pnp_rmse": float(
                    np.sqrt(np.mean(finite_errors * finite_errors))
                ),
                "holdout_pnp_median": float(np.median(finite_errors)),
                "K": trained["K"].tolist(),
                "dist": trained["dist"].reshape(-1).tolist(),
            }
        )
        final_progress(
            5,
            "Cross-validation",
            fold_index + 1,
            2,
            f"fold={fold_index + 1} "
            f"holdout={fold_results[-1]['holdout_pnp_rmse']:.6f}px",
        )

    fx_values = np.asarray([fold["K"][0][0] for fold in fold_results])
    fy_values = np.asarray([fold["K"][1][1] for fold in fold_results])
    cx_values = np.asarray([fold["K"][0][2] for fold in fold_results])
    cy_values = np.asarray([fold["K"][1][2] for fold in fold_results])
    return {
        "enabled": True,
        "folds": fold_results,
        "fx_range_px": float(np.ptp(fx_values)),
        "fy_range_px": float(np.ptp(fy_values)),
        "cx_range_px": float(np.ptp(cx_values)),
        "cy_range_px": float(np.ptp(cy_values)),
        "mean_holdout_pnp_rmse": float(
            np.mean([fold["holdout_pnp_rmse"] for fold in fold_results])
        ),
    }


def run_final_calibration_pipeline(
    samples: list[dict],
    image_size: tuple[int, int],
    board,
    camera_model: str,
    min_corners: int,
    squares_x: int,
    squares_y: int,
) -> dict:
    backend = configure_final_sharpness_backend()
    device_name = backend.get("opencl_device", {}).get("name", "CPU")
    final_progress(
        1,
        "Accelerator setup",
        1,
        1,
        f"{backend['gradient_backend']} ({device_name})",
    )

    if is_charuco_target():
        sharp_samples, blur_report = filter_motion_blur_samples(
            samples,
            int(squares_x),
            int(squares_y),
            min_corners,
            backend,
        )
        if len(sharp_samples) < MIN_SAMPLES:
            raise RuntimeError(
                f"Only {len(sharp_samples)} sharp ChArUco samples remain; "
                f"need at least {MIN_SAMPLES}."
            )
        diverse_samples, pose_rejected_count = select_pose_diverse_charuco_samples(
            sharp_samples,
            image_size,
            board,
        )
    else:
        final_progress(2, "Sharpness analysis", 1, 1, "not a ChArUco target")
        diverse_samples = list(samples)
        blur_report = {
            "valid_before_blur_filter": len(samples),
            "rejected_blur_count": 0,
        }
        pose_rejected_count = 0
        final_progress(3, "Pose diversity", 1, 1, "not a ChArUco target")

    results, robust_report = robust_calibrate_final_samples(
        diverse_samples,
        image_size,
        board,
        camera_model,
        min_corners,
    )
    selected_samples = [
        sample
        for sample in diverse_samples
        if sample.get("final_filter_status") == "selected"
    ]
    cross_validation = cross_validate_final_samples(
        selected_samples,
        image_size,
        board,
        camera_model,
        min_corners,
    )
    results["final_processing"] = {
        "enabled": True,
        "acceleration": backend,
        "input_sample_count": len(samples),
        **blur_report,
        "max_calibration_views": int(FINAL_MAX_CALIBRATION_VIEWS),
        "rejected_pose_redundancy_count": pose_rejected_count,
        "max_view_error_px": float(FINAL_MAX_VIEW_ERROR_PX),
        **robust_report,
        "cross_validation": cross_validation,
    }
    return results


def compute_mean_reproj_error(K, dist, rvecs, tvecs, objpoints, imgpoints) -> tuple[float, list[float]]:
    total_err_sq = 0.0
    total_pts = 0
    per_view_errors: list[float] = []

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err = cv2.norm(imgp, proj, cv2.NORM_L2)
        n = len(objp)
        per_view_errors.append(float(np.sqrt((err * err) / max(n, 1))))
        total_err_sq += err * err
        total_pts += n

    mean_err = float(np.sqrt(total_err_sq / max(total_pts, 1)))
    return mean_err, per_view_errors


def compute_fisheye_mean_reproj_error(K, D, rvecs, tvecs, objpoints, imgpoints) -> tuple[float, list[float]]:
    total_err_sq = 0.0
    total_pts = 0
    per_view_errors: list[float] = []

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
        err = cv2.norm(imgp.reshape(-1, 2), proj.reshape(-1, 2), cv2.NORM_L2)
        n = objp.reshape(-1, 3).shape[0]
        per_view_errors.append(float(np.sqrt((err * err) / max(n, 1))))
        total_err_sq += err * err
        total_pts += n

    mean_err = float(np.sqrt(total_err_sq / max(total_pts, 1)))
    return mean_err, per_view_errors


def initial_camera_matrix_for_fisheye(image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    # Equidistant fisheye projection is roughly r = f * theta. Use a 180-degree
    # circular-fisheye prior as a stable seed; cv2.fisheye.calibrate will refine it.
    f = min(width, height) / np.pi
    K = np.asarray(
        [
            [f, 0.0, (width - 1.0) * 0.5],
            [0.0, f, (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K.astype(np.float64)


def image_points_bbox_coverage(imgpoints: np.ndarray, image_size: tuple[int, int]) -> float:
    pts = np.asarray(imgpoints, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return 0.0
    width, height = image_size
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    bbox_wh = np.maximum(max_xy - min_xy, 0.0)
    return float((bbox_wh[0] * bbox_wh[1]) / max(float(width * height), 1.0))


def format_sample_summary(
    local_idx: int,
    global_idx: int,
    used_index: int,
    corner_count: int,
    metadata: dict,
    imgpoints: np.ndarray,
    image_size: tuple[int, int],
) -> str:
    return (
        f"local={local_idx} global={global_idx} captured_index={used_index} "
        f"corners={corner_count} "
        f"bbox_coverage={image_points_bbox_coverage(imgpoints, image_size):.3f} "
        f"image={metadata.get('image_path', '')}"
    )


def calibrate_pinhole_points(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    used_indices: list[int],
    corner_counts: list[int],
    sample_metadata: list[dict],
    image_size: tuple[int, int],
) -> dict:
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    mean_err, per_view_errors = compute_mean_reproj_error(
        K,
        dist,
        rvecs,
        tvecs,
        objpoints,
        imgpoints,
    )
    return {
        "camera_model": "pinhole",
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rms": float(rms),
        "mean_reproj_error": mean_err,
        "per_view_errors": per_view_errors,
        "used_indices": used_indices,
        "corner_counts": corner_counts,
        "sample_metadata": sample_metadata,
        "rejected_indices": [],
    }


def calibrate_fisheye_points(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    used_indices: list[int],
    corner_counts: list[int],
    sample_metadata: list[dict],
    image_size: tuple[int, int],
) -> dict:
    fisheye_objpoints = [
        obj.reshape(1, -1, 3).astype(np.float64)
        for obj in objpoints
    ]
    fisheye_imgpoints = [
        img.reshape(1, -1, 2).astype(np.float64)
        for img in imgpoints
    ]

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

    active_indices = list(range(len(fisheye_objpoints)))
    rejected_indices: list[int] = []
    last_error: Optional[Exception] = None

    while len(active_indices) >= MIN_SAMPLES:
        sub_objpoints = [fisheye_objpoints[i] for i in active_indices]
        sub_imgpoints = [fisheye_imgpoints[i] for i in active_indices]
        sub_used_indices = [used_indices[i] for i in active_indices]
        sub_corner_counts = [corner_counts[i] for i in active_indices]
        sub_sample_metadata = [sample_metadata[i] for i in active_indices]

        try:
            K_init = initial_camera_matrix_for_fisheye(image_size)
            D_init = np.zeros((4, 1), dtype=np.float64)
            rvecs_init = [np.zeros((1, 1, 3), dtype=np.float64) for _ in sub_objpoints]
            tvecs_init = [np.zeros((1, 1, 3), dtype=np.float64) for _ in sub_objpoints]
            print(f"[INFO] Fisheye initial K:\n{K_init}")
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                sub_objpoints,
                sub_imgpoints,
                image_size,
                K_init,
                D_init,
                rvecs_init,
                tvecs_init,
                flags=flags,
                criteria=criteria,
            )
            mean_err, per_view_errors = compute_fisheye_mean_reproj_error(
                K,
                D,
                rvecs,
                tvecs,
                sub_objpoints,
                sub_imgpoints,
            )
            return {
                "camera_model": "fisheye",
                "K": K,
                "dist": D,
                "D": D,
                "rvecs": rvecs,
                "tvecs": tvecs,
                "rms": float(rms),
                "mean_reproj_error": mean_err,
                "per_view_errors": per_view_errors,
                "used_indices": sub_used_indices,
                "corner_counts": sub_corner_counts,
                "sample_metadata": sub_sample_metadata,
                "rejected_indices": rejected_indices,
                "fisheye_flags": int(flags),
            }
        except cv2.error as exc:
            last_error = exc
            match = re.search(r"input array (\d+)", str(exc))
            if match is None:
                if len(active_indices) > MIN_SAMPLES:
                    coverage_scores = [
                        image_points_bbox_coverage(fisheye_imgpoints[i], image_size)
                        for i in active_indices
                    ]
                    bad_local_idx = int(np.argmin(coverage_scores))
                    bad_global_idx = active_indices.pop(bad_local_idx)
                    rejected_indices.append(used_indices[bad_global_idx])
                    print(
                        f"[WARN] Rejecting globally ill-conditioned fisheye {CALIBRATION_TARGET} sample "
                        f"{format_sample_summary(bad_local_idx, bad_global_idx, used_indices[bad_global_idx], corner_counts[bad_global_idx], sample_metadata[bad_global_idx], fisheye_imgpoints[bad_global_idx], image_size)}; "
                        f"remaining={len(active_indices)}"
                    )
                    continue
                print("[ERROR] Remaining fisheye samples are still ill-conditioned:")
                for local_idx, global_idx in enumerate(active_indices):
                    print(
                        "  "
                        + format_sample_summary(
                            local_idx,
                            global_idx,
                            used_indices[global_idx],
                            corner_counts[global_idx],
                            sample_metadata[global_idx],
                            fisheye_imgpoints[global_idx],
                            image_size,
                        )
                    )
                raise RuntimeError(
                    "cv2.fisheye.calibrate failed without identifying a bad input view. "
                    f"Retake more diverse {CALIBRATION_TARGET} samples with the board covering center, "
                    "edges, and corners."
                ) from exc

            bad_local_idx = int(match.group(1))
            if bad_local_idx < 0 or bad_local_idx >= len(active_indices):
                raise RuntimeError(
                    f"cv2.fisheye.calibrate reported invalid bad view index {bad_local_idx}."
                ) from exc
            bad_global_idx = active_indices.pop(bad_local_idx)
            rejected_indices.append(used_indices[bad_global_idx])
            print(
                f"[WARN] Rejecting ill-conditioned fisheye {CALIBRATION_TARGET} sample "
                f"{format_sample_summary(bad_local_idx, bad_global_idx, used_indices[bad_global_idx], corner_counts[bad_global_idx], sample_metadata[bad_global_idx], fisheye_imgpoints[bad_global_idx], image_size)}; "
                f"remaining={len(active_indices)}"
            )

    raise RuntimeError(
        "cv2.fisheye.calibrate rejected too many views. "
        f"Rejected captured frame indices: {rejected_indices}. "
        f"Detected valid frame indices: {used_indices}"
    ) from last_error


def calibrate_charuco_samples(
    samples: list[dict],
    image_size: tuple[int, int],
    board,
    camera_model: str,
    min_corners_per_sample: int,
):
    objpoints = []
    imgpoints = []
    used_indices = []
    corner_counts = []
    sample_metadata = []

    for sample in samples:
        quality_ok, _quality_reason = charuco_detection_quality(
            sample.get("charuco_ids"),
            min_corners_per_sample,
        )
        if not quality_ok:
            continue
        objp, imgp = charuco_to_calibration_points(
            board,
            sample["charuco_corners"],
            sample["charuco_ids"],
        )
        if objp is None or imgp is None:
            continue
        objpoints.append(objp)
        imgpoints.append(imgp)
        used_indices.append(sample["index"])
        corner_counts.append(int(len(objp)))
        metadata = {
            "sample_index": int(sample.get("sample_index", len(sample_metadata))),
            "frame_index": int(sample["index"]),
            "corner_count": int(sample.get("corner_count", len(objp))),
            "marker_count": int(sample.get("marker_count", 0)),
            "image_path": str(sample.get("image_path", "")),
            "capture_mode": str(sample.get("capture_mode", "unknown")),
            "timestamp": float(sample.get("timestamp", 0.0)),
            "final_filter_status": str(
                sample.get("final_filter_status", "not_processed")
            ),
        }
        for key in (
            "rectified_laplacian_var",
            "rectified_tenengrad_mean",
            "pixels_per_square",
            "sharpness_score",
            "sharpness_percentile",
            "final_reproj_error",
        ):
            value = sample.get(key)
            if value is not None and np.isfinite(float(value)):
                metadata[key] = float(value)
        sample_metadata.append(metadata)

    if len(objpoints) < MIN_SAMPLES:
        raise RuntimeError(
            f"Not enough valid ChArUco samples: {len(objpoints)}; need >= {MIN_SAMPLES}"
        )

    if camera_model == "pinhole":
        return calibrate_pinhole_points(
            objpoints,
            imgpoints,
            used_indices,
            corner_counts,
            sample_metadata,
            image_size,
        )
    if camera_model == "fisheye":
        return calibrate_fisheye_points(
            objpoints,
            imgpoints,
            used_indices,
            corner_counts,
            sample_metadata,
            image_size,
        )
    raise ValueError(f"Unsupported camera_model={camera_model}")


def calibrate_apriltag_grid_samples(
    samples: list[dict],
    image_size: tuple[int, int],
    camera_model: str,
    min_corners_per_sample: int,
):
    objpoints = []
    imgpoints = []
    used_indices = []
    corner_counts = []
    sample_metadata = []

    for sample in samples:
        if "object_points" not in sample or "image_points" not in sample:
            continue
        objp = np.asarray(sample["object_points"], dtype=np.float32).reshape(-1, 3)
        imgp = np.asarray(sample["image_points"], dtype=np.float32).reshape(-1, 1, 2)
        if objp.shape[0] < min_corners_per_sample or imgp.shape[0] != objp.shape[0]:
            continue
        objpoints.append(objp)
        imgpoints.append(imgp)
        used_indices.append(sample["index"])
        corner_counts.append(int(objp.shape[0]))
        sample_metadata.append(
            {
                "sample_index": int(sample.get("sample_index", len(sample_metadata))),
                "frame_index": int(sample["index"]),
                "corner_count": int(sample.get("corner_count", objp.shape[0])),
                "marker_count": int(sample.get("marker_count", 0)),
                "tag_ids": [int(v) for v in sample.get("tag_ids", [])],
                "image_path": str(sample.get("image_path", "")),
                "capture_mode": str(sample.get("capture_mode", "unknown")),
                "timestamp": float(sample.get("timestamp", 0.0)),
            }
        )

    if len(objpoints) < MIN_SAMPLES:
        raise RuntimeError(
            f"Not enough valid AprilGrid samples: {len(objpoints)}; need >= {MIN_SAMPLES}"
        )

    if camera_model == "pinhole":
        return calibrate_pinhole_points(
            objpoints,
            imgpoints,
            used_indices,
            corner_counts,
            sample_metadata,
            image_size,
        )
    if camera_model == "fisheye":
        return calibrate_fisheye_points(
            objpoints,
            imgpoints,
            used_indices,
            corner_counts,
            sample_metadata,
            image_size,
        )
    raise ValueError(f"Unsupported camera_model={camera_model}")


def calibrate_target_samples(
    samples: list[dict],
    image_size: tuple[int, int],
    target,
    camera_model: str,
    min_corners_per_sample: int,
):
    if is_charuco_target():
        return calibrate_charuco_samples(samples, image_size, target, camera_model, min_corners_per_sample)
    if CALIBRATION_TARGET == "apriltag_grid":
        return calibrate_apriltag_grid_samples(samples, image_size, camera_model, min_corners_per_sample)
    raise ValueError(f"Unsupported CALIBRATION_TARGET={CALIBRATION_TARGET}")


def draw_hud(
    frame_bgr: np.ndarray,
    detected_corners: int,
    detected_markers: int,
    sample_count: int,
    paused: bool,
    show_help: bool,
    last_auto_reason: str,
    min_corners_per_sample: int,
) -> np.ndarray:
    vis = frame_bgr.copy()
    status = "OK" if detected_corners >= min_corners_per_sample else "LOW"
    color = (40, 220, 40) if status == "OK" else (40, 40, 220)

    target_label = "AprilGrid" if CALIBRATION_TARGET == "apriltag_grid" else "ChArUco"
    cv2.putText(
        vis,
        f"{target_label}: {status} corners={detected_corners} markers={detected_markers}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"samples={sample_count} paused={paused} auto={AUTO_SAVE_VALID_IMAGES}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"auto: {last_auto_reason}",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    if show_help:
        cv2.putText(
            vis,
            "[s] manual store  [p] pause  [c] clear  [h] help  [q] calibrate+quit",
            (12, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    return vis


def finite_csv_value(value) -> float | str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return numeric if np.isfinite(numeric) else ""


def final_diagnostics_paths(output_path: str) -> dict[str, Path]:
    yaml_path = Path(output_path)
    diagnostics_dir = yaml_path.parent / f"{yaml_path.stem}_diagnostics"
    return {
        "directory": diagnostics_dir,
        "selection_csv": diagnostics_dir / "selection_report.csv",
        "rejected_blur_preview": diagnostics_dir
        / "rejected_blur_contact_sheet.jpg",
        "rejected_reprojection_preview": diagnostics_dir
        / "rejected_reprojection_contact_sheet.jpg",
        "selected_preview": diagnostics_dir / "selected_contact_sheet.jpg",
    }


def write_final_selection_csv(
    path: Path,
    samples: list[dict],
    min_corners: int,
) -> None:
    fieldnames = [
        "sample_index",
        "frame_index",
        "image_path",
        "status",
        "quality_ok",
        "quality_reason",
        "corner_count",
        "marker_count",
        "rectified_laplacian_var",
        "rectified_tenengrad_mean",
        "pixels_per_square",
        "sharpness_score",
        "sharpness_percentile",
        "joint_reproj_error",
        "sharpness_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            quality_ok, quality_reason = charuco_detection_quality(
                sample.get("charuco_ids"),
                min_corners,
            )
            writer.writerow(
                {
                    "sample_index": int(sample.get("sample_index", -1)),
                    "frame_index": int(sample.get("index", -1)),
                    "image_path": str(sample.get("image_path", "")),
                    "status": str(
                        sample.get("final_filter_status", "not_processed")
                    ),
                    "quality_ok": bool(quality_ok),
                    "quality_reason": str(
                        sample.get("final_quality_reason", quality_reason)
                    ),
                    "corner_count": int(sample.get("corner_count", 0)),
                    "marker_count": int(sample.get("marker_count", 0)),
                    "rectified_laplacian_var": finite_csv_value(
                        sample.get("rectified_laplacian_var")
                    ),
                    "rectified_tenengrad_mean": finite_csv_value(
                        sample.get("rectified_tenengrad_mean")
                    ),
                    "pixels_per_square": finite_csv_value(
                        sample.get("pixels_per_square")
                    ),
                    "sharpness_score": finite_csv_value(
                        sample.get("sharpness_score")
                    ),
                    "sharpness_percentile": finite_csv_value(
                        sample.get("sharpness_percentile")
                    ),
                    "joint_reproj_error": finite_csv_value(
                        sample.get("final_reproj_error")
                    ),
                    "sharpness_error": str(sample.get("sharpness_error", "")),
                }
            )


def sample_preview_error(sample: dict) -> float:
    value = sample.get("final_reproj_error", math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def compact_preview_metric(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric:.3f}" if np.isfinite(numeric) else "n/a"


def write_final_contact_sheet(
    path: Path,
    samples: list[dict],
    title: str,
    sort_key,
) -> None:
    chosen = sorted(samples, key=sort_key)[: int(FINAL_CONTACT_SHEET_MAX_IMAGES)]
    if not chosen:
        canvas = np.full((120, 900, 3), 245, dtype=np.uint8)
        cv2.putText(
            canvas,
            f"{title}: no samples",
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError(f"Failed to write contact sheet: {path}")
        return

    tile_width, tile_height = 360, 290
    columns = 4
    rows = int(math.ceil(len(chosen) / columns))
    canvas = np.full(
        (60 + rows * tile_height, columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        title,
        (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    for position, sample in enumerate(chosen):
        image = cv2.imread(str(sample.get("image_path", "")), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min(
            (tile_width - 12) / image.shape[1],
            220 / image.shape[0],
        )
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        row = position // columns
        column = position % columns
        x0 = column * tile_width + (tile_width - resized.shape[1]) // 2
        y0 = 60 + row * tile_height + 4
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        text_x = column * tile_width + 8
        text_y = 60 + row * tile_height + 235
        sample_label = (
            f"sample={int(sample.get('sample_index', -1)):04d} "
            f"frame={int(sample.get('index', -1)):06d}"
        )
        cv2.putText(
            canvas,
            sample_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"sharp={compact_preview_metric(sample.get('sharpness_score'))} "
                f"error={compact_preview_metric(sample_preview_error(sample))}"
            ),
            (text_x, text_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(sample.get("final_filter_status", "not_processed")),
            (text_x, text_y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to write contact sheet: {path}")


def write_final_diagnostics(
    paths: dict[str, Path],
    samples: list[dict],
    min_corners: int,
) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    write_final_selection_csv(paths["selection_csv"], samples, min_corners)
    final_progress(
        6,
        "Save outputs",
        2,
        5,
        str(paths["selection_csv"]),
    )

    blur_samples = [
        sample
        for sample in samples
        if sample.get("final_filter_status") == "rejected_blur"
    ]
    write_final_contact_sheet(
        paths["rejected_blur_preview"],
        blur_samples,
        "Rejected low-sharpness / motion-blur samples",
        sort_key=lambda sample: float(sample.get("sharpness_score", math.inf)),
    )
    final_progress(
        6,
        "Save outputs",
        3,
        5,
        str(paths["rejected_blur_preview"]),
    )

    reprojection_samples = [
        sample
        for sample in samples
        if sample.get("final_filter_status") == "rejected_reprojection"
    ]
    write_final_contact_sheet(
        paths["rejected_reprojection_preview"],
        reprojection_samples,
        "Rejected reprojection outliers",
        sort_key=lambda sample: -sample_preview_error(sample),
    )
    final_progress(
        6,
        "Save outputs",
        4,
        5,
        str(paths["rejected_reprojection_preview"]),
    )

    selected_samples = [
        sample
        for sample in samples
        if sample.get("final_filter_status") == "selected"
    ]
    write_final_contact_sheet(
        paths["selected_preview"],
        selected_samples,
        "Selected sharp, pose-diverse calibration samples",
        sort_key=lambda sample: (
            int(sample.get("index", -1)),
            int(sample.get("sample_index", -1)),
        ),
    )
    final_progress(
        6,
        "Save outputs",
        5,
        5,
        str(paths["selected_preview"]),
    )


def save_yaml(
    path: str,
    image_size: tuple[int, int],
    results: dict,
    args: argparse.Namespace,
) -> None:
    data = {
        "camera_model": str(results["camera_model"]),
        "calibration_target": str(CALIBRATION_TARGET),
        "capture": {
            "auto_save_valid_images": bool(AUTO_SAVE_VALID_IMAGES),
            "auto_save_cooldown_s": float(AUTO_SAVE_COOLDOWN_S),
            "min_corners_per_sample": int(args.min_corners),
            "sample_image_root": str(SAMPLE_IMAGE_ROOT),
            "sample_image_dir": str(results.get("sample_image_dir", "")),
        },
        "image_size": [int(image_size[0]), int(image_size[1])],
        "K": results["K"].tolist(),
        "dist": results["dist"].reshape(-1).tolist(),
        "fx": float(results["K"][0, 0]),
        "fy": float(results["K"][1, 1]),
        "cx": float(results["K"][0, 2]),
        "cy": float(results["K"][1, 2]),
        "rms": float(results["rms"]),
        "mean_reproj_error": float(results["mean_reproj_error"]),
        "num_samples": int(len(results["used_indices"])),
        "used_indices": [int(v) for v in results["used_indices"]],
        "corner_counts": [int(v) for v in results["corner_counts"]],
        "per_view_errors": [float(v) for v in results["per_view_errors"]],
        "rejected_indices": [int(v) for v in results.get("rejected_indices", [])],
        "samples": results.get("sample_metadata", []),
    }
    if is_charuco_target():
        data["charuco"] = {
            "squares_x": int(args.squares_x),
            "squares_y": int(args.squares_y),
            "square_length": float(args.square_length),
            "marker_length": float(args.marker_length),
            "dictionary": str(args.dictionary),
            "legacy_pattern": bool(args.legacy_pattern),
        }
        if CALIBRATION_TARGET == "charuco_scale0p25":
            data["charuco"]["mode_config"] = str(
                CHARUCO_SCALE0P25_CONFIG.expanduser().resolve()
            )
            data["charuco"]["linear_scale_from_source"] = 0.25
    elif CALIBRATION_TARGET == "apriltag_grid":
        data["apriltag_grid"] = {
            "yaml": str(APRILTAG_GRID_YAML.expanduser().resolve()),
            "target_type": str(results.get("apriltag_grid", {}).get("target_type", "apriltag_grid")),
            "tag_family": str(results.get("apriltag_grid", {}).get("tag_family", "")),
            "rows": int(results.get("apriltag_grid", {}).get("rows", 0)),
            "cols": int(results.get("apriltag_grid", {}).get("cols", 0)),
            "tag_size_m": float(results.get("apriltag_grid", {}).get("tag_size_m", 0.0)),
            "tag_gap_m": float(results.get("apriltag_grid", {}).get("tag_gap_m", 0.0)),
            "board_width_m": float(results.get("apriltag_grid", {}).get("board_width_m", 0.0)),
            "board_height_m": float(results.get("apriltag_grid", {}).get("board_height_m", 0.0)),
            "min_corners_per_sample": int(args.min_corners),
        }
    if results["camera_model"] == "fisheye":
        data["D"] = results["dist"].reshape(-1).tolist()
        data["fisheye_flags"] = int(results.get("fisheye_flags", 0))
    if results.get("final_processing"):
        data["final_processing"] = results["final_processing"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def default_output_path(camera_name: str, image_size: tuple[int, int], camera_model: str) -> str:
    width, height = image_size
    model_part = "" if camera_model == "pinhole" else f"_{camera_model}"
    target_part = str(CALIBRATION_TARGET)
    return f"outputs/intrinsics_{camera_name}{model_part}_{target_part}_{width}x{height}.yaml"


def run_interactive_calibration(args: argparse.Namespace) -> str:
    global APRILTAG_GRID_YAML, CALIBRATION_TARGET, CAMERA_MODEL, SAMPLE_IMAGE_ROOT
    CALIBRATION_TARGET = str(
        getattr(args, "target", CALIBRATION_TARGET)
    )
    CAMERA_MODEL = str(
        getattr(args, "camera_model", CAMERA_MODEL)
    ).lower()
    target_yaml = getattr(args, "target_yaml", None)
    if target_yaml is not None:
        APRILTAG_GRID_YAML = Path(target_yaml)
    sample_root = getattr(args, "sample_root", None)
    if sample_root is not None:
        SAMPLE_IMAGE_ROOT = Path(sample_root)
    elif CALIBRATION_TARGET == "apriltag_grid":
        SAMPLE_IMAGE_ROOT = Path("outputs/intrinsics_apriltag_grid_samples")
    elif CALIBRATION_TARGET == "charuco_scale0p25":
        SAMPLE_IMAGE_ROOT = Path("outputs/intrinsics_charuco_scale0p25_samples")
    else:
        SAMPLE_IMAGE_ROOT = Path("outputs/intrinsics_charuco_samples")

    if CAMERA_MODEL not in {"pinhole", "fisheye"}:
        raise ValueError(f"Unsupported CAMERA_MODEL={CAMERA_MODEL}; use 'pinhole' or 'fisheye'.")

    config = get_cv2_config(args.camera_name)
    src = parse_camera_source(args.src)
    if config:
        src = str(config["port"])
        args.width = args.width if args.width is not None else int(config["resolution"][0])
        args.height = args.height if args.height is not None else int(config["resolution"][1])
        args.fps = args.fps if args.fps is not None else int(config["fps"])
        args.fourcc = args.fourcc if args.fourcc is not None else config.get("fourcc")
    if args.port is not None:
        src = str(args.port)
    elif not config and DEFAULT_CV2_PORT is not None:
        src = str(DEFAULT_CV2_PORT)

    if is_charuco_target():
        board, dictionary = create_charuco_board(
            args.squares_x,
            args.squares_y,
            args.square_length,
            args.marker_length,
            args.dictionary,
            args.legacy_pattern,
        )
        detector = CharucoDetectorAdapter(board, dictionary)
    elif CALIBRATION_TARGET == "apriltag_grid":
        board = load_apriltag_grid_board(APRILTAG_GRID_YAML)
        detector = AprilTagGridDetectorAdapter(board)
        if args.min_corners is None:
            args.min_corners = int(board.min_corners_per_sample)
    else:
        raise ValueError(f"Unsupported CALIBRATION_TARGET={CALIBRATION_TARGET}")
    if args.min_corners is None:
        args.min_corners = MIN_CORNERS_PER_SAMPLE
    cap, resolved_src = start_capture(src, args.width, args.height, args.fps, args.fourcc)

    samples: list[dict] = []
    sample_image_dir = create_sample_image_dir()
    last_auto_time = 0.0
    last_auto_frame_index = -1
    last_auto_reason = "waiting"
    frame_index = 0
    paused = False
    show_help = True
    last_frame: Optional[np.ndarray] = None
    last_detection = None
    image_size: Optional[tuple[int, int]] = None

    print(f"[INFO] Calibration target: {CALIBRATION_TARGET}")
    if is_charuco_target():
        print(f"[INFO] ChArUco board: {args.squares_x}x{args.squares_y}, square={args.square_length}, marker={args.marker_length}, dict={args.dictionary}, legacy={args.legacy_pattern}")
        if CALIBRATION_TARGET == "charuco_scale0p25":
            print(f"[INFO] ChArUco mode config: {CHARUCO_SCALE0P25_CONFIG.expanduser().resolve()}")
    else:
        print(
            f"[INFO] AprilGrid board: yaml={board.path}, family={board.tag_family}, "
            f"grid={board.rows}x{board.cols}, tag={board.tag_size_m}m, gap={board.tag_gap_m}m"
        )
    print(f"[INFO] Min corners per sample: {args.min_corners}")
    print(f"[INFO] Camera model: {CAMERA_MODEL}")
    print(f"[INFO] CV2 source: requested={src}, resolved={resolved_src}, width={args.width}, height={args.height}, fps={args.fps}, fourcc={args.fourcc}")
    print(f"[INFO] Sample images will be cached under {sample_image_dir}")
    print(f"[INFO] Auto-save valid images: {AUTO_SAVE_VALID_IMAGES}, cooldown={AUTO_SAVE_COOLDOWN_S}s")
    print("[INFO] Store samples manually with 's'. Press 'q' to calibrate and save.")

    window_name = args.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    try:
        while True:
            if not paused or last_frame is None:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError("Failed to read frame from camera.")
                last_frame = frame_bgr
                height, width = frame_bgr.shape[:2]
                image_size = (int(width), int(height))
                frame_index += 1

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if is_charuco_target():
                    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detect(gray)
                    last_detection = {
                        "charuco_corners": charuco_corners,
                        "charuco_ids": charuco_ids,
                        "marker_corners": marker_corners,
                        "marker_ids": marker_ids,
                    }
                else:
                    object_points, image_points, marker_corners, marker_ids, used_tag_ids = detect_apriltag_grid_points(
                        gray,
                        detector,
                        board,
                    )
                    last_detection = {
                        "object_points": object_points,
                        "image_points": image_points,
                        "marker_corners": marker_corners,
                        "marker_ids": marker_ids,
                        "used_tag_ids": used_tag_ids,
                    }

            marker_corners = last_detection.get("marker_corners")
            marker_ids = last_detection.get("marker_ids")
            detected_markers = 0 if marker_ids is None else int(len(marker_ids))
            if is_charuco_target():
                charuco_corners = last_detection.get("charuco_corners")
                charuco_ids = last_detection.get("charuco_ids")
                detected_corners = 0 if charuco_ids is None else int(len(charuco_ids))
                target_quality_ok, target_quality_reason = charuco_detection_quality(
                    charuco_ids,
                    args.min_corners,
                )
            else:
                object_points = last_detection.get("object_points")
                image_points = last_detection.get("image_points")
                used_tag_ids = last_detection.get("used_tag_ids", [])
                detected_corners = 0 if image_points is None else int(np.asarray(image_points).reshape(-1, 2).shape[0])
                target_quality_ok = detected_corners >= args.min_corners
                target_quality_reason = f"corners={detected_corners}"

            auto_stored_this_frame = False
            now = time.time()
            if not AUTO_SAVE_VALID_IMAGES:
                last_auto_reason = "off"
            elif paused:
                last_auto_reason = "paused"
            elif not target_quality_ok:
                last_auto_reason = target_quality_reason
            elif now - last_auto_time < AUTO_SAVE_COOLDOWN_S:
                last_auto_reason = "cooldown"
            elif frame_index == last_auto_frame_index:
                last_auto_reason = "same frame"
            else:
                if is_charuco_target():
                    sample = store_sample(
                        samples,
                        sample_image_dir,
                        frame_index,
                        last_frame,
                        charuco_corners,
                        charuco_ids,
                        detected_corners,
                        detected_markers,
                        "auto",
                    )
                else:
                    sample = store_apriltag_grid_sample(
                        samples,
                        sample_image_dir,
                        frame_index,
                        last_frame,
                        object_points,
                        image_points,
                        used_tag_ids,
                        detected_markers,
                        "auto",
                    )
                last_auto_time = now
                last_auto_frame_index = frame_index
                auto_stored_this_frame = True
                last_auto_reason = f"stored sample {len(samples)}"
                print(
                    f"[INFO] Auto-stored sample {len(samples)}: "
                    f"frame={frame_index}, corners={detected_corners}, "
                    f"markers={detected_markers}, image={sample['image_path']}"
                )

            vis = last_frame.copy()
            if marker_corners is not None and marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            if is_charuco_target():
                if charuco_corners is not None and charuco_ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            elif marker_corners is not None and marker_ids is not None:
                for corners, marker_id_raw in zip(marker_corners, marker_ids.reshape(-1)):
                    marker_id = int(marker_id_raw)
                    if marker_id not in board.tag_object_points:
                        continue
                    center = np.mean(np.asarray(corners).reshape(4, 2), axis=0)
                    cv2.putText(
                        vis,
                        f"id={marker_id}",
                        (int(center[0]) + 4, int(center[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            vis = draw_hud(
                vis,
                detected_corners,
                detected_markers,
                len(samples),
                paused,
                show_help,
                last_auto_reason,
                args.min_corners,
            )

            if args.display_scale and args.display_scale > 0:
                vis = cv2.resize(
                    vis,
                    None,
                    fx=float(args.display_scale),
                    fy=float(args.display_scale),
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break
            if key == ord("h"):
                show_help = not show_help
            elif key == ord("p"):
                paused = not paused
            elif key == ord("c"):
                samples.clear()
                last_auto_time = 0.0
                last_auto_frame_index = -1
                last_auto_reason = "cleared"
                print("[INFO] Cleared samples.")
            elif key == ord("s"):
                if auto_stored_this_frame:
                    print("[INFO] Manual store skipped; auto already stored this frame.")
                    continue
                if not target_quality_ok:
                    print(
                        f"[WARN] Not stored: target quality failed: {target_quality_reason}."
                    )
                    continue
                if is_charuco_target():
                    sample = store_sample(
                        samples,
                        sample_image_dir,
                        frame_index,
                        last_frame,
                        charuco_corners,
                        charuco_ids,
                        detected_corners,
                        detected_markers,
                        "manual",
                    )
                else:
                    sample = store_apriltag_grid_sample(
                        samples,
                        sample_image_dir,
                        frame_index,
                        last_frame,
                        object_points,
                        image_points,
                        used_tag_ids,
                        detected_markers,
                        "manual",
                    )
                last_auto_frame_index = frame_index
                print(
                    f"[INFO] Manually stored sample {len(samples)}: "
                    f"frame={frame_index}, corners={detected_corners}, "
                    f"markers={detected_markers}, image={sample['image_path']}"
                )
    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    if image_size is None:
        raise RuntimeError("No image was captured.")
    if len(samples) == 0:
        raise RuntimeError(f"No {CALIBRATION_TARGET} samples were stored.")

    results = run_final_calibration_pipeline(
        samples,
        image_size,
        board,
        CAMERA_MODEL,
        args.min_corners,
        args.squares_x,
        args.squares_y,
    )
    results["sample_image_dir"] = str(sample_image_dir)
    if CALIBRATION_TARGET == "apriltag_grid":
        results["apriltag_grid"] = {
            "target_type": "apriltag_grid",
            "tag_family": board.tag_family,
            "rows": board.rows,
            "cols": board.cols,
            "tag_size_m": board.tag_size_m,
            "tag_gap_m": board.tag_gap_m,
            "board_width_m": board.board_width_m,
            "board_height_m": board.board_height_m,
        }
    print("[INFO] Calibration results:")
    print(f"  camera_model: {results['camera_model']}")
    print(f"  image_size: {image_size[0]}x{image_size[1]}")
    print(f"  samples: {len(results['used_indices'])}")
    if results.get("rejected_indices"):
        print(f"  rejected_indices: {results['rejected_indices']}")
    if results.get("final_processing"):
        final_processing = results["final_processing"]
        print(
            "  final_processing: "
            f"blur_rejected={final_processing.get('rejected_blur_count', 0)}, "
            "pose_redundant="
            f"{final_processing.get('rejected_pose_redundancy_count', 0)}, "
            "reproj_rejected="
            f"{final_processing.get('rejected_reprojection_count', 0)}"
        )
        cross_validation = final_processing.get("cross_validation", {})
        if cross_validation.get("enabled"):
            print(
                "  cross_validation: "
                f"mean_holdout_pnp_rmse="
                f"{cross_validation['mean_holdout_pnp_rmse']:.6f}px, "
                f"fx_range={cross_validation['fx_range_px']:.6f}px, "
                f"fy_range={cross_validation['fy_range_px']:.6f}px"
            )
    print(f"  rms: {results['rms']}")
    print(f"  mean_reproj_error: {results['mean_reproj_error']}")
    print(f"  K:\n{results['K']}")
    print(f"  dist: {results['dist'].reshape(-1)}")

    camera_name = args.camera_name or args.output_name
    output_path = args.output or default_output_path(camera_name, image_size, CAMERA_MODEL)
    if args.timestamp:
        output_path = append_timestamp_to_yaml_path(output_path)
    diagnostics_paths = final_diagnostics_paths(output_path)
    results["final_processing"]["diagnostics"] = {
        key: str(value)
        for key, value in diagnostics_paths.items()
    }
    final_progress(6, "Save outputs", 0, 5, output_path)
    save_yaml(output_path, image_size, results, args)
    final_progress(6, "Save outputs", 1, 5, output_path)
    write_final_diagnostics(
        diagnostics_paths,
        samples,
        args.min_corners,
    )
    print(f"[INFO] Saved intrinsics to {output_path}")
    print(f"[INFO] Saved final diagnostics to {diagnostics_paths['directory']}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    charuco_defaults = charuco_mode_defaults()
    parser = argparse.ArgumentParser(
        description="Interactive CV2 ChArUco intrinsics calibration."
    )
    parser.add_argument("--src", default=DEFAULT_CV2_SOURCE, help="CV2 source index, /dev/videoX, or USB port id.")
    parser.add_argument("--port", default=None, help="USB port id such as 3-10.1:1.0; overrides --src and the port in --camera-name.")
    parser.add_argument("--camera-name", default=DEFAULT_CV2_CAMERA_NAME, help="Use an OpenCV camera entry from configs/cameras.yaml.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Name used in default output path when --camera-name is not set.")
    parser.add_argument("--width", type=int, default=DEFAULT_CV2_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_CV2_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_CV2_FPS)
    parser.add_argument("--fourcc", default=DEFAULT_CV2_FOURCC)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--camera-model",
        choices=("pinhole", "fisheye"),
        default=CAMERA_MODEL,
    )
    parser.add_argument(
        "--target",
        choices=("charuco", "charuco_scale0p25", "apriltag_grid"),
        default=CALIBRATION_TARGET,
    )
    parser.add_argument(
        "--target-yaml",
        type=Path,
        default=None,
        help="AprilTag-grid YAML; used when --target=apriltag_grid.",
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=None,
        help="Override the directory used for captured calibration images.",
    )
    parser.add_argument("--no-timestamp", dest="timestamp", action="store_false")
    parser.set_defaults(timestamp=True)

    parser.add_argument("--squares-x", type=int, default=charuco_defaults["squares_x"])
    parser.add_argument("--squares-y", type=int, default=charuco_defaults["squares_y"])
    parser.add_argument("--square-length", type=float, default=charuco_defaults["square_length"])
    parser.add_argument("--marker-length", type=float, default=charuco_defaults["marker_length"])
    parser.add_argument("--dictionary", default=charuco_defaults["dictionary"])
    parser.add_argument(
        "--legacy-pattern",
        action="store_true",
        default=charuco_defaults["legacy_pattern"],
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=None,
        help="Minimum detected corners per stored sample. Defaults to the AprilGrid YAML value, or 8 if absent.",
    )
    parser.add_argument("--display-scale", type=float, default=DEFAULT_DISPLAY_SCALE)
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    return parser


if __name__ == "__main__":
    run_interactive_calibration(build_arg_parser().parse_args())
