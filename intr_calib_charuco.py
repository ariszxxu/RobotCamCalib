import argparse
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

# Same AprilGrid board used by extr_calib_fingertip_apriltag_grid.py.
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
DEFAULT_CV2_PORT: Optional[str] = "3-6:1.0"
DEFAULT_CV2_WIDTH: Optional[int] = 2592
DEFAULT_CV2_HEIGHT: Optional[int] = 1944
DEFAULT_CV2_FPS: Optional[int] = 50
DEFAULT_CV2_FOURCC: Optional[str] = "MJPG"
DEFAULT_OUTPUT_NAME: Optional[str] = None
DEFAULT_DISPLAY_SCALE: Optional[float] = 0.4
DEFAULT_WINDOW_NAME: str = "thumb_web_cam ChArUco intrinsics"

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

    try:
        from intr_calib import get_cv2_camera_config
    except Exception as exc:
        raise RuntimeError("Could not import CV2 camera configs from intr_calib.py.") from exc

    return get_cv2_camera_config(camera_name)


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
        sample_metadata.append(
            {
                "sample_index": int(sample.get("sample_index", len(sample_metadata))),
                "frame_index": int(sample["index"]),
                "corner_count": int(sample.get("corner_count", len(objp))),
                "marker_count": int(sample.get("marker_count", 0)),
                "image_path": str(sample.get("image_path", "")),
                "capture_mode": str(sample.get("capture_mode", "unknown")),
                "timestamp": float(sample.get("timestamp", 0.0)),
            }
        )

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

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def default_output_path(camera_name: str, image_size: tuple[int, int], camera_model: str) -> str:
    width, height = image_size
    model_part = "" if camera_model == "pinhole" else f"_{camera_model}"
    target_part = str(CALIBRATION_TARGET)
    return f"outputs/intrinsics_{camera_name}{model_part}_{target_part}_{width}x{height}.yaml"


def run_interactive_calibration(args: argparse.Namespace) -> str:
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
    print(f"[INFO] Store samples manually with 's'. Press 'q' to calibrate and save.")

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

    results = calibrate_target_samples(samples, image_size, board, CAMERA_MODEL, args.min_corners)
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
    print(f"  rms: {results['rms']}")
    print(f"  mean_reproj_error: {results['mean_reproj_error']}")
    print(f"  K:\n{results['K']}")
    print(f"  dist: {results['dist'].reshape(-1)}")

    camera_name = args.camera_name or args.output_name
    output_path = args.output or default_output_path(camera_name, image_size, CAMERA_MODEL)
    if args.timestamp:
        output_path = append_timestamp_to_yaml_path(output_path)
    save_yaml(output_path, image_size, results, args)
    print(f"[INFO] Saved intrinsics to {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    charuco_defaults = charuco_mode_defaults()
    parser = argparse.ArgumentParser(
        description="Interactive CV2 ChArUco intrinsics calibration."
    )
    parser.add_argument("--src", default=DEFAULT_CV2_SOURCE, help="CV2 source index, /dev/videoX, or USB port id.")
    parser.add_argument("--port", default=DEFAULT_CV2_PORT, help="USB port id such as 3-10.1:1.0; overrides --src and the port in --camera-name.")
    parser.add_argument("--camera-name", default=DEFAULT_CV2_CAMERA_NAME, help="Use CV2_CAMERA_CONFIGS entry from intr_calib.py.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Name used in default output path when --camera-name is not set.")
    parser.add_argument("--width", type=int, default=DEFAULT_CV2_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_CV2_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_CV2_FPS)
    parser.add_argument("--fourcc", default=DEFAULT_CV2_FOURCC)
    parser.add_argument("--output", default=None)
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
