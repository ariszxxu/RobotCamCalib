#!/usr/bin/env python3
"""Calibrate hand-back-palm_T_G305-raw-left-RGB from paired target views.

Physical arrangement
--------------------
* ``third_view_cam`` observes the AprilCube rigidly attached to the hand back.
* G305 raw left RGB observes a ChArUco board fixed relative to third_view_cam.
* G305 raw left RGB is rigidly attached to the hand-back-palm/AprilCube frame.

With ``T_A_B`` mapping coordinates from B into A, each stationary paired
observation obeys

    T_g305_left_charuco_i
    @ T_charuco_third_view
    @ T_third_view_hand_back_palm_i
    = T_g305_left_hand_back_palm.

The requested output is the inverse, ``T_hand_back_palm_g305_raw_left_rgb``.

The live stage automatically stores synchronized, stable, pose-diverse image
pairs.  At the requested sample count it stops both cameras, re-detects all
saved pairs with precise detectors in a CPU thread pool, evaluates target
sharpness with OpenCV/OpenCL (GPU when available), rejects bad samples, runs a
parallel multi-start robust SE(3) solve, and writes a timestamped YAML report.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot_cam_calib.targets import (
    AprilCubeDetectionContext,
    Intrinsics,
    PoseDetection,
    detect_aprilcube_pose,
    project_points_for_intrinsics,
    solve_pnp_for_intrinsics,
)
from intr_calib_charuco import (
    CharucoDetectorAdapter,
    charuco_to_calibration_points,
    create_charuco_board,
    start_capture,
)
from robot_cam_calib.capture import (
    FrameWorker as BufferedFrameWorker,
    TimedFrame,
    put_lines,
    resize_for_display,
    select_synchronized_pair as select_frame_pair,
)
from robot_cam_calib.geometry import (
    inv_T,
    make_T,
    params_to_transform,
    residual_stats,
    robust_limit as mad_robust_limit,
    so3_log,
    solve_x_given_y as _solve_X_given_Y,
    solve_y_given_x as _solve_Y_given_X,
    transform_delta,
    transform_to_params,
)
from robot_cam_calib.io import append_timestamp, atomic_yaml_dump


REPO_ROOT = Path(__file__).resolve().parent

THIRD_VIEW_CAMERA_NAME = "third_view_cam"
THIRD_VIEW_PORT = "7-4:1.0"
THIRD_VIEW_INTRINSICS_YAML = Path(
    "/home/ps/project/ConSensV2Lab/image2cube_pose/assets/intrinsics/"
    "intrinsics_None_charuco_2592x1944_0721_235457_"
    "offline_object_release.yaml"
)
THIRD_VIEW_FPS = 20
THIRD_VIEW_FOURCC = "MJPG"

G305_CAMERA_NAME = "g305_raw_left_rgb"
G305_AUTO_SERIAL = "auto"
CAPTURED_G305_SERIAL_20260730 = "CV27561000NC"
G305_WORK_MODE = "Dual Color Streams"
G305_WIDTH = 1280
G305_HEIGHT = 800
G305_FPS = 20
G305_FORMAT = "RGB"
G305_FRAME_TIMEOUT_MS = 1000

# Immutable snapshot printed by this exact device/profile during the
# 2026-07-30 capture.  It lets an interrupted session be solved from saved
# images without reopening the camera merely to query the same factory profile.
G305_FACTORY_K_1280X800 = np.asarray(
    [
        [614.49212646, 0.0, 639.62207031],
        [0.0, 614.45648193, 398.80517578],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
G305_FACTORY_DIST_1280X800 = np.asarray(
    [
        -1.10108125,
        0.551912010,
        0.000101657119,
        -0.000161002856,
        -0.0668475181,
        -1.08186996,
        0.525091767,
        -0.0563182458,
    ],
    dtype=np.float64,
)

APRILCUBE_SRC_DIR = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/src"
)
HAND_BACK_PALM_APRILCUBE_CONFIG = Path(
    "/home/ps/project/ConSensV2Lab/image2cube_pose/assets/cubes/"
    "cube_april_36h11_100_123_2x2x2_outer62p5mm/config.json"
)
CHARUCO_BOARD_YAML = REPO_ROOT / (
    "outputs/charuco_a4_0712_223646/"
    "charuco_7x5_40mm_marker30mm_DICT_5X5_50_"
    "A4_landscape_600dpi.yaml"
)

# Kept as provenance for the next calibration requested by the user.  This
# camera does not participate in the present closed-loop equation.
MIDDLE_FINGER_PORT_REFERENCE = "5-4:1.0"
MIDDLE_FINGER_INTRINSICS_REFERENCE = REPO_ROOT / (
    "outputs/intrinsics_None_charuco_2592x1944_0801_185224.yaml"
)

OUTPUT_PATH = REPO_ROOT / (
    "outputs/extrinsics_hand_back_palm_g305_raw_left_rgb.yaml"
)
SAMPLE_IMAGE_ROOT = REPO_ROOT / (
    "outputs/extrinsics_hand_back_palm_g305_raw_left_rgb_samples"
)

DEFAULT_MAX_SAMPLES = 80
MIN_SAMPLES_TO_SOLVE = 16
FRAME_BUFFER_SIZE = 30
MAX_PAIR_SKEW_S = 0.035
STABLE_REQUIRED_PAIRS = 3
STABLE_MAX_ROT_DELTA_DEG = 1.5
STABLE_MAX_TRANS_DELTA_M = 0.005
AUTO_CAPTURE_COOLDOWN_S = 0.7
MIN_SAMPLE_ROT_DELTA_DEG = 5.0
MIN_SAMPLE_TRANS_DELTA_M = 0.015

MIN_APRILCUBE_TAGS = 1
MAX_APRILCUBE_REPROJ_PX = 3.0
MIN_CHARUCO_CORNERS = 12
MAX_CHARUCO_REPROJ_PX = 2.0
CHARUCO_AXIS_LENGTH_M = 0.03

DISPLAY_SCALE_THIRD_VIEW = 0.35
DISPLAY_SCALE_G305 = 0.65

DEFAULT_OFFLINE_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))
BLUR_MAX_REJECT_FRACTION = 0.10
BLUR_ROBUST_Z_LIMIT = -2.5

# Rotation and translation residuals are normalized to similar magnitudes.
SOLVER_ROT_SCALE_DEG = 2.0
SOLVER_TRANS_SCALE_M = 0.005
OUTLIER_MAD_MULTIPLIER = 3.0
OUTLIER_MIN_ROT_DEG = 0.5
OUTLIER_MAX_ROT_DEG = 8.0
OUTLIER_MIN_TRANS_M = 0.002
OUTLIER_MAX_TRANS_M = 0.030
OUTLIER_MAX_ITERATIONS = 5


@dataclass
class CalibrationSample:
    index: int
    timestamp: float
    pair_skew_s: float
    third_view_frame_index: int
    g305_frame_index: int
    T_third_view_hand_back_palm: np.ndarray
    T_g305_left_charuco: np.ndarray
    cube_tags: int
    cube_reproj_error_px: float
    charuco_corners: int
    charuco_reproj_error_px: float
    third_view_image_path: str
    g305_image_path: str
    capture_mode: str
    third_view_device_timestamp_ms: Optional[float] = None
    g305_device_timestamp_ms: Optional[float] = None
    third_view_sharpness: float = math.nan
    g305_sharpness: float = math.nan
    offline_status: str = "online_only"
    offline_error: str = ""


@dataclass(frozen=True)
class G305ProfileInfo:
    serial: str
    device_name: str
    firmware: str
    connection_type: str
    previous_work_mode: str
    active_work_mode: str
    width: int
    height: int
    fps: int
    format_name: str
    K: np.ndarray
    dist: np.ndarray
    intrinsics_source: str = "Orbbec left VideoStreamProfile at runtime"

    def as_intrinsics(self) -> Intrinsics:
        return Intrinsics(
            path=Path(
                "orbbec://"
                f"{self.serial}/left_color/{self.width}x{self.height}"
                f"@{self.fps}/{self.format_name}"
            ),
            camera_model="pinhole",
            image_size=(self.width, self.height),
            K=self.K.copy(),
            dist=self.dist.copy(),
        )


class FrameWorker(BufferedFrameWorker):
    """Compatibility wrapper retaining this workflow's buffer size."""

    def __init__(
        self,
        name: str,
        read_fn: Callable[[], tuple[np.ndarray, Optional[float], Optional[int]]],
    ) -> None:
        super().__init__(
            name,
            read_fn,
            buffer_size=FRAME_BUFFER_SIZE,
            stop_timeout_s=3.0,
        )


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name) if isinstance(name, str) else str(value).split(".")[-1]


def intrinsic_matrix(intrinsic: Any) -> np.ndarray:
    return np.asarray(
        [
            [intrinsic.fx, 0.0, intrinsic.cx],
            [0.0, intrinsic.fy, intrinsic.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_vector(distortion: Any) -> np.ndarray:
    # Orbbec exposes Brown-Conrady coefficients.  OpenCV's 8-coefficient order
    # is k1, k2, p1, p2, k3, k4, k5, k6.
    return np.asarray(
        [
            distortion.k1,
            distortion.k2,
            distortion.p1,
            distortion.p2,
            distortion.k3,
            distortion.k4,
            distortion.k5,
            distortion.k6,
        ],
        dtype=np.float64,
    )


def rgb_frame_to_bgr(frame: Any, width: int, height: int) -> np.ndarray:
    video = frame.as_video_frame()
    actual_format = enum_name(video.get_format()).upper()
    if actual_format != "RGB":
        raise RuntimeError(f"Expected G305 RGB frame, got {actual_format}")
    flat = np.asanyarray(video.get_data()).reshape(-1)
    expected = width * height * 3
    if flat.size != expected:
        raise RuntimeError(
            f"G305 frame has {flat.size} bytes; expected {expected}"
        )
    rgb = flat.reshape(height, width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class G305RawLeftCamera:
    """Own the Orbbec device, dual-color work mode, profiles, and pipeline."""

    def __init__(
        self,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        format_name: str,
        work_mode: str,
        timeout_ms: int,
    ) -> None:
        self.serial = "" if serial is None else str(serial).strip()
        self.width = width
        self.height = height
        self.fps = fps
        self.format_name = format_name
        self.work_mode = work_mode
        self.timeout_ms = timeout_ms
        self.ob: Any = None
        self.context: Any = None
        self.device: Any = None
        self.pipeline: Any = None
        self.started = False
        self.previous_mode = ""
        self.profile_info: Optional[G305ProfileInfo] = None

    @staticmethod
    def _device_field(devices: Any, method: str, index: int) -> str:
        getter = getattr(devices, method, None)
        if not callable(getter):
            return ""
        try:
            value = getter(index)
        except Exception:
            return ""
        return "" if value is None else str(value)

    def _select_connected_device(self, devices: Any) -> tuple[Any, str]:
        """Resolve one freshly enumerated G305, optionally by explicit serial."""

        records: list[tuple[int, str, str, str]] = []
        for index in range(int(devices.get_count())):
            records.append(
                (
                    index,
                    self._device_field(
                        devices, "get_device_name_by_index", index
                    ),
                    self._device_field(
                        devices, "get_device_serial_number_by_index", index
                    ),
                    self._device_field(
                        devices,
                        "get_device_connection_type_by_index",
                        index,
                    ),
                )
            )
        available = ", ".join(
            f"{name or 'unknown'} serial={serial or 'unknown'} "
            f"connection={connection or 'unknown'}"
            for _index, name, serial, connection in records
        ) or "none"
        explicit_serial = self.serial.lower() not in {"", G305_AUTO_SERIAL}
        if explicit_serial:
            matches = [record for record in records if record[2] == self.serial]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Requested G305 serial {self.serial!r} is not connected; "
                    f"freshly enumerated devices: {available}"
                )
            selected_serial = matches[0][2]
        else:
            candidates = [
                record
                for record in records
                if "gemini 305" in record[1].lower() and record[2]
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "G305 automatic selection requires exactly one connected "
                    "Orbbec Gemini 305; "
                    f"found {len(candidates)} candidates; devices: {available}. "
                    "Connect only the intended G305 or pass --g305-serial."
                )
            _index, name, selected_serial, connection = candidates[0]
            print(
                f"[G305] auto-selected {name} serial={selected_serial} "
                f"connection={connection or 'unknown'}",
                flush=True,
            )
        return devices.get_device_by_serial_number(selected_serial), selected_serial

    @staticmethod
    def _select_profile(
        pipeline: Any,
        sensor_type: Any,
        width: int,
        height: int,
        fps: int,
        format_name: str,
    ) -> Any:
        profiles = pipeline.get_stream_profile_list(sensor_type)
        available: list[str] = []
        for index in range(profiles.get_count()):
            profile = profiles.get_stream_profile_by_index(
                index
            ).as_video_stream_profile()
            description = (
                f"{profile.get_width()}x{profile.get_height()}"
                f"@{profile.get_fps()} {enum_name(profile.get_format())}"
            )
            available.append(description)
            if (
                profile.get_width() == width
                and profile.get_height() == height
                and profile.get_fps() == fps
                and enum_name(profile.get_format()).upper()
                == format_name.upper()
            ):
                return profile
        raise RuntimeError(
            f"G305 does not expose {width}x{height}@{fps} {format_name} for "
            f"{enum_name(sensor_type)}. Available: {available}"
        )

    def open(self) -> G305ProfileInfo:
        try:
            import pyorbbecsdk as ob
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyorbbecsdk is unavailable in this Python environment. "
                "Run with /home/ps/miniconda3/envs/pyroki/bin/python."
            ) from exc

        self.ob = ob
        self.context = ob.Context()
        devices = self.context.query_devices()
        if devices.get_count() <= 0:
            raise RuntimeError("No Orbbec device found")
        self.device, selected_serial = self._select_connected_device(devices)
        info = self.device.get_device_info()
        actual_serial = str(info.get_serial_number())
        if actual_serial != selected_serial:
            raise RuntimeError(
                f"Fresh G305 enumeration selected serial={selected_serial}, "
                f"but opened device reports serial={actual_serial}"
            )

        self.previous_mode = str(self.device.get_depth_work_mode().name)
        if self.previous_mode != self.work_mode:
            print(
                f"[G305] switching work mode {self.previous_mode!r} "
                f"-> {self.work_mode!r}"
            )
            self.device.set_depth_work_mode(self.work_mode)
            time.sleep(1.0)

        self.pipeline = ob.Pipeline(self.device)
        left_profile = self._select_profile(
            self.pipeline,
            ob.OBSensorType.LEFT_COLOR_SENSOR,
            self.width,
            self.height,
            self.fps,
            self.format_name,
        )
        right_profile = self._select_profile(
            self.pipeline,
            ob.OBSensorType.RIGHT_COLOR_SENSOR,
            self.width,
            self.height,
            self.fps,
            self.format_name,
        )
        intrinsic = left_profile.get_intrinsic()
        distortion = left_profile.get_distortion()
        profile_info = G305ProfileInfo(
            serial=actual_serial,
            device_name=str(info.get_name()),
            firmware=str(info.get_firmware_version()),
            connection_type=str(info.get_connection_type()),
            previous_work_mode=self.previous_mode,
            active_work_mode=self.work_mode,
            width=int(left_profile.get_width()),
            height=int(left_profile.get_height()),
            fps=int(left_profile.get_fps()),
            format_name=enum_name(left_profile.get_format()),
            K=intrinsic_matrix(intrinsic),
            dist=distortion_vector(distortion),
        )

        stream_config = ob.Config()
        # The G305 dual-color work mode is configured with both color profiles.
        # Only the raw left RGB frame is used by this calibration.
        stream_config.enable_stream(left_profile)
        stream_config.enable_stream(right_profile)
        self.pipeline.start(stream_config)
        self.started = True
        self.profile_info = profile_info
        return profile_info

    def read_bgr(
        self,
    ) -> tuple[np.ndarray, Optional[float], Optional[int]]:
        if not self.started or self.pipeline is None:
            raise RuntimeError("G305 pipeline is not started")
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            raise RuntimeError("G305 wait_for_frames timed out")
        left = frames.get_left_color_frame()
        if left is None:
            raise RuntimeError("G305 frame set has no raw left color frame")
        bgr = rgb_frame_to_bgr(left, self.width, self.height)
        return (
            bgr,
            float(left.get_timestamp()),
            int(left.get_system_timestamp_us()),
        )

    def close(self) -> None:
        if self.started and self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.started = False
        if (
            self.device is not None
            and self.previous_mode
            and self.previous_mode != self.work_mode
        ):
            time.sleep(0.5)
            print(f"[G305] restoring work mode {self.previous_mode!r}")
            self.device.set_depth_work_mode(self.previous_mode)
            time.sleep(0.5)


def load_intrinsics(path: Path) -> Intrinsics:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return Intrinsics(
        path=resolved,
        camera_model=str(data.get("camera_model", "pinhole")).lower(),
        image_size=tuple(int(value) for value in data["image_size"]),
        K=np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        dist=np.asarray(
            data.get("dist", data.get("D", [0, 0, 0, 0, 0])),
            dtype=np.float64,
        ).reshape(-1),
    )


def load_charuco_target(
    path: Path,
) -> tuple[Any, CharucoDetectorAdapter, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or data.get("target_type") != "charuco":
        raise ValueError(f"Expected target_type=charuco in {resolved}")
    config = data.get("charuco")
    if not isinstance(config, dict):
        raise ValueError(f"Missing charuco mapping in {resolved}")
    required = (
        "squares_x",
        "squares_y",
        "square_length",
        "marker_length",
        "dictionary",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing ChArUco keys in {resolved}: {missing}")
    normalized = {
        "squares_x": int(config["squares_x"]),
        "squares_y": int(config["squares_y"]),
        "square_length": float(config["square_length"]),
        "marker_length": float(config["marker_length"]),
        "dictionary": str(config["dictionary"]),
        "legacy_pattern": bool(config.get("legacy_pattern", False)),
    }
    board, dictionary = create_charuco_board(
        normalized["squares_x"],
        normalized["squares_y"],
        normalized["square_length"],
        normalized["marker_length"],
        normalized["dictionary"],
        normalized["legacy_pattern"],
    )
    return board, CharucoDetectorAdapter(board, dictionary), normalized


def install_legacy_aruco_compatibility() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is unavailable; install opencv-contrib-python"
        )
    if not hasattr(cv2.aruco, "DetectorParameters"):
        cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters_create
    if not hasattr(cv2.aruco, "ArucoDetector"):

        class LegacyArucoDetector:
            def __init__(self, dictionary: Any, parameters: Any) -> None:
                self.dictionary = dictionary
                self.parameters = parameters

            def detectMarkers(self, image: np.ndarray):
                return cv2.aruco.detectMarkers(
                    image,
                    self.dictionary,
                    parameters=self.parameters,
                )

        cv2.aruco.ArucoDetector = LegacyArucoDetector


def create_aprilcube_context(
    intr: Intrinsics,
    *,
    fast: bool,
) -> AprilCubeDetectionContext:
    source = str(APRILCUBE_SRC_DIR.expanduser().resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    install_legacy_aruco_compatibility()
    import aprilcube  # type: ignore[import-not-found]

    detector = aprilcube.detector(
        HAND_BACK_PALM_APRILCUBE_CONFIG,
        intrinsic_cfg={
            "fx": float(intr.K[0, 0]),
            "fy": float(intr.K[1, 1]),
            "cx": float(intr.K[0, 2]),
            "cy": float(intr.K[1, 2]),
        },
        dist_coeffs=intr.dist,
        enable_filter=False,
        fast=fast,
    )
    # AprilCube stores a debug visualization inside process_frame().  On a
    # degenerate transient PnP result, projected axis coordinates can be NaN;
    # OpenCV then raises while drawing them before the caller can reject the
    # invalid pose.  Visualization must never terminate calibration.
    original_draw_result = detector.draw_result

    def guarded_draw_result(image: np.ndarray, result: dict[str, Any]):
        try:
            return original_draw_result(image, result)
        except (cv2.error, OverflowError, ValueError):
            return image.copy()

    detector.draw_result = guarded_draw_result
    face_id_sets = {
        str(face): {int(tag_id) for tag_id in ids}
        for face, ids in detector.face_id_sets.items()
    }
    return AprilCubeDetectionContext(
        detector=detector,
        face_id_sets=face_id_sets,
        tag_corner_map_mm={
            int(tag_id): np.asarray(corners, dtype=np.float64).reshape(4, 3)
            for tag_id, corners in detector.tag_corner_map.items()
        },
        multi_tag_faces={
            face for face, ids in face_id_sets.items() if len(ids) > 1
        },
    )


def detect_hand_back_palm_pose(
    frame_bgr: np.ndarray,
    context: AprilCubeDetectionContext,
    intr: Intrinsics,
) -> PoseDetection:
    """Run AprilCube and turn bad transient/debug poses into a rejected frame."""
    try:
        detection = detect_aprilcube_pose(frame_bgr, context, intr)
    except Exception as exc:
        return PoseDetection(
            ok=False,
            T=None,
            message=(
                "third-view/AprilCube exception rejected: "
                f"{type(exc).__name__}: {exc}"
            ),
            vis=frame_bgr.copy(),
        )
    if not detection.ok or detection.T is None:
        return detection
    transform = np.asarray(detection.T, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return PoseDetection(
            ok=False,
            T=None,
            n_points=detection.n_points,
            reproj_error=detection.reproj_error,
            message="third-view/AprilCube non-finite pose rejected",
            vis=(
                detection.vis
                if detection.vis is not None
                else frame_bgr.copy()
            ),
        )
    rotation = transform[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    ):
        return PoseDetection(
            ok=False,
            T=None,
            n_points=detection.n_points,
            reproj_error=detection.reproj_error,
            message="third-view/AprilCube invalid rotation rejected",
            vis=(
                detection.vis
                if detection.vis is not None
                else frame_bgr.copy()
            ),
        )
    return detection


def detect_charuco_pose(
    frame_bgr: np.ndarray,
    detector: CharucoDetectorAdapter,
    board: Any,
    intr: Intrinsics,
    label: str,
) -> PoseDetection:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detect(
        gray
    )
    vis = frame_bgr.copy()
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if charuco_corners is not None and charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            vis, charuco_corners, charuco_ids
        )

    objpoints, imgpoints = charuco_to_calibration_points(
        board, charuco_corners, charuco_ids
    )
    count = 0 if objpoints is None else int(len(objpoints))
    if (
        objpoints is None
        or imgpoints is None
        or count < MIN_CHARUCO_CORNERS
    ):
        return PoseDetection(
            ok=False,
            T=None,
            n_points=count,
            message=(
                f"{label}: ChArUco corners={count} "
                f"need>={MIN_CHARUCO_CORNERS}"
            ),
            vis=vis,
        )

    try:
        ok, rvec, tvec = solve_pnp_for_intrinsics(
            objpoints,
            imgpoints,
            intr.K,
            intr.dist,
            intr.camera_model,
        )
    except cv2.error as exc:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=count,
            message=f"{label}: solvePnP error: {exc.err}",
            vis=vis,
        )
    if not ok or rvec is None or tvec is None:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=count,
            message=f"{label}: solvePnP failed",
            vis=vis,
        )

    projected = project_points_for_intrinsics(
        objpoints,
        rvec,
        tvec,
        intr.K,
        intr.dist,
        intr.camera_model,
    )
    error = float(
        np.mean(
            np.linalg.norm(
                imgpoints.reshape(-1, 2) - projected.reshape(-1, 2),
                axis=1,
            )
        )
    )
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T_g305_left_charuco = make_T(
        R, np.asarray(tvec, dtype=np.float64).reshape(3)
    )
    try:
        cv2.drawFrameAxes(
            vis,
            intr.K,
            intr.dist,
            rvec,
            tvec,
            CHARUCO_AXIS_LENGTH_M,
        )
    except cv2.error:
        pass
    return PoseDetection(
        ok=True,
        T=T_g305_left_charuco,
        n_points=count,
        reproj_error=error,
        message=(
            f"{label}: ChArUco ok corners={count} err={error:.2f}px"
        ),
        vis=vis,
    )


def validate_configuration(third_view_intr: Intrinsics) -> None:
    required = (
        THIRD_VIEW_INTRINSICS_YAML,
        HAND_BACK_PALM_APRILCUBE_CONFIG,
        CHARUCO_BOARD_YAML,
        MIDDLE_FINGER_INTRINSICS_REFERENCE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required/reference files: {missing}")
    if third_view_intr.image_size != (2592, 1944):
        raise ValueError(
            "third_view intrinsics must be 2592x1944, got "
            f"{third_view_intr.image_size}"
        )
    if third_view_intr.camera_model != "pinhole":
        raise ValueError(
            "third_view intrinsics must use a pinhole model, got "
            f"{third_view_intr.camera_model}"
        )


def select_synchronized_pair(
    third_frames: list[TimedFrame],
    g305_frames: list[TimedFrame],
    last_pair: Optional[tuple[int, int]],
) -> Optional[tuple[TimedFrame, TimedFrame, float]]:
    return select_frame_pair(
        third_frames,
        g305_frames,
        last_pair,
        MAX_PAIR_SKEW_S,
    )


def detection_quality(
    cube: PoseDetection,
    charuco: PoseDetection,
    pair_skew_s: float,
) -> tuple[bool, str]:
    if pair_skew_s > MAX_PAIR_SKEW_S:
        return False, f"pair skew {pair_skew_s * 1000.0:.1f}ms"
    if not cube.ok or cube.T is None:
        return False, cube.message
    if cube.n_points < MIN_APRILCUBE_TAGS:
        return False, f"cube tags {cube.n_points} < {MIN_APRILCUBE_TAGS}"
    if cube.reproj_error > MAX_APRILCUBE_REPROJ_PX:
        return False, f"cube reproj {cube.reproj_error:.2f}px"
    if not charuco.ok or charuco.T is None:
        return False, charuco.message
    if charuco.n_points < MIN_CHARUCO_CORNERS:
        return False, (
            f"ChArUco corners {charuco.n_points} < {MIN_CHARUCO_CORNERS}"
        )
    if charuco.reproj_error > MAX_CHARUCO_REPROJ_PX:
        return False, f"ChArUco reproj {charuco.reproj_error:.2f}px"
    return True, "detections valid"


def is_stable_pair(
    previous: Optional[tuple[np.ndarray, np.ndarray]],
    T_third_view_hand_back_palm: np.ndarray,
    T_g305_left_charuco: np.ndarray,
) -> tuple[bool, str]:
    if previous is None:
        return False, "building stability history"
    cube_rot, cube_trans = transform_delta(
        previous[0], T_third_view_hand_back_palm
    )
    board_rot, board_trans = transform_delta(
        previous[1], T_g305_left_charuco
    )
    stable = (
        cube_rot <= STABLE_MAX_ROT_DELTA_DEG
        and board_rot <= STABLE_MAX_ROT_DELTA_DEG
        and cube_trans <= STABLE_MAX_TRANS_DELTA_M
        and board_trans <= STABLE_MAX_TRANS_DELTA_M
    )
    return stable, (
        f"motion cube={cube_rot:.2f}deg/{cube_trans * 1000.0:.1f}mm "
        f"board={board_rot:.2f}deg/{board_trans * 1000.0:.1f}mm"
    )


def is_diverse_from_all(
    samples: list[CalibrationSample],
    T_third_view_hand_back_palm: np.ndarray,
) -> tuple[bool, str]:
    if not samples:
        return True, "first pose"
    deltas = [
        transform_delta(
            sample.T_third_view_hand_back_palm,
            T_third_view_hand_back_palm,
        )
        for sample in samples
    ]
    nearest_rot, nearest_trans = min(
        deltas, key=lambda value: value[0] / 5.0 + value[1] / 0.015
    )
    diverse = all(
        rot >= MIN_SAMPLE_ROT_DELTA_DEG
        or trans >= MIN_SAMPLE_TRANS_DELTA_M
        for rot, trans in deltas
    )
    return diverse, (
        f"nearest={nearest_rot:.2f}deg/{nearest_trans * 1000.0:.1f}mm"
    )


def create_sample_dir() -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    path = SAMPLE_IMAGE_ROOT / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def store_sample(
    samples: list[CalibrationSample],
    sample_dir: Path,
    third_frame: TimedFrame,
    g305_frame: TimedFrame,
    pair_skew_s: float,
    cube: PoseDetection,
    charuco: PoseDetection,
    capture_mode: str,
) -> CalibrationSample:
    assert cube.T is not None and charuco.T is not None
    index = len(samples)
    third_path = (
        sample_dir / f"sample_{index:04d}_third_view_hand_back_palm.png"
    )
    g305_path = sample_dir / f"sample_{index:04d}_g305_left_charuco.png"
    if not cv2.imwrite(str(third_path), third_frame.frame_bgr):
        raise RuntimeError(f"Failed to save {third_path}")
    if not cv2.imwrite(str(g305_path), g305_frame.frame_bgr):
        raise RuntimeError(f"Failed to save {g305_path}")
    sample = CalibrationSample(
        index=index,
        timestamp=0.5 * (third_frame.timestamp + g305_frame.timestamp),
        pair_skew_s=float(pair_skew_s),
        third_view_frame_index=third_frame.index,
        g305_frame_index=g305_frame.index,
        T_third_view_hand_back_palm=cube.T.copy(),
        T_g305_left_charuco=charuco.T.copy(),
        cube_tags=int(cube.n_points),
        cube_reproj_error_px=float(cube.reproj_error),
        charuco_corners=int(charuco.n_points),
        charuco_reproj_error_px=float(charuco.reproj_error),
        third_view_image_path=str(third_path),
        g305_image_path=str(g305_path),
        capture_mode=capture_mode,
        third_view_device_timestamp_ms=third_frame.device_timestamp_ms,
        g305_device_timestamp_ms=g305_frame.device_timestamp_ms,
    )
    samples.append(sample)
    write_capture_manifest(sample_dir, samples)
    return sample


def sample_to_dict(sample: CalibrationSample) -> dict[str, Any]:
    return {
        "index": int(sample.index),
        "timestamp": float(sample.timestamp),
        "pair_skew_s": float(sample.pair_skew_s),
        "third_view_frame_index": int(sample.third_view_frame_index),
        "g305_frame_index": int(sample.g305_frame_index),
        "T_third_view_cam_hand_back_palm": (
            sample.T_third_view_hand_back_palm.tolist()
        ),
        "T_g305_raw_left_rgb_charuco": (
            sample.T_g305_left_charuco.tolist()
        ),
        "cube_tags": int(sample.cube_tags),
        "cube_reproj_error_px": float(sample.cube_reproj_error_px),
        "charuco_corners": int(sample.charuco_corners),
        "charuco_reproj_error_px": float(
            sample.charuco_reproj_error_px
        ),
        "third_view_image_path": sample.third_view_image_path,
        "g305_image_path": sample.g305_image_path,
        "capture_mode": sample.capture_mode,
        "third_view_device_timestamp_ms": (
            sample.third_view_device_timestamp_ms
        ),
        "g305_device_timestamp_ms": sample.g305_device_timestamp_ms,
        "third_view_sharpness": (
            None
            if not np.isfinite(sample.third_view_sharpness)
            else float(sample.third_view_sharpness)
        ),
        "g305_sharpness": (
            None
            if not np.isfinite(sample.g305_sharpness)
            else float(sample.g305_sharpness)
        ),
        "offline_status": sample.offline_status,
        "offline_error": sample.offline_error,
    }


def write_capture_manifest(
    sample_dir: Path,
    samples: list[CalibrationSample],
) -> None:
    atomic_yaml_dump(
        sample_dir / "capture_manifest.yaml",
        {
            "schema": "robot_cam_calib.g305_hand_back_capture.v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "frame_convention": (
                "T_A_B maps coordinates from frame B into frame A"
            ),
            "num_samples": len(samples),
            "samples": [sample_to_dict(sample) for sample in samples],
        },
    )


def load_samples_from_manifest(path: Path) -> list[CalibrationSample]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if (
        not isinstance(payload, dict)
        or payload.get("schema")
        != "robot_cam_calib.g305_hand_back_capture.v1"
    ):
        raise ValueError(f"Unsupported capture manifest: {resolved}")
    records = payload.get("samples")
    if not isinstance(records, list):
        raise ValueError(f"Manifest has no sample list: {resolved}")
    samples: list[CalibrationSample] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid sample record in {resolved}")
        third_path = Path(
            str(record["third_view_image_path"])
        ).expanduser()
        g305_path = Path(str(record["g305_image_path"])).expanduser()
        if not third_path.is_file() or not g305_path.is_file():
            raise FileNotFoundError(
                f"Missing saved image pair for sample {record.get('index')}: "
                f"{third_path}, {g305_path}"
            )
        samples.append(
            CalibrationSample(
                index=int(record["index"]),
                timestamp=float(record["timestamp"]),
                pair_skew_s=float(record["pair_skew_s"]),
                third_view_frame_index=int(
                    record["third_view_frame_index"]
                ),
                g305_frame_index=int(record["g305_frame_index"]),
                T_third_view_hand_back_palm=np.asarray(
                    record["T_third_view_cam_hand_back_palm"],
                    dtype=np.float64,
                ).reshape(4, 4),
                T_g305_left_charuco=np.asarray(
                    record["T_g305_raw_left_rgb_charuco"],
                    dtype=np.float64,
                ).reshape(4, 4),
                cube_tags=int(record["cube_tags"]),
                cube_reproj_error_px=float(
                    record["cube_reproj_error_px"]
                ),
                charuco_corners=int(record["charuco_corners"]),
                charuco_reproj_error_px=float(
                    record["charuco_reproj_error_px"]
                ),
                third_view_image_path=str(third_path.resolve()),
                g305_image_path=str(g305_path.resolve()),
                capture_mode=str(record.get("capture_mode", "offline")),
                third_view_device_timestamp_ms=record.get(
                    "third_view_device_timestamp_ms"
                ),
                g305_device_timestamp_ms=record.get(
                    "g305_device_timestamp_ms"
                ),
            )
        )
    if len(samples) != int(payload.get("num_samples", len(samples))):
        raise ValueError(
            f"Manifest sample count mismatch: header="
            f"{payload.get('num_samples')} records={len(samples)}"
        )
    return sorted(samples, key=lambda sample: sample.index)


def captured_g305_profile_snapshot() -> G305ProfileInfo:
    return G305ProfileInfo(
        serial=CAPTURED_G305_SERIAL_20260730,
        device_name="Orbbec Gemini 305",
        firmware="queried_during_capture_not_persisted",
        connection_type="USB3.2",
        previous_work_mode="Default",
        active_work_mode=G305_WORK_MODE,
        width=G305_WIDTH,
        height=G305_HEIGHT,
        fps=G305_FPS,
        format_name=G305_FORMAT,
        K=G305_FACTORY_K_1280X800.copy(),
        dist=G305_FACTORY_DIST_1280X800.copy(),
        intrinsics_source=(
            "snapshot from the interrupted 2026-07-30 capture log for "
            "serial CV27561000NC, raw-left 1280x800@20 RGB"
        ),
    )


@dataclass(frozen=True)
class DetectorBundle:
    cube_context: AprilCubeDetectionContext
    charuco_board: Any
    charuco_detector: CharucoDetectorAdapter


_DETECTOR_THREAD_LOCAL = threading.local()


def get_thread_detector_bundle(
    third_view_intr: Intrinsics,
) -> DetectorBundle:
    bundle = getattr(_DETECTOR_THREAD_LOCAL, "bundle", None)
    if bundle is None:
        board, detector, _config = load_charuco_target(
            CHARUCO_BOARD_YAML
        )
        bundle = DetectorBundle(
            cube_context=create_aprilcube_context(
                third_view_intr, fast=False
            ),
            charuco_board=board,
            charuco_detector=detector,
        )
        _DETECTOR_THREAD_LOCAL.bundle = bundle
    return bundle


def offline_redetect_one(
    sample: CalibrationSample,
    third_view_intr: Intrinsics,
    g305_intr: Intrinsics,
) -> CalibrationSample:
    try:
        third_image = cv2.imread(
            sample.third_view_image_path, cv2.IMREAD_COLOR
        )
        g305_image = cv2.imread(sample.g305_image_path, cv2.IMREAD_COLOR)
        if third_image is None:
            raise RuntimeError(
                f"could not read {sample.third_view_image_path}"
            )
        if g305_image is None:
            raise RuntimeError(f"could not read {sample.g305_image_path}")
        bundle = get_thread_detector_bundle(third_view_intr)
        cube = detect_hand_back_palm_pose(
            third_image,
            bundle.cube_context,
            third_view_intr,
        )
        charuco = detect_charuco_pose(
            g305_image,
            bundle.charuco_detector,
            bundle.charuco_board,
            g305_intr,
            "G305 raw-left/ChArUco offline",
        )
        valid, reason = detection_quality(cube, charuco, sample.pair_skew_s)
        if (
            not valid
            or cube.T is None
            or charuco.T is None
        ):
            return replace(
                sample,
                offline_status="rejected_redetection",
                offline_error=reason,
            )
        return replace(
            sample,
            T_third_view_hand_back_palm=cube.T.copy(),
            T_g305_left_charuco=charuco.T.copy(),
            cube_tags=int(cube.n_points),
            cube_reproj_error_px=float(cube.reproj_error),
            charuco_corners=int(charuco.n_points),
            charuco_reproj_error_px=float(charuco.reproj_error),
            offline_status="redetected",
            offline_error="",
        )
    except Exception as exc:
        return replace(
            sample,
            offline_status="rejected_redetection_exception",
            offline_error=f"{type(exc).__name__}: {exc}",
        )


def parallel_offline_redetect(
    samples: list[CalibrationSample],
    third_view_intr: Intrinsics,
    g305_intr: Intrinsics,
    workers: int,
) -> list[CalibrationSample]:
    workers = max(1, min(int(workers), len(samples)))
    print(
        f"[POST] precise offline redetection: "
        f"{len(samples)} pairs, {workers} CPU workers"
    )
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    completed: list[CalibrationSample] = []
    try:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="offline-detect",
        ) as executor:
            futures = {
                executor.submit(
                    offline_redetect_one,
                    sample,
                    third_view_intr,
                    g305_intr,
                ): sample.index
                for sample in samples
            }
            for position, future in enumerate(
                as_completed(futures), start=1
            ):
                refined = future.result()
                completed.append(refined)
                print(
                    f"\r[POST] redetection {position}/{len(samples)}",
                    end="",
                    flush=True,
                )
    finally:
        cv2.setNumThreads(previous_cv_threads)
    print()
    return sorted(completed, key=lambda sample: sample.index)


def configure_opencl_backend() -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": "cpu",
        "device": None,
    }
    try:
        if not cv2.ocl.haveOpenCL():
            return result
        device = cv2.ocl.Device_getDefault()
        if not device.available():
            return result
        cv2.ocl.setUseOpenCL(True)
        warmup = cv2.UMat(np.zeros((64, 64), dtype=np.uint8))
        cv2.Laplacian(warmup, cv2.CV_32F).get()
        result["backend"] = "opencl_umat"
        result["device"] = {
            "name": str(device.name()),
            "vendor": str(device.vendorName()),
            "version": str(device.version()),
        }
    except Exception as exc:
        cv2.ocl.setUseOpenCL(False)
        result["initialization_error"] = f"{type(exc).__name__}: {exc}"
    return result


def focus_metric(image: np.ndarray, use_opencl: bool) -> float:
    if image.size == 0:
        return math.nan
    if use_opencl:
        source = cv2.UMat(image)
        laplacian = cv2.Laplacian(source, cv2.CV_32F).get()
        grad_x = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
        gradient_sq = cv2.add(
            cv2.multiply(grad_x, grad_x),
            cv2.multiply(grad_y, grad_y),
        ).get()
    else:
        laplacian = cv2.Laplacian(image, cv2.CV_32F)
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        gradient_sq = grad_x * grad_x + grad_y * grad_y
    # Tenengrad is less sensitive than Laplacian variance to JPEG ringing.
    return float(np.mean(gradient_sq) + 0.05 * np.var(laplacian))


def transform_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(
        np.asarray(T[:3, :3], dtype=np.float64)
    )
    return rvec, np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1)


def project_target_points(
    object_points: np.ndarray,
    T_camera_target: np.ndarray,
    intr: Intrinsics,
) -> np.ndarray:
    rvec, tvec = transform_to_rvec_tvec(T_camera_target)
    return project_points_for_intrinsics(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        rvec,
        tvec,
        intr.K,
        intr.dist,
        intr.camera_model,
    ).reshape(-1, 2)


def canonical_charuco_patch(
    image_path: str,
    T_g305_left_charuco: np.ndarray,
    g305_intr: Intrinsics,
    charuco_config: dict[str, Any],
    use_opencl: bool,
) -> np.ndarray:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"could not read {image_path}")
    width_m = (
        charuco_config["squares_x"] * charuco_config["square_length"]
    )
    height_m = (
        charuco_config["squares_y"] * charuco_config["square_length"]
    )
    outer = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [width_m, 0.0, 0.0],
            [width_m, height_m, 0.0],
            [0.0, height_m, 0.0],
        ],
        dtype=np.float64,
    )
    projected = project_target_points(
        outer, T_g305_left_charuco, g305_intr
    ).astype(np.float32)
    target_size = (700, 500)
    destination = np.asarray(
        [
            [0.0, 0.0],
            [target_size[0] - 1.0, 0.0],
            [target_size[0] - 1.0, target_size[1] - 1.0],
            [0.0, target_size[1] - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(projected, destination)
    source: Any = cv2.UMat(gray) if use_opencl else gray
    patch = cv2.warpPerspective(
        source,
        homography,
        target_size,
        flags=cv2.INTER_LINEAR,
    )
    return patch.get() if isinstance(patch, cv2.UMat) else patch


def canonical_cube_patch(
    image_path: str,
    T_third_view_hand_back_palm: np.ndarray,
    third_view_intr: Intrinsics,
    cube_object_points_m: np.ndarray,
    use_opencl: bool,
) -> np.ndarray:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"could not read {image_path}")
    projected = project_target_points(
        cube_object_points_m,
        T_third_view_hand_back_palm,
        third_view_intr,
    )
    height, width = gray.shape
    x0 = max(0, int(math.floor(float(np.min(projected[:, 0])))) - 12)
    y0 = max(0, int(math.floor(float(np.min(projected[:, 1])))) - 12)
    x1 = min(width, int(math.ceil(float(np.max(projected[:, 0])))) + 13)
    y1 = min(height, int(math.ceil(float(np.max(projected[:, 1])))) + 13)
    if x1 - x0 < 20 or y1 - y0 < 20:
        raise RuntimeError("projected AprilCube ROI is too small")
    crop = gray[y0:y1, x0:x1]
    source: Any = cv2.UMat(crop) if use_opencl else crop
    patch = cv2.resize(
        source,
        (512, 512),
        interpolation=cv2.INTER_CUBIC,
    )
    return patch.get() if isinstance(patch, cv2.UMat) else patch


def robust_low_score_indices(
    indexed_scores: list[tuple[int, float]],
) -> tuple[set[int], dict[str, Any]]:
    finite = [
        (index, score)
        for index, score in indexed_scores
        if np.isfinite(score) and score > 0.0
    ]
    if len(finite) < 8:
        return set(), {
            "finite_count": len(finite),
            "rejected_indices": [],
            "reason": "too_few_scores",
        }
    values = np.log(
        np.asarray([score for _index, score in finite], dtype=np.float64)
    )
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-9)
    robust_z = (values - median) / scale
    candidates = [
        (finite[position][0], float(robust_z[position]))
        for position in range(len(finite))
        if robust_z[position] < BLUR_ROBUST_Z_LIMIT
    ]
    maximum = max(
        1, int(math.floor(len(finite) * BLUR_MAX_REJECT_FRACTION))
    )
    rejected = {
        index
        for index, _z in sorted(candidates, key=lambda item: item[1])[
            :maximum
        ]
    }
    return rejected, {
        "finite_count": len(finite),
        "log_median": median,
        "log_mad": mad,
        "robust_z_limit": BLUR_ROBUST_Z_LIMIT,
        "max_reject_fraction": BLUR_MAX_REJECT_FRACTION,
        "rejected_indices": sorted(rejected),
    }


def gpu_sharpness_filter(
    samples: list[CalibrationSample],
    third_view_intr: Intrinsics,
    g305_intr: Intrinsics,
    charuco_config: dict[str, Any],
    cube_object_points_m: np.ndarray,
) -> tuple[list[CalibrationSample], dict[str, Any]]:
    backend = configure_opencl_backend()
    use_opencl = backend["backend"] == "opencl_umat"
    print(
        f"[POST] target sharpness backend={backend['backend']} "
        f"device={backend.get('device')}"
    )
    scored: list[CalibrationSample] = []
    for position, sample in enumerate(samples, start=1):
        if sample.offline_status != "redetected":
            scored.append(sample)
            continue
        try:
            third_patch = canonical_cube_patch(
                sample.third_view_image_path,
                sample.T_third_view_hand_back_palm,
                third_view_intr,
                cube_object_points_m,
                use_opencl,
            )
            g305_patch = canonical_charuco_patch(
                sample.g305_image_path,
                sample.T_g305_left_charuco,
                g305_intr,
                charuco_config,
                use_opencl,
            )
            scored.append(
                replace(
                    sample,
                    third_view_sharpness=focus_metric(
                        third_patch, use_opencl
                    ),
                    g305_sharpness=focus_metric(g305_patch, use_opencl),
                )
            )
        except Exception as exc:
            scored.append(
                replace(
                    sample,
                    offline_status="rejected_sharpness_exception",
                    offline_error=f"{type(exc).__name__}: {exc}",
                )
            )
        print(
            f"\r[POST] GPU/CPU sharpness {position}/{len(samples)}",
            end="",
            flush=True,
        )
    print()

    third_rejected, third_stats = robust_low_score_indices(
        [
            (sample.index, sample.third_view_sharpness)
            for sample in scored
            if sample.offline_status == "redetected"
        ]
    )
    g305_rejected, g305_stats = robust_low_score_indices(
        [
            (sample.index, sample.g305_sharpness)
            for sample in scored
            if sample.offline_status == "redetected"
        ]
    )
    blur_rejected = third_rejected | g305_rejected
    final: list[CalibrationSample] = []
    for sample in scored:
        if (
            sample.offline_status == "redetected"
            and sample.index in blur_rejected
        ):
            final.append(
                replace(
                    sample,
                    offline_status="rejected_blur",
                    offline_error=(
                        "robust target sharpness outlier: "
                        f"third={sample.index in third_rejected}, "
                        f"g305={sample.index in g305_rejected}"
                    ),
                )
            )
        elif sample.offline_status == "redetected":
            final.append(replace(sample, offline_status="usable"))
        else:
            final.append(sample)
    report = {
        **backend,
        "third_view": third_stats,
        "g305_raw_left_rgb": g305_stats,
        "rejected_indices": sorted(blur_rejected),
    }
    return final, report


def initialize_joint_solution(
    T_third_view_hand_back_palm_list: list[np.ndarray],
    T_g305_left_charuco_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    # A_i X B_i = Y -> A_i X = Y inv(B_i).
    left_list = T_g305_left_charuco_list
    right_list = [
        inv_T(transform)
        for transform in T_third_view_hand_back_palm_list
    ]
    X = np.eye(4, dtype=np.float64)
    Y = _solve_Y_given_X(left_list, right_list, X)
    for _iteration in range(8):
        X = _solve_X_given_Y(left_list, right_list, Y)
        Y = _solve_Y_given_X(left_list, right_list, X)
    return X, Y


def joint_residual_vector(
    params: np.ndarray,
    cube_list: list[np.ndarray],
    charuco_list: list[np.ndarray],
    normalized: bool = True,
) -> np.ndarray:
    X_charuco_third_view = params_to_transform(params[:6])
    Y_g305_left_hand_back_palm = params_to_transform(params[6:])
    rot_scale = np.radians(SOLVER_ROT_SCALE_DEG) if normalized else 1.0
    trans_scale = SOLVER_TRANS_SCALE_M if normalized else 1.0
    residuals: list[float] = []
    for T_third_cube, T_g305_charuco in zip(cube_list, charuco_list):
        closure = (
            inv_T(Y_g305_left_hand_back_palm)
            @ T_g305_charuco
            @ X_charuco_third_view
            @ T_third_cube
        )
        residuals.extend(so3_log(closure[:3, :3]) / rot_scale)
        residuals.extend(closure[:3, 3] / trans_scale)
    return np.asarray(residuals, dtype=np.float64)


def run_joint_least_squares(
    params0: np.ndarray,
    cube_list: list[np.ndarray],
    charuco_list: list[np.ndarray],
):
    return least_squares(
        joint_residual_vector,
        params0,
        args=(cube_list, charuco_list, True),
        loss="huber",
        f_scale=1.0,
        max_nfev=1000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )


def multistart_subsets(
    num_samples: int,
) -> list[tuple[str, list[int]]]:
    all_indices = list(range(num_samples))
    candidates: list[tuple[str, list[int]]] = [("full", all_indices)]
    for name, fraction in (("third", 1.0 / 3.0), ("half", 0.5)):
        window = max(MIN_SAMPLES_TO_SOLVE, int(round(num_samples * fraction)))
        if window >= num_samples:
            continue
        starts = (0, (num_samples - window) // 2, num_samples - window)
        for start in starts:
            indices = list(range(start, start + window))
            candidates.append((f"{name}_{start}_{start + window}", indices))
    unique: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for label, indices in candidates:
        key = tuple(indices)
        if key not in seen:
            seen.add(key)
            unique.append((label, indices))
    return unique


def solve_candidate(
    label: str,
    indices: list[int],
    cube_list: list[np.ndarray],
    charuco_list: list[np.ndarray],
) -> tuple[str, Any]:
    subset_cube = [cube_list[index] for index in indices]
    subset_charuco = [charuco_list[index] for index in indices]
    X_init, Y_init = initialize_joint_solution(
        subset_cube, subset_charuco
    )
    params0 = np.hstack(
        [transform_to_params(X_init), transform_to_params(Y_init)]
    )
    if len(indices) != len(cube_list):
        subset_result = run_joint_least_squares(
            params0, subset_cube, subset_charuco
        )
        params0 = subset_result.x
    return label, run_joint_least_squares(
        params0, cube_list, charuco_list
    )


def solve_once(
    samples: list[CalibrationSample],
    solver_workers: int,
) -> dict[str, Any]:
    cube_list = [
        sample.T_third_view_hand_back_palm for sample in samples
    ]
    charuco_list = [sample.T_g305_left_charuco for sample in samples]
    starts = multistart_subsets(len(samples))
    workers = max(1, min(int(solver_workers), len(starts)))
    candidate_results: list[tuple[str, Any]] = []
    if workers == 1:
        candidate_results = [
            solve_candidate(label, indices, cube_list, charuco_list)
            for label, indices in starts
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="se3-multistart",
        ) as executor:
            futures = [
                executor.submit(
                    solve_candidate,
                    label,
                    indices,
                    cube_list,
                    charuco_list,
                )
                for label, indices in starts
            ]
            candidate_results = [future.result() for future in futures]

    selected_label, result = min(
        candidate_results, key=lambda item: float(item[1].cost)
    )
    X_charuco_third_view = params_to_transform(result.x[:6])
    Y_g305_left_hand_back_palm = params_to_transform(result.x[6:])
    T_hand_back_palm_g305_left = inv_T(Y_g305_left_hand_back_palm)

    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        closure = (
            inv_T(Y_g305_left_hand_back_palm)
            @ sample.T_g305_left_charuco
            @ X_charuco_third_view
            @ sample.T_third_view_hand_back_palm
        )
        per_sample.append(
            {
                "index": int(sample.index),
                "rot_deg": float(
                    np.degrees(
                        np.linalg.norm(so3_log(closure[:3, :3]))
                    )
                ),
                "trans_m": float(np.linalg.norm(closure[:3, 3])),
            }
        )

    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    positive = singular_values[singular_values > 1e-10]
    condition = (
        float(positive[0] / positive[-1])
        if positive.size
        else float("inf")
    )
    return {
        "T_charuco_third_view_cam": X_charuco_third_view,
        "T_g305_raw_left_rgb_hand_back_palm": (
            Y_g305_left_hand_back_palm
        ),
        "T_hand_back_palm_g305_raw_left_rgb": (
            T_hand_back_palm_g305_left
        ),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "optimizer_num_starts": len(candidate_results),
        "optimizer_workers": workers,
        "optimizer_selected_start": selected_label,
        "optimizer_candidate_costs": {
            label: float(candidate.cost)
            for label, candidate in candidate_results
        },
        "jacobian_rank": int(np.linalg.matrix_rank(result.jac, tol=1e-8)),
        "jacobian_condition": condition,
        "jacobian_singular_values": singular_values.tolist(),
        "per_sample_residuals": per_sample,
    }


def robust_limit(
    values: list[float],
    minimum: float,
    maximum: float,
) -> float:
    return mad_robust_limit(
        values,
        minimum,
        maximum,
        multiplier=OUTLIER_MAD_MULTIPLIER,
    )


def solve_with_outlier_rejection(
    samples: list[CalibrationSample],
    solver_workers: int,
) -> dict[str, Any]:
    active = list(samples)
    iterations: list[dict[str, Any]] = []
    for iteration in range(OUTLIER_MAX_ITERATIONS):
        solution = solve_once(active, solver_workers)
        residuals = solution["per_sample_residuals"]
        rot_limit = robust_limit(
            [item["rot_deg"] for item in residuals],
            OUTLIER_MIN_ROT_DEG,
            OUTLIER_MAX_ROT_DEG,
        )
        trans_limit = robust_limit(
            [item["trans_m"] for item in residuals],
            OUTLIER_MIN_TRANS_M,
            OUTLIER_MAX_TRANS_M,
        )
        kept = {
            item["index"]
            for item in residuals
            if item["rot_deg"] <= rot_limit
            and item["trans_m"] <= trans_limit
        }
        next_active = [
            sample for sample in active if sample.index in kept
        ]
        iterations.append(
            {
                "iteration": iteration,
                "input_count": len(active),
                "output_count": len(next_active),
                "rot_limit_deg": rot_limit,
                "trans_limit_m": trans_limit,
                "rejected_indices": [
                    sample.index
                    for sample in active
                    if sample.index not in kept
                ],
            }
        )
        if (
            len(next_active) < MIN_SAMPLES_TO_SOLVE
            or len(next_active) == len(active)
        ):
            break
        active = next_active

    solution = solve_once(active, solver_workers)
    inliers = [sample.index for sample in active]
    inlier_set = set(inliers)
    solution["inlier_indices"] = inliers
    solution["outlier_indices"] = [
        sample.index
        for sample in samples
        if sample.index not in inlier_set
    ]
    solution["outlier_rejection_iterations"] = iterations
    solution["residual_rot_deg"] = residual_stats(
        [
            item["rot_deg"]
            for item in solution["per_sample_residuals"]
        ]
    )
    solution["residual_trans_m"] = residual_stats(
        [
            item["trans_m"]
            for item in solution["per_sample_residuals"]
        ]
    )
    return solution


def serialize_solution(solution: dict[str, Any]) -> dict[str, Any]:
    return {
        "T_charuco_third_view_cam": (
            solution["T_charuco_third_view_cam"].tolist()
        ),
        "T_third_view_cam_charuco": inv_T(
            solution["T_charuco_third_view_cam"]
        ).tolist(),
        "T_g305_raw_left_rgb_hand_back_palm": (
            solution["T_g305_raw_left_rgb_hand_back_palm"].tolist()
        ),
        "T_hand_back_palm_g305_raw_left_rgb": (
            solution["T_hand_back_palm_g305_raw_left_rgb"].tolist()
        ),
        "requested_output": {
            "name": "T_hand_back_palm_g305_raw_left_rgb",
            "meaning": (
                "G305 original/raw left RGB optical-frame pose expressed "
                "in the hand-back-palm/wrist-Q AprilCube object frame"
            ),
            "units": "meters",
        },
        "optimizer_success": solution["optimizer_success"],
        "optimizer_message": solution["optimizer_message"],
        "optimizer_nfev": solution["optimizer_nfev"],
        "optimizer_num_starts": solution["optimizer_num_starts"],
        "optimizer_workers": solution["optimizer_workers"],
        "optimizer_selected_start": (
            solution["optimizer_selected_start"]
        ),
        "optimizer_candidate_costs": (
            solution["optimizer_candidate_costs"]
        ),
        "jacobian_rank": solution["jacobian_rank"],
        "jacobian_condition": solution["jacobian_condition"],
        "jacobian_singular_values": (
            solution["jacobian_singular_values"]
        ),
        "residual_rot_deg": solution["residual_rot_deg"],
        "residual_trans_m": solution["residual_trans_m"],
        "inlier_indices": solution["inlier_indices"],
        "outlier_indices": solution["outlier_indices"],
        "outlier_rejection_iterations": (
            solution["outlier_rejection_iterations"]
        ),
        "per_sample_residuals": solution["per_sample_residuals"],
    }


def save_results(
    output_path: Path,
    all_samples: list[CalibrationSample],
    usable_samples: list[CalibrationSample],
    solution: dict[str, Any],
    sample_dir: Path,
    third_view_active_device: int | str,
    g305_profile: G305ProfileInfo,
    charuco_config: dict[str, Any],
    postprocess_report: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    timestamped = append_timestamp(output_path)
    payload = {
        "schema": (
            "robot_cam_calib.hand_back_palm_g305_raw_left_rgb.v1"
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_convention": (
            "T_A_B maps coordinates from frame B into frame A; "
            "translation is meters"
        ),
        "measurement_equation": (
            "T_g305_raw_left_rgb_charuco_i @ "
            "T_charuco_third_view_cam @ "
            "T_third_view_cam_hand_back_palm_i = "
            "T_g305_raw_left_rgb_hand_back_palm"
        ),
        "frames": {
            "hand_back_palm": (
                "wrist_Q/hand-back AprilCube config object frame"
            ),
            "g305_raw_left_rgb": (
                "G305 original unrectified left RGB optical frame"
            ),
            "third_view_cam": "third-view camera optical frame",
            "charuco": "7x5 40mm/30mm ChArUco board frame",
        },
        "inputs": {
            "third_view": {
                "usb_port": args.third_view_port,
                "active_device": str(third_view_active_device),
                "resolution": list(
                    load_intrinsics(
                        THIRD_VIEW_INTRINSICS_YAML
                    ).image_size
                ),
                "fps": int(args.third_view_fps),
                "fourcc": THIRD_VIEW_FOURCC,
                "intrinsics_yaml": str(
                    THIRD_VIEW_INTRINSICS_YAML.resolve()
                ),
            },
            "g305_raw_left_rgb": {
                "device_name": g305_profile.device_name,
                "serial": g305_profile.serial,
                "firmware": g305_profile.firmware,
                "connection_type": g305_profile.connection_type,
                "previous_work_mode": g305_profile.previous_work_mode,
                "capture_work_mode": g305_profile.active_work_mode,
                "resolution": [
                    g305_profile.width,
                    g305_profile.height,
                ],
                "fps": g305_profile.fps,
                "format": g305_profile.format_name,
                "raw_unrectified": True,
                "intrinsics_source": g305_profile.intrinsics_source,
                "distortion_model": (
                    "OpenCV Brown-Conrady coefficient order assumed from "
                    "Orbbec profile"
                ),
                "K": g305_profile.K.tolist(),
                "dist": g305_profile.dist.tolist(),
            },
            "targets": {
                "hand_back_palm_aprilcube_config": str(
                    HAND_BACK_PALM_APRILCUBE_CONFIG.resolve()
                ),
                "charuco_board_yaml": str(
                    CHARUCO_BOARD_YAML.resolve()
                ),
                "charuco": charuco_config,
            },
            "middle_finger_cam_future_reference": {
                "used_in_this_calibration": False,
                "usb_port": MIDDLE_FINGER_PORT_REFERENCE,
                "intrinsics_yaml": str(
                    MIDDLE_FINGER_INTRINSICS_REFERENCE.resolve()
                ),
                "future_requested_transform": (
                    "T_hand_back_cube_middle_finger_cam"
                ),
            },
        },
        "capture": {
            "sample_image_dir": str(sample_dir),
            "num_raw_samples": len(all_samples),
            "num_usable_after_postprocess": len(usable_samples),
            "automatic_capture": True,
            "automatic_stop_sample_count": int(args.max_samples),
            "software_timestamp_pairing": True,
            "max_pair_skew_s": MAX_PAIR_SKEW_S,
            "stable_required_pairs": STABLE_REQUIRED_PAIRS,
            "stable_max_rot_delta_deg": STABLE_MAX_ROT_DELTA_DEG,
            "stable_max_trans_delta_m": STABLE_MAX_TRANS_DELTA_M,
            "min_sample_rot_delta_deg": MIN_SAMPLE_ROT_DELTA_DEG,
            "min_sample_trans_delta_m": MIN_SAMPLE_TRANS_DELTA_M,
            "warning": (
                "The cameras are not hardware synchronized. Samples are "
                "stored only after both target poses remain stationary."
            ),
        },
        "postprocess": postprocess_report,
        "solution": serialize_solution(solution),
        "samples": [sample_to_dict(sample) for sample in all_samples],
    }
    atomic_yaml_dump(timestamped, payload)
    return timestamped


def run_self_test(solver_workers: int) -> None:
    rng = np.random.default_rng(20260730)
    X_true = make_T(
        Rotation.from_rotvec(
            np.radians([15.0, -20.0, 8.0])
        ).as_matrix(),
        [0.08, -0.03, 0.12],
    )
    Y_true = make_T(
        Rotation.from_rotvec(
            np.radians([-10.0, 12.0, 25.0])
        ).as_matrix(),
        [-0.04, 0.06, 0.09],
    )
    samples: list[CalibrationSample] = []
    for index in range(48):
        B = make_T(
            Rotation.from_rotvec(
                rng.normal(0.0, np.radians(25.0), 3)
            ).as_matrix(),
            rng.uniform(-0.25, 0.25, 3)
            + np.asarray([0.0, 0.0, 0.6]),
        )
        A = Y_true @ inv_T(B) @ inv_T(X_true)
        noise_A = make_T(
            Rotation.from_rotvec(
                rng.normal(0.0, np.radians(0.12), 3)
            ).as_matrix(),
            rng.normal(0.0, 0.0006, 3),
        )
        noise_B = make_T(
            Rotation.from_rotvec(
                rng.normal(0.0, np.radians(0.12), 3)
            ).as_matrix(),
            rng.normal(0.0, 0.0006, 3),
        )
        samples.append(
            CalibrationSample(
                index=index,
                timestamp=float(index),
                pair_skew_s=0.0,
                third_view_frame_index=index,
                g305_frame_index=index,
                T_third_view_hand_back_palm=noise_B @ B,
                T_g305_left_charuco=noise_A @ A,
                cube_tags=4,
                cube_reproj_error_px=0.4,
                charuco_corners=20,
                charuco_reproj_error_px=0.4,
                third_view_image_path="",
                g305_image_path="",
                capture_mode="synthetic",
                offline_status="usable",
            )
        )
    solution = solve_with_outlier_rejection(samples, solver_workers)
    x_rot, x_trans = transform_delta(
        X_true, solution["T_charuco_third_view_cam"]
    )
    y_rot, y_trans = transform_delta(
        Y_true,
        solution["T_g305_raw_left_rgb_hand_back_palm"],
    )
    requested_rot, requested_trans = transform_delta(
        inv_T(Y_true),
        solution["T_hand_back_palm_g305_raw_left_rgb"],
    )
    print(
        f"[SELF-TEST] X error={x_rot:.4f}deg/"
        f"{x_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] Y error={y_rot:.4f}deg/"
        f"{y_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] requested transform error="
        f"{requested_rot:.4f}deg/{requested_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] rank={solution['jacobian_rank']} "
        f"condition={solution['jacobian_condition']:.3e}"
    )
    if (
        x_rot > 0.5
        or x_trans > 0.005
        or y_rot > 0.5
        or y_trans > 0.005
        or solution["jacobian_rank"] < 12
    ):
        raise AssertionError(
            "Synthetic recovery exceeded 0.5deg/5mm or is rank deficient"
        )


def postprocess_and_solve(
    samples: list[CalibrationSample],
    sample_dir: Path,
    third_view_intr: Intrinsics,
    g305_intr: Intrinsics,
    charuco_config: dict[str, Any],
    cube_object_points_m: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    list[CalibrationSample],
    list[CalibrationSample],
    dict[str, Any],
    dict[str, Any],
]:
    redetected = parallel_offline_redetect(
        samples,
        third_view_intr,
        g305_intr,
        args.offline_workers,
    )
    checked, sharpness_report = gpu_sharpness_filter(
        redetected,
        third_view_intr,
        g305_intr,
        charuco_config,
        cube_object_points_m,
    )
    usable = [
        sample for sample in checked if sample.offline_status == "usable"
    ]
    write_capture_manifest(sample_dir, checked)
    if len(usable) < MIN_SAMPLES_TO_SOLVE:
        rejected = {
            sample.index: {
                "status": sample.offline_status,
                "error": sample.offline_error,
            }
            for sample in checked
            if sample.offline_status != "usable"
        }
        raise RuntimeError(
            f"Only {len(usable)} usable samples remain after offline "
            f"validation; need {MIN_SAMPLES_TO_SOLVE}. Rejected={rejected}"
        )
    print(
        f"[POST] robust parallel multi-start solve: "
        f"{len(usable)} usable samples, "
        f"{args.solver_workers} requested workers"
    )
    started = time.perf_counter()
    solution = solve_with_outlier_rejection(
        usable, args.solver_workers
    )
    solve_seconds = time.perf_counter() - started
    postprocess_report = {
        "precise_redetection": {
            "workers": int(args.offline_workers),
            "input_count": len(samples),
            "redetected_count": sum(
                sample.offline_status
                in {"usable", "rejected_blur"}
                for sample in checked
            ),
            "rejected_count": sum(
                sample.offline_status.startswith("rejected_redetection")
                for sample in checked
            ),
        },
        "sharpness": sharpness_report,
        "solver": {
            "workers_requested": int(args.solver_workers),
            "workers_used": int(solution["optimizer_workers"]),
            "elapsed_seconds": float(solve_seconds),
        },
        "sample_status_counts": {
            status: sum(
                sample.offline_status == status for sample in checked
            )
            for status in sorted(
                {sample.offline_status for sample in checked}
            )
        },
    }
    return checked, usable, solution, postprocess_report


def open_third_view(
    intr: Intrinsics,
    port: str,
    fps: int,
) -> tuple[Any, int | str]:
    capture, active_device = start_capture(
        port,
        intr.image_size[0],
        intr.image_size[1],
        fps,
        THIRD_VIEW_FOURCC,
    )
    actual_size = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    if actual_size != intr.image_size:
        capture.release()
        raise RuntimeError(
            f"third_view opened at {actual_size}, but intrinsics require "
            f"{intr.image_size}"
        )
    return capture, active_device


def main(args: argparse.Namespace) -> None:
    global CHARUCO_BOARD_YAML, HAND_BACK_PALM_APRILCUBE_CONFIG
    global MIDDLE_FINGER_INTRINSICS_REFERENCE, SAMPLE_IMAGE_ROOT
    global THIRD_VIEW_INTRINSICS_YAML
    THIRD_VIEW_INTRINSICS_YAML = Path(args.third_view_intrinsics)
    HAND_BACK_PALM_APRILCUBE_CONFIG = Path(args.aprilcube_config)
    CHARUCO_BOARD_YAML = Path(args.charuco_board)
    MIDDLE_FINGER_INTRINSICS_REFERENCE = Path(
        args.middle_finger_reference
    )
    SAMPLE_IMAGE_ROOT = Path(args.sample_root)

    third_view_intr = load_intrinsics(THIRD_VIEW_INTRINSICS_YAML)
    validate_configuration(third_view_intr)
    board, charuco_detector, charuco_config = load_charuco_target(
        CHARUCO_BOARD_YAML
    )
    online_cube_context = create_aprilcube_context(
        third_view_intr, fast=True
    )
    cube_object_points_m = np.asarray(
        online_cube_context.detector.box_corners_3d,
        dtype=np.float64,
    ).reshape(-1, 3) / 1000.0

    print("[INFO] Measurement equation:")
    print(
        "  T_g305_raw_left_rgb_charuco @ T_charuco_third_view_cam @"
    )
    print(
        "  T_third_view_cam_hand_back_palm = "
        "T_g305_raw_left_rgb_hand_back_palm"
    )
    print(
        "[INFO] Requested output: "
        "T_hand_back_palm_g305_raw_left_rgb"
    )
    print(
        f"[INFO] third-view intrinsics={third_view_intr.path} "
        f"size={third_view_intr.image_size}"
    )
    print(
        f"[INFO] hand-back-palm AprilCube="
        f"{HAND_BACK_PALM_APRILCUBE_CONFIG.resolve()}"
    )
    print(f"[INFO] ChArUco={CHARUCO_BOARD_YAML.resolve()}")
    print(
        "[INFO] middle-finger intrinsics remembered for the next "
        f"calibration: {MIDDLE_FINGER_INTRINSICS_REFERENCE.resolve()}"
    )

    if args.check_config:
        try:
            __import__("pyorbbecsdk")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Configuration is valid, but pyorbbecsdk is missing from "
                "this Python environment. Use the pyroki conda Python."
            ) from exc
        print("[INFO] Configuration, targets, detectors, and SDK import OK.")
        return

    if args.offline_manifest is not None:
        manifest = args.offline_manifest.expanduser().resolve()
        samples = load_samples_from_manifest(manifest)
        if len(samples) < MIN_SAMPLES_TO_SOLVE:
            raise RuntimeError(
                f"Offline manifest has {len(samples)} samples; need at least "
                f"{MIN_SAMPLES_TO_SOLVE}"
            )
        sample_dir = manifest.parent
        profile = captured_g305_profile_snapshot()
        g305_intr = profile.as_intrinsics()
        print(
            f"[OFFLINE] loaded {len(samples)} saved image pairs from "
            f"{manifest}"
        )
        checked, usable, solution, postprocess_report = (
            postprocess_and_solve(
                samples,
                sample_dir,
                third_view_intr,
                g305_intr,
                charuco_config,
                cube_object_points_m,
                args,
            )
        )
        output = save_results(
            args.output,
            checked,
            usable,
            solution,
            sample_dir,
            "offline_saved_images",
            profile,
            charuco_config,
            {
                **postprocess_report,
                "offline_source_manifest": str(manifest),
                "offline_reason": (
                    "resumed after AprilCube debug visualization exception"
                ),
            },
            args,
        )
        print(f"[INFO] Saved {output}")
        print("[RESULT] T_hand_back_palm_g305_raw_left_rgb:")
        print(solution["T_hand_back_palm_g305_raw_left_rgb"])
        print("[DIAGNOSTICS]")
        print(
            f"  usable={len(usable)}/{len(checked)} "
            f"inliers={len(solution['inlier_indices'])} "
            f"outliers={solution['outlier_indices']}"
        )
        print(
            f"  rotation residual deg={solution['residual_rot_deg']}"
        )
        print(
            f"  translation residual m={solution['residual_trans_m']}"
        )
        print(
            f"  rank={solution['jacobian_rank']} "
            f"condition={solution['jacobian_condition']:.3e}"
        )
        return

    g305 = G305RawLeftCamera(
        serial=args.g305_serial,
        width=G305_WIDTH,
        height=G305_HEIGHT,
        fps=G305_FPS,
        format_name=G305_FORMAT,
        work_mode=G305_WORK_MODE,
        timeout_ms=G305_FRAME_TIMEOUT_MS,
    )
    third_capture: Any = None
    third_active_device: int | str = ""
    third_worker: Optional[FrameWorker] = None
    g305_worker: Optional[FrameWorker] = None
    samples: list[CalibrationSample] = []
    sample_dir: Optional[Path] = None

    try:
        profile = g305.open()
        g305_intr = profile.as_intrinsics()
        print(
            f"[G305] serial={profile.serial} "
            f"profile={profile.width}x{profile.height}@{profile.fps} "
            f"{profile.format_name}"
        )
        print(f"[G305] runtime raw-left K=\n{profile.K}")
        print(f"[G305] runtime raw-left dist={profile.dist}")

        third_capture, third_active_device = open_third_view(
            third_view_intr,
            args.third_view_port,
            args.third_view_fps,
        )

        def read_third() -> tuple[
            np.ndarray, Optional[float], Optional[int]
        ]:
            ok, frame = third_capture.read()
            if not ok or frame is None:
                raise RuntimeError("third-view read failed")
            return frame, None, None

        third_worker = FrameWorker("third-view-capture", read_third)
        g305_worker = FrameWorker("g305-left-capture", g305.read_bgr)
        third_worker.start()
        g305_worker.start()

        if args.check_hardware:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                if third_worker.snapshot() and g305_worker.snapshot():
                    print(
                        "[INFO] Hardware check OK: both cameras produced "
                        "frames at the required profiles."
                    )
                    return
                time.sleep(0.05)
            raise RuntimeError(
                "Hardware check timed out before both cameras produced frames"
            )

        sample_dir = create_sample_dir()
        last_pair: Optional[tuple[int, int]] = None
        previous_valid: Optional[tuple[np.ndarray, np.ndarray]] = None
        stable_count = 0
        last_capture_time = -float("inf")
        last_status = "waiting for synchronized frames"
        stop_requested = False
        print(f"[INFO] Samples: {sample_dir}")
        print(
            "[INFO] Move the rigid hand/G305 assembly to a new pose and "
            "hold still. Capture and stop/solve are automatic."
        )
        print("[INFO] [s] manual store  [q/esc] solve early")

        while not stop_requested:
            pair = select_synchronized_pair(
                third_worker.snapshot(),
                g305_worker.snapshot(),
                last_pair,
            )
            if pair is None:
                if third_worker.last_error or g305_worker.last_error:
                    last_status = (
                        f"camera errors third={third_worker.last_error} "
                        f"g305={g305_worker.last_error}"
                    )
                key = cv2.waitKey(5) & 0xFF if args.preview else 255
                if key in (ord("q"), 27):
                    break
                continue

            third_frame, g305_frame, skew = pair
            last_pair = (third_frame.index, g305_frame.index)
            cube = detect_hand_back_palm_pose(
                third_frame.frame_bgr,
                online_cube_context,
                third_view_intr,
            )
            charuco = detect_charuco_pose(
                g305_frame.frame_bgr,
                charuco_detector,
                board,
                g305_intr,
                "G305 raw-left/ChArUco",
            )
            valid, quality_reason = detection_quality(
                cube, charuco, skew
            )

            stable_reason = "detections invalid"
            if (
                valid
                and cube.T is not None
                and charuco.T is not None
            ):
                stable, stable_reason = is_stable_pair(
                    previous_valid, cube.T, charuco.T
                )
                stable_count = stable_count + 1 if stable else 0
                previous_valid = (cube.T.copy(), charuco.T.copy())
            else:
                stable_count = 0
                previous_valid = None

            diverse = False
            diversity_reason = "waiting for valid pose"
            if valid and cube.T is not None:
                diverse, diversity_reason = is_diverse_from_all(
                    samples, cube.T
                )

            now = time.monotonic()
            auto_store = (
                valid
                and stable_count >= STABLE_REQUIRED_PAIRS
                and diverse
                and now - last_capture_time >= AUTO_CAPTURE_COOLDOWN_S
            )
            stored_this_pair = False
            if auto_store:
                sample = store_sample(
                    samples,
                    sample_dir,
                    third_frame,
                    g305_frame,
                    skew,
                    cube,
                    charuco,
                    "auto",
                )
                stored_this_pair = True
                stable_count = 0
                last_capture_time = now
                last_status = f"auto stored sample {len(samples)}"
                print(
                    f"[CAPTURE] {last_status}: "
                    f"skew={sample.pair_skew_s * 1000.0:.1f}ms "
                    f"cube={sample.cube_tags}tags/"
                    f"{sample.cube_reproj_error_px:.2f}px "
                    f"charuco={sample.charuco_corners}corners/"
                    f"{sample.charuco_reproj_error_px:.2f}px"
                )
                if len(samples) >= args.max_samples:
                    print(
                        f"[CAPTURE] reached {len(samples)} samples; "
                        "stopping cameras and starting automatic postprocess"
                    )
                    stop_requested = True
            else:
                last_status = (
                    f"{quality_reason}; stable={stable_count}/"
                    f"{STABLE_REQUIRED_PAIRS} {stable_reason}; "
                    f"{diversity_reason}"
                )

            if args.preview:
                lines = [
                    f"samples={len(samples)}/{args.max_samples} "
                    f"skew={skew * 1000.0:.1f}ms",
                    last_status,
                    cube.message,
                    charuco.message,
                    "Move, then hold still | [s] manual | [q] solve early",
                ]
                third_vis = put_lines(
                    cube.vis
                    if cube.vis is not None
                    else third_frame.frame_bgr,
                    lines,
                )
                g305_vis = put_lines(
                    charuco.vis
                    if charuco.vis is not None
                    else g305_frame.frame_bgr,
                    lines,
                )
                cv2.imshow(
                    "third_view / hand-back-palm AprilCube",
                    resize_for_display(
                        third_vis, DISPLAY_SCALE_THIRD_VIEW
                    ),
                )
                cv2.imshow(
                    "G305 raw left RGB / ChArUco",
                    resize_for_display(g305_vis, DISPLAY_SCALE_G305),
                )
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if stored_this_pair:
                    print(
                        "[INFO] Manual store skipped: auto capture already "
                        "stored this pair."
                    )
                elif not valid:
                    print(
                        f"[WARN] Manual sample rejected: {quality_reason}"
                    )
                else:
                    sample = store_sample(
                        samples,
                        sample_dir,
                        third_frame,
                        g305_frame,
                        skew,
                        cube,
                        charuco,
                        "manual",
                    )
                    stable_count = 0
                    last_capture_time = now
                    print(
                        f"[CAPTURE] manually stored sample "
                        f"{len(samples)} skew="
                        f"{sample.pair_skew_s * 1000.0:.1f}ms"
                    )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; solving collected samples if possible.")
    finally:
        if third_worker is not None:
            third_worker.stop()
        if g305_worker is not None:
            g305_worker.stop()
        if third_capture is not None:
            third_capture.release()
        g305.close()
        cv2.destroyAllWindows()

    if args.check_hardware:
        return
    if sample_dir is None:
        raise RuntimeError("Capture did not create a sample directory")
    if len(samples) < MIN_SAMPLES_TO_SOLVE:
        print(
            f"[WARN] Only {len(samples)} samples; need at least "
            f"{MIN_SAMPLES_TO_SOLVE}. Images and manifest were preserved at "
            f"{sample_dir}; no extrinsics YAML was written."
        )
        return

    assert g305.profile_info is not None
    g305_intr = g305.profile_info.as_intrinsics()
    checked, usable, solution, postprocess_report = (
        postprocess_and_solve(
            samples,
            sample_dir,
            third_view_intr,
            g305_intr,
            charuco_config,
            cube_object_points_m,
            args,
        )
    )
    if solution["jacobian_rank"] < 12:
        print(
            f"[WARN] Jacobian rank={solution['jacobian_rank']} < 12. "
            "Pose excitation is degenerate; result is saved with diagnostics."
        )
    output = save_results(
        args.output,
        checked,
        usable,
        solution,
        sample_dir,
        third_active_device,
        g305.profile_info,
        charuco_config,
        postprocess_report,
        args,
    )
    print(f"[INFO] Saved {output}")
    print("[RESULT] T_hand_back_palm_g305_raw_left_rgb:")
    print(solution["T_hand_back_palm_g305_raw_left_rgb"])
    print("[DIAGNOSTICS]")
    print(
        f"  usable={len(usable)}/{len(checked)} "
        f"inliers={len(solution['inlier_indices'])} "
        f"outliers={solution['outlier_indices']}"
    )
    print(f"  rotation residual deg={solution['residual_rot_deg']}")
    print(f"  translation residual m={solution['residual_trans_m']}")
    print(
        f"  rank={solution['jacobian_rank']} "
        f"condition={solution['jacobian_condition']:.3e}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--third-view-port",
        default=THIRD_VIEW_PORT,
        help="USB topology port or /dev/video node for third-view camera.",
    )
    parser.add_argument(
        "--third-view-fps",
        type=int,
        default=THIRD_VIEW_FPS,
    )
    parser.add_argument(
        "--third-view-intrinsics",
        type=Path,
        default=THIRD_VIEW_INTRINSICS_YAML,
    )
    parser.add_argument(
        "--aprilcube-config",
        type=Path,
        default=HAND_BACK_PALM_APRILCUBE_CONFIG,
    )
    parser.add_argument(
        "--charuco-board",
        type=Path,
        default=CHARUCO_BOARD_YAML,
    )
    parser.add_argument(
        "--middle-finger-reference",
        type=Path,
        default=MIDDLE_FINGER_INTRINSICS_REFERENCE,
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=SAMPLE_IMAGE_ROOT,
    )
    parser.add_argument(
        "--g305-serial",
        default=G305_AUTO_SERIAL,
        help=(
            "G305 serial override; default 'auto' freshly enumerates hardware "
            "and requires exactly one connected Orbbec Gemini 305"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Automatically stop capture and solve at this sample count.",
    )
    parser.add_argument(
        "--offline-workers",
        type=int,
        default=DEFAULT_OFFLINE_WORKERS,
        help="CPU workers for precise saved-image redetection.",
    )
    parser.add_argument(
        "--solver-workers",
        type=int,
        default=min(7, DEFAULT_OFFLINE_WORKERS),
        help="CPU workers for independent robust multi-start solves.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Load files, targets, detectors, and SDK without opening cameras.",
    )
    parser.add_argument(
        "--check-hardware",
        action="store_true",
        help=(
            "Open both required camera profiles, receive frames, then exit "
            "without collecting samples."
        ),
    )
    parser.add_argument(
        "--offline-manifest",
        type=Path,
        help=(
            "Resume precise postprocessing and solve from a saved "
            "capture_manifest.yaml without opening cameras."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic synthetic SE(3) recovery without cameras.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    if cli_args.max_samples < MIN_SAMPLES_TO_SOLVE:
        raise SystemExit(
            f"--max-samples must be >= {MIN_SAMPLES_TO_SOLVE}"
        )
    if cli_args.offline_workers < 1 or cli_args.solver_workers < 1:
        raise SystemExit("worker counts must be >= 1")
    if cli_args.self_test:
        run_self_test(cli_args.solver_workers)
    else:
        main(cli_args)
