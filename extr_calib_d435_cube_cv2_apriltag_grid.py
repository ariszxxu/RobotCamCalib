from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cameras import RealsenseCamera
from extr_calib_fingertip_apriltag_grid import (
    AprilCubeDetectionContext,
    Intrinsics,
    PoseDetection,
    detect_aprilcube_pose,
    load_intrinsics,
)
from intr_calib_charuco import (
    CharucoDetectorAdapter,
    charuco_to_calibration_points,
    create_charuco_board,
    start_capture,
)


# ---------------------------- User macros ---------------------------- #
D435_SERIAL = "244222070135"
D435_INTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/intrinsics_realsense_1280x720_0707_171032.yaml"
)
D435_WIDTH = 1280
D435_HEIGHT = 720
D435_FPS = 15  # D435 exposes 1280x720 color at 6/10/15 FPS, not 30 FPS.
D435_FORMAT = "bgra8"

# Previous thumb_web_cam / AprilTag-grid configuration:
# CV2_CAMERA_NAME = "thumb_web_cam"
# CV2_PORT = "3-9:1.0"
# CV2_INTRINSICS_YAML = Path(
#     "/home/ps/RobotCamCalib1/outputs/"
#     "intrinsics_cam0_fisheye_2592x1944_0703_230535.yaml"
# )
# APRILTAG_GRID_YAML = Path(
#     "/home/ps/RobotCamCalib1/outputs/apriltag_grid_36h10_a4_full/"
#     "apriltag_36h10_grid_8x11_ids_87_to_0_tag20mm_gap5mm_a4_full.yaml"
# )

CV2_CAMERA_NAME = "middle_finger_cam"
CV2_PORT = "3-8:1.0"
CV2_INTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/"
    "intrinsics_charuco_scale0p25_2592x1944_0712_225925.yaml"
)
CV2_FPS = 50
CV2_FOURCC = "MJPG"

APRILCUBE_SRC_DIR = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/src"
)
APRILCUBE_CONFIG = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/"
    "cube_april_36h11_100_123_2x2x2_outer62p5mm/config.json"
)
CHARUCO_BOARD_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/charuco_a4_scale0p25/"
    "charuco_7x5_scale0p25_square10mm_marker7p5mm_"
    "DICT_5X5_50_A4_landscape.yaml"
)

OUTPUT_PATH = Path(
    "outputs/extrinsics_middle_finger_cam_cube_d435_charuco_scale0p25.yaml"
)
SAMPLE_IMAGE_ROOT = Path(
    "outputs/extrinsics_middle_finger_cam_cube_d435_charuco_scale0p25_samples"
)

# The two cameras have no shared hardware clock. Arrival-time pairing is only
# accepted while both detected poses are stable, which prevents hand motion
# from turning software timestamp skew into an extrinsic bias.
MAX_PAIR_SKEW_S = 0.030
FRAME_BUFFER_SIZE = 20
STABLE_REQUIRED_PAIRS = 3
STABLE_MAX_ROT_DELTA_DEG = 2.0
STABLE_MAX_TRANS_DELTA_M = 0.008

AUTO_CAPTURE = True
AUTO_CAPTURE_COOLDOWN_S = 0.6
MIN_SAMPLE_ROT_DELTA_DEG = 5.0
MIN_SAMPLE_TRANS_DELTA_M = 0.020
MIN_SAMPLES_TO_SOLVE = 12
AUTO_STOP_SAMPLE_COUNT = 80

MIN_APRILCUBE_TAGS = 2
MAX_APRILCUBE_REPROJ_PX = 3.0
MIN_CHARUCO_CORNERS = 12
MAX_CHARUCO_REPROJ_PX = 2.0
CHARUCO_AXIS_LENGTH_M = 0.02

# These residual scales follow extr_calib.py. They balance radians and meters
# inside the robust joint SE(3) solve; they are not claimed sensor covariances.
SOLVER_ROT_SCALE_DEG = 3.0
SOLVER_TRANS_SCALE_M = 0.010
OUTLIER_MIN_ROT_DEG = 2.0
OUTLIER_MAX_ROT_DEG = 10.0
OUTLIER_MIN_TRANS_M = 0.010
OUTLIER_MAX_TRANS_M = 0.050
OUTLIER_MAD_MULTIPLIER = 3.0
OUTLIER_MAX_ITERATIONS = 5

DISPLAY_SCALE_D435 = 0.75
DISPLAY_SCALE_CV2 = 0.35


@dataclass(frozen=True)
class TimedFrame:
    index: int
    timestamp: float
    frame_bgr: np.ndarray


class FrameWorker:
    def __init__(self, name: str, read_fn: Callable[[], np.ndarray]) -> None:
        self.name = name
        self.read_fn = read_fn
        self.frames: deque[TimedFrame] = deque(maxlen=FRAME_BUFFER_SIZE)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.last_error: Optional[str] = None
        self._index = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def snapshot(self) -> list[TimedFrame]:
        with self.lock:
            return list(self.frames)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                start = time.monotonic()
                frame = self.read_fn()
                end = time.monotonic()
                if frame is None:
                    raise RuntimeError("read returned None")
                timed = TimedFrame(
                    index=self._index,
                    timestamp=0.5 * (start + end),
                    frame_bgr=frame,
                )
                self._index += 1
                with self.lock:
                    self.frames.append(timed)
                self.last_error = None
            except Exception as exc:  # camera errors must remain visible to HUD
                self.last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.02)


@dataclass
class CalibrationSample:
    index: int
    timestamp: float
    pair_skew_s: float
    d435_frame_index: int
    cv2_frame_index: int
    T_d435_cube: np.ndarray
    T_cv2_charuco: np.ndarray
    cube_tags: int
    cube_reproj_error_px: float
    charuco_corners: int
    charuco_reproj_error_px: float
    d435_image_path: str
    cv2_image_path: str
    capture_mode: str


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def load_charuco_target(
    path: Path,
) -> tuple[Any, CharucoDetectorAdapter, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
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
    n_corners = 0 if objpoints is None else int(len(objpoints))
    if (
        objpoints is None
        or imgpoints is None
        or n_corners < MIN_CHARUCO_CORNERS
    ):
        return PoseDetection(
            ok=False,
            T=None,
            n_points=n_corners,
            message=(
                f"{label}: ChArUco corners={n_corners} "
                f"need>={MIN_CHARUCO_CORNERS}"
            ),
            vis=vis,
        )

    try:
        ok, rvec, tvec = cv2.solvePnP(
            objpoints,
            imgpoints,
            intr.K,
            intr.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=n_corners,
            message=f"{label}: solvePnP error: {exc.err}",
            vis=vis,
        )
    if not ok:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=n_corners,
            message=f"{label}: solvePnP failed",
            vis=vis,
        )
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            objpoints,
            imgpoints,
            intr.K,
            intr.dist,
            rvec,
            tvec,
        )
    except cv2.error:
        pass

    projected, _ = cv2.projectPoints(
        objpoints, rvec, tvec, intr.K, intr.dist
    )
    reproj_error = float(
        np.mean(
            np.linalg.norm(
                imgpoints.reshape(-1, 2) - projected.reshape(-1, 2),
                axis=1,
            )
        )
    )
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T_cv2_charuco = make_T(R, np.asarray(tvec, dtype=np.float64).reshape(3))
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
        T=T_cv2_charuco,
        n_points=n_corners,
        reproj_error=reproj_error,
        message=(
            f"{label}: ChArUco ok corners={n_corners} "
            f"err={reproj_error:.2f}px"
        ),
        vis=vis,
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_rotvec()


def transform_delta(T_ref: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    delta = inv_T(T_ref) @ T
    rot_deg = float(np.degrees(np.linalg.norm(so3_log(delta[:3, :3]))))
    trans_m = float(np.linalg.norm(delta[:3, 3]))
    return rot_deg, trans_m


def transform_to_params(T: np.ndarray) -> np.ndarray:
    return np.hstack(
        [Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]
    )


def params_to_transform(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64).reshape(6)
    return make_T(Rotation.from_rotvec(params[:3]).as_matrix(), params[3:])


def _wahba(rotations_src: list[np.ndarray], rotations_tgt: list[np.ndarray]) -> np.ndarray:
    H = np.zeros((3, 3), dtype=np.float64)
    for R_src, R_tgt in zip(rotations_src, rotations_tgt):
        H += R_tgt @ R_src.T
    U, _S, Vt = np.linalg.svd(H)
    R_out = U @ Vt
    if np.linalg.det(R_out) < 0:
        U[:, -1] *= -1
        R_out = U @ Vt
    return R_out


def _solve_Y_given_X(
    left_list: list[np.ndarray],
    right_list: list[np.ndarray],
    X: np.ndarray,
) -> np.ndarray:
    # Solve left_i X ~= Y right_i.
    R_targets = [left[:3, :3] @ X[:3, :3] for left in left_list]
    R_sources = [right[:3, :3] for right in right_list]
    R_Y = _wahba(R_sources, R_targets)
    translations = [
        left[:3, :3] @ X[:3, 3]
        + left[:3, 3]
        - R_Y @ right[:3, 3]
        for left, right in zip(left_list, right_list)
    ]
    return make_T(R_Y, np.mean(translations, axis=0))


def _solve_X_given_Y(
    left_list: list[np.ndarray],
    right_list: list[np.ndarray],
    Y: np.ndarray,
) -> np.ndarray:
    # Solve left_i X ~= Y right_i.
    R_sources = [left[:3, :3] for left in left_list]
    R_targets = [Y[:3, :3] @ right[:3, :3] for right in right_list]
    R_X = _wahba(R_sources, R_targets)
    M = np.vstack([left[:3, :3] for left in left_list])
    v = np.hstack(
        [
            Y[:3, :3] @ right[:3, 3]
            + Y[:3, 3]
            - left[:3, 3]
            for left, right in zip(left_list, right_list)
        ]
    )
    t_X, *_ = np.linalg.lstsq(M, v, rcond=None)
    return make_T(R_X, t_X)


def initialize_joint_solution(
    T_d435_cube_list: list[np.ndarray],
    T_cv2_charuco_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    # Measurement closure:
    #   T_cv2_charuco_i * T_charuco_d435 * T_d435_cube_i = T_cv2_cube
    # Map it to left_i X = Y right_i with right_i=inv(T_d435_cube_i).
    left_list = T_cv2_charuco_list
    right_list = [inv_T(T) for T in T_d435_cube_list]
    X = np.eye(4, dtype=np.float64)
    Y = _solve_Y_given_X(left_list, right_list, X)
    for _ in range(8):
        X = _solve_X_given_Y(left_list, right_list, Y)
        Y = _solve_Y_given_X(left_list, right_list, X)
    return X, Y


def joint_residual_vector(
    params: np.ndarray,
    T_d435_cube_list: list[np.ndarray],
    T_cv2_charuco_list: list[np.ndarray],
    normalized: bool = True,
) -> np.ndarray:
    X_charuco_d435 = params_to_transform(params[:6])
    Y_cv2_cube = params_to_transform(params[6:])
    rot_scale = np.radians(SOLVER_ROT_SCALE_DEG) if normalized else 1.0
    trans_scale = SOLVER_TRANS_SCALE_M if normalized else 1.0
    residuals = []
    for T_d435_cube, T_cv2_charuco in zip(
        T_d435_cube_list, T_cv2_charuco_list
    ):
        closure = (
            inv_T(Y_cv2_cube)
            @ T_cv2_charuco
            @ X_charuco_d435
            @ T_d435_cube
        )
        residuals.extend(so3_log(closure[:3, :3]) / rot_scale)
        residuals.extend(closure[:3, 3] / trans_scale)
    return np.asarray(residuals, dtype=np.float64)


def _run_joint_least_squares(
    params0: np.ndarray,
    T_d435_cube_list: list[np.ndarray],
    T_cv2_charuco_list: list[np.ndarray],
):
    return least_squares(
        joint_residual_vector,
        params0,
        args=(T_d435_cube_list, T_cv2_charuco_list, True),
        loss="huber",
        f_scale=1.0,
        max_nfev=1000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )


def _multistart_subsets(num_samples: int) -> list[tuple[str, list[int]]]:
    all_indices = list(range(num_samples))
    candidates: list[tuple[str, list[int]]] = [("full", all_indices)]
    for fraction_name, fraction in (("third", 1.0 / 3.0), ("half", 0.5)):
        window = max(MIN_SAMPLES_TO_SOLVE, int(round(num_samples * fraction)))
        if window >= num_samples:
            continue
        starts = (0, (num_samples - window) // 2, num_samples - window)
        for start in starts:
            indices = list(range(start, start + window))
            candidates.append((f"{fraction_name}_{start}_{start + window}", indices))

    unique: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for label, indices in candidates:
        key = tuple(indices)
        if key not in seen:
            unique.append((label, indices))
            seen.add(key)
    return unique


def solve_once(samples: list[CalibrationSample]) -> dict:
    T_d435_cube_list = [s.T_d435_cube for s in samples]
    T_cv2_charuco_list = [s.T_cv2_charuco for s in samples]
    candidate_results = []
    for label, indices in _multistart_subsets(len(samples)):
        subset_d435 = [T_d435_cube_list[i] for i in indices]
        subset_charuco = [T_cv2_charuco_list[i] for i in indices]
        X_init, Y_init = initialize_joint_solution(
            subset_d435, subset_charuco
        )
        params0 = np.hstack(
            [transform_to_params(X_init), transform_to_params(Y_init)]
        )
        if len(indices) != len(samples):
            subset_result = _run_joint_least_squares(
                params0, subset_d435, subset_charuco
            )
            params0 = subset_result.x
        result = _run_joint_least_squares(
            params0, T_d435_cube_list, T_cv2_charuco_list
        )
        candidate_results.append((label, result))

    selected_label, result = min(
        candidate_results, key=lambda item: float(item[1].cost)
    )
    X_charuco_d435 = params_to_transform(result.x[:6])
    Y_cv2_cube = params_to_transform(result.x[6:])

    per_sample = []
    for sample in samples:
        closure = (
            inv_T(Y_cv2_cube)
            @ sample.T_cv2_charuco
            @ X_charuco_d435
            @ sample.T_d435_cube
        )
        per_sample.append(
            {
                "index": int(sample.index),
                "rot_deg": float(
                    np.degrees(np.linalg.norm(so3_log(closure[:3, :3])))
                ),
                "trans_m": float(np.linalg.norm(closure[:3, 3])),
            }
        )

    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    positive = singular_values[singular_values > 1e-10]
    condition = float(positive[0] / positive[-1]) if positive.size else float("inf")
    return {
        "T_charuco_d435": X_charuco_d435,
        "T_cv2_cube": Y_cv2_cube,
        "T_cube_cv2": inv_T(Y_cv2_cube),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "optimizer_num_starts": len(candidate_results),
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
    values: list[float], minimum: float, maximum: float
) -> float:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    data_limit = median + OUTLIER_MAD_MULTIPLIER * 1.4826 * mad
    return float(np.clip(data_limit, minimum, maximum))


def residual_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def solve_with_outlier_rejection(samples: list[CalibrationSample]) -> dict:
    active = list(samples)
    iterations = []
    for iteration in range(OUTLIER_MAX_ITERATIONS):
        solution = solve_once(active)
        residuals = solution["per_sample_residuals"]
        rot_limit = robust_limit(
            [r["rot_deg"] for r in residuals],
            OUTLIER_MIN_ROT_DEG,
            OUTLIER_MAX_ROT_DEG,
        )
        trans_limit = robust_limit(
            [r["trans_m"] for r in residuals],
            OUTLIER_MIN_TRANS_M,
            OUTLIER_MAX_TRANS_M,
        )
        kept_indices = {
            r["index"]
            for r in residuals
            if r["rot_deg"] <= rot_limit and r["trans_m"] <= trans_limit
        }
        next_active = [s for s in active if s.index in kept_indices]
        iterations.append(
            {
                "iteration": iteration,
                "input_count": len(active),
                "output_count": len(next_active),
                "rot_limit_deg": rot_limit,
                "trans_limit_m": trans_limit,
                "rejected_indices": [
                    s.index for s in active if s.index not in kept_indices
                ],
            }
        )
        if len(next_active) < MIN_SAMPLES_TO_SOLVE or len(next_active) == len(active):
            break
        active = next_active

    solution = solve_once(active)
    inlier_indices = [s.index for s in active]
    inlier_set = set(inlier_indices)
    solution["inlier_indices"] = inlier_indices
    solution["outlier_indices"] = [
        s.index for s in samples if s.index not in inlier_set
    ]
    solution["outlier_rejection_iterations"] = iterations
    rot_values = [r["rot_deg"] for r in solution["per_sample_residuals"]]
    trans_values = [r["trans_m"] for r in solution["per_sample_residuals"]]
    solution["residual_rot_deg"] = residual_stats(rot_values)
    solution["residual_trans_m"] = residual_stats(trans_values)
    return solution


def ensure_aprilcube_on_path() -> None:
    src = str(APRILCUBE_SRC_DIR.expanduser().resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def install_legacy_aruco_compatibility() -> None:
    """Expose the OpenCV 4.7 ArUco interface on the project's OpenCV 4.5.

    AprilCube uses ``DetectorParameters()`` and ``ArucoDetector.detectMarkers``.
    OpenCV 4.5 provides the same detector through the older procedural API.
    This adapter is local to the Python process and does not patch AprilCube.
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable; install opencv-contrib-python")
    if not hasattr(cv2.aruco, "DetectorParameters"):
        cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters_create
    if not hasattr(cv2.aruco, "ArucoDetector"):
        class LegacyArucoDetector:
            def __init__(self, dictionary, parameters) -> None:
                self.dictionary = dictionary
                self.parameters = parameters

            def detectMarkers(self, image):
                return cv2.aruco.detectMarkers(
                    image,
                    self.dictionary,
                    parameters=self.parameters,
                )

        cv2.aruco.ArucoDetector = LegacyArucoDetector


def create_aprilcube_context(intr: Intrinsics) -> AprilCubeDetectionContext:
    ensure_aprilcube_on_path()
    install_legacy_aruco_compatibility()
    import aprilcube  # type: ignore[import-not-found]

    detector = aprilcube.detector(
        APRILCUBE_CONFIG,
        intrinsic_cfg={
            "fx": float(intr.K[0, 0]),
            "fy": float(intr.K[1, 1]),
            "cx": float(intr.K[0, 2]),
            "cy": float(intr.K[1, 2]),
        },
        dist_coeffs=intr.dist,
        enable_filter=False,
        fast=False,
    )
    face_id_sets = {
        str(face): {int(tag_id) for tag_id in ids}
        for face, ids in detector.face_id_sets.items()
    }
    multi_tag_faces = {
        face for face, ids in face_id_sets.items() if len(ids) > 1
    }
    return AprilCubeDetectionContext(
        detector=detector,
        face_id_sets=face_id_sets,
        tag_corner_map_mm={
            int(tag_id): np.asarray(corners, dtype=np.float64).reshape(4, 3)
            for tag_id, corners in detector.tag_corner_map.items()
        },
        multi_tag_faces=multi_tag_faces,
    )


def validate_configuration(
    d435_intr: Intrinsics,
    cv2_intr: Intrinsics,
) -> None:
    required_paths = (
        D435_INTRINSICS_YAML,
        CV2_INTRINSICS_YAML,
        APRILCUBE_CONFIG,
        CHARUCO_BOARD_YAML,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required calibration files: {missing}")
    if d435_intr.image_size != (D435_WIDTH, D435_HEIGHT):
        raise ValueError(
            f"D435 intrinsics are {d435_intr.image_size}, expected "
            f"{D435_WIDTH}x{D435_HEIGHT}"
        )
    if d435_intr.camera_model != "pinhole" or d435_intr.dist.size != 5:
        raise ValueError(
            "D435 calibration must be pinhole with 5 OpenCV distortion coefficients; "
            f"got model={d435_intr.camera_model}, dist={d435_intr.dist.size}"
        )
    if cv2_intr.camera_model != "pinhole" or cv2_intr.dist.size != 5:
        raise ValueError(
            "middle_finger_cam calibration must be pinhole with 5 OpenCV "
            "distortion coefficients; "
            f"got model={cv2_intr.camera_model}, dist={cv2_intr.dist.size}"
        )


def select_synchronized_pair(
    d435_frames: list[TimedFrame],
    cv2_frames: list[TimedFrame],
    last_pair: Optional[tuple[int, int]],
) -> Optional[tuple[TimedFrame, TimedFrame, float]]:
    if not d435_frames or not cv2_frames:
        return None
    last_d435, last_cv2 = last_pair if last_pair is not None else (-1, -1)
    candidates = []
    for d435_frame in d435_frames:
        if d435_frame.index <= last_d435:
            continue
        for cv2_frame in cv2_frames:
            if cv2_frame.index <= last_cv2:
                continue
            skew = abs(d435_frame.timestamp - cv2_frame.timestamp)
            if skew <= MAX_PAIR_SKEW_S:
                candidates.append((d435_frame, cv2_frame, skew))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            min(item[0].timestamp, item[1].timestamp),
            -item[2],
        ),
    )


def detection_quality(
    cube_det: PoseDetection,
    charuco_det: PoseDetection,
    pair_skew_s: float,
) -> tuple[bool, str]:
    if pair_skew_s > MAX_PAIR_SKEW_S:
        return False, f"pair skew {pair_skew_s * 1000.0:.1f}ms"
    if not cube_det.ok or cube_det.T is None:
        return False, cube_det.message
    if cube_det.n_points < MIN_APRILCUBE_TAGS:
        return False, f"cube tags {cube_det.n_points} < {MIN_APRILCUBE_TAGS}"
    if cube_det.reproj_error > MAX_APRILCUBE_REPROJ_PX:
        return False, f"cube reproj {cube_det.reproj_error:.2f}px"
    if not charuco_det.ok or charuco_det.T is None:
        return False, charuco_det.message
    if charuco_det.n_points < MIN_CHARUCO_CORNERS:
        return False, (
            f"ChArUco corners {charuco_det.n_points} < {MIN_CHARUCO_CORNERS}"
        )
    if charuco_det.reproj_error > MAX_CHARUCO_REPROJ_PX:
        return False, f"ChArUco reproj {charuco_det.reproj_error:.2f}px"
    return True, "detections valid"


def is_stable_pair(
    previous: Optional[tuple[np.ndarray, np.ndarray]],
    T_d435_cube: np.ndarray,
    T_cv2_charuco: np.ndarray,
) -> tuple[bool, str]:
    if previous is None:
        return False, "building stability history"
    cube_rot, cube_trans = transform_delta(previous[0], T_d435_cube)
    charuco_rot, charuco_trans = transform_delta(
        previous[1], T_cv2_charuco
    )
    stable = (
        cube_rot <= STABLE_MAX_ROT_DELTA_DEG
        and charuco_rot <= STABLE_MAX_ROT_DELTA_DEG
        and cube_trans <= STABLE_MAX_TRANS_DELTA_M
        and charuco_trans <= STABLE_MAX_TRANS_DELTA_M
    )
    reason = (
        f"motion cube={cube_rot:.2f}deg/{cube_trans * 1000.0:.1f}mm "
        f"charuco={charuco_rot:.2f}deg/{charuco_trans * 1000.0:.1f}mm"
    )
    return stable, reason


def is_diverse_from_last(
    samples: list[CalibrationSample], T_d435_cube: np.ndarray
) -> tuple[bool, str]:
    if not samples:
        return True, "first pose"
    rot_deg, trans_m = transform_delta(samples[-1].T_d435_cube, T_d435_cube)
    ok = (
        rot_deg >= MIN_SAMPLE_ROT_DELTA_DEG
        or trans_m >= MIN_SAMPLE_TRANS_DELTA_M
    )
    return ok, f"diversity={rot_deg:.2f}deg/{trans_m * 1000.0:.1f}mm"


def create_sample_dir() -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    path = SAMPLE_IMAGE_ROOT / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def store_sample(
    samples: list[CalibrationSample],
    sample_dir: Path,
    d435_frame: TimedFrame,
    cv2_frame: TimedFrame,
    pair_skew_s: float,
    cube_det: PoseDetection,
    charuco_det: PoseDetection,
    capture_mode: str,
) -> CalibrationSample:
    assert cube_det.T is not None and charuco_det.T is not None
    index = len(samples)
    d435_path = sample_dir / f"sample_{index:04d}_d435_cube.png"
    cv2_path = sample_dir / f"sample_{index:04d}_middle_finger_charuco.png"
    if not cv2.imwrite(str(d435_path), d435_frame.frame_bgr):
        raise RuntimeError(f"Failed to save {d435_path}")
    if not cv2.imwrite(str(cv2_path), cv2_frame.frame_bgr):
        raise RuntimeError(f"Failed to save {cv2_path}")
    sample = CalibrationSample(
        index=index,
        timestamp=0.5 * (d435_frame.timestamp + cv2_frame.timestamp),
        pair_skew_s=float(pair_skew_s),
        d435_frame_index=d435_frame.index,
        cv2_frame_index=cv2_frame.index,
        T_d435_cube=cube_det.T.copy(),
        T_cv2_charuco=charuco_det.T.copy(),
        cube_tags=int(cube_det.n_points),
        cube_reproj_error_px=float(cube_det.reproj_error),
        charuco_corners=int(charuco_det.n_points),
        charuco_reproj_error_px=float(charuco_det.reproj_error),
        d435_image_path=str(d435_path),
        cv2_image_path=str(cv2_path),
        capture_mode=capture_mode,
    )
    samples.append(sample)
    return sample


def put_lines(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    for row, line in enumerate(lines):
        y = 28 + row * 28
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def append_timestamp(path: Path) -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def sample_to_dict(sample: CalibrationSample) -> dict:
    return {
        "index": int(sample.index),
        "timestamp": float(sample.timestamp),
        "pair_skew_s": float(sample.pair_skew_s),
        "d435_frame_index": int(sample.d435_frame_index),
        "cv2_frame_index": int(sample.cv2_frame_index),
        "T_d435_cube": sample.T_d435_cube.tolist(),
        "T_middle_finger_cam_charuco": sample.T_cv2_charuco.tolist(),
        "cube_tags": int(sample.cube_tags),
        "cube_reproj_error_px": float(sample.cube_reproj_error_px),
        "charuco_corners": int(sample.charuco_corners),
        "charuco_reproj_error_px": float(sample.charuco_reproj_error_px),
        "d435_image_path": sample.d435_image_path,
        "cv2_image_path": sample.cv2_image_path,
        "capture_mode": sample.capture_mode,
    }


def serialize_solution(solution: dict) -> dict:
    return {
        "T_charuco_d435": solution["T_charuco_d435"].tolist(),
        "T_d435_charuco": inv_T(solution["T_charuco_d435"]).tolist(),
        "T_middle_finger_cam_cube": solution["T_cv2_cube"].tolist(),
        "T_cube_middle_finger_cam": solution["T_cube_cv2"].tolist(),
        "T_cv2_cube": solution["T_cv2_cube"].tolist(),
        "T_cube_cv2": solution["T_cube_cv2"].tolist(),
        "compatibility_aliases": {
            "T_cv2_cube": "T_middle_finger_cam_cube",
            "T_cube_cv2": "T_cube_middle_finger_cam",
        },
        "requested_output": {
            "name": "T_cube_middle_finger_cam",
            "meaning": (
                "middle_finger_cam optical frame pose/offset expressed in "
                "AprilCube frame"
            ),
            "units": "meters",
        },
        "optimizer_success": solution["optimizer_success"],
        "optimizer_message": solution["optimizer_message"],
        "optimizer_nfev": solution["optimizer_nfev"],
        "optimizer_num_starts": solution["optimizer_num_starts"],
        "optimizer_selected_start": solution["optimizer_selected_start"],
        "optimizer_candidate_costs": solution["optimizer_candidate_costs"],
        "jacobian_rank": solution["jacobian_rank"],
        "jacobian_condition": solution["jacobian_condition"],
        "jacobian_singular_values": solution["jacobian_singular_values"],
        "residual_rot_deg": solution["residual_rot_deg"],
        "residual_trans_m": solution["residual_trans_m"],
        "inlier_indices": solution["inlier_indices"],
        "outlier_indices": solution["outlier_indices"],
        "outlier_rejection_iterations": solution[
            "outlier_rejection_iterations"
        ],
        "per_sample_residuals": solution["per_sample_residuals"],
    }


def save_results(
    output_path: Path,
    samples: list[CalibrationSample],
    solution: dict,
    sample_dir: Path,
    cv2_device: int | str,
    d435_serial: str,
    cv2_port: str,
) -> Path:
    output_path = append_timestamp(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": "robot_cam_calib.d435_cube_middle_finger_charuco.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_convention": (
            "T_A_B maps coordinates from frame B into frame A; translation is meters"
        ),
        "measurement_equation": (
            "T_middle_finger_cam_charuco_i @ T_charuco_d435 @ "
            "T_d435_cube_i = T_middle_finger_cam_cube"
        ),
        "frames": {
            "d435": "RealSense D435 color optical frame",
            "middle_finger_cam": "middle_finger_cam optical frame",
            "cube": "AprilCube config object frame",
            "charuco": "scale-0.25 ChArUco YAML board frame",
        },
        "inputs": {
            "d435_serial": d435_serial,
            "d435_intrinsics_yaml": str(D435_INTRINSICS_YAML.resolve()),
            "cv2_port": cv2_port,
            "cv2_camera_name": CV2_CAMERA_NAME,
            "cv2_active_device": str(cv2_device),
            "cv2_intrinsics_yaml": str(CV2_INTRINSICS_YAML.resolve()),
            "aprilcube_config": str(APRILCUBE_CONFIG.resolve()),
            "charuco_board_yaml": str(CHARUCO_BOARD_YAML.resolve()),
        },
        "capture": {
            "sample_image_dir": str(sample_dir),
            "num_raw_samples": len(samples),
            "software_timestamp_pairing": True,
            "max_pair_skew_s": float(MAX_PAIR_SKEW_S),
            "stable_required_pairs": int(STABLE_REQUIRED_PAIRS),
            "stable_max_rot_delta_deg": float(STABLE_MAX_ROT_DELTA_DEG),
            "stable_max_trans_delta_m": float(STABLE_MAX_TRANS_DELTA_M),
            "min_sample_rot_delta_deg": float(MIN_SAMPLE_ROT_DELTA_DEG),
            "min_sample_trans_delta_m": float(MIN_SAMPLE_TRANS_DELTA_M),
            "min_charuco_corners": int(MIN_CHARUCO_CORNERS),
            "max_charuco_reproj_px": float(MAX_CHARUCO_REPROJ_PX),
            "warning": (
                "The cameras are not hardware-synchronized. Auto-capture only stores "
                "stable poses; do not use samples captured during continuous motion."
            ),
        },
        "solution": serialize_solution(solution),
        "samples": [sample_to_dict(sample) for sample in samples],
    }
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return output_path


def run_self_test() -> None:
    rng = np.random.default_rng(20260712)
    X_true = make_T(
        Rotation.from_rotvec(np.radians([15.0, -20.0, 8.0])).as_matrix(),
        [0.08, -0.03, 0.12],
    )
    Y_true = make_T(
        Rotation.from_rotvec(np.radians([-10.0, 12.0, 25.0])).as_matrix(),
        [-0.04, 0.06, 0.09],
    )
    samples = []
    for index in range(40):
        A = make_T(
            Rotation.from_rotvec(rng.normal(0.0, np.radians(25.0), 3)).as_matrix(),
            rng.uniform(-0.25, 0.25, 3) + np.array([0.0, 0.0, 0.6]),
        )
        B = Y_true @ inv_T(A) @ inv_T(X_true)
        noise_A = make_T(
            Rotation.from_rotvec(rng.normal(0.0, np.radians(0.15), 3)).as_matrix(),
            rng.normal(0.0, 0.0008, 3),
        )
        noise_B = make_T(
            Rotation.from_rotvec(rng.normal(0.0, np.radians(0.15), 3)).as_matrix(),
            rng.normal(0.0, 0.0008, 3),
        )
        samples.append(
            CalibrationSample(
                index=index,
                timestamp=float(index),
                pair_skew_s=0.0,
                d435_frame_index=index,
                cv2_frame_index=index,
                T_d435_cube=noise_A @ A,
                T_cv2_charuco=noise_B @ B,
                cube_tags=4,
                cube_reproj_error_px=0.5,
                charuco_corners=20,
                charuco_reproj_error_px=0.5,
                d435_image_path="",
                cv2_image_path="",
                capture_mode="synthetic",
            )
        )

    solved = solve_with_outlier_rejection(samples)
    x_rot, x_trans = transform_delta(X_true, solved["T_charuco_d435"])
    y_rot, y_trans = transform_delta(Y_true, solved["T_cv2_cube"])
    q_rot, q_trans = transform_delta(inv_T(Y_true), solved["T_cube_cv2"])
    print(
        f"[SELF-TEST] T_charuco_d435 error={x_rot:.4f}deg/"
        f"{x_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] T_cv2_cube error={y_rot:.4f}deg/"
        f"{y_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] requested T_cube_cv2 error={q_rot:.4f}deg/"
        f"{q_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] jacobian rank={solved['jacobian_rank']} "
        f"condition={solved['jacobian_condition']:.3e}"
    )
    if x_rot > 0.5 or x_trans > 0.005 or y_rot > 0.5 or y_trans > 0.005:
        raise AssertionError("Synthetic recovery exceeded 0.5deg/5mm tolerance")


def main(args: argparse.Namespace) -> None:
    d435_intr = load_intrinsics(D435_INTRINSICS_YAML)
    cv2_intr = load_intrinsics(CV2_INTRINSICS_YAML)
    validate_configuration(d435_intr, cv2_intr)
    board, charuco_detector, charuco_config = load_charuco_target(
        CHARUCO_BOARD_YAML
    )
    cube_context = create_aprilcube_context(d435_intr)

    print("[INFO] Coordinate equation:")
    print(
        "  T_middle_finger_cam_charuco @ T_charuco_d435 @ "
        "T_d435_cube = T_middle_finger_cam_cube"
    )
    print(
        "[INFO] Requested output: T_cube_middle_finger_cam = "
        "inv(T_middle_finger_cam_cube)"
    )
    print(f"[INFO] D435 intrinsics={d435_intr.path} model={d435_intr.camera_model}")
    print(
        f"[INFO] {CV2_CAMERA_NAME} intrinsics={cv2_intr.path} "
        f"model={cv2_intr.camera_model}"
    )
    print(f"[INFO] AprilCube config={APRILCUBE_CONFIG}")
    print(f"[INFO] ChArUco board={CHARUCO_BOARD_YAML.resolve()}")
    print(f"[INFO] ChArUco config={charuco_config}")
    print(
        f"[INFO] ChArUco quality: corners>={MIN_CHARUCO_CORNERS}, "
        f"reprojection<={MAX_CHARUCO_REPROJ_PX:.2f}px"
    )

    if args.check_config:
        print("[INFO] Configuration and detectors loaded successfully.")
        return

    d435_camera = RealsenseCamera(
        serial=args.d435_serial,
        width=D435_WIDTH,
        height=D435_HEIGHT,
        fps=D435_FPS,
        format=D435_FORMAT,
    )
    d435_camera.start()
    try:
        cv2_cap, cv2_device = start_capture(
            args.cv2_port,
            cv2_intr.image_size[0],
            cv2_intr.image_size[1],
            CV2_FPS,
            CV2_FOURCC,
        )
    except Exception:
        d435_camera.stop()
        raise
    cv2_actual_size = (
        int(round(cv2_cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(cv2_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    if cv2_actual_size != cv2_intr.image_size:
        cv2_cap.release()
        d435_camera.stop()
        raise RuntimeError(
            f"CV2 camera opened at {cv2_actual_size}, but its intrinsics require "
            f"{cv2_intr.image_size}. Refusing to calibrate with mismatched geometry."
        )

    def read_d435_bgr() -> np.ndarray:
        return cv2.cvtColor(d435_camera.read(), cv2.COLOR_RGB2BGR)

    def read_cv2_bgr() -> np.ndarray:
        ok, frame = cv2_cap.read()
        if not ok or frame is None:
            raise RuntimeError("CV2 camera read failed")
        return frame

    d435_worker = FrameWorker("d435-capture", read_d435_bgr)
    cv2_worker = FrameWorker("cv2-capture", read_cv2_bgr)
    d435_worker.start()
    cv2_worker.start()

    sample_dir = create_sample_dir()
    samples: list[CalibrationSample] = []
    last_pair: Optional[tuple[int, int]] = None
    previous_valid_poses: Optional[tuple[np.ndarray, np.ndarray]] = None
    stable_count = 0
    last_capture_time = -float("inf")
    last_status = "waiting for synchronized frames"

    print(
        f"[INFO] D435 serial={args.d435_serial} "
        f"{D435_WIDTH}x{D435_HEIGHT}@{D435_FPS}"
    )
    print(
        f"[INFO] {CV2_CAMERA_NAME} port={args.cv2_port}, "
        f"active_device={cv2_device}, "
        f"requested={cv2_intr.image_size[0]}x{cv2_intr.image_size[1]}@{CV2_FPS}"
    )
    print(f"[INFO] Samples will be saved under {sample_dir}")
    print(
        f"[INFO] Move the cube+{CV2_CAMERA_NAME} rigid assembly to a new pose, "
        "then hold it still briefly. Auto-capture stores only stable, diverse poses."
    )
    print("[INFO] [s] manual store  [c] clear  [q/esc] solve and save")

    try:
        stop_requested = False
        while not stop_requested:
            pair = select_synchronized_pair(
                d435_worker.snapshot(), cv2_worker.snapshot(), last_pair
            )
            if pair is None:
                if d435_worker.last_error or cv2_worker.last_error:
                    last_status = (
                        f"camera error d435={d435_worker.last_error} "
                        f"cv2={cv2_worker.last_error}"
                    )
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            d435_frame, cv2_frame, pair_skew_s = pair
            last_pair = (d435_frame.index, cv2_frame.index)
            cube_det = detect_aprilcube_pose(
                d435_frame.frame_bgr, cube_context, d435_intr
            )
            charuco_det = detect_charuco_pose(
                cv2_frame.frame_bgr,
                charuco_detector,
                board,
                cv2_intr,
                f"{CV2_CAMERA_NAME}/ChArUco",
            )
            quality_ok, quality_reason = detection_quality(
                cube_det, charuco_det, pair_skew_s
            )

            stable = False
            stable_reason = "detections invalid"
            if (
                quality_ok
                and cube_det.T is not None
                and charuco_det.T is not None
            ):
                stable, stable_reason = is_stable_pair(
                    previous_valid_poses, cube_det.T, charuco_det.T
                )
                stable_count = stable_count + 1 if stable else 0
                previous_valid_poses = (
                    cube_det.T.copy(),
                    charuco_det.T.copy(),
                )
            else:
                stable_count = 0
                previous_valid_poses = None

            diverse = False
            diversity_reason = "waiting for valid pose"
            if quality_ok and cube_det.T is not None:
                diverse, diversity_reason = is_diverse_from_last(
                    samples, cube_det.T
                )

            now = time.monotonic()
            auto_ok = (
                AUTO_CAPTURE
                and quality_ok
                and stable_count >= STABLE_REQUIRED_PAIRS
                and diverse
                and now - last_capture_time >= AUTO_CAPTURE_COOLDOWN_S
            )
            auto_stored_this_pair = False
            if auto_ok:
                sample = store_sample(
                    samples,
                    sample_dir,
                    d435_frame,
                    cv2_frame,
                    pair_skew_s,
                    cube_det,
                    charuco_det,
                    "auto",
                )
                last_capture_time = now
                stable_count = 0
                auto_stored_this_pair = True
                last_status = f"auto stored sample {len(samples)}"
                print(
                    f"[INFO] {last_status}: skew={sample.pair_skew_s * 1000.0:.1f}ms "
                    f"cube={sample.cube_tags}tags/{sample.cube_reproj_error_px:.2f}px "
                    f"charuco={sample.charuco_corners}corners/"
                    f"{sample.charuco_reproj_error_px:.2f}px"
                )
                if len(samples) >= args.max_samples:
                    print(f"[INFO] Reached {len(samples)} samples; solving.")
                    stop_requested = True
            else:
                last_status = (
                    f"{quality_reason}; stable={stable_count}/{STABLE_REQUIRED_PAIRS} "
                    f"{stable_reason}; {diversity_reason}"
                )

            status_lines = [
                f"samples={len(samples)}/{args.max_samples} "
                f"pair_skew={pair_skew_s * 1000.0:.1f}ms",
                last_status,
                cube_det.message,
                charuco_det.message,
                "Move to a diverse pose, then hold still | [s] store [c] clear [q] solve",
            ]
            d435_vis = put_lines(
                cube_det.vis
                if cube_det.vis is not None
                else d435_frame.frame_bgr,
                status_lines,
            )
            cv2_vis = put_lines(
                charuco_det.vis
                if charuco_det.vis is not None
                else cv2_frame.frame_bgr,
                status_lines,
            )
            cv2.imshow(
                "D435 / AprilCube",
                resize_for_display(d435_vis, DISPLAY_SCALE_D435),
            )
            cv2.imshow(
                f"{CV2_CAMERA_NAME} pinhole / ChArUco",
                resize_for_display(cv2_vis, DISPLAY_SCALE_CV2),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                for sample in samples:
                    for image_path in (
                        sample.d435_image_path,
                        sample.cv2_image_path,
                    ):
                        Path(image_path).unlink(missing_ok=True)
                samples.clear()
                previous_valid_poses = None
                stable_count = 0
                last_capture_time = -float("inf")
                print("[INFO] Cleared samples.")
            elif key == ord("s"):
                if auto_stored_this_pair:
                    print("[INFO] Manual store skipped; auto already stored this pair.")
                elif not quality_ok:
                    print(f"[WARN] Manual sample rejected: {quality_reason}")
                else:
                    sample = store_sample(
                        samples,
                        sample_dir,
                        d435_frame,
                        cv2_frame,
                        pair_skew_s,
                        cube_det,
                        charuco_det,
                        "manual",
                    )
                    last_capture_time = now
                    stable_count = 0
                    print(
                        f"[INFO] Manually stored sample {len(samples)} "
                        f"skew={sample.pair_skew_s * 1000.0:.1f}ms"
                    )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; solving collected samples.")
    finally:
        d435_worker.stop()
        cv2_worker.stop()
        cv2_cap.release()
        d435_camera.stop()
        cv2.destroyAllWindows()

    if len(samples) < MIN_SAMPLES_TO_SOLVE:
        print(
            f"[WARN] Only {len(samples)} samples; need at least "
            f"{MIN_SAMPLES_TO_SOLVE}. No extrinsics YAML saved."
        )
        return

    solution = solve_with_outlier_rejection(samples)
    if solution["jacobian_rank"] < 12:
        print(
            f"[WARN] Solver Jacobian rank={solution['jacobian_rank']} < 12; "
            "pose excitation is degenerate. Result will still be saved with warning."
        )
    output_path = save_results(
        args.output,
        samples,
        solution,
        sample_dir,
        cv2_device,
        args.d435_serial,
        args.cv2_port,
    )
    print(f"[INFO] Saved {output_path}")
    print(
        "[RESULT] T_cube_middle_finger_cam "
        "(middle_finger_cam frame offset expressed in cube frame):"
    )
    print(solution["T_cube_cv2"])
    print("[DIAGNOSTICS]")
    print(f"  inliers={len(solution['inlier_indices'])}/{len(samples)}")
    print(f"  outliers={solution['outlier_indices']}")
    print(f"  rotation residual deg={solution['residual_rot_deg']}")
    print(f"  translation residual m={solution['residual_trans_m']}")
    print(
        f"  jacobian rank={solution['jacobian_rank']} "
        f"condition={solution['jacobian_condition']:.3e}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly calibrate fixed D435<->ChArUco and "
            "AprilCube<->middle_finger_cam transforms from paired observations."
        )
    )
    parser.add_argument("--d435-serial", default=D435_SERIAL)
    parser.add_argument("--cv2-port", default=CV2_PORT)
    parser.add_argument(
        "--max-samples", type=int, default=AUTO_STOP_SAMPLE_COUNT
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Load intrinsics, target layouts, and detectors without opening cameras.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a deterministic synthetic recovery test without cameras.",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    if cli_args.self_test:
        run_self_test()
    else:
        main(cli_args)
