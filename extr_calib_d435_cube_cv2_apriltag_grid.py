from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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
    load_intrinsics,
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
from robot_cam_calib.io import append_timestamp


# ---------------------------- User macros ---------------------------- #
THIRD_VIEW_CAMERA_NAME = "third_view_cam"
THIRD_VIEW_PORT = "3-6:1.0"
THIRD_VIEW_INTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/"
    "intrinsics_charuco_offline_eval_0721_235144/"
    "intrinsics_None_charuco_2592x1944_0721_235457_"
    "offline_object_release.yaml"
)
THIRD_VIEW_FPS = 50
THIRD_VIEW_FOURCC = "MJPG"

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

THUMB_WEB_CAMERA_NAME = "thumb_web_cam"
THUMB_WEB_PORT = "3-10:1.0"
THUMB_WEB_INTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/"
    "intrinsics_thumb_web_cam_fisheye_charuco_2592x1944_0708_020331.yaml"
)
THUMB_WEB_FPS = 50
THUMB_WEB_FOURCC = "MJPG"

APRILCUBE_SRC_DIR = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/src"
)
APRILCUBE_CONFIG = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/"
    "cube_april_36h11_100_123_2x2x2_outer62p5mm/config.json"
)
CHARUCO_BOARD_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/charuco_a4_0712_223646/"
    "charuco_7x5_40mm_marker30mm_DICT_5X5_50_"
    "A4_landscape_600dpi.yaml"
)

OUTPUT_PATH = Path(
    "outputs/extrinsics_thumb_web_cam_cube_third_view_cam_"
    "charuco_40mm.yaml"
)
SAMPLE_IMAGE_ROOT = Path(
    "outputs/extrinsics_thumb_web_cam_cube_third_view_cam_"
    "charuco_40mm_samples"
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

# These residual scales balance radians and meters inside the robust joint
# SE(3) solve; they are not claimed sensor covariances.
SOLVER_ROT_SCALE_DEG = 3.0
SOLVER_TRANS_SCALE_M = 0.010
OUTLIER_MIN_ROT_DEG = 2.0
OUTLIER_MAX_ROT_DEG = 10.0
OUTLIER_MIN_TRANS_M = 0.010
OUTLIER_MAX_TRANS_M = 0.050
OUTLIER_MAD_MULTIPLIER = 3.0
OUTLIER_MAX_ITERATIONS = 5

DISPLAY_SCALE_THIRD_VIEW = 0.35
DISPLAY_SCALE_THUMB_WEB = 0.35


class FrameWorker(BufferedFrameWorker):
    """Compatibility wrapper retaining this script's buffer size."""

    def __init__(self, name: str, read_fn: Callable[[], np.ndarray]) -> None:
        super().__init__(
            name,
            read_fn,
            buffer_size=FRAME_BUFFER_SIZE,
            stop_timeout_s=2.0,
        )


@dataclass
class CalibrationSample:
    index: int
    timestamp: float
    pair_skew_s: float
    third_view_frame_index: int
    thumb_web_frame_index: int
    T_third_view_cube: np.ndarray
    T_thumb_web_charuco: np.ndarray
    cube_tags: int
    cube_reproj_error_px: float
    charuco_corners: int
    charuco_reproj_error_px: float
    third_view_image_path: str
    thumb_web_image_path: str
    capture_mode: str


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
    assert rvec is not None and tvec is not None

    projected = project_points_for_intrinsics(
        objpoints,
        rvec,
        tvec,
        intr.K,
        intr.dist,
        intr.camera_model,
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
    T_thumb_web_charuco = make_T(
        R, np.asarray(tvec, dtype=np.float64).reshape(3)
    )
    try:
        if intr.camera_model == "fisheye":
            axis = np.float64(
                [
                    [0.0, 0.0, 0.0],
                    [CHARUCO_AXIS_LENGTH_M, 0.0, 0.0],
                    [0.0, CHARUCO_AXIS_LENGTH_M, 0.0],
                    [0.0, 0.0, CHARUCO_AXIS_LENGTH_M],
                ]
            )
            axis_points = project_points_for_intrinsics(
                axis,
                rvec,
                tvec,
                intr.K,
                intr.dist,
                intr.camera_model,
            ).reshape(-1, 2)
            origin = tuple(np.round(axis_points[0]).astype(int))
            cv2.line(
                vis,
                origin,
                tuple(np.round(axis_points[1]).astype(int)),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                vis,
                origin,
                tuple(np.round(axis_points[2]).astype(int)),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                vis,
                origin,
                tuple(np.round(axis_points[3]).astype(int)),
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        else:
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
        T=T_thumb_web_charuco,
        n_points=n_corners,
        reproj_error=reproj_error,
        message=(
            f"{label}: ChArUco ok corners={n_corners} "
            f"err={reproj_error:.2f}px"
        ),
        vis=vis,
    )


def initialize_joint_solution(
    T_third_view_cube_list: list[np.ndarray],
    T_thumb_web_charuco_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    # Measurement closure:
    #   T_thumb_web_charuco_i * T_charuco_third_view
    #       * T_third_view_cube_i = T_thumb_web_cube
    # Map it to left_i X = Y right_i with
    # right_i=inv(T_third_view_cube_i).
    left_list = T_thumb_web_charuco_list
    right_list = [inv_T(T) for T in T_third_view_cube_list]
    X = np.eye(4, dtype=np.float64)
    Y = _solve_Y_given_X(left_list, right_list, X)
    for _ in range(8):
        X = _solve_X_given_Y(left_list, right_list, Y)
        Y = _solve_Y_given_X(left_list, right_list, X)
    return X, Y


def joint_residual_vector(
    params: np.ndarray,
    T_third_view_cube_list: list[np.ndarray],
    T_thumb_web_charuco_list: list[np.ndarray],
    normalized: bool = True,
) -> np.ndarray:
    X_charuco_third_view = params_to_transform(params[:6])
    Y_thumb_web_cube = params_to_transform(params[6:])
    rot_scale = np.radians(SOLVER_ROT_SCALE_DEG) if normalized else 1.0
    trans_scale = SOLVER_TRANS_SCALE_M if normalized else 1.0
    residuals = []
    for T_third_view_cube, T_thumb_web_charuco in zip(
        T_third_view_cube_list, T_thumb_web_charuco_list
    ):
        closure = (
            inv_T(Y_thumb_web_cube)
            @ T_thumb_web_charuco
            @ X_charuco_third_view
            @ T_third_view_cube
        )
        residuals.extend(so3_log(closure[:3, :3]) / rot_scale)
        residuals.extend(closure[:3, 3] / trans_scale)
    return np.asarray(residuals, dtype=np.float64)


def _run_joint_least_squares(
    params0: np.ndarray,
    T_third_view_cube_list: list[np.ndarray],
    T_thumb_web_charuco_list: list[np.ndarray],
):
    return least_squares(
        joint_residual_vector,
        params0,
        args=(
            T_third_view_cube_list,
            T_thumb_web_charuco_list,
            True,
        ),
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
    T_third_view_cube_list = [s.T_third_view_cube for s in samples]
    T_thumb_web_charuco_list = [
        s.T_thumb_web_charuco for s in samples
    ]
    candidate_results = []
    for label, indices in _multistart_subsets(len(samples)):
        subset_third_view = [T_third_view_cube_list[i] for i in indices]
        subset_charuco = [
            T_thumb_web_charuco_list[i] for i in indices
        ]
        X_init, Y_init = initialize_joint_solution(
            subset_third_view, subset_charuco
        )
        params0 = np.hstack(
            [transform_to_params(X_init), transform_to_params(Y_init)]
        )
        if len(indices) != len(samples):
            subset_result = _run_joint_least_squares(
                params0, subset_third_view, subset_charuco
            )
            params0 = subset_result.x
        result = _run_joint_least_squares(
            params0,
            T_third_view_cube_list,
            T_thumb_web_charuco_list,
        )
        candidate_results.append((label, result))

    selected_label, result = min(
        candidate_results, key=lambda item: float(item[1].cost)
    )
    X_charuco_third_view = params_to_transform(result.x[:6])
    Y_thumb_web_cube = params_to_transform(result.x[6:])

    per_sample = []
    for sample in samples:
        closure = (
            inv_T(Y_thumb_web_cube)
            @ sample.T_thumb_web_charuco
            @ X_charuco_third_view
            @ sample.T_third_view_cube
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
        "T_charuco_third_view_cam": X_charuco_third_view,
        "T_thumb_web_cam_cube": Y_thumb_web_cube,
        "T_cube_thumb_web_cam": inv_T(Y_thumb_web_cube),
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
    return mad_robust_limit(
        values,
        minimum,
        maximum,
        multiplier=OUTLIER_MAD_MULTIPLIER,
    )


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
    third_view_intr: Intrinsics,
    thumb_web_intr: Intrinsics,
) -> None:
    required_paths = (
        THIRD_VIEW_INTRINSICS_YAML,
        THUMB_WEB_INTRINSICS_YAML,
        APRILCUBE_CONFIG,
        CHARUCO_BOARD_YAML,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required calibration files: {missing}")
    if (
        third_view_intr.camera_model != "pinhole"
        or third_view_intr.dist.size != 5
    ):
        raise ValueError(
            "third_view_cam calibration must be pinhole with 5 OpenCV "
            "distortion coefficients; "
            f"got model={third_view_intr.camera_model}, "
            f"dist={third_view_intr.dist.size}"
        )
    valid_thumb_web_model = (
        thumb_web_intr.camera_model == "pinhole"
        and thumb_web_intr.dist.size == 5
    ) or (
        thumb_web_intr.camera_model == "fisheye"
        and thumb_web_intr.dist.size == 4
    )
    if not valid_thumb_web_model:
        raise ValueError(
            "thumb_web_cam calibration must be pinhole/5-dist or "
            "fisheye/4-dist; "
            f"got model={thumb_web_intr.camera_model}, "
            f"dist={thumb_web_intr.dist.size}"
        )


def select_synchronized_pair(
    third_view_frames: list[TimedFrame],
    thumb_web_frames: list[TimedFrame],
    last_pair: Optional[tuple[int, int]],
) -> Optional[tuple[TimedFrame, TimedFrame, float]]:
    return select_frame_pair(
        third_view_frames,
        thumb_web_frames,
        last_pair,
        MAX_PAIR_SKEW_S,
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
    T_third_view_cube: np.ndarray,
    T_thumb_web_charuco: np.ndarray,
) -> tuple[bool, str]:
    if previous is None:
        return False, "building stability history"
    cube_rot, cube_trans = transform_delta(previous[0], T_third_view_cube)
    charuco_rot, charuco_trans = transform_delta(
        previous[1], T_thumb_web_charuco
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
    samples: list[CalibrationSample], T_third_view_cube: np.ndarray
) -> tuple[bool, str]:
    if not samples:
        return True, "first pose"
    rot_deg, trans_m = transform_delta(
        samples[-1].T_third_view_cube, T_third_view_cube
    )
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
    third_view_frame: TimedFrame,
    thumb_web_frame: TimedFrame,
    pair_skew_s: float,
    cube_det: PoseDetection,
    charuco_det: PoseDetection,
    capture_mode: str,
) -> CalibrationSample:
    assert cube_det.T is not None and charuco_det.T is not None
    index = len(samples)
    third_view_path = sample_dir / f"sample_{index:04d}_third_view_cube.png"
    thumb_web_path = (
        sample_dir / f"sample_{index:04d}_thumb_web_charuco.png"
    )
    if not cv2.imwrite(str(third_view_path), third_view_frame.frame_bgr):
        raise RuntimeError(f"Failed to save {third_view_path}")
    if not cv2.imwrite(str(thumb_web_path), thumb_web_frame.frame_bgr):
        raise RuntimeError(f"Failed to save {thumb_web_path}")
    sample = CalibrationSample(
        index=index,
        timestamp=0.5
        * (third_view_frame.timestamp + thumb_web_frame.timestamp),
        pair_skew_s=float(pair_skew_s),
        third_view_frame_index=third_view_frame.index,
        thumb_web_frame_index=thumb_web_frame.index,
        T_third_view_cube=cube_det.T.copy(),
        T_thumb_web_charuco=charuco_det.T.copy(),
        cube_tags=int(cube_det.n_points),
        cube_reproj_error_px=float(cube_det.reproj_error),
        charuco_corners=int(charuco_det.n_points),
        charuco_reproj_error_px=float(charuco_det.reproj_error),
        third_view_image_path=str(third_view_path),
        thumb_web_image_path=str(thumb_web_path),
        capture_mode=capture_mode,
    )
    samples.append(sample)
    return sample


def sample_to_dict(sample: CalibrationSample) -> dict:
    return {
        "index": int(sample.index),
        "timestamp": float(sample.timestamp),
        "pair_skew_s": float(sample.pair_skew_s),
        "third_view_frame_index": int(sample.third_view_frame_index),
        "thumb_web_frame_index": int(sample.thumb_web_frame_index),
        "T_third_view_cam_cube": sample.T_third_view_cube.tolist(),
        "T_thumb_web_cam_charuco": (
            sample.T_thumb_web_charuco.tolist()
        ),
        "cube_tags": int(sample.cube_tags),
        "cube_reproj_error_px": float(sample.cube_reproj_error_px),
        "charuco_corners": int(sample.charuco_corners),
        "charuco_reproj_error_px": float(sample.charuco_reproj_error_px),
        "third_view_image_path": sample.third_view_image_path,
        "thumb_web_image_path": sample.thumb_web_image_path,
        "capture_mode": sample.capture_mode,
    }


def serialize_solution(solution: dict) -> dict:
    return {
        "T_charuco_third_view_cam": solution[
            "T_charuco_third_view_cam"
        ].tolist(),
        "T_third_view_cam_charuco": inv_T(
            solution["T_charuco_third_view_cam"]
        ).tolist(),
        "T_thumb_web_cam_cube": solution[
            "T_thumb_web_cam_cube"
        ].tolist(),
        "T_cube_thumb_web_cam": solution[
            "T_cube_thumb_web_cam"
        ].tolist(),
        "requested_output": {
            "name": "T_cube_thumb_web_cam",
            "meaning": (
                "thumb_web_cam optical frame pose/offset expressed in "
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
    third_view_device: int | str,
    thumb_web_device: int | str,
    third_view_port: str,
    thumb_web_port: str,
) -> Path:
    output_path = append_timestamp(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": (
            "robot_cam_calib.third_view_cube_thumb_web_charuco.v1"
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_convention": (
            "T_A_B maps coordinates from frame B into frame A; translation is meters"
        ),
        "measurement_equation": (
            "T_thumb_web_cam_charuco_i @ T_charuco_third_view_cam @ "
            "T_third_view_cam_cube_i = T_thumb_web_cam_cube"
        ),
        "frames": {
            "third_view_cam": "third_view_cam optical frame",
            "thumb_web_cam": "thumb_web_cam optical frame",
            "cube": "AprilCube config object frame",
            "charuco": "40 mm-square ChArUco YAML board frame",
        },
        "inputs": {
            "third_view_port": third_view_port,
            "third_view_camera_name": THIRD_VIEW_CAMERA_NAME,
            "third_view_active_device": str(third_view_device),
            "third_view_intrinsics_yaml": str(
                THIRD_VIEW_INTRINSICS_YAML.resolve()
            ),
            "thumb_web_port": thumb_web_port,
            "thumb_web_camera_name": THUMB_WEB_CAMERA_NAME,
            "thumb_web_active_device": str(thumb_web_device),
            "thumb_web_intrinsics_yaml": str(
                THUMB_WEB_INTRINSICS_YAML.resolve()
            ),
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
                third_view_frame_index=index,
                thumb_web_frame_index=index,
                T_third_view_cube=noise_A @ A,
                T_thumb_web_charuco=noise_B @ B,
                cube_tags=4,
                cube_reproj_error_px=0.5,
                charuco_corners=20,
                charuco_reproj_error_px=0.5,
                third_view_image_path="",
                thumb_web_image_path="",
                capture_mode="synthetic",
            )
        )

    solved = solve_with_outlier_rejection(samples)
    x_rot, x_trans = transform_delta(
        X_true, solved["T_charuco_third_view_cam"]
    )
    y_rot, y_trans = transform_delta(
        Y_true, solved["T_thumb_web_cam_cube"]
    )
    q_rot, q_trans = transform_delta(
        inv_T(Y_true), solved["T_cube_thumb_web_cam"]
    )
    print(
        f"[SELF-TEST] T_charuco_third_view_cam error={x_rot:.4f}deg/"
        f"{x_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] T_thumb_web_cam_cube error={y_rot:.4f}deg/"
        f"{y_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] requested T_cube_thumb_web_cam "
        f"error={q_rot:.4f}deg/"
        f"{q_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] jacobian rank={solved['jacobian_rank']} "
        f"condition={solved['jacobian_condition']:.3e}"
    )
    if x_rot > 0.5 or x_trans > 0.005 or y_rot > 0.5 or y_trans > 0.005:
        raise AssertionError("Synthetic recovery exceeded 0.5deg/5mm tolerance")


def main(args: argparse.Namespace) -> None:
    global APRILCUBE_CONFIG, CHARUCO_BOARD_YAML, SAMPLE_IMAGE_ROOT
    global THIRD_VIEW_INTRINSICS_YAML, THUMB_WEB_INTRINSICS_YAML
    THIRD_VIEW_INTRINSICS_YAML = Path(args.third_view_intrinsics)
    THUMB_WEB_INTRINSICS_YAML = Path(args.thumb_web_intrinsics)
    APRILCUBE_CONFIG = Path(args.aprilcube_config)
    CHARUCO_BOARD_YAML = Path(args.charuco_board)
    SAMPLE_IMAGE_ROOT = Path(args.sample_root)

    third_view_intr = load_intrinsics(THIRD_VIEW_INTRINSICS_YAML)
    thumb_web_intr = load_intrinsics(THUMB_WEB_INTRINSICS_YAML)
    validate_configuration(third_view_intr, thumb_web_intr)
    board, charuco_detector, charuco_config = load_charuco_target(
        CHARUCO_BOARD_YAML
    )
    cube_context = create_aprilcube_context(third_view_intr)

    print("[INFO] Coordinate equation:")
    print(
        "  T_thumb_web_cam_charuco @ T_charuco_third_view_cam @ "
        "T_third_view_cam_cube = T_thumb_web_cam_cube"
    )
    print(
        "[INFO] Requested output: T_cube_thumb_web_cam = "
        "inv(T_thumb_web_cam_cube)"
    )
    print(
        f"[INFO] {THIRD_VIEW_CAMERA_NAME} intrinsics={third_view_intr.path} "
        f"model={third_view_intr.camera_model}"
    )
    print(
        f"[INFO] {THUMB_WEB_CAMERA_NAME} "
        f"intrinsics={thumb_web_intr.path} "
        f"model={thumb_web_intr.camera_model}"
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

    third_view_cap, third_view_device = start_capture(
        args.third_view_port,
        third_view_intr.image_size[0],
        third_view_intr.image_size[1],
        THIRD_VIEW_FPS,
        THIRD_VIEW_FOURCC,
    )
    try:
        thumb_web_cap, thumb_web_device = start_capture(
            args.thumb_web_port,
            thumb_web_intr.image_size[0],
            thumb_web_intr.image_size[1],
            THUMB_WEB_FPS,
            THUMB_WEB_FOURCC,
        )
    except Exception:
        third_view_cap.release()
        raise

    third_view_actual_size = (
        int(round(third_view_cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(third_view_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    thumb_web_actual_size = (
        int(round(thumb_web_cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(thumb_web_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    if third_view_actual_size != third_view_intr.image_size:
        thumb_web_cap.release()
        third_view_cap.release()
        raise RuntimeError(
            f"{THIRD_VIEW_CAMERA_NAME} opened at {third_view_actual_size}, "
            f"but its intrinsics require {third_view_intr.image_size}. "
            "Refusing to calibrate with mismatched geometry."
        )
    if thumb_web_actual_size != thumb_web_intr.image_size:
        thumb_web_cap.release()
        third_view_cap.release()
        raise RuntimeError(
            f"{THUMB_WEB_CAMERA_NAME} opened at "
            f"{thumb_web_actual_size}, but its intrinsics require "
            f"{thumb_web_intr.image_size}. Refusing to calibrate with "
            "mismatched geometry."
        )

    def read_third_view_bgr() -> np.ndarray:
        ok, frame = third_view_cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"{THIRD_VIEW_CAMERA_NAME} read failed")
        return frame

    def read_thumb_web_bgr() -> np.ndarray:
        ok, frame = thumb_web_cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"{THUMB_WEB_CAMERA_NAME} read failed")
        return frame

    third_view_worker = FrameWorker(
        "third-view-capture", read_third_view_bgr
    )
    thumb_web_worker = FrameWorker(
        "thumb-web-capture", read_thumb_web_bgr
    )
    third_view_worker.start()
    thumb_web_worker.start()

    sample_dir = create_sample_dir()
    samples: list[CalibrationSample] = []
    last_pair: Optional[tuple[int, int]] = None
    previous_valid_poses: Optional[tuple[np.ndarray, np.ndarray]] = None
    stable_count = 0
    last_capture_time = -float("inf")
    last_status = "waiting for synchronized frames"

    print(
        f"[INFO] {THIRD_VIEW_CAMERA_NAME} port={args.third_view_port}, "
        f"active_device={third_view_device}, "
        f"requested={third_view_intr.image_size[0]}x"
        f"{third_view_intr.image_size[1]}@{THIRD_VIEW_FPS}"
    )
    print(
        f"[INFO] {THUMB_WEB_CAMERA_NAME} "
        f"port={args.thumb_web_port}, "
        f"active_device={thumb_web_device}, "
        f"requested={thumb_web_intr.image_size[0]}x"
        f"{thumb_web_intr.image_size[1]}@{THUMB_WEB_FPS}"
    )
    print(f"[INFO] Samples will be saved under {sample_dir}")
    print(
        f"[INFO] Move the cube+{THUMB_WEB_CAMERA_NAME} rigid assembly "
        "to a new pose, then hold it still briefly. Auto-capture stores only "
        "stable, diverse poses."
    )
    print("[INFO] [s] manual store  [c] clear  [q/esc] solve and save")

    try:
        stop_requested = False
        while not stop_requested:
            pair = select_synchronized_pair(
                third_view_worker.snapshot(),
                thumb_web_worker.snapshot(),
                last_pair,
            )
            if pair is None:
                if (
                    third_view_worker.last_error
                    or thumb_web_worker.last_error
                ):
                    last_status = (
                        "camera error "
                        f"third_view={third_view_worker.last_error} "
                        f"thumb_web={thumb_web_worker.last_error}"
                    )
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            third_view_frame, thumb_web_frame, pair_skew_s = pair
            last_pair = (third_view_frame.index, thumb_web_frame.index)
            cube_det = detect_aprilcube_pose(
                third_view_frame.frame_bgr,
                cube_context,
                third_view_intr,
            )
            charuco_det = detect_charuco_pose(
                thumb_web_frame.frame_bgr,
                charuco_detector,
                board,
                thumb_web_intr,
                f"{THUMB_WEB_CAMERA_NAME}/ChArUco",
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
                    third_view_frame,
                    thumb_web_frame,
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
            third_view_vis = put_lines(
                cube_det.vis
                if cube_det.vis is not None
                else third_view_frame.frame_bgr,
                status_lines,
            )
            thumb_web_vis = put_lines(
                charuco_det.vis
                if charuco_det.vis is not None
                else thumb_web_frame.frame_bgr,
                status_lines,
            )
            cv2.imshow(
                f"{THIRD_VIEW_CAMERA_NAME} / AprilCube",
                resize_for_display(
                    third_view_vis, DISPLAY_SCALE_THIRD_VIEW
                ),
            )
            cv2.imshow(
                f"{THUMB_WEB_CAMERA_NAME} "
                f"{thumb_web_intr.camera_model} / ChArUco",
                resize_for_display(
                    thumb_web_vis, DISPLAY_SCALE_THUMB_WEB
                ),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                for sample in samples:
                    for image_path in (
                        sample.third_view_image_path,
                        sample.thumb_web_image_path,
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
                        third_view_frame,
                        thumb_web_frame,
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
        third_view_worker.stop()
        thumb_web_worker.stop()
        thumb_web_cap.release()
        third_view_cap.release()
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
        third_view_device,
        thumb_web_device,
        args.third_view_port,
        args.thumb_web_port,
    )
    print(f"[INFO] Saved {output_path}")
    print(
        "[RESULT] T_cube_thumb_web_cam "
        "(thumb_web_cam frame offset expressed in cube frame):"
    )
    print(solution["T_cube_thumb_web_cam"])
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
            "Jointly calibrate fixed third_view_cam<->ChArUco and "
            "AprilCube<->thumb_web_cam transforms from paired observations."
        )
    )
    parser.add_argument("--third-view-port", default=THIRD_VIEW_PORT)
    parser.add_argument("--thumb-web-port", default=THUMB_WEB_PORT)
    parser.add_argument(
        "--third-view-intrinsics",
        type=Path,
        default=THIRD_VIEW_INTRINSICS_YAML,
    )
    parser.add_argument(
        "--thumb-web-intrinsics",
        type=Path,
        default=THUMB_WEB_INTRINSICS_YAML,
    )
    parser.add_argument(
        "--aprilcube-config",
        type=Path,
        default=APRILCUBE_CONFIG,
    )
    parser.add_argument(
        "--charuco-board",
        type=Path,
        default=CHARUCO_BOARD_YAML,
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=SAMPLE_IMAGE_ROOT,
    )
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
