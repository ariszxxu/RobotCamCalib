#!/usr/bin/env python3
"""Calibrate two camera-to-grid mounting offsets from mutual observations.

Physical arrangement
--------------------
* ``third_view_cam`` is fixed and carries ``third_grid``.
* ``thumb_web_cam`` moves and carries ``thumb_grid``.
* third_view_cam observes thumb_grid.
* thumb_web_cam observes third_grid.

Both grids use the same AprilTag layout, but they are two different physical
targets.  With ``T_A_B`` mapping coordinates from B into A, every synchronized
sample obeys the closed loop

    T_third_grid_third_view_cam
    @ T_third_view_cam_thumb_grid_i
    @ T_thumb_grid_thumb_web_cam
    @ T_thumb_web_cam_third_grid_i
    = I.

The program previews both cameras, automatically stores stable and sufficiently
different poses, robustly solves the two fixed camera-to-grid transforms, and
writes a timestamped YAML result plus the accepted image pairs.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot_cam_calib.targets import (
    AprilTagGridBoard,
    AprilTagGridDetector,
    Intrinsics,
    PoseDetection,
    detect_apriltag_grid_pose,
    load_apriltag_grid_board,
    load_intrinsics,
)
from intr_calib_charuco import start_capture
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
    solve_x_given_y as _solve_U_given_V,
    solve_y_given_x as _solve_V_given_U,
    transform_delta,
    transform_to_params,
)


REPO_ROOT = Path(__file__).resolve().parent

THIRD_VIEW_CAMERA_NAME = "third_view_cam"
THIRD_VIEW_PORT = "3-6:1.0"
THIRD_VIEW_INTRINSICS_YAML = Path(
    "/home/ps/project/ConSensV2Lab/image2cube_pose/assets/intrinsics/"
    "intrinsics_None_charuco_2592x1944_0721_235457_"
    "offline_object_release.yaml"
)

THUMB_WEB_CAMERA_NAME = "thumb_web_cam"
THUMB_WEB_PORT = "3-8:1.0"
THUMB_WEB_INTRINSICS_YAML = REPO_ROOT / (
    "outputs/intrinsics_None_fisheye_charuco_2592x1944_0722_180005.yaml"
)

APRILTAG_GRID_YAML = REPO_ROOT / (
    "outputs/tiny_physical_optics_frame_offset/"
    "tiny_physical_optics_frame_offset.yaml"
)

CAMERA_FPS = 50
CAMERA_FOURCC = "MJPG"
FRAME_BUFFER_SIZE = 20
MAX_PAIR_SKEW_S = 0.030

# Automatic sampling is deliberately restricted to briefly stationary poses.
# The two UVC cameras do not share a hardware clock.
# One valid pose establishes the reference; two subsequent stable detections
# trigger capture.  At 5 MP, dual AprilTag detection is much slower than the
# camera FPS, so requiring four stable comparisons makes the UI feel stalled.
STABLE_REQUIRED_PAIRS = 2
STABLE_MAX_ROT_DELTA_DEG = 1.5
STABLE_MAX_TRANS_DELTA_M = 0.004
AUTO_CAPTURE_COOLDOWN_S = 0.7
MIN_SAMPLE_ROT_DELTA_DEG = 5.0
MIN_SAMPLE_TRANS_DELTA_M = 0.015

MIN_TAGS_PER_CAMERA = 4
MIN_CORNERS_PER_CAMERA = MIN_TAGS_PER_CAMERA * 4
MAX_THIRD_VIEW_REPROJ_PX = 2.0
MAX_THUMB_WEB_REPROJ_PX = 3.0

MIN_SAMPLES_TO_SOLVE = 12
DEFAULT_MAX_SAMPLES = 50

# Rotation and translation residuals need comparable numerical scales.
SOLVER_ROT_SCALE_DEG = 2.0
SOLVER_TRANS_SCALE_M = 0.005
OUTLIER_MAD_MULTIPLIER = 3.0
OUTLIER_MIN_ROT_DEG = 0.5
OUTLIER_MAX_ROT_DEG = 8.0
OUTLIER_MIN_TRANS_M = 0.002
OUTLIER_MAX_TRANS_M = 0.030
OUTLIER_MAX_ITERATIONS = 5

DISPLAY_SCALE_THIRD_VIEW = 0.35
DISPLAY_SCALE_THUMB_WEB = 0.35
OUTPUT_ROOT = REPO_ROOT / "outputs/dual_camera_rigid_grid_offsets"


class FrameWorker(BufferedFrameWorker):
    """Compatibility wrapper retaining this workflow's buffer size."""

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
    # third_view_cam observes the grid rigidly attached to thumb_web_cam.
    T_third_view_cam_thumb_grid: np.ndarray
    # thumb_web_cam observes the grid rigidly attached to third_view_cam.
    T_thumb_web_cam_third_grid: np.ndarray
    third_view_corners: int
    third_view_reproj_error_px: float
    thumb_web_corners: int
    thumb_web_reproj_error_px: float
    third_view_image_path: str
    thumb_web_image_path: str
    capture_mode: str


def initialize_solution(
    T_third_view_cam_thumb_grid: list[np.ndarray],
    T_thumb_web_cam_third_grid: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize the two constant mounts with alternating closed-form solves.

    Let

    * U = T_thumb_grid_thumb_web_cam (requested)
    * V = T_third_view_cam_third_grid
    * M_i = T_third_view_cam_thumb_grid_i
    * N_i = T_thumb_web_cam_third_grid_i

    The loop equation is equivalent to ``M_i U = V inv(N_i)``.
    """
    left_list = T_third_view_cam_thumb_grid
    right_list = [inv_T(T) for T in T_thumb_web_cam_third_grid]
    U = np.eye(4, dtype=np.float64)
    V = _solve_V_given_U(left_list, right_list, U)
    for _ in range(10):
        U = _solve_U_given_V(left_list, right_list, V)
        V = _solve_V_given_U(left_list, right_list, U)
    return U, V


def joint_residual_vector(
    params: np.ndarray,
    T_third_view_cam_thumb_grid: list[np.ndarray],
    T_thumb_web_cam_third_grid: list[np.ndarray],
    normalized: bool = True,
) -> np.ndarray:
    U_thumb_grid_thumb_web_cam = params_to_transform(params[:6])
    V_third_view_cam_third_grid = params_to_transform(params[6:])
    rot_scale = np.radians(SOLVER_ROT_SCALE_DEG) if normalized else 1.0
    trans_scale = SOLVER_TRANS_SCALE_M if normalized else 1.0
    residuals: list[float] = []

    for M_third_view_cam_thumb_grid, N_thumb_web_cam_third_grid in zip(
        T_third_view_cam_thumb_grid,
        T_thumb_web_cam_third_grid,
    ):
        closure = (
            inv_T(V_third_view_cam_third_grid)
            @ M_third_view_cam_thumb_grid
            @ U_thumb_grid_thumb_web_cam
            @ N_thumb_web_cam_third_grid
        )
        residuals.extend(so3_log(closure[:3, :3]) / rot_scale)
        residuals.extend(closure[:3, 3] / trans_scale)
    return np.asarray(residuals, dtype=np.float64)


def _run_least_squares(
    params0: np.ndarray,
    M_list: list[np.ndarray],
    N_list: list[np.ndarray],
):
    return least_squares(
        joint_residual_vector,
        params0,
        args=(M_list, N_list, True),
        loss="huber",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=1500,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )


def _initialization_subsets(num_samples: int) -> list[tuple[str, list[int]]]:
    all_indices = list(range(num_samples))
    candidates: list[tuple[str, list[int]]] = [("full", all_indices)]
    window = max(MIN_SAMPLES_TO_SOLVE, num_samples // 2)
    if window < num_samples:
        for start in (0, (num_samples - window) // 2, num_samples - window):
            candidates.append(
                (f"window_{start}_{start + window}", list(range(start, start + window)))
            )
    unique: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for label, indices in candidates:
        key = tuple(indices)
        if key not in seen:
            unique.append((label, indices))
            seen.add(key)
    return unique


def solve_once(samples: list[CalibrationSample]) -> dict:
    M_list = [s.T_third_view_cam_thumb_grid for s in samples]
    N_list = [s.T_thumb_web_cam_third_grid for s in samples]
    candidates = []

    for label, indices in _initialization_subsets(len(samples)):
        M_subset = [M_list[i] for i in indices]
        N_subset = [N_list[i] for i in indices]
        U_init, V_init = initialize_solution(M_subset, N_subset)
        params0 = np.hstack(
            [transform_to_params(U_init), transform_to_params(V_init)]
        )
        if len(indices) != len(samples):
            subset_result = _run_least_squares(
                params0,
                M_subset,
                N_subset,
            )
            params0 = subset_result.x
        result = _run_least_squares(params0, M_list, N_list)
        candidates.append((label, result))

    selected_label, result = min(
        candidates, key=lambda item: float(item[1].cost)
    )
    U = params_to_transform(result.x[:6])
    V = params_to_transform(result.x[6:])
    T_third_grid_third_view_cam = inv_T(V)

    per_sample = []
    for sample in samples:
        closure = (
            T_third_grid_third_view_cam
            @ sample.T_third_view_cam_thumb_grid
            @ U
            @ sample.T_thumb_web_cam_third_grid
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
    condition = (
        float(positive[0] / positive[-1])
        if positive.size
        else float("inf")
    )
    return {
        "T_thumb_grid_thumb_web_cam": U,
        "T_thumb_web_cam_thumb_grid": inv_T(U),
        "T_third_grid_third_view_cam": T_third_grid_third_view_cam,
        "T_third_view_cam_third_grid": V,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "optimizer_cost": float(result.cost),
        "optimizer_num_starts": len(candidates),
        "optimizer_selected_start": selected_label,
        "optimizer_candidate_costs": {
            label: float(candidate.cost) for label, candidate in candidates
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
            if r["rot_deg"] <= rot_limit
            and r["trans_m"] <= trans_limit
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
        if (
            len(next_active) < MIN_SAMPLES_TO_SOLVE
            or len(next_active) == len(active)
        ):
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
    rot_values = [
        r["rot_deg"] for r in solution["per_sample_residuals"]
    ]
    trans_values = [
        r["trans_m"] for r in solution["per_sample_residuals"]
    ]
    solution["residual_rot_deg"] = residual_stats(rot_values)
    solution["residual_trans_m"] = residual_stats(trans_values)
    return solution


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
    third_det: PoseDetection,
    thumb_det: PoseDetection,
    pair_skew_s: float,
) -> tuple[bool, str]:
    if pair_skew_s > MAX_PAIR_SKEW_S:
        return False, f"pair skew {pair_skew_s * 1000.0:.1f}ms"
    if not third_det.ok or third_det.T is None:
        return False, third_det.message
    if third_det.n_points < MIN_CORNERS_PER_CAMERA:
        return False, (
            f"third corners {third_det.n_points} < {MIN_CORNERS_PER_CAMERA}"
        )
    if third_det.reproj_error > MAX_THIRD_VIEW_REPROJ_PX:
        return False, (
            f"third reproj {third_det.reproj_error:.2f}px > "
            f"{MAX_THIRD_VIEW_REPROJ_PX:.2f}px"
        )
    if not thumb_det.ok or thumb_det.T is None:
        return False, thumb_det.message
    if thumb_det.n_points < MIN_CORNERS_PER_CAMERA:
        return False, (
            f"thumb corners {thumb_det.n_points} < {MIN_CORNERS_PER_CAMERA}"
        )
    if thumb_det.reproj_error > MAX_THUMB_WEB_REPROJ_PX:
        return False, (
            f"thumb reproj {thumb_det.reproj_error:.2f}px > "
            f"{MAX_THUMB_WEB_REPROJ_PX:.2f}px"
        )
    return True, "detections valid"


def is_stable_pair(
    previous: Optional[tuple[np.ndarray, np.ndarray]],
    T_third_view_cam_thumb_grid: np.ndarray,
    T_thumb_web_cam_third_grid: np.ndarray,
) -> tuple[bool, str]:
    if previous is None:
        return False, "building stability history"
    third_rot, third_trans = transform_delta(
        previous[0], T_third_view_cam_thumb_grid
    )
    thumb_rot, thumb_trans = transform_delta(
        previous[1], T_thumb_web_cam_third_grid
    )
    stable = (
        third_rot <= STABLE_MAX_ROT_DELTA_DEG
        and thumb_rot <= STABLE_MAX_ROT_DELTA_DEG
        and third_trans <= STABLE_MAX_TRANS_DELTA_M
        and thumb_trans <= STABLE_MAX_TRANS_DELTA_M
    )
    reason = (
        f"motion third={third_rot:.2f}deg/{third_trans * 1000.0:.1f}mm "
        f"thumb={thumb_rot:.2f}deg/{thumb_trans * 1000.0:.1f}mm"
    )
    return stable, reason


def is_diverse_from_last(
    samples: list[CalibrationSample],
    T_third_view_cam_thumb_grid: np.ndarray,
) -> tuple[bool, str]:
    if not samples:
        return True, "first pose"
    rot_deg, trans_m = transform_delta(
        samples[-1].T_third_view_cam_thumb_grid,
        T_third_view_cam_thumb_grid,
    )
    diverse = (
        rot_deg >= MIN_SAMPLE_ROT_DELTA_DEG
        or trans_m >= MIN_SAMPLE_TRANS_DELTA_M
    )
    return (
        diverse,
        f"diversity={rot_deg:.2f}deg/{trans_m * 1000.0:.1f}mm",
    )


def create_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "images").mkdir()
    return run_dir


def store_sample(
    samples: list[CalibrationSample],
    run_dir: Path,
    third_frame: TimedFrame,
    thumb_frame: TimedFrame,
    pair_skew_s: float,
    third_det: PoseDetection,
    thumb_det: PoseDetection,
    capture_mode: str,
) -> CalibrationSample:
    assert third_det.T is not None and thumb_det.T is not None
    index = len(samples)
    image_dir = run_dir / "images"
    third_path = image_dir / f"sample_{index:04d}_third_sees_thumb_grid.jpg"
    thumb_path = image_dir / f"sample_{index:04d}_thumb_sees_third_grid.jpg"
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    if not cv2.imwrite(
        str(third_path), third_frame.frame_bgr, jpeg_params
    ):
        raise RuntimeError(f"Failed to save {third_path}")
    if not cv2.imwrite(
        str(thumb_path), thumb_frame.frame_bgr, jpeg_params
    ):
        raise RuntimeError(f"Failed to save {thumb_path}")

    sample = CalibrationSample(
        index=index,
        timestamp=0.5 * (third_frame.timestamp + thumb_frame.timestamp),
        pair_skew_s=float(pair_skew_s),
        third_view_frame_index=third_frame.index,
        thumb_web_frame_index=thumb_frame.index,
        T_third_view_cam_thumb_grid=third_det.T.copy(),
        T_thumb_web_cam_third_grid=thumb_det.T.copy(),
        third_view_corners=int(third_det.n_points),
        third_view_reproj_error_px=float(third_det.reproj_error),
        thumb_web_corners=int(thumb_det.n_points),
        thumb_web_reproj_error_px=float(thumb_det.reproj_error),
        third_view_image_path=str(third_path),
        thumb_web_image_path=str(thumb_path),
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
        "T_third_view_cam_thumb_grid": (
            sample.T_third_view_cam_thumb_grid.tolist()
        ),
        "T_thumb_web_cam_third_grid": (
            sample.T_thumb_web_cam_third_grid.tolist()
        ),
        "third_view_tags": int(sample.third_view_corners // 4),
        "third_view_corners": int(sample.third_view_corners),
        "third_view_reproj_error_px": float(
            sample.third_view_reproj_error_px
        ),
        "thumb_web_tags": int(sample.thumb_web_corners // 4),
        "thumb_web_corners": int(sample.thumb_web_corners),
        "thumb_web_reproj_error_px": float(
            sample.thumb_web_reproj_error_px
        ),
        "third_view_image_path": sample.third_view_image_path,
        "thumb_web_image_path": sample.thumb_web_image_path,
        "capture_mode": sample.capture_mode,
    }


def solution_to_dict(solution: dict) -> dict:
    T_thumb = solution["T_thumb_grid_thumb_web_cam"]
    T_third = solution["T_third_grid_third_view_cam"]
    return {
        "requested_transforms": {
            # These aliases match the names requested by the user.  Each
            # ``grid`` is the physical target attached to that camera.
            "T_grid_thumb_web_cam": T_thumb.tolist(),
            "T_grid_third_view_cam": T_third.tolist(),
        },
        "explicit_transforms": {
            "T_thumb_grid_thumb_web_cam": T_thumb.tolist(),
            "T_thumb_web_cam_thumb_grid": solution[
                "T_thumb_web_cam_thumb_grid"
            ].tolist(),
            "T_third_grid_third_view_cam": T_third.tolist(),
            "T_third_view_cam_third_grid": solution[
                "T_third_view_cam_third_grid"
            ].tolist(),
        },
        "optimizer_success": solution["optimizer_success"],
        "optimizer_message": solution["optimizer_message"],
        "optimizer_nfev": solution["optimizer_nfev"],
        "optimizer_cost": solution["optimizer_cost"],
        "optimizer_num_starts": solution["optimizer_num_starts"],
        "optimizer_selected_start": solution["optimizer_selected_start"],
        "optimizer_candidate_costs": solution[
            "optimizer_candidate_costs"
        ],
        "jacobian_rank": solution["jacobian_rank"],
        "jacobian_condition": solution["jacobian_condition"],
        "jacobian_singular_values": solution[
            "jacobian_singular_values"
        ],
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
    run_dir: Path,
    samples: list[CalibrationSample],
    solution: dict,
    board: AprilTagGridBoard,
    third_intr: Intrinsics,
    thumb_intr: Intrinsics,
    third_device: int | str,
    thumb_device: int | str,
    third_port: str,
    thumb_port: str,
) -> Path:
    output_path = run_dir / "dual_camera_rigid_grid_offsets.yaml"
    data = {
        "schema": "robot_cam_calib.dual_rigid_apriltag_grid_offsets.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_convention": (
            "T_A_B maps coordinates expressed in frame B into frame A; "
            "translations are metres"
        ),
        "physical_arrangement": {
            "third_view_cam": "fixed camera carrying third_grid",
            "thumb_web_cam": "moving camera carrying thumb_grid",
            "third_view_cam_observes": "thumb_grid",
            "thumb_web_cam_observes": "third_grid",
            "grids_are_distinct_physical_targets": True,
            "grids_share_the_same_layout": True,
        },
        "measurement_equation": (
            "T_third_grid_third_view_cam @ "
            "T_third_view_cam_thumb_grid_i @ "
            "T_thumb_grid_thumb_web_cam @ "
            "T_thumb_web_cam_third_grid_i = I"
        ),
        "frames": {
            "third_view_cam": (
                "third camera optical frame: +x image right, +y image down, "
                "+z forward"
            ),
            "thumb_web_cam": (
                "thumb camera optical frame: +x image right, +y image down, "
                "+z forward"
            ),
            "third_grid": (
                "50 mm board attached to third_view_cam; origin at complete "
                "outer-boundary center, +x print right, +y print down"
            ),
            "thumb_grid": (
                "50 mm board attached to thumb_web_cam; origin at complete "
                "outer-boundary center, +x print right, +y print down"
            ),
        },
        "inputs": {
            "apriltag_grid_yaml": str(board.path),
            "third_view_port": third_port,
            "third_view_active_device": str(third_device),
            "third_view_intrinsics_yaml": str(third_intr.path),
            "third_view_camera_model": third_intr.camera_model,
            "thumb_web_port": thumb_port,
            "thumb_web_active_device": str(thumb_device),
            "thumb_web_intrinsics_yaml": str(thumb_intr.path),
            "thumb_web_camera_model": thumb_intr.camera_model,
        },
        "capture": {
            "run_dir": str(run_dir),
            "image_dir": str(run_dir / "images"),
            "num_raw_samples": len(samples),
            "software_timestamp_pairing": True,
            "max_pair_skew_s": float(MAX_PAIR_SKEW_S),
            "stable_required_pairs": int(STABLE_REQUIRED_PAIRS),
            "stable_max_rot_delta_deg": float(
                STABLE_MAX_ROT_DELTA_DEG
            ),
            "stable_max_trans_delta_m": float(
                STABLE_MAX_TRANS_DELTA_M
            ),
            "min_sample_rot_delta_deg": float(
                MIN_SAMPLE_ROT_DELTA_DEG
            ),
            "min_sample_trans_delta_m": float(
                MIN_SAMPLE_TRANS_DELTA_M
            ),
            "warning": (
                "The cameras have no shared hardware clock. Automatic samples "
                "are accepted only after both observed poses are stable."
            ),
        },
        "solution": solution_to_dict(solution),
        "samples": [sample_to_dict(sample) for sample in samples],
    }
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    return output_path


def validate_configuration(
    third_intr: Intrinsics,
    thumb_intr: Intrinsics,
    board: AprilTagGridBoard,
) -> None:
    if third_intr.camera_model != "pinhole" or third_intr.dist.size != 5:
        raise ValueError(
            "third_view_cam must use pinhole/5-dist intrinsics; got "
            f"{third_intr.camera_model}/{third_intr.dist.size}"
        )
    if thumb_intr.camera_model != "fisheye" or thumb_intr.dist.size != 4:
        raise ValueError(
            "thumb_web_cam must use fisheye/4-dist intrinsics; got "
            f"{thumb_intr.camera_model}/{thumb_intr.dist.size}"
        )
    if third_intr.image_size != (2592, 1944):
        raise ValueError(
            f"Unexpected third_view image size {third_intr.image_size}"
        )
    if thumb_intr.image_size != (2592, 1944):
        raise ValueError(
            f"Unexpected thumb_web image size {thumb_intr.image_size}"
        )
    if board.tag_family != "DICT_APRILTAG_36h11":
        raise ValueError(
            f"Expected DICT_APRILTAG_36h11, got {board.tag_family}"
        )
    if len(board.tag_object_points) != 9:
        raise ValueError(
            f"Expected 9 tags in 3x3 board, got {len(board.tag_object_points)}"
        )
    if not np.allclose(
        [board.board_width_m, board.board_height_m],
        [0.05, 0.05],
    ):
        raise ValueError(
            "Expected a 50 x 50 mm board, got "
            f"{board.board_width_m} x {board.board_height_m} m"
        )


def run_self_test() -> None:
    rng = np.random.default_rng(20260723)
    U_true = make_T(
        Rotation.from_rotvec(
            np.radians([8.0, -14.0, 21.0])
        ).as_matrix(),
        [0.026, -0.018, 0.011],
    )
    V_true = make_T(
        Rotation.from_rotvec(
            np.radians([-12.0, 7.0, -18.0])
        ).as_matrix(),
        [-0.031, 0.022, 0.016],
    )
    samples: list[CalibrationSample] = []
    for index in range(45):
        M = make_T(
            Rotation.from_rotvec(
                rng.normal(0.0, np.radians(30.0), 3)
            ).as_matrix(),
            rng.uniform(-0.20, 0.20, 3) + np.array([0.0, 0.0, 0.45]),
        )
        N = inv_T(U_true) @ inv_T(M) @ V_true
        noise_M = make_T(
            Rotation.from_rotvec(
                rng.normal(0.0, np.radians(0.08), 3)
            ).as_matrix(),
            rng.normal(0.0, 0.0004, 3),
        )
        noise_N = make_T(
            Rotation.from_rotvec(
                rng.normal(0.0, np.radians(0.08), 3)
            ).as_matrix(),
            rng.normal(0.0, 0.0004, 3),
        )
        if index in {7, 24, 39}:
            noise_N = (
                make_T(
                    Rotation.from_rotvec(
                        np.radians([3.5, -2.5, 2.0])
                    ).as_matrix(),
                    [0.012, -0.009, 0.010],
                )
                @ noise_N
            )
        samples.append(
            CalibrationSample(
                index=index,
                timestamp=float(index),
                pair_skew_s=0.0,
                third_view_frame_index=index,
                thumb_web_frame_index=index,
                T_third_view_cam_thumb_grid=noise_M @ M,
                T_thumb_web_cam_third_grid=noise_N @ N,
                third_view_corners=36,
                third_view_reproj_error_px=0.4,
                thumb_web_corners=36,
                thumb_web_reproj_error_px=0.5,
                third_view_image_path="",
                thumb_web_image_path="",
                capture_mode="synthetic",
            )
        )

    solution = solve_with_outlier_rejection(samples)
    U_rot, U_trans = transform_delta(
        U_true, solution["T_thumb_grid_thumb_web_cam"]
    )
    X_rot, X_trans = transform_delta(
        inv_T(V_true),
        solution["T_third_grid_third_view_cam"],
    )
    print(
        "[SELF-TEST] T_thumb_grid_thumb_web_cam "
        f"error={U_rot:.4f}deg/{U_trans * 1000.0:.3f}mm"
    )
    print(
        "[SELF-TEST] T_third_grid_third_view_cam "
        f"error={X_rot:.4f}deg/{X_trans * 1000.0:.3f}mm"
    )
    print(
        f"[SELF-TEST] jacobian rank={solution['jacobian_rank']} "
        f"condition={solution['jacobian_condition']:.3e}"
    )
    print(
        f"[SELF-TEST] rejected synthetic outliers="
        f"{solution['outlier_indices']}"
    )
    if (
        U_rot > 0.3
        or U_trans > 0.003
        or X_rot > 0.3
        or X_trans > 0.003
    ):
        raise AssertionError(
            "Synthetic recovery exceeded 0.3 deg / 3 mm tolerance"
        )
    if not {7, 24, 39}.issubset(set(solution["outlier_indices"])):
        raise AssertionError("Synthetic outliers were not all rejected")


def main(args: argparse.Namespace) -> None:
    if args.self_test:
        run_self_test()
        return

    required_paths = (
        args.third_view_intrinsics,
        args.thumb_web_intrinsics,
        args.grid_yaml,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    third_intr = load_intrinsics(args.third_view_intrinsics)
    thumb_intr = load_intrinsics(args.thumb_web_intrinsics)
    board = load_apriltag_grid_board(args.grid_yaml)
    validate_configuration(third_intr, thumb_intr, board)
    third_detector = AprilTagGridDetector(board)
    thumb_detector = AprilTagGridDetector(board)

    print("[INFO] Mutual rigid-grid calibration")
    print(
        "  third_view_cam sees thumb_grid; "
        "thumb_web_cam sees third_grid"
    )
    print("[INFO] Closed-loop equation:")
    print(
        "  T_third_grid_third_view_cam @ "
        "T_third_view_cam_thumb_grid_i @ "
        "T_thumb_grid_thumb_web_cam @ "
        "T_thumb_web_cam_third_grid_i = I"
    )
    print(
        f"[INFO] third intrinsics={third_intr.path} "
        f"model={third_intr.camera_model}"
    )
    print(
        f"[INFO] thumb intrinsics={thumb_intr.path} "
        f"model={thumb_intr.camera_model}"
    )
    print(f"[INFO] grid={board.path}")

    if args.check_config:
        print("[INFO] Configuration and AprilTag detectors loaded successfully.")
        return

    third_cap, third_device = start_capture(
        args.third_view_port,
        third_intr.image_size[0],
        third_intr.image_size[1],
        CAMERA_FPS,
        CAMERA_FOURCC,
    )
    try:
        thumb_cap, thumb_device = start_capture(
            args.thumb_web_port,
            thumb_intr.image_size[0],
            thumb_intr.image_size[1],
            CAMERA_FPS,
            CAMERA_FOURCC,
        )
    except Exception:
        third_cap.release()
        raise

    third_actual_size = (
        int(round(third_cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(third_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    thumb_actual_size = (
        int(round(thumb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(thumb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    if third_actual_size != third_intr.image_size:
        thumb_cap.release()
        third_cap.release()
        raise RuntimeError(
            f"third_view_cam opened at {third_actual_size}, but intrinsics "
            f"require {third_intr.image_size}"
        )
    if thumb_actual_size != thumb_intr.image_size:
        thumb_cap.release()
        third_cap.release()
        raise RuntimeError(
            f"thumb_web_cam opened at {thumb_actual_size}, but intrinsics "
            f"require {thumb_intr.image_size}"
        )

    def read_third() -> np.ndarray:
        ok, frame = third_cap.read()
        if not ok or frame is None:
            raise RuntimeError("third_view_cam read failed")
        return frame

    def read_thumb() -> np.ndarray:
        ok, frame = thumb_cap.read()
        if not ok or frame is None:
            raise RuntimeError("thumb_web_cam read failed")
        return frame

    third_worker = FrameWorker("third-view-capture", read_third)
    thumb_worker = FrameWorker("thumb-web-capture", read_thumb)
    third_worker.start()
    thumb_worker.start()

    run_dir = create_run_dir(args.output_root)
    samples: list[CalibrationSample] = []
    last_pair: Optional[tuple[int, int]] = None
    previous_valid: Optional[tuple[np.ndarray, np.ndarray]] = None
    stable_count = 0
    last_capture_time = -float("inf")
    last_status = "waiting for synchronized frames"

    print(
        f"[INFO] third port={args.third_view_port}, "
        f"active_device={third_device}"
    )
    print(
        f"[INFO] thumb port={args.thumb_web_port}, "
        f"active_device={thumb_device}"
    )
    print(f"[INFO] Run directory: {run_dir}")
    print(
        "[INFO] Move thumb_web_cam to a new orientation and position, then "
        "hold it still briefly. Sampling is automatic."
    )
    print("[INFO] [s] manual store  [c] clear samples  [q/esc] solve and save")

    try:
        stop_requested = False
        while not stop_requested:
            pair = select_synchronized_pair(
                third_worker.snapshot(),
                thumb_worker.snapshot(),
                last_pair,
            )
            if pair is None:
                if third_worker.last_error or thumb_worker.last_error:
                    last_status = (
                        f"camera errors third={third_worker.last_error} "
                        f"thumb={thumb_worker.last_error}"
                    )
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            third_frame, thumb_frame, pair_skew_s = pair
            last_pair = (third_frame.index, thumb_frame.index)

            third_det = detect_apriltag_grid_pose(
                third_frame.frame_bgr,
                third_detector,
                board,
                third_intr,
                "third_cam/thumb_grid",
            )
            thumb_det = detect_apriltag_grid_pose(
                thumb_frame.frame_bgr,
                thumb_detector,
                board,
                thumb_intr,
                "thumb_cam/third_grid",
            )
            quality_ok, quality_reason = detection_quality(
                third_det,
                thumb_det,
                pair_skew_s,
            )

            stable = False
            stable_reason = "detections invalid"
            if (
                quality_ok
                and third_det.T is not None
                and thumb_det.T is not None
            ):
                stable, stable_reason = is_stable_pair(
                    previous_valid,
                    third_det.T,
                    thumb_det.T,
                )
                stable_count = stable_count + 1 if stable else 0
                previous_valid = (
                    third_det.T.copy(),
                    thumb_det.T.copy(),
                )
            else:
                stable_count = 0
                previous_valid = None

            diverse = False
            diversity_reason = "waiting for valid pose"
            if quality_ok and third_det.T is not None:
                diverse, diversity_reason = is_diverse_from_last(
                    samples,
                    third_det.T,
                )

            now = time.monotonic()
            auto_ok = (
                quality_ok
                and stable_count >= STABLE_REQUIRED_PAIRS
                and diverse
                and now - last_capture_time >= AUTO_CAPTURE_COOLDOWN_S
            )
            auto_stored = False
            if auto_ok:
                sample = store_sample(
                    samples,
                    run_dir,
                    third_frame,
                    thumb_frame,
                    pair_skew_s,
                    third_det,
                    thumb_det,
                    "auto",
                )
                last_capture_time = now
                stable_count = 0
                auto_stored = True
                last_status = f"auto stored sample {len(samples)}"
                print(
                    f"[INFO] {last_status}: "
                    f"skew={sample.pair_skew_s * 1000.0:.1f}ms "
                    f"third={sample.third_view_corners // 4}tags/"
                    f"{sample.third_view_reproj_error_px:.2f}px "
                    f"thumb={sample.thumb_web_corners // 4}tags/"
                    f"{sample.thumb_web_reproj_error_px:.2f}px"
                )
                if len(samples) >= args.max_samples:
                    print(
                        f"[INFO] Reached {len(samples)} samples; solving."
                    )
                    stop_requested = True
            else:
                if not quality_ok:
                    action = "WAIT FOR BOTH GRIDS"
                elif not diverse:
                    action = "MOVE TO A NEW POSE"
                else:
                    action = (
                        "HOLD STILL "
                        f"{stable_count}/{STABLE_REQUIRED_PAIRS}"
                    )
                last_status = (
                    f"{action}: {quality_reason}; {stable_reason}; "
                    f"{diversity_reason}"
                )

            status_lines = [
                f"samples={len(samples)}/{args.max_samples} "
                f"pair_skew={pair_skew_s * 1000.0:.1f}ms",
                last_status,
                third_det.message,
                thumb_det.message,
                "Move, then hold still | [s] store [c] clear [q] solve",
            ]
            third_vis = put_lines(
                third_det.vis
                if third_det.vis is not None
                else third_frame.frame_bgr,
                status_lines,
            )
            thumb_vis = put_lines(
                thumb_det.vis
                if thumb_det.vis is not None
                else thumb_frame.frame_bgr,
                status_lines,
            )
            cv2.imshow(
                "third_view_cam sees thumb_grid",
                resize_for_display(
                    third_vis, DISPLAY_SCALE_THIRD_VIEW
                ),
            )
            cv2.imshow(
                "thumb_web_cam sees third_grid",
                resize_for_display(
                    thumb_vis, DISPLAY_SCALE_THUMB_WEB
                ),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                for sample in samples:
                    Path(sample.third_view_image_path).unlink(
                        missing_ok=True
                    )
                    Path(sample.thumb_web_image_path).unlink(
                        missing_ok=True
                    )
                samples.clear()
                previous_valid = None
                stable_count = 0
                last_capture_time = -float("inf")
                print("[INFO] Cleared samples.")
            elif key == ord("s"):
                if auto_stored:
                    print(
                        "[INFO] Manual store skipped; auto already stored "
                        "this pair."
                    )
                elif not quality_ok:
                    print(
                        f"[WARN] Manual sample rejected: {quality_reason}"
                    )
                else:
                    sample = store_sample(
                        samples,
                        run_dir,
                        third_frame,
                        thumb_frame,
                        pair_skew_s,
                        third_det,
                        thumb_det,
                        "manual",
                    )
                    stable_count = 0
                    last_capture_time = now
                    print(
                        f"[INFO] Manually stored sample {len(samples)} "
                        f"skew={sample.pair_skew_s * 1000.0:.1f}ms"
                    )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; solving collected samples.")
    finally:
        third_worker.stop()
        thumb_worker.stop()
        thumb_cap.release()
        third_cap.release()
        cv2.destroyAllWindows()

    if len(samples) < MIN_SAMPLES_TO_SOLVE:
        print(
            f"[WARN] Only {len(samples)} samples; need at least "
            f"{MIN_SAMPLES_TO_SOLVE}. No result YAML saved."
        )
        return

    solution = solve_with_outlier_rejection(samples)
    output_path = save_results(
        run_dir,
        samples,
        solution,
        board,
        third_intr,
        thumb_intr,
        third_device,
        thumb_device,
        args.third_view_port,
        args.thumb_web_port,
    )
    print(f"[INFO] Saved {output_path}")
    print("[RESULT] T_grid_thumb_web_cam:")
    print(solution["T_thumb_grid_thumb_web_cam"])
    print("[RESULT] T_grid_third_view_cam:")
    print(solution["T_third_grid_third_view_cam"])
    print("[DIAGNOSTICS]")
    print(
        f"  inliers={len(solution['inlier_indices'])}/{len(samples)} "
        f"outliers={solution['outlier_indices']}"
    )
    print(
        f"  rotation residual deg={solution['residual_rot_deg']}"
    )
    print(
        f"  translation residual m={solution['residual_trans_m']}"
    )
    print(
        f"  jacobian rank={solution['jacobian_rank']}/12 "
        f"condition={solution['jacobian_condition']:.3e}"
    )
    if solution["jacobian_rank"] < 12:
        print(
            "[WARN] Rank-deficient motion set. Repeat with more rotation "
            "about multiple axes and more translation diversity."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the fixed grid-to-camera offsets for two cameras "
            "that carry identical but physically distinct AprilTag grids."
        )
    )
    parser.add_argument(
        "--third-view-port",
        default=THIRD_VIEW_PORT,
    )
    parser.add_argument(
        "--thumb-web-port",
        default=THUMB_WEB_PORT,
    )
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
        "--grid-yaml",
        type=Path,
        default=APRILTAG_GRID_YAML,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate inputs and construct detectors without opening cameras.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic synthetic recovery without cameras.",
    )
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
