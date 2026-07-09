from __future__ import annotations

import sys
import time
import shutil
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from cameras import AVCameraManager
from intr_calib_charuco import (
    CHARUCO_DICTIONARY,
    CHARUCO_LEGACY_PATTERN,
    CHARUCO_MARKER_LENGTH,
    CHARUCO_SQUARE_LENGTH,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    CharucoDetectorAdapter,
    charuco_to_calibration_points,
    create_charuco_board,
    start_capture,
)


# ---------------------------- User macros ---------------------------- #
THIRD_VIEW_PORT = "3-11.1:1.0"
FINGERTIP_AV_CAMERA_TO_PORT = {"I": "3-5.4.4.4:1.0"}
FINGERTIP_AV_LEFT_RIGHT_ORDER = {"I": ["tip", "root"]}

THIRD_VIEW_INTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/intrinsics_third_view_cv2_charuco_1920x1080_0704_222919.yaml"
)
ROOT_INTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/intrinsics_i_root_av_320x240_0704_200627.yaml"
)
TIP_INTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/intrinsics_i_tip_av_320x240_0704_200143.yaml"
)

APRILCUBE_CFG_DIR = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/cube_april_36h11_12_17_1x1x1_10mm"
)
APRILCUBE_SRC_DIR = Path("/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/src")

THIRD_VIEW_FPS = 50
THIRD_VIEW_FOURCC = "MJPG"
FINGERTIP_AV_FPS = 25
FINGERTIP_AV_FOURCC = "mjpeg"

DISPLAY_SCALE_THIRD = 0.45
DISPLAY_SCALE_FINGERTIP = 1.5
MIN_SAMPLES_TO_SAVE = 8
OUTPUT_PATH = Path("outputs/extrinsics_fingertip_Q_root_tip.yaml")
MIN_CHARUCO_PNP_POINTS = 6
CHARUCO_AXIS_LENGTH_M = 0.02

AUTO_CAPTURE = True
AUTO_CAPTURE_COOLDOWN_S = 0.8
MAX_CHARUCO_REPROJ_PX = 1.5
MAX_APRILCUBE_REPROJ_PX = 2.0
MIN_APRILCUBE_TAGS = 1
MIN_BOARD_ROT_DELTA_DEG = 4.0
MIN_BOARD_TRANS_DELTA_M = 0.015
SAMPLE_IMAGE_ROOT = Path("outputs/extrinsics_fingertip_samples")
USABLE_SAMPLE_IMAGE_ROOT = Path("outputs/extrinsics_fingertip_usable_samples")
USABLE_SAMPLE_PICKLE_NAME = "usable_samples.pkl"
AUTO_STOP_AFTER_USABLE_SAMPLE_GROUPS = True
MAX_USABLE_SAMPLE_GROUPS = 50

OUTLIER_REJECTION = True
OUTLIER_MAX_ROT_DEG = 10.0
OUTLIER_MAX_TRANS_M = 0.05
OUTLIER_MAX_ITERATIONS = 5

# aprilcube.detector(...).process_frame() returns object/cube -> camera in mm.
# Keep this explicit because this convention is the main thing to validate.
APRILCUBE_POSE_CONVENTION = "T_E_Q"  # supported: "T_E_Q", "T_Q_E"


@dataclass
class Intrinsics:
    path: Path
    image_size: tuple[int, int]
    K: np.ndarray
    dist: np.ndarray


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_th = (np.trace(R) - 1.0) / 2.0
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-12:
        return np.zeros(3, dtype=np.float64)
    w_hat = (R - R.T) / (2.0 * np.sin(th))
    return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=np.float64) * th


def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


@dataclass
class PoseDetection:
    ok: bool
    T: Optional[np.ndarray]
    n_points: int = 0
    reproj_error: float = float("inf")
    message: str = ""
    vis: Optional[np.ndarray] = None


@dataclass
class Sample:
    index: int
    timestamp: float
    T_root_B: np.ndarray
    T_tip_B: np.ndarray
    T_E_B: np.ndarray
    T_E_Q: np.ndarray
    T_Q_root: np.ndarray
    T_Q_tip: np.ndarray
    errors: dict[str, float]
    image_paths: dict[str, str]
    capture_mode: str


def append_timestamp(path: Path) -> Path:
    root = path.with_suffix("")
    suffix = path.suffix
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    return root.parent / f"{root.name}_{stamp}{suffix}"


def load_intrinsics(path: Path) -> Intrinsics:
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Intrinsics(
        path=path.expanduser().resolve(),
        image_size=tuple(int(v) for v in data["image_size"]),
        K=np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        dist=np.asarray(data.get("dist", data.get("D", [0, 0, 0, 0, 0])), dtype=np.float64).reshape(-1),
    )


def scale_intrinsics(intr: Intrinsics, new_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    old_w, old_h = intr.image_size
    new_w, new_h = new_size
    if (old_w, old_h) == (new_w, new_h):
        return intr.K.copy(), intr.dist.copy()

    sx = new_w / old_w
    sy = new_h / old_h
    K = intr.K.copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K, intr.dist.copy()


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return make_T(R, np.asarray(tvec, dtype=np.float64).reshape(3))


def reproj_error(
    objpoints: np.ndarray,
    imgpoints: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
    return float(np.mean(np.linalg.norm(imgpoints.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)))


def detect_charuco_pose(
    frame_bgr: np.ndarray,
    detector: CharucoDetectorAdapter,
    board: Any,
    intr: Intrinsics,
    label: str,
) -> PoseDetection:
    h, w = frame_bgr.shape[:2]
    K, dist = scale_intrinsics(intr, (w, h))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detect(gray)

    vis = frame_bgr.copy()
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if charuco_corners is not None and charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

    objpoints, imgpoints = charuco_to_calibration_points(board, charuco_corners, charuco_ids)
    n = 0 if objpoints is None else int(len(objpoints))
    if objpoints is None or imgpoints is None or n < MIN_CHARUCO_PNP_POINTS:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=n,
            message=f"{label}: charuco corners={n} need>={MIN_CHARUCO_PNP_POINTS}",
            vis=vis,
        )

    try:
        ok, rvec, tvec = cv2.solvePnP(
            objpoints,
            imgpoints,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=n,
            message=f"{label}: solvePnP error corners={n}: {exc.err}",
            vis=vis,
        )
    if not ok:
        return PoseDetection(ok=False, T=None, n_points=n, message=f"{label}: solvePnP failed", vis=vis)

    try:
        rvec, tvec = cv2.solvePnPRefineLM(objpoints, imgpoints, K, dist, rvec, tvec)
    except cv2.error:
        pass

    err = reproj_error(objpoints, imgpoints, rvec, tvec, K, dist)
    T_cam_board = rvec_tvec_to_T(rvec, tvec)
    try:
        cv2.drawFrameAxes(vis, K, dist, rvec, tvec, CHARUCO_AXIS_LENGTH_M)
    except cv2.error:
        pass
    return PoseDetection(
        ok=True,
        T=T_cam_board,
        n_points=n,
        reproj_error=err,
        message=f"{label}: B ok corners={n} err={err:.2f}px",
        vis=vis,
    )


def ensure_aprilcube_on_path() -> None:
    src = str(APRILCUBE_SRC_DIR.expanduser().resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def create_aprilcube_detector(intr: Intrinsics):
    ensure_aprilcube_on_path()
    import aprilcube  # noqa: PLC0415

    intrinsic_cfg = {
        "fx": float(intr.K[0, 0]),
        "fy": float(intr.K[1, 1]),
        "cx": float(intr.K[0, 2]),
        "cy": float(intr.K[1, 2]),
    }
    return aprilcube.detector(
        APRILCUBE_CFG_DIR,
        intrinsic_cfg=intrinsic_cfg,
        dist_coeffs=intr.dist,
        enable_filter=False,
        fast=False,
    )


def detect_aprilcube_pose(frame_bgr: np.ndarray, detector: Any) -> PoseDetection:
    result = detector.process_frame(frame_bgr)
    vis = detector.draw_result(frame_bgr, result)
    if not result.get("success", False) or result.get("T") is None:
        n_tags = int(result.get("n_tags", 0))
        return PoseDetection(ok=False, T=None, n_points=n_tags, message=f"E: Q not found tags={n_tags}", vis=vis)

    T = np.asarray(result["T"], dtype=np.float64).reshape(4, 4)
    T[:3, 3] *= 0.001  # AprilCube model uses millimeters; project uses meters elsewhere.
    if APRILCUBE_POSE_CONVENTION == "T_Q_E":
        T = inv_T(T)
    elif APRILCUBE_POSE_CONVENTION != "T_E_Q":
        raise ValueError(f"Unsupported APRILCUBE_POSE_CONVENTION={APRILCUBE_POSE_CONVENTION}")

    return PoseDetection(
        ok=True,
        T=T,
        n_points=int(result.get("n_tags", 0)),
        reproj_error=float(result.get("reproj_error", float("inf"))),
        message=f"E: Q ok tags={int(result.get('n_tags', 0))} err={float(result.get('reproj_error', 0.0)):.2f}px",
        vis=vis,
    )


def open_third_view_camera(intr: Intrinsics):
    width, height = intr.image_size
    return start_capture(
        THIRD_VIEW_PORT,
        width,
        height,
        THIRD_VIEW_FPS,
        THIRD_VIEW_FOURCC,
    )


def open_fingertip_av_manager(root_intr: Intrinsics) -> AVCameraManager:
    root_w, root_h = root_intr.image_size
    default_opts = {
        "input_format": FINGERTIP_AV_FOURCC,
        "video_size": f"{root_w * 2}x{root_h}",
        "framerate": str(FINGERTIP_AV_FPS),
    }
    manager = AVCameraManager(
        camera_to_port=FINGERTIP_AV_CAMERA_TO_PORT,
        camera_left_right_order=FINGERTIP_AV_LEFT_RIGHT_ORDER,
        default_options=default_opts,
        stream_index=0,
    )
    manager.start()
    return manager


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("No transforms to average.")

    rotations = [T[:3, :3] for T in transforms]
    M = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        M += R
    U, _S, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    t_avg = np.mean([T[:3, 3] for T in transforms], axis=0)
    return make_T(R_avg, t_avg)


def transform_residual(T_ref: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    dT = inv_T(T_ref) @ T
    rot_deg = float(np.degrees(np.linalg.norm(so3_log(dT[:3, :3]))))
    trans_m = float(np.linalg.norm(dT[:3, 3]))
    return rot_deg, trans_m


DIAGNOSTIC_KEYS = (
    "root_residual_rot_deg",
    "root_residual_trans_m",
    "tip_residual_rot_deg",
    "tip_residual_trans_m",
    "root_tip_consistency_rot_deg",
    "root_tip_consistency_trans_m",
)


def summarize_samples(samples: list[Sample]) -> dict[str, Any]:
    Q_T_root_list = [s.T_Q_root for s in samples]
    Q_T_tip_list = [s.T_Q_tip for s in samples]
    Q_T_root = average_transforms(Q_T_root_list)
    Q_T_tip = average_transforms(Q_T_tip_list)

    root_res = [transform_residual(Q_T_root, T) for T in Q_T_root_list]
    tip_res = [transform_residual(Q_T_tip, T) for T in Q_T_tip_list]

    root_tip_board = [s.T_root_B @ inv_T(s.T_tip_B) for s in samples]
    root_tip_cube = [inv_T(s.T_Q_root) @ s.T_Q_tip for s in samples]
    root_tip_delta = [
        transform_residual(a, b)
        for a, b in zip(root_tip_board, root_tip_cube)
    ]

    def stats(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "Q_T_root": Q_T_root,
        "Q_T_tip": Q_T_tip,
        "root_residual_rot_deg": stats([x[0] for x in root_res]),
        "root_residual_trans_m": stats([x[1] for x in root_res]),
        "tip_residual_rot_deg": stats([x[0] for x in tip_res]),
        "tip_residual_trans_m": stats([x[1] for x in tip_res]),
        "root_tip_consistency_rot_deg": stats([x[0] for x in root_tip_delta]),
        "root_tip_consistency_trans_m": stats([x[1] for x in root_tip_delta]),
    }


def residuals_against_solution(
    samples: list[Sample],
    Q_T_root: np.ndarray,
    Q_T_tip: np.ndarray,
) -> dict[int, dict[str, float]]:
    residuals: dict[int, dict[str, float]] = {}
    for s in samples:
        root_rot, root_trans = transform_residual(Q_T_root, s.T_Q_root)
        tip_rot, tip_trans = transform_residual(Q_T_tip, s.T_Q_tip)
        residuals[s.index] = {
            "root_rot_deg": root_rot,
            "root_trans_m": root_trans,
            "tip_rot_deg": tip_rot,
            "tip_trans_m": tip_trans,
            "max_rot_deg": max(root_rot, tip_rot),
            "max_trans_m": max(root_trans, tip_trans),
        }
    return residuals


def solve_with_outlier_rejection(samples: list[Sample]) -> dict[str, Any]:
    raw_solution = summarize_samples(samples)
    if not OUTLIER_REJECTION:
        raw_solution["inlier_indices"] = [s.index for s in samples]
        raw_solution["outlier_indices"] = []
        raw_solution["sample_residuals"] = residuals_against_solution(
            samples,
            raw_solution["Q_T_root"],
            raw_solution["Q_T_tip"],
        )
        return raw_solution

    min_inliers = max(MIN_SAMPLES_TO_SAVE, 1)
    inlier_indices = [s.index for s in samples]
    iterations: list[dict[str, Any]] = []

    for iteration in range(OUTLIER_MAX_ITERATIONS):
        active = [s for s in samples if s.index in set(inlier_indices)]
        solution = summarize_samples(active)
        residuals = residuals_against_solution(samples, solution["Q_T_root"], solution["Q_T_tip"])
        next_inliers = [
            s.index
            for s in samples
            if residuals[s.index]["max_rot_deg"] <= OUTLIER_MAX_ROT_DEG
            and residuals[s.index]["max_trans_m"] <= OUTLIER_MAX_TRANS_M
        ]

        iterations.append(
            {
                "iteration": iteration,
                "num_input_inliers": len(inlier_indices),
                "num_next_inliers": len(next_inliers),
                "outlier_indices": [s.index for s in samples if s.index not in next_inliers],
            }
        )

        if len(next_inliers) < min_inliers:
            break
        if next_inliers == inlier_indices:
            break
        inlier_indices = next_inliers

    inlier_set = set(inlier_indices)
    inlier_samples = [s for s in samples if s.index in inlier_set]
    filtered_solution = summarize_samples(inlier_samples)
    filtered_residuals = residuals_against_solution(
        samples,
        filtered_solution["Q_T_root"],
        filtered_solution["Q_T_tip"],
    )
    outlier_indices = [s.index for s in samples if s.index not in inlier_set]
    rejection_reasons = {
        idx: (
            f"max_rot={filtered_residuals[idx]['max_rot_deg']:.2f}deg "
            f"or max_trans={filtered_residuals[idx]['max_trans_m']:.4f}m exceeds "
            f"{OUTLIER_MAX_ROT_DEG:.1f}deg/{OUTLIER_MAX_TRANS_M:.3f}m"
        )
        for idx in outlier_indices
    }

    filtered_solution["raw_diagnostics_before_filter"] = {
        k: raw_solution[k] for k in DIAGNOSTIC_KEYS
    }
    filtered_solution["outlier_rejection"] = {
        "enabled": True,
        "max_rot_deg": float(OUTLIER_MAX_ROT_DEG),
        "max_trans_m": float(OUTLIER_MAX_TRANS_M),
        "max_iterations": int(OUTLIER_MAX_ITERATIONS),
        "num_raw_samples": len(samples),
        "num_inliers": len(inlier_indices),
        "num_outliers": len(outlier_indices),
        "inlier_indices": inlier_indices,
        "outlier_indices": outlier_indices,
        "rejection_reasons": rejection_reasons,
        "iterations": iterations,
    }
    filtered_solution["inlier_indices"] = inlier_indices
    filtered_solution["outlier_indices"] = outlier_indices
    filtered_solution["sample_residuals"] = filtered_residuals
    return filtered_solution


def export_usable_sample_cache(
    output_path: Path,
    samples: list[Sample],
    solution: dict[str, Any],
) -> dict[str, Any]:
    inlier_indices = set(solution.get("inlier_indices", [s.index for s in samples]))
    sample_residuals = solution.get("sample_residuals", {})
    cache_dir = USABLE_SAMPLE_IMAGE_ROOT / output_path.with_suffix("").name
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    usable_paths_by_index: dict[int, dict[str, str]] = {}
    pickle_samples: list[dict[str, Any]] = []

    for s in samples:
        if s.index not in inlier_indices:
            continue

        copied_paths: dict[str, str] = {}
        encoded_images: dict[str, bytes] = {}
        image_shapes: dict[str, list[int]] = {}
        for camera_name, src_path_str in s.image_paths.items():
            src_path = Path(src_path_str)
            dst_path = cache_dir / f"sample_{s.index:04d}_{camera_name}.png"
            shutil.copy2(src_path, dst_path)
            copied_paths[camera_name] = str(dst_path)

            image_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read usable sample image for pickle: {src_path}")
            ok, encoded = cv2.imencode(".png", image_bgr)
            if not ok:
                raise RuntimeError(f"Failed to PNG-encode usable sample image for pickle: {src_path}")
            encoded_images[camera_name] = encoded.tobytes()
            image_shapes[camera_name] = [int(v) for v in image_bgr.shape]

        usable_paths_by_index[s.index] = copied_paths
        entries.append(
            {
                "index": int(s.index),
                "timestamp": float(s.timestamp),
                "capture_mode": str(s.capture_mode),
                "usable_image_paths": copied_paths,
                "source_image_paths": s.image_paths,
                "errors": s.errors,
                "solution_residual": sample_residuals.get(s.index),
            }
        )
        pickle_samples.append(
            {
                "index": int(s.index),
                "timestamp": float(s.timestamp),
                "capture_mode": str(s.capture_mode),
                "image_encoding": "png",
                "images_png": encoded_images,
                "image_shapes_bgr": image_shapes,
                "usable_image_paths": copied_paths,
                "source_image_paths": s.image_paths,
                "errors": s.errors,
                "solution_residual": sample_residuals.get(s.index),
                "T_root_B": s.T_root_B,
                "T_tip_B": s.T_tip_B,
                "T_E_B": s.T_E_B,
                "T_E_Q": s.T_E_Q,
                "T_Q_root": s.T_Q_root,
                "T_Q_tip": s.T_Q_tip,
            }
        )

    pickle_path = cache_dir / USABLE_SAMPLE_PICKLE_NAME
    pickle_payload = {
        "version": 1,
        "description": (
            "Final inlier sample images for fingertip extrinsics. "
            "Images are PNG-encoded BGR frames; decode with cv2.imdecode."
        ),
        "source_extrinsics_yaml": str(output_path),
        "cache_dir": str(cache_dir),
        "num_usable_samples": len(pickle_samples),
        "frame_convention": {
            "Q": "AprilCube frame / rig frame",
            "root": "root camera optical frame",
            "tip": "tip camera optical frame",
            "E": "third-view calibration camera optical frame",
            "B": "ChArUco board frame",
            "pose_notation": "A_T_B maps points from B frame into A frame",
        },
        "Q_T_root": solution["Q_T_root"],
        "Q_T_tip": solution["Q_T_tip"],
        "diagnostics": {k: solution[k] for k in DIAGNOSTIC_KEYS},
        "samples": pickle_samples,
    }
    with pickle_path.open("wb") as f:
        pickle.dump(pickle_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "source_extrinsics_yaml": str(output_path),
        "num_usable_samples": len(entries),
        "cache_dir": str(cache_dir),
        "pickle": str(pickle_path),
        "samples": entries,
    }
    manifest_path = cache_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return {
        "cache_dir": str(cache_dir),
        "manifest": str(manifest_path),
        "pickle": str(pickle_path),
        "num_usable_samples": len(entries),
        "paths_by_sample_index": usable_paths_by_index,
    }


def save_results(path: Path, samples: list[Sample], solution: dict[str, Any]) -> Path:
    output_path = append_timestamp(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inlier_indices = set(solution.get("inlier_indices", [s.index for s in samples]))
    sample_residuals = solution.get("sample_residuals", {})
    rejection_reasons = solution.get("outlier_rejection", {}).get("rejection_reasons", {})
    usable_cache = export_usable_sample_cache(output_path, samples, solution)
    usable_paths_by_index = usable_cache.get("paths_by_sample_index", {})

    data = {
        "frame_convention": {
            "Q": "AprilCube frame / rig frame",
            "root": "root camera optical frame",
            "tip": "tip camera optical frame",
            "E": "third-view calibration camera optical frame",
            "B": "ChArUco board frame",
            "pose_notation": "A_T_B maps points from B frame into A frame",
            "aprilcube_pose_convention": APRILCUBE_POSE_CONVENTION,
        },
        "inputs": {
            "third_view_port": THIRD_VIEW_PORT,
            "fingertip_av_camera_to_port": FINGERTIP_AV_CAMERA_TO_PORT,
            "fingertip_av_left_right_order": FINGERTIP_AV_LEFT_RIGHT_ORDER,
            "third_view_intrinsics_yaml": str(THIRD_VIEW_INTRINSICS_YAML),
            "root_intrinsics_yaml": str(ROOT_INTRINSICS_YAML),
            "tip_intrinsics_yaml": str(TIP_INTRINSICS_YAML),
            "aprilcube_cfg_dir": str(APRILCUBE_CFG_DIR),
            "auto_capture": {
                "enabled": bool(AUTO_CAPTURE),
                "cooldown_s": float(AUTO_CAPTURE_COOLDOWN_S),
                "max_charuco_reproj_px": float(MAX_CHARUCO_REPROJ_PX),
                "max_aprilcube_reproj_px": float(MAX_APRILCUBE_REPROJ_PX),
                "min_aprilcube_tags": int(MIN_APRILCUBE_TAGS),
                "min_board_rot_delta_deg": float(MIN_BOARD_ROT_DELTA_DEG),
                "min_board_trans_delta_m": float(MIN_BOARD_TRANS_DELTA_M),
                "auto_stop_after_usable_sample_groups": bool(AUTO_STOP_AFTER_USABLE_SAMPLE_GROUPS),
                "max_usable_sample_groups": int(MAX_USABLE_SAMPLE_GROUPS),
            },
            "usable_sample_cache": {
                "root": str(USABLE_SAMPLE_IMAGE_ROOT),
                "cache_dir": usable_cache["cache_dir"],
                "manifest": usable_cache["manifest"],
                "pickle": usable_cache["pickle"],
            },
            "outlier_rejection": {
                "enabled": bool(OUTLIER_REJECTION),
                "max_rot_deg": float(OUTLIER_MAX_ROT_DEG),
                "max_trans_m": float(OUTLIER_MAX_TRANS_M),
                "max_iterations": int(OUTLIER_MAX_ITERATIONS),
            },
            "charuco": {
                "squares_x": int(CHARUCO_SQUARES_X),
                "squares_y": int(CHARUCO_SQUARES_Y),
                "square_length": float(CHARUCO_SQUARE_LENGTH),
                "marker_length": float(CHARUCO_MARKER_LENGTH),
                "dictionary": str(CHARUCO_DICTIONARY),
                "legacy_pattern": bool(CHARUCO_LEGACY_PATTERN),
            },
        },
        "num_samples": len(inlier_indices),
        "num_raw_samples": len(samples),
        "Q_T_root": solution["Q_T_root"].tolist(),
        "Q_T_tip": solution["Q_T_tip"].tolist(),
        "diagnostics": {k: solution[k] for k in DIAGNOSTIC_KEYS},
        "raw_diagnostics_before_filter": solution.get("raw_diagnostics_before_filter"),
        "outlier_rejection": solution.get("outlier_rejection"),
        "usable_sample_cache": {
            "cache_dir": usable_cache["cache_dir"],
            "manifest": usable_cache["manifest"],
            "pickle": usable_cache["pickle"],
            "num_usable_samples": usable_cache["num_usable_samples"],
        },
        "samples": [
            {
                "index": s.index,
                "timestamp": float(s.timestamp),
                "used_for_solution": s.index in inlier_indices,
                "rejection_reason": rejection_reasons.get(s.index),
                "T_root_B": s.T_root_B.tolist(),
                "T_tip_B": s.T_tip_B.tolist(),
                "T_E_B": s.T_E_B.tolist(),
                "T_E_Q": s.T_E_Q.tolist(),
                "T_Q_root": s.T_Q_root.tolist(),
                "T_Q_tip": s.T_Q_tip.tolist(),
                "solution_residual": sample_residuals.get(s.index),
                "errors": s.errors,
                "image_paths": s.image_paths,
                "usable_image_paths": usable_paths_by_index.get(s.index),
                "capture_mode": s.capture_mode,
            }
            for s in samples
        ],
    }
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return output_path


def put_lines(img: np.ndarray, lines: list[str], color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 24
    return out


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def create_sample_image_dir() -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    path = SAMPLE_IMAGE_ROOT / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_sample_images(
    sample_index: int,
    image_dir: Path,
    E_frame: np.ndarray,
    root_frame: np.ndarray,
    tip_frame: np.ndarray,
) -> dict[str, str]:
    paths = {
        "third_view": image_dir / f"sample_{sample_index:04d}_third_view.png",
        "root": image_dir / f"sample_{sample_index:04d}_root.png",
        "tip": image_dir / f"sample_{sample_index:04d}_tip.png",
    }
    frames = {
        "third_view": E_frame,
        "root": root_frame,
        "tip": tip_frame,
    }

    saved: dict[str, str] = {}
    for name, path in paths.items():
        if not cv2.imwrite(str(path), frames[name]):
            raise RuntimeError(f"Failed to save sample image: {path}")
        saved[name] = str(path)

    return saved


def quality_ok(
    root_det: PoseDetection,
    tip_det: PoseDetection,
    E_B_det: PoseDetection,
    E_Q_det: PoseDetection,
) -> tuple[bool, str]:
    charuco_errors = {
        "root": root_det.reproj_error,
        "tip": tip_det.reproj_error,
        "E/B": E_B_det.reproj_error,
    }
    for name, err in charuco_errors.items():
        if not np.isfinite(err) or err > MAX_CHARUCO_REPROJ_PX:
            return False, f"{name} charuco err {err:.2f}px > {MAX_CHARUCO_REPROJ_PX:.2f}px"

    if not np.isfinite(E_Q_det.reproj_error) or E_Q_det.reproj_error > MAX_APRILCUBE_REPROJ_PX:
        return False, f"E/Q cube err {E_Q_det.reproj_error:.2f}px > {MAX_APRILCUBE_REPROJ_PX:.2f}px"

    if E_Q_det.n_points < MIN_APRILCUBE_TAGS:
        return False, f"E/Q cube tags {E_Q_det.n_points} < {MIN_APRILCUBE_TAGS}"

    return True, "quality ok"


def should_auto_capture(
    *,
    now: float,
    last_auto_time: float,
    last_saved_T_E_B: Optional[np.ndarray],
    all_ok: bool,
    root_det: PoseDetection,
    tip_det: PoseDetection,
    E_B_det: PoseDetection,
    E_Q_det: PoseDetection,
) -> tuple[bool, str]:
    if not AUTO_CAPTURE:
        return False, "auto off"
    if not all_ok:
        return False, "waiting for all poses"
    if now - last_auto_time < AUTO_CAPTURE_COOLDOWN_S:
        return False, "cooldown"

    ok, reason = quality_ok(root_det, tip_det, E_B_det, E_Q_det)
    if not ok:
        return False, reason

    if last_saved_T_E_B is not None and E_B_det.T is not None:
        rot_deg, trans_m = transform_residual(last_saved_T_E_B, E_B_det.T)
        if rot_deg < MIN_BOARD_ROT_DELTA_DEG and trans_m < MIN_BOARD_TRANS_DELTA_M:
            return (
                False,
                f"duplicate board pose dR={rot_deg:.1f}deg dt={trans_m:.3f}m",
            )

    return True, "auto capture ready"


def sample_from_detections(
    sample_index: int,
    root_det: PoseDetection,
    tip_det: PoseDetection,
    E_B_det: PoseDetection,
    E_Q_det: PoseDetection,
    image_paths: dict[str, str],
    capture_mode: str,
) -> Sample:
    assert root_det.T is not None
    assert tip_det.T is not None
    assert E_B_det.T is not None
    assert E_Q_det.T is not None

    T_root_B = root_det.T
    T_tip_B = tip_det.T
    T_E_B = E_B_det.T
    T_E_Q = E_Q_det.T
    T_Q_root = inv_T(T_E_Q) @ T_E_B @ inv_T(T_root_B)
    T_Q_tip = inv_T(T_E_Q) @ T_E_B @ inv_T(T_tip_B)

    return Sample(
        index=sample_index,
        timestamp=time.time(),
        T_root_B=T_root_B.copy(),
        T_tip_B=T_tip_B.copy(),
        T_E_B=T_E_B.copy(),
        T_E_Q=T_E_Q.copy(),
        T_Q_root=T_Q_root,
        T_Q_tip=T_Q_tip,
        errors={
            "root_charuco_reproj_px": float(root_det.reproj_error),
            "tip_charuco_reproj_px": float(tip_det.reproj_error),
            "E_charuco_reproj_px": float(E_B_det.reproj_error),
            "E_aprilcube_reproj_px": float(E_Q_det.reproj_error),
        },
        image_paths=dict(image_paths),
        capture_mode=str(capture_mode),
    )


def store_current_sample(
    *,
    samples: list[Sample],
    image_dir: Path,
    capture_mode: str,
    E_frame: np.ndarray,
    root_frame: np.ndarray,
    tip_frame: np.ndarray,
    root_det: PoseDetection,
    tip_det: PoseDetection,
    E_B_det: PoseDetection,
    E_Q_det: PoseDetection,
) -> Sample:
    sample_index = len(samples)
    image_paths = save_sample_images(
        sample_index,
        image_dir,
        E_frame,
        root_frame,
        tip_frame,
    )
    sample = sample_from_detections(
        sample_index,
        root_det,
        tip_det,
        E_B_det,
        E_Q_det,
        image_paths,
        capture_mode,
    )
    samples.append(sample)
    return sample


def main() -> None:
    third_intr = load_intrinsics(THIRD_VIEW_INTRINSICS_YAML)
    root_intr = load_intrinsics(ROOT_INTRINSICS_YAML)
    tip_intr = load_intrinsics(TIP_INTRINSICS_YAML)

    board, dictionary = create_charuco_board(
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_MARKER_LENGTH,
        CHARUCO_DICTIONARY,
        CHARUCO_LEGACY_PATTERN,
    )
    charuco_detector = CharucoDetectorAdapter(board, dictionary)
    aprilcube_detector = create_aprilcube_detector(third_intr)

    third_cap, third_device = open_third_view_camera(third_intr)
    av_manager = open_fingertip_av_manager(root_intr)

    samples: list[Sample] = []
    sample_image_dir = create_sample_image_dir()
    last_auto_time = 0.0
    last_saved_T_E_B: Optional[np.ndarray] = None
    last_auto_reason = "not evaluated"
    frame_idx = 0
    print(f"[INFO] third-view active_device={third_device}, intrinsics={third_intr.path}")
    print(f"[INFO] root intrinsics={root_intr.path}, tip intrinsics={tip_intr.path}")
    print(f"[INFO] AV ports={FINGERTIP_AV_CAMERA_TO_PORT}, split={FINGERTIP_AV_LEFT_RIGHT_ORDER}")
    print(f"[INFO] sample images will be saved under {sample_image_dir}")
    print(f"[INFO] usable inlier images will be copied under {USABLE_SAMPLE_IMAGE_ROOT}")
    print(f"[INFO] auto stop after {MAX_USABLE_SAMPLE_GROUPS} valid sample groups: {AUTO_STOP_AFTER_USABLE_SAMPLE_GROUPS}")
    print("[INFO] Press s to manually store a valid synchronized sample; c clears; q solves+saves+quits.")

    try:
        stop_requested = False
        while True:
            frame_idx += 1
            ok, E_frame = third_cap.read()
            if not ok or E_frame is None:
                print("[WARN] no third-view frame")
                time.sleep(0.02)
                continue

            av_frames = av_manager.get_frames(img_size=root_intr.image_size)
            root_frame = av_frames.get("root")
            tip_frame = av_frames.get("tip")
            if root_frame is None or tip_frame is None:
                print("[WARN] missing root/tip AV frames")
                time.sleep(0.02)
                continue

            root_det = detect_charuco_pose(root_frame, charuco_detector, board, root_intr, "root")
            tip_det = detect_charuco_pose(tip_frame, charuco_detector, board, tip_intr, "tip")
            E_B_det = detect_charuco_pose(E_frame, charuco_detector, board, third_intr, "E/B")
            E_Q_det = detect_aprilcube_pose(E_frame, aprilcube_detector)

            all_ok = root_det.ok and tip_det.ok and E_B_det.ok and E_Q_det.ok
            now = time.time()
            auto_ok, last_auto_reason = should_auto_capture(
                now=now,
                last_auto_time=last_auto_time,
                last_saved_T_E_B=last_saved_T_E_B,
                all_ok=all_ok,
                root_det=root_det,
                tip_det=tip_det,
                E_B_det=E_B_det,
                E_Q_det=E_Q_det,
            )
            auto_stored_this_frame = False
            if auto_ok:
                sample = store_current_sample(
                    samples=samples,
                    image_dir=sample_image_dir,
                    capture_mode="auto",
                    E_frame=E_frame,
                    root_frame=root_frame,
                    tip_frame=tip_frame,
                    root_det=root_det,
                    tip_det=tip_det,
                    E_B_det=E_B_det,
                    E_Q_det=E_Q_det,
                )
                last_auto_time = now
                last_saved_T_E_B = sample.T_E_B.copy()
                last_auto_reason = f"stored auto sample {len(samples)}"
                auto_stored_this_frame = True
                print(f"[INFO] auto stored sample {len(samples)} errors={sample.errors}")
                if AUTO_STOP_AFTER_USABLE_SAMPLE_GROUPS and len(samples) >= MAX_USABLE_SAMPLE_GROUPS:
                    print(
                        f"[INFO] reached {len(samples)} valid sample groups; "
                        "auto-stopping capture and solving."
                    )
                    stop_requested = True

            status = [
                f"samples={len(samples)}/{MAX_USABLE_SAMPLE_GROUPS} frame={frame_idx} all_ok={all_ok} auto={AUTO_CAPTURE}",
                f"auto: {last_auto_reason}",
                root_det.message,
                tip_det.message,
                E_B_det.message,
                E_Q_det.message,
                "[s] store  [c] clear  [q] solve/save/quit",
            ]

            if E_Q_det.vis is not None and E_B_det.vis is not None:
                # The third-view camera must show both detections: AprilCube Q
                # and ChArUco board B. Each detector draws on the raw frame, so
                # blend the two overlays before adding the status panel.
                E_vis = cv2.addWeighted(E_Q_det.vis, 0.65, E_B_det.vis, 0.35, 0.0)
            elif E_Q_det.vis is not None:
                E_vis = E_Q_det.vis.copy()
            elif E_B_det.vis is not None:
                E_vis = E_B_det.vis.copy()
            else:
                E_vis = E_frame.copy()
            E_vis = put_lines(E_vis, status, color=(0, 255, 255) if all_ok else (0, 0, 255))

            root_vis = put_lines(root_det.vis if root_det.vis is not None else root_frame, [root_det.message])
            tip_vis = put_lines(tip_det.vis if tip_det.vis is not None else tip_frame, [tip_det.message])

            cv2.imshow("third-view E: ChArUco B + AprilCube Q", resize_for_display(E_vis, DISPLAY_SCALE_THIRD))
            cv2.imshow("root", resize_for_display(root_vis, DISPLAY_SCALE_FINGERTIP))
            cv2.imshow("tip", resize_for_display(tip_vis, DISPLAY_SCALE_FINGERTIP))

            key = cv2.waitKey(1) & 0xFF
            if stop_requested:
                break
            if key == ord("s"):
                if auto_stored_this_frame:
                    print("[INFO] manual store skipped; auto already stored this frame.")
                    continue
                if not all_ok:
                    print("[WARN] sample not stored; one or more poses are invalid.")
                    for line in status[2:6]:
                        print(f"  {line}")
                    continue
                sample = store_current_sample(
                    samples=samples,
                    image_dir=sample_image_dir,
                    capture_mode="manual",
                    E_frame=E_frame,
                    root_frame=root_frame,
                    tip_frame=tip_frame,
                    root_det=root_det,
                    tip_det=tip_det,
                    E_B_det=E_B_det,
                    E_Q_det=E_Q_det,
                )
                last_saved_T_E_B = sample.T_E_B.copy()
                print(f"[INFO] manually stored sample {len(samples)} errors={sample.errors}")
                if AUTO_STOP_AFTER_USABLE_SAMPLE_GROUPS and len(samples) >= MAX_USABLE_SAMPLE_GROUPS:
                    print(
                        f"[INFO] reached {len(samples)} valid sample groups; "
                        "auto-stopping capture and solving."
                    )
                    break
            elif key == ord("c"):
                samples.clear()
                last_saved_T_E_B = None
                last_auto_time = 0.0
                print("[INFO] cleared samples")
            elif key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    finally:
        third_cap.release()
        av_manager.release_all()
        cv2.destroyAllWindows()

    if len(samples) < MIN_SAMPLES_TO_SAVE:
        print(f"[WARN] only {len(samples)} samples; need >= {MIN_SAMPLES_TO_SAVE}. Nothing saved.")
        return

    solution = solve_with_outlier_rejection(samples)
    output_path = save_results(OUTPUT_PATH, samples, solution)
    print(f"[INFO] saved {output_path}")
    cache_info = solution.get("outlier_rejection", {})
    print(
        f"[INFO] usable inlier images copied to "
        f"{USABLE_SAMPLE_IMAGE_ROOT / output_path.with_suffix('').name} "
        f"(inliers={cache_info.get('num_inliers', len(samples))})"
    )
    print(
        f"[INFO] usable inlier sample pickle: "
        f"{USABLE_SAMPLE_IMAGE_ROOT / output_path.with_suffix('').name / USABLE_SAMPLE_PICKLE_NAME}"
    )
    print("[INFO] diagnostics:")
    for key in DIAGNOSTIC_KEYS:
        print(f"  {key}: {solution[key]}")
    print(f"  outlier_rejection: {solution.get('outlier_rejection')}")


if __name__ == "__main__":
    main()
