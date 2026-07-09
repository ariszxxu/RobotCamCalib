from __future__ import annotations

import sys
import os
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

from intr_calib_charuco import start_capture


# ---------------------------- User macros ---------------------------- #
THIRD_VIEW_PORT = "3-11.1:1.0"
THIRD_VIEW_INTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/intrinsics_third_view_cv2_charuco_1920x1080_0704_222919.yaml"
)

# supported: "av_split" keeps the original wide-frame split camera path;
# "cv2_pair" opens two independent cv2 cameras by USB port.
CALIB_CAMERA_INPUT_MODE = "cv2_pair"
ROOT_CAMERA_NAME = "thumb_web_cam"
TIP_CAMERA_NAME = "middle_finger_cam"
CV2_CAMERA_TO_PORT: dict[str, str] = {
    "thumb_web_cam": "3-5.4.3.4.4:1.0",
    "middle_finger_cam": "3-10:1.0",
}
CV2_CAMERA_TO_INTRINSICS_YAML: dict[str, str] = {
    # "thumb_web_cam": "/home/ps/RobotCamCalib1/outputs/intrinsics_cv2_fisheye_charuco_2592x1944_0707_215704.yaml",
    "thumb_web_cam": "/home/ps/RobotCamCalib1/outputs/intrinsics_thumb_web_cam_fisheye_charuco_2592x1944_0708_020331.yaml",
    "middle_finger_cam": "/home/ps/RobotCamCalib1/outputs/intrinsics_apriltag_grid_offline_eval_combined_0707_233141_233955/intrinsics_cv2_apriltag_grid_1920x1080_combined_0707_233141_233955_best_filtered.yaml",
}
CV2_CAMERA_FPS = 50
CV2_CAMERA_FOURCC = "MJPG"

FINGERTIP_AV_LEFT_RIGHT_ORDER = {"cam": [TIP_CAMERA_NAME, ROOT_CAMERA_NAME]}
APRILTAG_GRID_BOARD_NAME = "near_8mm"
APRILTAG_GRID_BOARD_YAMLS: dict[str, str] = {
    "near_8mm": (
        "/home/ps/RobotCamCalib1/outputs/apriltag_grid_36h10_a4_near_8mm/"
        "apriltag_36h10_grid_20x29_ids_579_to_0_tag8mm_gap2mm_margin3mm_a4_near.yaml"
    ),
    "a4_full_20mm": (
        "/home/ps/RobotCamCalib1/outputs/apriltag_grid_36h10_a4_full/"
        "apriltag_36h10_grid_8x11_ids_87_to_0_tag20mm_gap5mm_a4_full.yaml"
    ),
    "legacy_25h9_20mm": (
        "/home/ps/RobotCamCalib1/outputs/apriltag_grid_25h9/"
        "apriltag_25h9_grid_4x4_ids_15_to_0.yaml"
    ),
}
APRILTAG_GRID_YAML = Path(APRILTAG_GRID_BOARD_YAMLS[APRILTAG_GRID_BOARD_NAME])
APRILCUBE_SRC_DIR = Path("/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/src")


# I
# FINGERTIP_AV_CAMERA_TO_PORT = {"cam": "3-5.4.4.4:1.0"}
# ROOT_INTRINSICS_YAML = Path(
#     "/home/ps/RobotCamCalib1/outputs/intrinsics_i_root_av_320x240_0704_200627.yaml"
# )
# TIP_INTRINSICS_YAML = Path(
#     "/home/ps/RobotCamCalib1/outputs/intrinsics_i_tip_av_320x240_0704_200143.yaml"
# )
# APRILCUBE_CFG_DIR = Path(
#     "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/cube_april_36h11_12_17_1x1x1_10mm"
# )


# T
# FINGERTIP_AV_CAMERA_TO_PORT = {"cam": "3-5.4.4.2:1.0"}
# ROOT_INTRINSICS_YAML = Path(
#     "/home/ps/RobotCamCalib1/outputs/intrinsics_t_root_av_320x240_0704_201000.yaml"
# )
# TIP_INTRINSICS_YAML = Path(
#     "/home/ps/RobotCamCalib1/outputs/intrinsics_t_tip_av_320x240_0704_201253.yaml"
# )
# APRILCUBE_CFG_DIR = Path(
#     "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/cube_april_36h11_6_11_1x1x1_10mm"
# )


# M
# FINGERTIP_AV_CAMERA_TO_PORT = {"cam": "3-5.4.4.3:1.0"}
# ROOT_INTRINSICS_YAML = Path(CV2_CAMERA_TO_INTRINSICS_YAML[ROOT_CAMERA_NAME])
# TIP_INTRINSICS_YAML = Path(CV2_CAMERA_TO_INTRINSICS_YAML[TIP_CAMERA_NAME])
# APRILCUBE_CFG_DIR = Path(
#     "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/cube_april_36h11_0_5_1x1x1_10mm"
# )

FINGERTIP_AV_CAMERA_TO_PORT = {"cam": "3-5.4.4.3:1.0"}
ROOT_INTRINSICS_YAML = Path(CV2_CAMERA_TO_INTRINSICS_YAML[ROOT_CAMERA_NAME])
TIP_INTRINSICS_YAML = Path(CV2_CAMERA_TO_INTRINSICS_YAML[TIP_CAMERA_NAME])
APRILCUBE_CFG_DIR = Path(
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/cube_april_36h11_100_105_1x1x1_50mm"
)


THIRD_VIEW_FPS = 50
THIRD_VIEW_FOURCC = "MJPG"
FINGERTIP_AV_FPS = 25
FINGERTIP_AV_FOURCC = "mjpeg"

DISPLAY_SCALE_THIRD = 0.5
DISPLAY_SCALE_FINGERTIP = 0.5
THIRD_VIEW_WINDOW_NAME = "third-view E: AprilTag grid B + AprilCube Q"
_DISPLAY_WINDOW_THREAD_STARTED = False
MIN_SAMPLES_TO_SAVE = 8
OUTPUT_PATH = Path("outputs/extrinsics_wrist_Q_thumb_web_cam_middle_finger_cam_apriltag_grid.yaml")
MIN_APRILTAG_GRID_TAGS = 2
MIN_APRILTAG_GRID_CORNERS = 8
BOARD_AXIS_LENGTH_M = 0.02

AUTO_CAPTURE = True
AUTO_CAPTURE_COOLDOWN_S = 0.8
STATUS_LOG_INTERVAL_S = 1.0
PROFILE_DETECTION_TIMING = True
MAX_APRILTAG_GRID_REPROJ_PX = 2.0
MAX_FISHEYE_APRILTAG_GRID_REPROJ_PX = 3.0
MAX_APRILCUBE_REPROJ_PX = 2.0
MIN_APRILCUBE_TAGS = 1
MIN_BOARD_ROT_DELTA_DEG = 4.0
MIN_BOARD_TRANS_DELTA_M = 0.015
SAMPLE_IMAGE_ROOT = Path("outputs/extrinsics_fingertip_apriltag_grid_samples")
USABLE_SAMPLE_IMAGE_ROOT = Path("outputs/extrinsics_fingertip_apriltag_grid_usable_samples")
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
    camera_model: str
    image_size: tuple[int, int]
    K: np.ndarray
    dist: np.ndarray


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
        camera_model=str(data.get("camera_model", "pinhole")).lower(),
        image_size=tuple(int(v) for v in data["image_size"]),
        K=np.asarray(data["K"], dtype=np.float64).reshape(3, 3),
        dist=np.asarray(data.get("dist", data.get("D", [0, 0, 0, 0, 0])), dtype=np.float64).reshape(-1),
    )


def load_apriltag_grid_board(path: Path) -> AprilTagGridBoard:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data.get("target_type") != "apriltag_grid":
        raise ValueError(f"Expected target_type=apriltag_grid in {resolved}")

    tag_object_points = {
        int(tag_id): np.asarray(points, dtype=np.float64).reshape(4, 3)
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


def project_points_for_intrinsics(
    objpoints: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    if camera_model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            objpoints.reshape(1, -1, 3).astype(np.float64),
            rvec,
            tvec,
            K,
            dist.reshape(-1, 1).astype(np.float64),
        )
        return projected.reshape(-1, 1, 2)

    projected, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
    return projected


def reproj_error_for_intrinsics(
    objpoints: np.ndarray,
    imgpoints: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_model: str,
) -> float:
    projected = project_points_for_intrinsics(objpoints, rvec, tvec, K, dist, camera_model)
    return float(np.mean(np.linalg.norm(imgpoints.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)))


def solve_pnp_for_intrinsics(
    objpoints: np.ndarray,
    imgpoints: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_model: str,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    def has_positive_depth(rvec: np.ndarray, tvec: np.ndarray) -> bool:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        pts_cam = objpoints.reshape(-1, 3) @ R.T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
        return bool(np.all(pts_cam[:, 2] > 1e-4))

    if camera_model == "fisheye":
        fisheye_dist = dist.reshape(-1, 1).astype(np.float64)
        zero_dist = np.zeros(5, dtype=np.float64)
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

        def add_candidate(rvec: np.ndarray, tvec: np.ndarray) -> None:
            if not has_positive_depth(rvec, tvec):
                return
            err = reproj_error_for_intrinsics(
                objpoints,
                imgpoints,
                rvec,
                tvec,
                K,
                dist,
                camera_model,
            )
            candidates.append((err, rvec, tvec))

        undistorted_pixel = cv2.fisheye.undistortPoints(
            imgpoints.reshape(-1, 1, 2).astype(np.float64),
            K,
            fisheye_dist,
            P=K,
        )
        undistorted_normalized = cv2.fisheye.undistortPoints(
            imgpoints.reshape(-1, 1, 2).astype(np.float64),
            K,
            fisheye_dist,
            P=None,
        )
        eye_K = np.eye(3, dtype=np.float64)

        for solve_imgpoints, solve_K in (
            (undistorted_pixel, K),
            (undistorted_normalized, eye_K),
        ):
            for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
                try:
                    ok, rvec, tvec = cv2.solvePnP(
                        objpoints,
                        solve_imgpoints,
                        solve_K,
                        zero_dist,
                        flags=flag,
                    )
                except cv2.error:
                    continue
                if not ok:
                    continue
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        objpoints,
                        solve_imgpoints,
                        solve_K,
                        zero_dist,
                        rvec,
                        tvec,
                    )
                except cv2.error:
                    pass
                add_candidate(rvec, tvec)

                if flag == cv2.SOLVEPNP_IPPE:
                    try:
                        generic_ret = cv2.solvePnPGeneric(
                            objpoints,
                            solve_imgpoints,
                            solve_K,
                            zero_dist,
                            flags=cv2.SOLVEPNP_IPPE,
                        )
                    except cv2.error:
                        continue
                    if len(generic_ret) < 3:
                        continue
                    generic_ok, rvecs, tvecs = generic_ret[:3]
                    if not generic_ok:
                        continue
                    for generic_rvec, generic_tvec in zip(rvecs, tvecs):
                        add_candidate(generic_rvec, generic_tvec)

        if not candidates:
            return False, None, None
        _err, best_rvec, best_tvec = min(candidates, key=lambda x: x[0])
        return True, best_rvec, best_tvec

    ok, rvec, tvec = cv2.solvePnP(
        objpoints,
        imgpoints,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(objpoints, imgpoints, K, dist, rvec, tvec)
        except cv2.error:
            pass
    return ok, rvec, tvec


def T_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    return rvec, tvec


def load_aprilcube_tag_object_points_m() -> dict[int, np.ndarray]:
    ensure_aprilcube_on_path()
    from aprilcube.detect import build_tag_corner_map, load_cube_config  # noqa: PLC0415

    config, _face_id_sets = load_cube_config(str(APRILCUBE_CFG_DIR / "config.json"))
    tag_corner_map_mm = build_tag_corner_map(config)
    return {
        int(tag_id): np.asarray(corners_mm, dtype=np.float64).reshape(4, 3) * 0.001
        for tag_id, corners_mm in tag_corner_map_mm.items()
    }


def draw_aprilcube_reprojection(
    frame_bgr: np.ndarray,
    T_E_Q: np.ndarray,
    intr: Intrinsics,
    tag_object_points_m: dict[int, np.ndarray],
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    K, dist = scale_intrinsics(intr, (w, h))
    rvec, tvec = T_to_rvec_tvec(T_E_Q)
    out = frame_bgr.copy()

    for tag_id, objpoints in sorted(tag_object_points_m.items()):
        projected, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
        pts = np.round(projected.reshape(4, 2)).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
        for corner_idx, pt in enumerate(pts):
            color = (0, 255, 0) if corner_idx == 0 else (0, 128, 255)
            cv2.circle(out, tuple(pt), 4, color, -1, cv2.LINE_AA)
        center = np.mean(pts, axis=0).astype(np.int32)
        cv2.putText(
            out,
            f"Q id={tag_id}",
            (int(center[0]) + 4, int(center[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    try:
        cv2.drawFrameAxes(out, K, dist, rvec, tvec, 0.02)
    except cv2.error:
        pass
    return put_lines(
        out,
        [
            "AprilCube Q reprojection from saved T_E_Q",
            "yellow = projected tag boundary, green = projected corner 0",
        ],
        color=(0, 255, 255),
    )


def draw_apriltag_grid_reprojection(
    frame_bgr: np.ndarray,
    T_E_B: np.ndarray,
    intr: Intrinsics,
    board: AprilTagGridBoard,
    *,
    draw_all_tags: bool = True,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    K, dist = scale_intrinsics(intr, (w, h))
    rvec, tvec = T_to_rvec_tvec(T_E_B)
    out = frame_bgr.copy()

    if draw_all_tags:
        for tag_id, objpoints in sorted(board.tag_object_points.items()):
            projected, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
            pts = np.round(projected.reshape(4, 2)).astype(np.int32)
            cv2.polylines(out, [pts], True, (255, 180, 0), 2, cv2.LINE_AA)
            center = np.mean(pts, axis=0).astype(np.int32)
            cv2.putText(
                out,
                f"B id={tag_id}",
                (int(center[0]) + 4, int(center[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

    try:
        cv2.drawFrameAxes(out, K, dist, rvec, tvec, BOARD_AXIS_LENGTH_M)
    except cv2.error:
        pass
    return out


def draw_third_view_Q_and_B_reprojection(
    frame_bgr: np.ndarray,
    T_E_Q: np.ndarray,
    T_E_B: np.ndarray,
    intr: Intrinsics,
    board: AprilTagGridBoard,
    aprilcube_tag_object_points_m: dict[int, np.ndarray],
) -> np.ndarray:
    out = draw_aprilcube_reprojection(frame_bgr, T_E_Q, intr, aprilcube_tag_object_points_m)
    out = draw_apriltag_grid_reprojection(out, T_E_B, intr, board, draw_all_tags=True)
    return put_lines(
        out,
        [
            "AprilCube Q + AprilTag-grid B reprojection from saved poses",
            "Q: yellow cube/tag outlines; B: cyan/orange grid tag outlines",
            "axes colors: x=red, y=green, z=blue",
        ],
        color=(0, 255, 255),
    )


class AprilTagGridDetector:
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


def detect_apriltag_grid_pose(
    frame_bgr: np.ndarray,
    detector: AprilTagGridDetector,
    board: AprilTagGridBoard,
    intr: Intrinsics,
    label: str,
) -> PoseDetection:
    h, w = frame_bgr.shape[:2]
    K, dist = scale_intrinsics(intr, (w, h))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _rejected = detector.detect(gray)

    vis = frame_bgr.copy()
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    objpoints_list: list[np.ndarray] = []
    imgpoints_list: list[np.ndarray] = []
    used_ids: list[int] = []

    if marker_corners is not None and marker_ids is not None:
        for corners, marker_id_raw in zip(marker_corners, marker_ids.reshape(-1)):
            marker_id = int(marker_id_raw)
            if marker_id not in board.tag_object_points:
                continue
            objpoints_list.append(board.tag_object_points[marker_id])
            imgpoints_list.append(np.asarray(corners, dtype=np.float64).reshape(4, 2))
            used_ids.append(marker_id)

            center = np.mean(imgpoints_list[-1], axis=0)
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

    n_tags = len(used_ids)
    n_corners = n_tags * 4
    if n_tags < MIN_APRILTAG_GRID_TAGS or n_corners < MIN_APRILTAG_GRID_CORNERS:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=n_corners,
            message=(
                f"{label}: AprilTag grid tags={n_tags} corners={n_corners} "
                f"need>={MIN_APRILTAG_GRID_TAGS} tags/{MIN_APRILTAG_GRID_CORNERS} corners"
            ),
            vis=vis,
        )

    objpoints = np.concatenate(objpoints_list, axis=0).reshape(-1, 3)
    imgpoints = np.concatenate(imgpoints_list, axis=0).reshape(-1, 1, 2)
    try:
        ok, rvec, tvec = solve_pnp_for_intrinsics(
            objpoints,
            imgpoints,
            K,
            dist,
            intr.camera_model,
        )
    except cv2.error as exc:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=n_corners,
            message=f"{label}: solvePnP error corners={n_corners}: {exc.err}",
            vis=vis,
        )
    if not ok:
        return PoseDetection(ok=False, T=None, n_points=n_corners, message=f"{label}: solvePnP failed", vis=vis)

    assert rvec is not None and tvec is not None
    err = reproj_error_for_intrinsics(objpoints, imgpoints, rvec, tvec, K, dist, intr.camera_model)
    projected = project_points_for_intrinsics(objpoints, rvec, tvec, K, dist, intr.camera_model).reshape(-1, 2)
    measured = imgpoints.reshape(-1, 2)
    corner_errors = np.linalg.norm(measured - projected, axis=1)
    per_tag_errors = [
        (tag_id, float(np.mean(corner_errors[i * 4 : (i + 1) * 4])))
        for i, tag_id in enumerate(used_ids)
    ]
    worst_tag_errors = sorted(per_tag_errors, key=lambda x: x[1], reverse=True)[:5]
    worst_tag_text = ", ".join(f"{tag_id}:{tag_err:.1f}" for tag_id, tag_err in worst_tag_errors)

    if err > 10.0:
        for tag_id, tag_err in worst_tag_errors:
            tag_idx = used_ids.index(tag_id)
            center = np.mean(imgpoints_list[tag_idx], axis=0)
            cv2.putText(
                vis,
                f"e={tag_err:.0f}",
                (int(center[0]) + 4, int(center[1]) + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    T_cam_board = rvec_tvec_to_T(rvec, tvec)
    try:
        if intr.camera_model == "fisheye":
            axis = np.float64(
                [
                    [0.0, 0.0, 0.0],
                    [BOARD_AXIS_LENGTH_M, 0.0, 0.0],
                    [0.0, BOARD_AXIS_LENGTH_M, 0.0],
                    [0.0, 0.0, BOARD_AXIS_LENGTH_M],
                ]
            )
            pts = project_points_for_intrinsics(axis, rvec, tvec, K, dist, intr.camera_model).reshape(-1, 2)
            origin = tuple(np.round(pts[0]).astype(int))
            cv2.line(vis, origin, tuple(np.round(pts[1]).astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(vis, origin, tuple(np.round(pts[2]).astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(vis, origin, tuple(np.round(pts[3]).astype(int)), (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.drawFrameAxes(vis, K, dist, rvec, tvec, BOARD_AXIS_LENGTH_M)
    except cv2.error:
        pass
    return PoseDetection(
        ok=True,
        T=T_cam_board,
        n_points=n_corners,
        reproj_error=err,
        message=(
            f"{label}: B ok tags={n_tags} corners={n_corners} err={err:.2f}px "
            f"z={float(tvec.reshape(3)[2]):.3f}m worst=[{worst_tag_text}] ids={sorted(used_ids)}"
        ),
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


def open_fingertip_av_manager(root_intr: Intrinsics):
    from cameras import AVCameraManager

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


class CV2CameraPairManager:
    def __init__(
        self,
        root_name: str,
        tip_name: str,
        root_intr: Intrinsics,
        tip_intr: Intrinsics,
    ) -> None:
        self.root_name = root_name
        self.tip_name = tip_name
        self.root_intr = root_intr
        self.tip_intr = tip_intr
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.active_devices: dict[str, int | str] = {}
        self.requested_sizes: dict[str, tuple[int, int]] = {}
        self.actual_sizes: dict[str, tuple[int, int]] = {}

    def start(self) -> None:
        for internal_name, camera_name, intr in (
            (self.root_name, self.root_name, self.root_intr),
            (self.tip_name, self.tip_name, self.tip_intr),
        ):
            if camera_name not in CV2_CAMERA_TO_PORT:
                available = ", ".join(CV2_CAMERA_TO_PORT.keys())
                raise ValueError(f"Unknown CV2 camera '{camera_name}'. Available: {available}")
            cap, device = start_capture(
                CV2_CAMERA_TO_PORT[camera_name],
                intr.image_size[0],
                intr.image_size[1],
                CV2_CAMERA_FPS,
                CV2_CAMERA_FOURCC,
            )
            self.caps[internal_name] = cap
            self.active_devices[internal_name] = device
            self.requested_sizes[internal_name] = intr.image_size
            self.actual_sizes[internal_name] = (
                int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )

    def get_frames(self, img_size: Optional[tuple[int, int]] = None) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for internal_name, cap in self.caps.items():
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            target_size = img_size
            if target_size is not None:
                target_w, target_h = target_size
                h, w = frame.shape[:2]
                if (w, h) != (target_w, target_h):
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames[internal_name] = frame
        return frames

    def release_all(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()


def open_wrist_camera_manager(root_intr: Intrinsics, tip_intr: Intrinsics):
    if CALIB_CAMERA_INPUT_MODE == "av_split":
        manager = open_fingertip_av_manager(root_intr)
        return manager, {
            "mode": CALIB_CAMERA_INPUT_MODE,
            ROOT_CAMERA_NAME: next(iter(FINGERTIP_AV_CAMERA_TO_PORT.values()), None),
            TIP_CAMERA_NAME: next(iter(FINGERTIP_AV_CAMERA_TO_PORT.values()), None),
        }

    if CALIB_CAMERA_INPUT_MODE == "cv2_pair":
        manager = CV2CameraPairManager(ROOT_CAMERA_NAME, TIP_CAMERA_NAME, root_intr, tip_intr)
        manager.start()
        return manager, {
            "mode": CALIB_CAMERA_INPUT_MODE,
            "active_devices": dict(manager.active_devices),
            "requested_sizes": dict(manager.requested_sizes),
            "actual_sizes": dict(manager.actual_sizes),
        }

    raise ValueError(f"Unsupported CALIB_CAMERA_INPUT_MODE={CALIB_CAMERA_INPUT_MODE}")


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
    f"{ROOT_CAMERA_NAME}_residual_rot_deg",
    f"{ROOT_CAMERA_NAME}_residual_trans_m",
    f"{TIP_CAMERA_NAME}_residual_rot_deg",
    f"{TIP_CAMERA_NAME}_residual_trans_m",
    f"{ROOT_CAMERA_NAME}_{TIP_CAMERA_NAME}_consistency_rot_deg",
    f"{ROOT_CAMERA_NAME}_{TIP_CAMERA_NAME}_consistency_trans_m",
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
        f"Q_T_{ROOT_CAMERA_NAME}": Q_T_root,
        f"Q_T_{TIP_CAMERA_NAME}": Q_T_tip,
        f"{ROOT_CAMERA_NAME}_residual_rot_deg": stats([x[0] for x in root_res]),
        f"{ROOT_CAMERA_NAME}_residual_trans_m": stats([x[1] for x in root_res]),
        f"{TIP_CAMERA_NAME}_residual_rot_deg": stats([x[0] for x in tip_res]),
        f"{TIP_CAMERA_NAME}_residual_trans_m": stats([x[1] for x in tip_res]),
        f"{ROOT_CAMERA_NAME}_{TIP_CAMERA_NAME}_consistency_rot_deg": stats([x[0] for x in root_tip_delta]),
        f"{ROOT_CAMERA_NAME}_{TIP_CAMERA_NAME}_consistency_trans_m": stats([x[1] for x in root_tip_delta]),
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
            f"{ROOT_CAMERA_NAME}_rot_deg": root_rot,
            f"{ROOT_CAMERA_NAME}_trans_m": root_trans,
            f"{TIP_CAMERA_NAME}_rot_deg": tip_rot,
            f"{TIP_CAMERA_NAME}_trans_m": tip_trans,
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
            raw_solution[f"Q_T_{ROOT_CAMERA_NAME}"],
            raw_solution[f"Q_T_{TIP_CAMERA_NAME}"],
        )
        return raw_solution

    min_inliers = max(MIN_SAMPLES_TO_SAVE, 1)
    inlier_indices = [s.index for s in samples]
    iterations: list[dict[str, Any]] = []

    for iteration in range(OUTLIER_MAX_ITERATIONS):
        active = [s for s in samples if s.index in set(inlier_indices)]
        solution = summarize_samples(active)
        residuals = residuals_against_solution(
            samples,
            solution[f"Q_T_{ROOT_CAMERA_NAME}"],
            solution[f"Q_T_{TIP_CAMERA_NAME}"],
        )
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
        filtered_solution[f"Q_T_{ROOT_CAMERA_NAME}"],
        filtered_solution[f"Q_T_{TIP_CAMERA_NAME}"],
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
    board: AprilTagGridBoard,
) -> dict[str, Any]:
    inlier_indices = set(solution.get("inlier_indices", [s.index for s in samples]))
    sample_residuals = solution.get("sample_residuals", {})
    cache_dir = USABLE_SAMPLE_IMAGE_ROOT / output_path.with_suffix("").name
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    usable_paths_by_index: dict[int, dict[str, str]] = {}
    pickle_samples: list[dict[str, Any]] = []
    third_intr = load_intrinsics(THIRD_VIEW_INTRINSICS_YAML)
    aprilcube_tag_object_points_m = load_aprilcube_tag_object_points_m()

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

        third_view_bgr = cv2.imread(str(Path(s.image_paths["third_view"])), cv2.IMREAD_COLOR)
        if third_view_bgr is None:
            raise RuntimeError(f"Failed to read third-view image for reprojection: {s.image_paths['third_view']}")
        aprilcube_reproj_bgr = draw_aprilcube_reprojection(
            third_view_bgr,
            s.T_E_Q,
            third_intr,
            aprilcube_tag_object_points_m,
        )
        aprilcube_reproj_path = cache_dir / f"sample_{s.index:04d}_third_view_aprilcube_reprojection.png"
        if not cv2.imwrite(str(aprilcube_reproj_path), aprilcube_reproj_bgr):
            raise RuntimeError(f"Failed to save AprilCube reprojection image: {aprilcube_reproj_path}")
        ok, encoded_reproj = cv2.imencode(".png", aprilcube_reproj_bgr)
        if not ok:
            raise RuntimeError(
                f"Failed to PNG-encode AprilCube reprojection image: {aprilcube_reproj_path}"
            )
        encoded_images["third_view_aprilcube_reprojection"] = encoded_reproj.tobytes()
        image_shapes["third_view_aprilcube_reprojection"] = [int(v) for v in aprilcube_reproj_bgr.shape]
        copied_paths["third_view_aprilcube_reprojection"] = str(aprilcube_reproj_path)

        third_view_Q_B_bgr = draw_third_view_Q_and_B_reprojection(
            third_view_bgr,
            s.T_E_Q,
            s.T_E_B,
            third_intr,
            board,
            aprilcube_tag_object_points_m,
        )
        third_view_Q_B_path = cache_dir / f"sample_{s.index:04d}_third_view_Q_and_B_reprojection.png"
        if not cv2.imwrite(str(third_view_Q_B_path), third_view_Q_B_bgr):
            raise RuntimeError(f"Failed to save Q+B reprojection image: {third_view_Q_B_path}")
        ok, encoded_Q_B = cv2.imencode(".png", third_view_Q_B_bgr)
        if not ok:
            raise RuntimeError(f"Failed to PNG-encode Q+B reprojection image: {third_view_Q_B_path}")
        encoded_images["third_view_Q_and_B_reprojection"] = encoded_Q_B.tobytes()
        image_shapes["third_view_Q_and_B_reprojection"] = [int(v) for v in third_view_Q_B_bgr.shape]
        copied_paths["third_view_Q_and_B_reprojection"] = str(third_view_Q_B_path)

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
                f"T_{ROOT_CAMERA_NAME}_B": s.T_root_B,
                f"T_{TIP_CAMERA_NAME}_B": s.T_tip_B,
                "T_E_B": s.T_E_B,
                "T_E_Q": s.T_E_Q,
                f"T_Q_{ROOT_CAMERA_NAME}": s.T_Q_root,
                f"T_Q_{TIP_CAMERA_NAME}": s.T_Q_tip,
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
            ROOT_CAMERA_NAME: f"{ROOT_CAMERA_NAME} optical frame",
            TIP_CAMERA_NAME: f"{TIP_CAMERA_NAME} optical frame",
            "E": "third-view calibration camera optical frame",
            "B": "AprilTag grid board frame",
            "pose_notation": "A_T_B maps points from B frame into A frame",
        },
        f"Q_T_{ROOT_CAMERA_NAME}": solution[f"Q_T_{ROOT_CAMERA_NAME}"],
        f"Q_T_{TIP_CAMERA_NAME}": solution[f"Q_T_{TIP_CAMERA_NAME}"],
        "diagnostics": {k: solution[k] for k in DIAGNOSTIC_KEYS},
        "image_channels": {
            "third_view": "raw third-view BGR frame",
            "third_view_aprilcube_reprojection": (
                "third-view BGR frame with AprilCube Q projected from saved T_E_Q"
            ),
            "third_view_Q_and_B_reprojection": (
                "third-view BGR frame with AprilCube Q and AprilTag-grid B projected from saved T_E_Q/T_E_B"
            ),
            ROOT_CAMERA_NAME: f"raw {ROOT_CAMERA_NAME} BGR frame",
            TIP_CAMERA_NAME: f"raw {TIP_CAMERA_NAME} BGR frame",
        },
        "aprilcube_reprojection": {
            "source_pose": "T_E_Q",
            "tag_corner_units": "meters",
            "tag_ids": sorted(int(v) for v in aprilcube_tag_object_points_m.keys()),
            "third_view_intrinsics_yaml": str(THIRD_VIEW_INTRINSICS_YAML),
        },
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


def save_results(
    path: Path,
    samples: list[Sample],
    solution: dict[str, Any],
    board: AprilTagGridBoard,
) -> Path:
    output_path = append_timestamp(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inlier_indices = set(solution.get("inlier_indices", [s.index for s in samples]))
    sample_residuals = solution.get("sample_residuals", {})
    rejection_reasons = solution.get("outlier_rejection", {}).get("rejection_reasons", {})
    usable_cache = export_usable_sample_cache(output_path, samples, solution, board)
    usable_paths_by_index = usable_cache.get("paths_by_sample_index", {})

    data = {
        "frame_convention": {
            "Q": "AprilCube frame / rig frame",
            ROOT_CAMERA_NAME: f"{ROOT_CAMERA_NAME} optical frame",
            TIP_CAMERA_NAME: f"{TIP_CAMERA_NAME} optical frame",
            "E": "third-view calibration camera optical frame",
            "B": "AprilTag grid board frame",
            "pose_notation": "A_T_B maps points from B frame into A frame",
            "aprilcube_pose_convention": APRILCUBE_POSE_CONVENTION,
        },
        "inputs": {
            "calib_camera_input_mode": CALIB_CAMERA_INPUT_MODE,
            "camera_names": [ROOT_CAMERA_NAME, TIP_CAMERA_NAME],
            "cv2_camera_to_port": CV2_CAMERA_TO_PORT,
            "cv2_camera_to_intrinsics_yaml": CV2_CAMERA_TO_INTRINSICS_YAML,
            "third_view_port": THIRD_VIEW_PORT,
            "fingertip_av_camera_to_port": FINGERTIP_AV_CAMERA_TO_PORT,
            "fingertip_av_left_right_order": FINGERTIP_AV_LEFT_RIGHT_ORDER,
            "third_view_intrinsics_yaml": str(THIRD_VIEW_INTRINSICS_YAML),
            f"{ROOT_CAMERA_NAME}_intrinsics_yaml": str(ROOT_INTRINSICS_YAML),
            f"{TIP_CAMERA_NAME}_intrinsics_yaml": str(TIP_INTRINSICS_YAML),
            "aprilcube_cfg_dir": str(APRILCUBE_CFG_DIR),
            "apriltag_grid_board_name": str(APRILTAG_GRID_BOARD_NAME),
            "auto_capture": {
                "enabled": bool(AUTO_CAPTURE),
                "cooldown_s": float(AUTO_CAPTURE_COOLDOWN_S),
                "max_apriltag_grid_reproj_px": float(MAX_APRILTAG_GRID_REPROJ_PX),
                "max_fisheye_apriltag_grid_reproj_px": float(MAX_FISHEYE_APRILTAG_GRID_REPROJ_PX),
                "min_apriltag_grid_tags": int(MIN_APRILTAG_GRID_TAGS),
                "min_apriltag_grid_corners": int(MIN_APRILTAG_GRID_CORNERS),
                "max_aprilcube_reproj_px": float(MAX_APRILCUBE_REPROJ_PX),
                "min_aprilcube_tags": int(MIN_APRILCUBE_TAGS),
                "min_board_rot_delta_deg": float(MIN_BOARD_ROT_DELTA_DEG),
                "min_board_trans_delta_m": float(MIN_BOARD_TRANS_DELTA_M),
                "auto_stop_after_usable_sample_groups": bool(AUTO_STOP_AFTER_USABLE_SAMPLE_GROUPS),
                "max_usable_sample_groups": int(MAX_USABLE_SAMPLE_GROUPS),
            },
            "usable_sample_cache": {
                "cache_root": str(USABLE_SAMPLE_IMAGE_ROOT),
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
            "apriltag_grid": {
                "yaml": str(board.path),
                "tag_family": str(board.tag_family),
                "id_grid": board.id_grid,
                "rows": int(board.rows),
                "cols": int(board.cols),
                "tag_size_m": float(board.tag_size_m),
                "tag_gap_m": float(board.tag_gap_m),
                "board_width_m": float(board.board_width_m),
                "board_height_m": float(board.board_height_m),
            },
        },
        "num_samples": len(inlier_indices),
        "num_raw_samples": len(samples),
        f"Q_T_{ROOT_CAMERA_NAME}": solution[f"Q_T_{ROOT_CAMERA_NAME}"].tolist(),
        f"Q_T_{TIP_CAMERA_NAME}": solution[f"Q_T_{TIP_CAMERA_NAME}"].tolist(),
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
                f"T_{ROOT_CAMERA_NAME}_B": s.T_root_B.tolist(),
                f"T_{TIP_CAMERA_NAME}_B": s.T_tip_B.tolist(),
                "T_E_B": s.T_E_B.tolist(),
                "T_E_Q": s.T_E_Q.tolist(),
                f"T_Q_{ROOT_CAMERA_NAME}": s.T_Q_root.tolist(),
                f"T_Q_{TIP_CAMERA_NAME}": s.T_Q_tip.tolist(),
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


def create_display_window(name: str, image_size: tuple[int, int], scale: float, position: tuple[int, int]) -> None:
    global _DISPLAY_WINDOW_THREAD_STARTED
    if not _DISPLAY_WINDOW_THREAD_STARTED:
        print(
            "[INFO] starting cv2 window thread "
            f"DISPLAY={os.environ.get('DISPLAY')} "
            f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM')}",
            flush=True,
        )
        thread_result = cv2.startWindowThread()
        print(f"[INFO] cv2.startWindowThread() -> {thread_result}", flush=True)
        _DISPLAY_WINDOW_THREAD_STARTED = True
    width = max(1, int(round(image_size[0] * scale)))
    height = max(1, int(round(image_size[1] * scale)))
    print(f"[INFO] creating cv2 window {name!r} size={width}x{height} pos={position}", flush=True)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.moveWindow(name, position[0], position[1])
    visible = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
    print(f"[INFO] cv2 window {name!r} visible_prop={visible}", flush=True)


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
        ROOT_CAMERA_NAME: image_dir / f"sample_{sample_index:04d}_{ROOT_CAMERA_NAME}.png",
        TIP_CAMERA_NAME: image_dir / f"sample_{sample_index:04d}_{TIP_CAMERA_NAME}.png",
    }
    frames = {
        "third_view": E_frame,
        ROOT_CAMERA_NAME: root_frame,
        TIP_CAMERA_NAME: tip_frame,
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
    root_intr: Intrinsics,
    tip_intr: Intrinsics,
    third_intr: Intrinsics,
) -> tuple[bool, str]:
    grid_errors = {
        ROOT_CAMERA_NAME: (root_det.reproj_error, root_intr),
        TIP_CAMERA_NAME: (tip_det.reproj_error, tip_intr),
        "E/B": (E_B_det.reproj_error, third_intr),
    }
    for name, (err, intr) in grid_errors.items():
        max_err = (
            MAX_FISHEYE_APRILTAG_GRID_REPROJ_PX
            if intr.camera_model == "fisheye"
            else MAX_APRILTAG_GRID_REPROJ_PX
        )
        if not np.isfinite(err) or err > max_err:
            return False, f"{name} AprilTag grid err {err:.2f}px > {max_err:.2f}px"

    grid_points = {
        ROOT_CAMERA_NAME: root_det.n_points,
        TIP_CAMERA_NAME: tip_det.n_points,
        "E/B": E_B_det.n_points,
    }
    for name, n_points in grid_points.items():
        if n_points < MIN_APRILTAG_GRID_CORNERS:
            return False, f"{name} AprilTag grid corners {n_points} < {MIN_APRILTAG_GRID_CORNERS}"

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
    root_intr: Intrinsics,
    tip_intr: Intrinsics,
    third_intr: Intrinsics,
) -> tuple[bool, str]:
    if not AUTO_CAPTURE:
        return False, "auto off"
    if not all_ok:
        return False, "waiting for all poses"
    if now - last_auto_time < AUTO_CAPTURE_COOLDOWN_S:
        return False, "cooldown"

    ok, reason = quality_ok(root_det, tip_det, E_B_det, E_Q_det, root_intr, tip_intr, third_intr)
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
            f"{ROOT_CAMERA_NAME}_apriltag_grid_reproj_px": float(root_det.reproj_error),
            f"{TIP_CAMERA_NAME}_apriltag_grid_reproj_px": float(tip_det.reproj_error),
            "E_apriltag_grid_reproj_px": float(E_B_det.reproj_error),
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

    board = load_apriltag_grid_board(APRILTAG_GRID_YAML)
    grid_detector = AprilTagGridDetector(board)
    aprilcube_detector = create_aprilcube_detector(third_intr)

    third_cap, third_device = open_third_view_camera(third_intr)
    wrist_camera_manager, wrist_devices = open_wrist_camera_manager(root_intr, tip_intr)

    samples: list[Sample] = []
    sample_image_dir = create_sample_image_dir()
    last_auto_time = 0.0
    last_saved_T_E_B: Optional[np.ndarray] = None
    last_auto_reason = "not evaluated"
    last_status_log_time = 0.0
    frame_idx = 0
    third_actual_size = (
        int(round(third_cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(third_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    print(f"[INFO] third-view active_device={third_device}, intrinsics={third_intr.path}")
    print(
        f"[INFO] third-view frame_size requested={third_intr.image_size[0]}x{third_intr.image_size[1]} "
        f"actual={third_actual_size[0]}x{third_actual_size[1]} "
        f"match={third_actual_size == third_intr.image_size}"
    )
    print(
        f"[INFO] input_mode={CALIB_CAMERA_INPUT_MODE}, "
        f"cameras={ROOT_CAMERA_NAME}, {TIP_CAMERA_NAME}"
    )
    print(
        f"[INFO] {ROOT_CAMERA_NAME} intrinsics={root_intr.path} model={root_intr.camera_model}, "
        f"{TIP_CAMERA_NAME} intrinsics={tip_intr.path} model={tip_intr.camera_model}"
    )
    print(
        f"[INFO] AprilTag grid board_name={APRILTAG_GRID_BOARD_NAME}, path={board.path}, family={board.tag_family}, "
        f"grid={board.cols}x{board.rows}, tag={board.tag_size_m * 1000.0:.1f}mm, "
        f"gap={board.tag_gap_m * 1000.0:.1f}mm"
    )
    print(f"[INFO] wrist camera devices={wrist_devices}")
    if CALIB_CAMERA_INPUT_MODE == "cv2_pair":
        requested_sizes = wrist_devices.get("requested_sizes", {})
        actual_sizes = wrist_devices.get("actual_sizes", {})
        for camera_name in (ROOT_CAMERA_NAME, TIP_CAMERA_NAME):
            req = requested_sizes.get(camera_name)
            actual = actual_sizes.get(camera_name)
            if req is None or actual is None:
                continue
            print(
                f"[INFO] {camera_name} frame_size requested={req[0]}x{req[1]} "
                f"actual={actual[0]}x{actual[1]} match={actual == req}"
            )
    print(f"[INFO] sample images will be saved under {sample_image_dir}")
    print(f"[INFO] usable inlier images will be copied under {USABLE_SAMPLE_IMAGE_ROOT}")
    print(f"[INFO] auto stop after {MAX_USABLE_SAMPLE_GROUPS} valid sample groups: {AUTO_STOP_AFTER_USABLE_SAMPLE_GROUPS}")
    print("[INFO] Press s to manually store a valid synchronized sample; c clears; q solves+saves+quits.")

    create_display_window(THIRD_VIEW_WINDOW_NAME, third_intr.image_size, DISPLAY_SCALE_THIRD, (0, 0))
    create_display_window(ROOT_CAMERA_NAME, root_intr.image_size, DISPLAY_SCALE_FINGERTIP, (980, 0))
    create_display_window(TIP_CAMERA_NAME, tip_intr.image_size, DISPLAY_SCALE_FINGERTIP, (0, 580))

    try:
        stop_requested = False
        while True:
            frame_idx += 1
            ok, E_frame = third_cap.read()
            if not ok or E_frame is None:
                print("[WARN] no third-view frame")
                time.sleep(0.02)
                continue

            wrist_img_size = root_intr.image_size if CALIB_CAMERA_INPUT_MODE == "av_split" else None
            av_frames = wrist_camera_manager.get_frames(img_size=wrist_img_size)
            root_frame = av_frames.get(ROOT_CAMERA_NAME)
            tip_frame = av_frames.get(TIP_CAMERA_NAME)
            if root_frame is None or tip_frame is None:
                print(f"[WARN] missing {ROOT_CAMERA_NAME}/{TIP_CAMERA_NAME} wrist camera frames")
                time.sleep(0.02)
                continue

            detect_start = time.perf_counter()
            root_det = detect_apriltag_grid_pose(root_frame, grid_detector, board, root_intr, ROOT_CAMERA_NAME)
            root_dt = time.perf_counter()
            tip_det = detect_apriltag_grid_pose(tip_frame, grid_detector, board, tip_intr, TIP_CAMERA_NAME)
            tip_dt = time.perf_counter()
            E_B_det = detect_apriltag_grid_pose(E_frame, grid_detector, board, third_intr, "E/B")
            E_B_dt = time.perf_counter()
            E_Q_det = detect_aprilcube_pose(E_frame, aprilcube_detector)
            E_Q_dt = time.perf_counter()
            timing_now = time.time()
            if PROFILE_DETECTION_TIMING and timing_now - last_status_log_time >= STATUS_LOG_INTERVAL_S:
                print(
                    "[TIMING] detection "
                    f"{ROOT_CAMERA_NAME}={(root_dt - detect_start) * 1000.0:.1f}ms "
                    f"{TIP_CAMERA_NAME}={(tip_dt - root_dt) * 1000.0:.1f}ms "
                    f"E/B={(E_B_dt - tip_dt) * 1000.0:.1f}ms "
                    f"E/Q={(E_Q_dt - E_B_dt) * 1000.0:.1f}ms "
                    f"total={(E_Q_dt - detect_start) * 1000.0:.1f}ms"
                )

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
                root_intr=root_intr,
                tip_intr=tip_intr,
                third_intr=third_intr,
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
            if now - last_status_log_time >= STATUS_LOG_INTERVAL_S:
                last_status_log_time = now
                print("[STATUS] " + status[0])
                for line in status[1:6]:
                    print(f"  {line}")

            if E_Q_det.vis is not None and E_B_det.vis is not None:
                # The third-view camera must show both detections: AprilCube Q
                # and AprilTag grid board B. Each detector draws on the raw frame, so
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

            cv2.imshow(THIRD_VIEW_WINDOW_NAME, resize_for_display(E_vis, DISPLAY_SCALE_THIRD))
            cv2.imshow(ROOT_CAMERA_NAME, resize_for_display(root_vis, DISPLAY_SCALE_FINGERTIP))
            cv2.imshow(TIP_CAMERA_NAME, resize_for_display(tip_vis, DISPLAY_SCALE_FINGERTIP))

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
        wrist_camera_manager.release_all()
        cv2.destroyAllWindows()

    if len(samples) < MIN_SAMPLES_TO_SAVE:
        print(f"[WARN] only {len(samples)} samples; need >= {MIN_SAMPLES_TO_SAVE}. Nothing saved.")
        return

    solution = solve_with_outlier_rejection(samples)
    output_path = save_results(OUTPUT_PATH, samples, solution, board)
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
