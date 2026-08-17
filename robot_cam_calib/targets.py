"""Camera models and target pose detectors used by current workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from .geometry import inv_T, make_T


MIN_APRILTAG_GRID_TAGS = 2
MIN_APRILTAG_GRID_CORNERS = 8
BOARD_AXIS_LENGTH_M = 0.02
MAX_APRILCUBE_MULTI_TAG_FACE_REPROJ_PX = 5.0
APRILCUBE_POSE_CONVENTION = "T_E_Q"


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


@dataclass
class PoseDetection:
    ok: bool
    T: Optional[np.ndarray]
    n_points: int = 0
    reproj_error: float = float("inf")
    message: str = ""
    vis: Optional[np.ndarray] = None


@dataclass
class AprilCubeDetectionContext:
    detector: Any
    face_id_sets: dict[str, set[int]]
    tag_corner_map_mm: dict[int, np.ndarray]
    multi_tag_faces: set[str]


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


def load_apriltag_grid_board(path: Path) -> AprilTagGridBoard:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data.get("target_type") != "apriltag_grid":
        raise ValueError(f"Expected target_type=apriltag_grid in {resolved}")
    return AprilTagGridBoard(
        path=resolved,
        tag_family=str(data["tag_family"]),
        id_grid=[[int(value) for value in row] for row in data["id_grid"]],
        tag_object_points={
            int(tag_id): np.asarray(points, dtype=np.float64).reshape(4, 3)
            for tag_id, points in data["tag_object_points"].items()
        },
        rows=int(data["rows"]),
        cols=int(data["cols"]),
        tag_size_m=float(data["tag_size_m"]),
        tag_gap_m=float(data["tag_gap_m"]),
        board_width_m=float(data["board_width_m"]),
        board_height_m=float(data["board_height_m"]),
    )


def scale_intrinsics(
    intr: Intrinsics,
    new_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    old_width, old_height = intr.image_size
    new_width, new_height = new_size
    if (old_width, old_height) == (new_width, new_height):
        return intr.K.copy(), intr.dist.copy()
    scale_x = new_width / old_width
    scale_y = new_height / old_height
    matrix = intr.K.copy()
    matrix[0, 0] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[0, 2] *= scale_x
    matrix[1, 2] *= scale_y
    return matrix, intr.dist.copy()


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(
        np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    )
    return make_T(rotation, np.asarray(tvec, dtype=np.float64).reshape(3))


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
    projected = project_points_for_intrinsics(
        objpoints,
        rvec,
        tvec,
        K,
        dist,
        camera_model,
    )
    return float(
        np.mean(
            np.linalg.norm(
                imgpoints.reshape(-1, 2) - projected.reshape(-1, 2),
                axis=1,
            )
        )
    )


def solve_pnp_for_intrinsics(
    objpoints: np.ndarray,
    imgpoints: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_model: str,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    def has_positive_depth(rvec: np.ndarray, tvec: np.ndarray) -> bool:
        rotation, _ = cv2.Rodrigues(
            np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        )
        points_camera = (
            objpoints.reshape(-1, 3) @ rotation.T
            + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
        )
        return bool(np.all(points_camera[:, 2] > 1e-4))

    if camera_model == "fisheye":
        fisheye_dist = dist.reshape(-1, 1).astype(np.float64)
        zero_dist = np.zeros(5, dtype=np.float64)
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

        def add_candidate(rvec: np.ndarray, tvec: np.ndarray) -> None:
            if not has_positive_depth(rvec, tvec):
                return
            error = reproj_error_for_intrinsics(
                objpoints,
                imgpoints,
                rvec,
                tvec,
                K,
                dist,
                camera_model,
            )
            candidates.append((error, rvec, tvec))

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
        for solve_imgpoints, solve_K in (
            (undistorted_pixel, K),
            (undistorted_normalized, np.eye(3, dtype=np.float64)),
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
                if flag != cv2.SOLVEPNP_IPPE:
                    continue
                try:
                    generic = cv2.solvePnPGeneric(
                        objpoints,
                        solve_imgpoints,
                        solve_K,
                        zero_dist,
                        flags=cv2.SOLVEPNP_IPPE,
                    )
                except cv2.error:
                    continue
                if len(generic) >= 3 and generic[0]:
                    for generic_rvec, generic_tvec in zip(
                        generic[1],
                        generic[2],
                    ):
                        add_candidate(generic_rvec, generic_tvec)
        if not candidates:
            return False, None, None
        _error, best_rvec, best_tvec = min(
            candidates,
            key=lambda candidate: candidate[0],
        )
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
            rvec, tvec = cv2.solvePnPRefineLM(
                objpoints,
                imgpoints,
                K,
                dist,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
    return ok, rvec, tvec


class AprilTagGridDetector:
    def __init__(self, board: AprilTagGridBoard) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco is missing. Install opencv-contrib-python."
            )
        if not hasattr(cv2.aruco, board.tag_family):
            raise ValueError(
                f"OpenCV does not provide AprilTag dictionary {board.tag_family}"
            )
        self.board = board
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, board.tag_family)
        )
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.params = cv2.aruco.DetectorParameters()
        else:
            self.params = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )

    def detect(self, gray: np.ndarray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.params,
        )


def detect_apriltag_grid_pose(
    frame_bgr: np.ndarray,
    detector: AprilTagGridDetector,
    board: AprilTagGridBoard,
    intr: Intrinsics,
    label: str,
) -> PoseDetection:
    height, width = frame_bgr.shape[:2]
    K, dist = scale_intrinsics(intr, (width, height))
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _rejected = detector.detect(gray)
    vis = frame_bgr.copy()
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_ids: list[int] = []
    if marker_corners is not None and marker_ids is not None:
        for corners, raw_id in zip(marker_corners, marker_ids.reshape(-1)):
            tag_id = int(raw_id)
            if tag_id not in board.tag_object_points:
                continue
            object_points.append(board.tag_object_points[tag_id])
            points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
            image_points.append(points)
            used_ids.append(tag_id)
            center = np.mean(points, axis=0)
            cv2.putText(
                vis,
                f"id={tag_id}",
                (int(center[0]) + 4, int(center[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    tag_count = len(used_ids)
    corner_count = tag_count * 4
    if (
        tag_count < MIN_APRILTAG_GRID_TAGS
        or corner_count < MIN_APRILTAG_GRID_CORNERS
    ):
        return PoseDetection(
            ok=False,
            T=None,
            n_points=corner_count,
            message=(
                f"{label}: AprilTag grid tags={tag_count} "
                f"corners={corner_count} need>={MIN_APRILTAG_GRID_TAGS} "
                f"tags/{MIN_APRILTAG_GRID_CORNERS} corners"
            ),
            vis=vis,
        )

    objpoints = np.concatenate(object_points).reshape(-1, 3)
    imgpoints = np.concatenate(image_points).reshape(-1, 1, 2)
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
            n_points=corner_count,
            message=f"{label}: solvePnP error corners={corner_count}: {exc.err}",
            vis=vis,
        )
    if not ok or rvec is None or tvec is None:
        return PoseDetection(
            ok=False,
            T=None,
            n_points=corner_count,
            message=f"{label}: solvePnP failed",
            vis=vis,
        )

    error = reproj_error_for_intrinsics(
        objpoints,
        imgpoints,
        rvec,
        tvec,
        K,
        dist,
        intr.camera_model,
    )
    projected = project_points_for_intrinsics(
        objpoints,
        rvec,
        tvec,
        K,
        dist,
        intr.camera_model,
    ).reshape(-1, 2)
    corner_errors = np.linalg.norm(
        imgpoints.reshape(-1, 2) - projected,
        axis=1,
    )
    per_tag_errors = [
        (tag_id, float(np.mean(corner_errors[index * 4 : (index + 1) * 4])))
        for index, tag_id in enumerate(used_ids)
    ]
    worst_errors = sorted(
        per_tag_errors,
        key=lambda value: value[1],
        reverse=True,
    )[:5]
    worst_text = ", ".join(
        f"{tag_id}:{tag_error:.1f}" for tag_id, tag_error in worst_errors
    )
    if error > 10.0:
        for tag_id, tag_error in worst_errors:
            tag_index = used_ids.index(tag_id)
            center = np.mean(image_points[tag_index], axis=0)
            cv2.putText(
                vis,
                f"e={tag_error:.0f}",
                (int(center[0]) + 4, int(center[1]) + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    transform = rvec_tvec_to_T(rvec, tvec)
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
            points = project_points_for_intrinsics(
                axis,
                rvec,
                tvec,
                K,
                dist,
                intr.camera_model,
            ).reshape(-1, 2)
            origin = tuple(np.round(points[0]).astype(int))
            for point, color in zip(
                points[1:],
                ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
            ):
                cv2.line(
                    vis,
                    origin,
                    tuple(np.round(point).astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        else:
            cv2.drawFrameAxes(
                vis,
                K,
                dist,
                rvec,
                tvec,
                BOARD_AXIS_LENGTH_M,
            )
    except cv2.error:
        pass
    return PoseDetection(
        ok=True,
        T=transform,
        n_points=corner_count,
        reproj_error=error,
        message=(
            f"{label}: B ok tags={tag_count} corners={corner_count} "
            f"err={error:.2f}px z={float(tvec.reshape(3)[2]):.3f}m "
            f"worst=[{worst_text}] ids={sorted(used_ids)}"
        ),
        vis=vis,
    )


def solve_aprilcube_multi_tag_face_pose(
    result: dict[str, Any],
    context: AprilCubeDetectionContext,
    intr: Intrinsics,
    frame_size: tuple[int, int],
) -> Optional[tuple[np.ndarray, np.ndarray, float, str, list[int]]]:
    visible_faces = {
        str(face_name) for face_name in result.get("visible_faces", set())
    }
    if len(visible_faces) != 1:
        return None
    face_name = next(iter(visible_faces))
    if face_name not in context.multi_tag_faces:
        return None
    face_ids = context.face_id_sets[face_name]
    detections = [
        (int(tag_id), np.asarray(corners, dtype=np.float64).reshape(4, 2))
        for tag_id, corners in result.get("detections", [])
        if int(tag_id) in face_ids
        and int(tag_id) in context.tag_corner_map_mm
    ]
    if len(detections) < 2:
        return None
    object_points = np.vstack(
        [context.tag_corner_map_mm[tag_id] for tag_id, _ in detections]
    ).astype(np.float64)
    image_points = np.vstack([corners for _, corners in detections]).astype(
        np.float64
    )
    K, dist = scale_intrinsics(intr, frame_size)
    solve_points = image_points
    solve_dist = dist
    if intr.camera_model == "fisheye":
        solve_points = cv2.fisheye.undistortPoints(
            image_points.reshape(-1, 1, 2),
            K,
            dist.reshape(-1, 1),
            P=K,
        ).reshape(-1, 2)
        solve_dist = np.zeros(5, dtype=np.float64)

    face_normals = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }
    face_normal = face_normals.get(face_name)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    def add_candidate(rvec: np.ndarray, tvec: np.ndarray) -> None:
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        rotation, _ = cv2.Rodrigues(rvec)
        points_camera = object_points @ rotation.T + tvec.reshape(1, 3)
        if np.any(points_camera[:, 2] <= 1e-4):
            return
        if face_normal is not None and float((rotation @ face_normal)[2]) > 0:
            return
        error = reproj_error_for_intrinsics(
            object_points,
            image_points,
            rvec,
            tvec,
            K,
            dist,
            intr.camera_model,
        )
        if np.isfinite(error):
            candidates.append((error, rvec, tvec))

    try:
        generic = cv2.solvePnPGeneric(
            object_points,
            solve_points,
            K,
            solve_dist,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if len(generic) >= 3 and generic[0]:
            for rvec, tvec in zip(generic[1], generic[2]):
                add_candidate(rvec, tvec)
    except cv2.error:
        pass
    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_SQPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                solve_points,
                K,
                solve_dist,
                flags=flag,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                solve_points,
                K,
                solve_dist,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
        add_candidate(rvec, tvec)
    if not candidates:
        return None
    error, rvec, tvec = min(candidates, key=lambda value: value[0])
    if error > MAX_APRILCUBE_MULTI_TAG_FACE_REPROJ_PX:
        return None
    return rvec, tvec, error, face_name, [tag_id for tag_id, _ in detections]


def detect_aprilcube_pose(
    frame_bgr: np.ndarray,
    context: AprilCubeDetectionContext,
    intr: Intrinsics,
) -> PoseDetection:
    detector = context.detector
    result = detector.process_frame(frame_bgr)
    pose_mode = "single-tag-face" if not context.multi_tag_faces else "native"
    if context.multi_tag_faces:
        height, width = frame_bgr.shape[:2]
        multi_tag_pose = solve_aprilcube_multi_tag_face_pose(
            result,
            context,
            intr,
            (width, height),
        )
        if multi_tag_pose is not None:
            rvec, tvec, error, face_name, tag_ids = multi_tag_pose
            result.update(
                {
                    "success": True,
                    "failure_reason": "",
                    "rvec": rvec,
                    "tvec": tvec,
                    "T": rvec_tvec_to_T(rvec, tvec),
                    "reproj_error": error,
                    "n_inliers": len(tag_ids) * 4,
                    "multi_tag_face_pose": True,
                    "multi_tag_face": face_name,
                    "multi_tag_face_ids": tag_ids,
                }
            )
            pose_mode = f"multi-tag-face:{face_name}"

    vis = detector.draw_result(frame_bgr, result)
    if not result.get("success", False) or result.get("T") is None:
        tag_count = int(result.get("n_tags", 0))
        faces = (
            ",".join(
                sorted(str(face) for face in result.get("visible_faces", set()))
            )
            or "none"
        )
        reason = str(result.get("failure_reason", "unknown"))
        return PoseDetection(
            ok=False,
            T=None,
            n_points=tag_count,
            message=(
                f"E: Q not found tags={tag_count} faces={faces} reason={reason}"
            ),
            vis=vis,
        )

    transform = np.asarray(result["T"], dtype=np.float64).reshape(4, 4)
    transform[:3, 3] *= 0.001
    if APRILCUBE_POSE_CONVENTION == "T_Q_E":
        transform = inv_T(transform)
    elif APRILCUBE_POSE_CONVENTION != "T_E_Q":
        raise ValueError(
            f"Unsupported APRILCUBE_POSE_CONVENTION="
            f"{APRILCUBE_POSE_CONVENTION}"
        )
    return PoseDetection(
        ok=True,
        T=transform,
        n_points=int(result.get("n_tags", 0)),
        reproj_error=float(result.get("reproj_error", float("inf"))),
        message=(
            f"E: Q ok tags={int(result.get('n_tags', 0))} "
            f"err={float(result.get('reproj_error', 0.0)):.2f}px "
            f"mode={pose_mode}"
        ),
        vis=vis,
    )
