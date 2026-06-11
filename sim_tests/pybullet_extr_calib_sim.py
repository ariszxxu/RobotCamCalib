#!/usr/bin/env python3
"""PyBullet simulation harness for xArm camera/tag extrinsics calibration.

This script mirrors the real-world flow in ``extr_calib.py``:

1. Render an xArm, a calibrated camera, and an AprilTag target.
2. Feed rendered RGB frames and joint states into ``CamTagCalibrator``.
3. Append valid AprilTag observations and solve incrementally.
4. Compare every solved round against known simulation ground truth.
5. Visualize estimated/GT frames and an estimated colored point cloud in Viser.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_TESTS_DIR = Path(__file__).resolve().parent
SIM_OUTPUTS_DIR = SIM_TESTS_DIR / "outputs"
DEFAULT_INTRINSICS_PATH = REPO_ROOT / "outputs" / "intrinsics.yaml"
DEFAULT_ROBOT_URDF = REPO_ROOT / "assets" / "robots" / "xarm6" / "xarm6_wo_ee.urdf"
DEFAULT_COMPACT_BOARD_YAML = REPO_ROOT / "assets" / "apriltag_grid" / "compact_apriltag_grid_4x4_tag48mm.yaml"
DEFAULT_COMPACT_BOARD_IMAGE = REPO_ROOT / "assets" / "apriltag_grid" / "compact_apriltag_grid_4x4_tag48mm_board_only.png"
APRILTAG_MARKER_FRACTION = 0.82
XARM6_EXPECTED_DOF = 6


@dataclass
class RuntimeTools:
    CamTagCalibrator: object
    ExtrinsicsCalibConfig: object
    pose_err_deg_m: object


@dataclass
class RenderFrame:
    rgb: np.ndarray
    depth: np.ndarray
    X_WorldCam_gt: np.ndarray
    X_WorldTag_gt: np.ndarray


@dataclass
class GroundTruth:
    mode: str
    cammount_link_name: str
    tagmount_link_name: str
    X_CammountCam: np.ndarray
    X_TagmountTag: np.ndarray


@dataclass
class ErrorSample:
    sample_count: int
    cam_rot_deg: float
    cam_trans_m: float
    tag_rot_deg: float
    tag_trans_m: float


@dataclass
class AprilTagGridBoard:
    rows: int
    cols: int
    tag_size: float
    tag_id_start: int
    tag_gap_ratio: float
    marker_fraction: float = APRILTAG_MARKER_FRACTION

    @property
    def tag_ids(self) -> List[int]:
        return [self.tag_id_start + i for i in range(self.rows * self.cols)]

    @property
    def paper_size(self) -> float:
        return self.tag_size / self.marker_fraction

    @property
    def gap(self) -> float:
        return self.tag_size * self.tag_gap_ratio

    @property
    def pitch(self) -> float:
        return self.paper_size + self.gap

    @property
    def extent_x(self) -> float:
        return (self.cols - 1) * self.pitch + self.paper_size

    @property
    def extent_y(self) -> float:
        return (self.rows - 1) * self.pitch + self.paper_size

    def X_board_tag(self, tag_id: int) -> np.ndarray:
        offset = tag_id - self.tag_id_start
        if offset < 0 or offset >= self.rows * self.cols:
            raise KeyError(f"Tag id {tag_id} is not on this board.")
        row = offset // self.cols
        col = offset % self.cols
        x = ((self.cols - 1) / 2.0 - col) * self.pitch
        y = ((self.rows - 1) / 2.0 - row) * self.pitch
        return make_T(translation=[x, y, 0.0])

    def contains(self, tag_id: int) -> bool:
        return self.tag_id_start <= tag_id < self.tag_id_start + self.rows * self.cols


def require_runtime_dependencies() -> None:
    """Fail early with actionable dependency names before importing extr_calib."""
    missing = []
    for module_name, package_name in (
        ("pybullet", "pybullet"),
        ("viser", "viser"),
        ("yourdfpy", "yourdfpy"),
        ("pupil_apriltags", "pupil-apriltags"),
    ):
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements.txt`."
        )

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV was installed without cv2.aruco. Install opencv-contrib-python "
            "or use an OpenCV build that includes ArUco/AprilTag dictionaries."
        )


def load_runtime_tools() -> RuntimeTools:
    require_runtime_dependencies()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from extr_calib import CamTagCalibrator, ExtrinsicsCalibConfig, pose_err_deg_m

    return RuntimeTools(
        CamTagCalibrator=CamTagCalibrator,
        ExtrinsicsCalibConfig=ExtrinsicsCalibConfig,
        pose_err_deg_m=pose_err_deg_m,
    )


def assert_xarm6_sim_model(robot_urdf_path: Path, joint_names: Sequence[str]) -> None:
    if len(joint_names) != XARM6_EXPECTED_DOF:
        raise AssertionError(
            f"The PyBullet sim is configured for the bundled xArm6 model, not xArm7. "
            f"Loaded {len(joint_names)} actuated joints from {robot_urdf_path}. "
            "For xArm7 or another robot, change the URDF, mount link names, "
            "ground-truth transforms, and sampled joint configurations."
        )


def make_T(rotation: Optional[np.ndarray] = None, translation: Optional[Sequence[float]] = None) -> np.ndarray:
    X = np.eye(4)
    if rotation is not None:
        X[:3, :3] = np.asarray(rotation, dtype=float)
    if translation is not None:
        X[:3, 3] = np.asarray(translation, dtype=float)
    return X


def inv_T(X: np.ndarray) -> np.ndarray:
    Y = np.eye(4)
    Y[:3, :3] = X[:3, :3].T
    Y[:3, 3] = -X[:3, :3].T @ X[:3, 3]
    return Y


def T_from_pos_quat(pos: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    return make_T(R.from_quat(quat_xyzw).as_matrix(), pos)


def pos_quat_from_T(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X[:3, 3], R.from_matrix(X[:3, :3]).as_quat()


def look_at_cv(eye: Sequence[float], target: Sequence[float], up_hint: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Return a camera pose with OpenCV axes: +x right, +y down, +z forward."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up_hint = np.asarray(up_hint, dtype=float)

    z_axis = target - eye
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        raise ValueError("look_at eye and target are too close.")
    z_axis /= z_norm

    x_axis = np.cross(up_hint, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return make_T(np.column_stack([x_axis, y_axis, z_axis]), eye)


def detector_tag_pose_facing_camera(tag_position: Sequence[float], camera_position: Sequence[float]) -> np.ndarray:
    """Pose convention returned by pupil_apriltags for a visible frontal tag."""
    X = look_at_cv(camera_position, tag_position)
    X[:3, 3] = np.asarray(tag_position, dtype=float)
    X[:3, :3] = X[:3, :3] @ R.from_euler("z", 180.0, degrees=True).as_matrix()
    return X


def load_intrinsics(path: Path, width: Optional[int], height: Optional[int]) -> Tuple[np.ndarray, int, int]:
    if path.exists():
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        K = np.asarray(data["K"], dtype=float).reshape(3, 3)
        src_width, src_height = data.get("image_size", [1280, 720])
        width = src_width if width is None else width
        height = src_height if height is None else height
        if width != src_width or height != src_height:
            sx = width / float(src_width)
            sy = height / float(src_height)
            K = K.copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
        return K, int(width), int(height)

    width = 1280 if width is None else width
    height = 720 if height is None else height
    focal = 0.7 * width
    K = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return K, int(width), int(height)


def projection_matrix_from_K(K: np.ndarray, width: int, height: int, near: float, far: float) -> List[float]:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return [
        2.0 * fx / width,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 * fy / height,
        0.0,
        0.0,
        1.0 - 2.0 * cx / width,
        2.0 * cy / height - 1.0,
        (far + near) / (near - far),
        -1.0,
        0.0,
        0.0,
        2.0 * far * near / (near - far),
        0.0,
    ]


def april_tag_dictionary():
    aruco = cv2.aruco
    dict_id = getattr(aruco, "DICT_APRILTAG_36h11", None)
    if dict_id is None:
        dict_id = getattr(aruco, "DICT_APRILTAG_36H11")
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def write_apriltag_texture(
    path: Path,
    tag_id: int,
    texture_px: int = 1024,
    marker_fraction: float = APRILTAG_MARKER_FRACTION,
) -> None:
    dictionary = april_tag_dictionary()
    marker_px = int(texture_px * marker_fraction)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_px)
    else:
        marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, tag_id, marker_px, marker, 1)

    canvas = np.full((texture_px, texture_px), 255, dtype=np.uint8)
    offset = (texture_px - marker_px) // 2
    canvas[offset : offset + marker_px, offset : offset + marker_px] = marker
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to write AprilTag texture to {path}.")


def write_textured_square_obj(path: Path) -> None:
    # A double-sided unit square in the local XY plane. Scaling sets the board size.
    path.write_text(
        "\n".join(
            [
                "v -0.5 -0.5 0.0",
                "v 0.5 -0.5 0.0",
                "v 0.5 0.5 0.0",
                "v -0.5 0.5 0.0",
                "vt 0.0 1.0",
                "vt 1.0 1.0",
                "vt 1.0 0.0",
                "vt 0.0 0.0",
                "f 1/1 2/2 3/3",
                "f 1/1 3/3 4/4",
                "f 3/3 2/2 1/1",
                "f 4/4 3/3 1/1",
                "",
            ]
        )
    )


def load_image_physical_size_m(path: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        from PIL import Image
    except ImportError:
        return None, None

    with Image.open(path) as image:
        width_px, height_px = image.size
        dpi = image.info.get("dpi")
    if dpi is None:
        return None, None
    dpi_x, dpi_y = dpi if isinstance(dpi, tuple) else (dpi, dpi)
    if dpi_x <= 0 or dpi_y <= 0:
        return None, None
    return width_px / float(dpi_x) * 0.0254, height_px / float(dpi_y) * 0.0254


class PyBulletXarmSim:
    def __init__(
        self,
        robot_urdf_path: Path,
        K: np.ndarray,
        width: int,
        height: int,
        tag_size: float,
        tag_id: int,
        tag_marker_fraction: float,
        tag_rows: int,
        tag_cols: int,
        tag_gap_ratio: float,
        tag_board_yaml: Optional[Path],
        tag_board_image: Optional[Path],
        tag_board_image_width_m: Optional[float],
        tag_board_image_height_m: Optional[float],
        near: float,
        far: float,
        use_gui: bool,
        renderer_name: str,
        tmpdir: Path,
    ) -> None:
        import pybullet as p

        self.p = p
        self.K = K
        self.width = width
        self.height = height
        self.tag_size = tag_size
        self.tag_id = tag_id
        self.tag_marker_fraction = tag_marker_fraction
        self.near = near
        self.far = far
        self.renderer_name = renderer_name
        self.grid_board: Optional[AprilTagGridBoard] = None
        self.board_layout = None
        self.board_image: Optional[np.ndarray] = None
        self.board_image_size_m: Optional[Tuple[float, float]] = None

        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Failed to connect to PyBullet.")

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)

        self.robot = p.loadURDF(
            str(robot_urdf_path),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client,
        )
        self.joint_names, self.joint_indices, self.lower_limits, self.upper_limits = self._actuated_joints()
        assert_xarm6_sim_model(robot_urdf_path, self.joint_names)
        self.link_indices = self._link_indices()
        if "link_eef" not in self.link_indices:
            raise RuntimeError("The loaded xArm URDF does not expose link_eef in PyBullet.")

        plane_collision = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=plane_collision, physicsClientId=self.client)

        obj_path = tmpdir / "apriltag_board.obj"
        write_textured_square_obj(obj_path)
        self.tag_images: Dict[int, np.ndarray] = {}

        # `tag_size` is the physical AprilTag marker edge used by the detector.
        # The rendered tile is larger because the texture includes white paper
        # around the marker. Therefore: marker_edge_m = tile_size * fraction.
        self.tag_tile_size = tag_size / tag_marker_fraction
        marker_edge_m = self.tag_tile_size * tag_marker_fraction
        if not np.isclose(marker_edge_m, tag_size):
            raise RuntimeError(
                f"AprilTag size mismatch: detector tag_size={tag_size}, "
                f"rendered marker edge={marker_edge_m}."
            )

        if (tag_board_yaml is None) != (tag_board_image is None):
            raise ValueError("Provide both tag_board_yaml and tag_board_image, or neither.")

        if tag_board_yaml is not None and tag_board_image is not None:
            from apriltag_board import load_apriltag_board_yaml

            self.board_layout = load_apriltag_board_yaml(tag_board_yaml)
            if not np.isclose(self.board_layout.tag_size_m, tag_size, atol=1e-9):
                raise RuntimeError(
                    f"YAML tag size {self.board_layout.tag_size_m} m does not match detector tag_size={tag_size} m."
                )
            board_image = cv2.imread(str(tag_board_image), cv2.IMREAD_GRAYSCALE)
            if board_image is None:
                raise RuntimeError(f"Failed to read board image at {tag_board_image}.")
            self.board_image = board_image

            image_width_m, image_height_m = load_image_physical_size_m(tag_board_image)
            image_width_m = tag_board_image_width_m if tag_board_image_width_m is not None else image_width_m
            image_height_m = tag_board_image_height_m if tag_board_image_height_m is not None else image_height_m
            if image_width_m is None or image_height_m is None:
                raise RuntimeError(
                    "Could not infer board image physical size from DPI. "
                    "Pass --tag-board-image-width-m and --tag-board-image-height-m."
                )
            board_width_m, board_height_m = self.board_layout.board_size_m
            if image_width_m < board_width_m or image_height_m < board_height_m:
                raise RuntimeError(
                    f"Board image physical size {(image_width_m, image_height_m)} m is smaller than YAML board "
                    f"size {(board_width_m, board_height_m)} m."
                )
            self.board_image_size_m = (image_width_m, image_height_m)
            self.board_size = max(image_width_m, image_height_m)
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[image_width_m / 2.0, image_height_m / 2.0, 0.001],
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                physicsClientId=self.client,
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[image_width_m / 2.0, image_height_m / 2.0, 0.001],
                physicsClientId=self.client,
            )
        else:
            self.grid_board = AprilTagGridBoard(
                rows=tag_rows,
                cols=tag_cols,
                tag_size=tag_size,
                tag_id_start=tag_id,
                tag_gap_ratio=tag_gap_ratio,
                marker_fraction=tag_marker_fraction,
            )
            for tid in self.grid_board.tag_ids:
                texture_path = tmpdir / f"apriltag36h11_{tid}.png"
                write_apriltag_texture(texture_path, tag_id=tid, marker_fraction=tag_marker_fraction)
                tag_image = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
                if tag_image is None:
                    raise RuntimeError(f"Failed to read generated AprilTag texture at {texture_path}.")
                self.tag_images[tid] = tag_image
            self.tag_image = self.tag_images[tag_id]
            self.board_size = max(self.grid_board.extent_x, self.grid_board.extent_y)
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.grid_board.extent_x / 2.0, self.grid_board.extent_y / 2.0, 0.001],
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                physicsClientId=self.client,
            )
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.grid_board.extent_x / 2.0, self.grid_board.extent_y / 2.0, 0.001],
                physicsClientId=self.client,
            )

        self.tag_body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            physicsClientId=self.client,
        )

        self.projection_matrix = projection_matrix_from_K(K, width, height, near, far)

    def disconnect(self) -> None:
        if self.p.isConnected(self.client):
            self.p.disconnect(self.client)

    def _actuated_joints(self) -> Tuple[List[str], List[int], np.ndarray, np.ndarray]:
        p = self.p
        names = []
        indices = []
        lower = []
        upper = []
        for joint_index in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, joint_index, physicsClientId=self.client)
            joint_type = info[2]
            if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue
            names.append(info[1].decode("utf-8"))
            indices.append(joint_index)
            lower.append(info[8])
            upper.append(info[9])
        return names, indices, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def _link_indices(self) -> Dict[str, int]:
        p = self.p
        mapping = {p.getBodyInfo(self.robot, physicsClientId=self.client)[0].decode("utf-8"): -1}
        for joint_index in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, joint_index, physicsClientId=self.client)
            mapping[info[12].decode("utf-8")] = joint_index
        return mapping

    def reset_qpos(self, qpos: np.ndarray) -> None:
        for joint_index, value in zip(self.joint_indices, qpos):
            self.p.resetJointState(
                self.robot,
                joint_index,
                float(value),
                targetVelocity=0.0,
                physicsClientId=self.client,
            )
        self.p.stepSimulation(physicsClientId=self.client)

    def frame_pose(self, link_name: str) -> np.ndarray:
        p = self.p
        if link_name not in self.link_indices:
            if link_name == "link_base":
                pos, quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
                return T_from_pos_quat(pos, quat)
            raise KeyError(f"Unknown link name in PyBullet model: {link_name}")
        link_index = self.link_indices[link_name]
        if link_index < 0:
            pos, quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
            return T_from_pos_quat(pos, quat)
        state = p.getLinkState(
            self.robot,
            link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        return T_from_pos_quat(state[4], state[5])

    def set_tag_pose(self, X_WorldTag: np.ndarray) -> None:
        pos, quat = pos_quat_from_T(X_WorldTag)
        self.p.resetBasePositionAndOrientation(
            self.tag_body,
            pos,
            quat,
            physicsClientId=self.client,
        )

    def solve_ik(self, X_WorldEef: np.ndarray, rest_qpos: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.p
        pos, quat = pos_quat_from_T(X_WorldEef)
        rest = np.zeros(len(self.joint_indices)) if rest_qpos is None else rest_qpos
        ranges = np.maximum(self.upper_limits - self.lower_limits, 1e-3)
        qpos = p.calculateInverseKinematics(
            self.robot,
            self.link_indices["link_eef"],
            targetPosition=pos.tolist(),
            targetOrientation=quat.tolist(),
            lowerLimits=self.lower_limits.tolist(),
            upperLimits=self.upper_limits.tolist(),
            jointRanges=ranges.tolist(),
            restPoses=rest.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-5,
            physicsClientId=self.client,
        )
        qpos = np.asarray(qpos[: len(self.joint_indices)], dtype=float)
        return np.clip(qpos, self.lower_limits, self.upper_limits)

    def render(self, gt: GroundTruth, qpos: np.ndarray) -> RenderFrame:
        p = self.p
        self.reset_qpos(qpos)

        X_WorldCammount = self.frame_pose(gt.cammount_link_name)
        X_WorldTagmount = self.frame_pose(gt.tagmount_link_name)
        X_WorldCam = X_WorldCammount @ gt.X_CammountCam
        X_WorldTag = X_WorldTagmount @ gt.X_TagmountTag
        self.set_tag_pose(X_WorldTag)

        eye = X_WorldCam[:3, 3]
        target = eye + X_WorldCam[:3, 2]
        up = -X_WorldCam[:3, 1]
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        renderer = p.ER_TINY_RENDERER if self.renderer_name == "tiny" else p.ER_BULLET_HARDWARE_OPENGL
        _, _, rgba, depth, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=renderer,
            physicsClientId=self.client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3].copy()
        depth = np.asarray(depth, dtype=np.float32).reshape(self.height, self.width).copy()
        self.overlay_target(rgb, X_WorldCam, X_WorldTag)
        return RenderFrame(rgb=rgb, depth=depth, X_WorldCam_gt=X_WorldCam, X_WorldTag_gt=X_WorldTag)

    def overlay_target(self, rgb: np.ndarray, X_WorldCam: np.ndarray, X_WorldTarget: np.ndarray) -> None:
        if self.board_image is not None:
            self.overlay_board_image(rgb, X_WorldCam, X_WorldTarget)
            return

        if self.grid_board is None:
            raise RuntimeError("Grid tag board was not initialized.")
        for tag_id in self.grid_board.tag_ids:
            X_WorldTag = X_WorldTarget @ self.grid_board.X_board_tag(tag_id)
            self.overlay_apriltag(
                rgb,
                X_WorldCam,
                X_WorldTag,
                self.tag_images[tag_id],
                self.grid_board.paper_size,
            )

    def overlay_apriltag(
        self,
        rgb: np.ndarray,
        X_WorldCam: np.ndarray,
        X_WorldTag: np.ndarray,
        tag_image: np.ndarray,
        tile_size: float,
    ) -> None:
        X_CamTag = inv_T(X_WorldCam) @ X_WorldTag
        if X_CamTag[:3, 2][2] < 0.05:
            return

        half = tile_size / 2.0
        corners_tag = np.array(
            [
                [half, half, 0.0],
                [-half, half, 0.0],
                [-half, -half, 0.0],
                [half, -half, 0.0],
            ],
            dtype=float,
        )
        corners_cam = (X_CamTag[:3, :3] @ corners_tag.T).T + X_CamTag[:3, 3]
        if np.any(corners_cam[:, 2] <= self.near):
            return

        projected = np.column_stack(
            [
                self.K[0, 0] * corners_cam[:, 0] / corners_cam[:, 2] + self.K[0, 2],
                self.K[1, 1] * corners_cam[:, 1] / corners_cam[:, 2] + self.K[1, 2],
            ]
        ).astype(np.float32)

        area = cv2.contourArea(projected)
        if area < 25.0:
            return

        h, w = tag_image.shape
        src = np.array(
            [
                [0.0, 0.0],
                [w - 1.0, 0.0],
                [w - 1.0, h - 1.0],
                [0.0, h - 1.0],
            ],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(src, projected)
        tag_rgb = cv2.cvtColor(tag_image, cv2.COLOR_GRAY2RGB)
        warped = cv2.warpPerspective(
            tag_rgb,
            H,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        mask = cv2.warpPerspective(
            np.full_like(tag_image, 255),
            H,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rgb[mask > 0] = warped[mask > 0]

    def overlay_board_image(self, rgb: np.ndarray, X_WorldCam: np.ndarray, X_WorldBoard: np.ndarray) -> None:
        if self.board_image is None or self.board_image_size_m is None:
            raise RuntimeError("Board image overlay requested before board image initialization.")

        X_CamBoard = inv_T(X_WorldCam) @ X_WorldBoard
        if X_CamBoard[:3, 2][2] < 0.05:
            return

        width_m, height_m = self.board_image_size_m
        half_x = width_m / 2.0
        half_y = height_m / 2.0
        corners_board = np.array(
            [
                [half_x, half_y, 0.0],
                [-half_x, half_y, 0.0],
                [-half_x, -half_y, 0.0],
                [half_x, -half_y, 0.0],
            ],
            dtype=float,
        )
        corners_cam = (X_CamBoard[:3, :3] @ corners_board.T).T + X_CamBoard[:3, 3]
        if np.any(corners_cam[:, 2] <= self.near):
            return

        projected = np.column_stack(
            [
                self.K[0, 0] * corners_cam[:, 0] / corners_cam[:, 2] + self.K[0, 2],
                self.K[1, 1] * corners_cam[:, 1] / corners_cam[:, 2] + self.K[1, 2],
            ]
        ).astype(np.float32)
        if cv2.contourArea(projected) < 25.0:
            return

        h, w = self.board_image.shape
        src = np.array(
            [
                [0.0, 0.0],
                [w - 1.0, 0.0],
                [w - 1.0, h - 1.0],
                [0.0, h - 1.0],
            ],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(src, projected)
        board_rgb = cv2.cvtColor(self.board_image, cv2.COLOR_GRAY2RGB)
        warped = cv2.warpPerspective(
            board_rgb,
            H,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        mask = cv2.warpPerspective(
            np.full_like(self.board_image, 255),
            H,
            (self.width, self.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rgb[mask > 0] = warped[mask > 0]


def build_ground_truth(mode: str) -> GroundTruth:
    if mode == "thirdview":
        X_WorldCam = look_at_cv([0.82, -0.85, 0.58], [0.32, 0.0, 0.34])
        return GroundTruth(
            mode=mode,
            cammount_link_name="link_base",
            tagmount_link_name="link_eef",
            X_CammountCam=X_WorldCam,
            X_TagmountTag=make_T(translation=[0.0, 0.0, -0.08]),
        )

    if mode == "wrist":
        tag_position = np.array([0.46, -0.02, 0.25])
        nominal_camera_eye = np.array([0.82, -0.55, 0.48])
        X_Tag = detector_tag_pose_facing_camera(tag_position, nominal_camera_eye)
        return GroundTruth(
            mode=mode,
            cammount_link_name="link_eef",
            tagmount_link_name="link_base",
            X_CammountCam=make_T(translation=[0.0, 0.0, 0.08]),
            X_TagmountTag=X_Tag,
        )

    raise ValueError(f"Unknown mode: {mode}")


def random_local_rotation(rng: np.random.Generator, tilt_deg: float = 22.0, yaw_deg: float = 35.0) -> np.ndarray:
    roll = rng.uniform(-tilt_deg, tilt_deg)
    pitch = rng.uniform(-tilt_deg, tilt_deg)
    yaw = rng.uniform(-yaw_deg, yaw_deg)
    return R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()


def sample_candidate_qpos(
    sim: PyBulletXarmSim,
    gt: GroundTruth,
    rng: np.random.Generator,
    rest_qpos: Optional[np.ndarray],
) -> np.ndarray:
    if gt.mode == "thirdview":
        center = np.array([0.34, 0.0, 0.34])
        target_pos = center + rng.uniform([-0.12, -0.18, -0.10], [0.16, 0.18, 0.16])
        camera_eye = gt.X_CammountCam[:3, 3]
        X_WorldTag = detector_tag_pose_facing_camera(target_pos, camera_eye)
        X_WorldTag[:3, :3] = X_WorldTag[:3, :3] @ random_local_rotation(rng)
        X_WorldTag[:3, 3] = target_pos
        X_WorldEef = X_WorldTag @ inv_T(gt.X_TagmountTag)
        return sim.solve_ik(X_WorldEef, rest_qpos=rest_qpos)

    tag_pos = gt.X_TagmountTag[:3, 3]
    eye = tag_pos + rng.uniform([0.22, -0.48, 0.06], [0.52, -0.24, 0.32])
    X_WorldCam = look_at_cv(eye, tag_pos)
    X_WorldCam[:3, :3] = X_WorldCam[:3, :3] @ random_local_rotation(rng, tilt_deg=12.0, yaw_deg=22.0)
    X_WorldEef = X_WorldCam @ inv_T(gt.X_CammountCam)
    return sim.solve_ik(X_WorldEef, rest_qpos=rest_qpos)


def tag_object_corners_pupil_order(tag_size: float) -> np.ndarray:
    """3D tag corners matching pupil_apriltags Detection.corners order."""
    half = tag_size / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def detect_grid_board_cam_target(
    detector,
    rgb: np.ndarray,
    K: np.ndarray,
    tag_size: float,
    board: AprilTagGridBoard,
    min_tags: int,
) -> Optional[np.ndarray]:
    """Bundle-PnP estimate of X_CamBoardCenter from all detected grid tags."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detections = detector.detect(gray, False)
    if len(detections) < min_tags:
        return None

    obj_corners_tag = tag_object_corners_pupil_order(tag_size)
    all_obj_pts = []
    all_img_pts = []
    used_tag_ids = set()

    for det in detections:
        tag_id = int(getattr(det, "tag_id", -1))
        if not board.contains(tag_id):
            continue
        X_board_tag = board.X_board_tag(tag_id)
        obj_pts = (X_board_tag[:3, :3] @ obj_corners_tag.T).T + X_board_tag[:3, 3]
        all_obj_pts.append(obj_pts)
        all_img_pts.append(np.asarray(det.corners, dtype=np.float32))
        used_tag_ids.add(tag_id)

    if len(used_tag_ids) < min_tags:
        return None

    obj_pts_np = np.vstack(all_obj_pts).astype(np.float32)
    img_pts_np = np.vstack(all_img_pts).astype(np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        obj_pts_np,
        img_pts_np,
        K.astype(np.float64),
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts_np,
            img_pts_np,
            K.astype(np.float64),
            dist_coeffs,
            rvec,
            tvec,
        )
    except cv2.error:
        pass

    R_cam_target, _ = cv2.Rodrigues(rvec)
    return make_T(R_cam_target, tvec.reshape(3))


def detect_yaml_board_cam_target(
    detector,
    rgb: np.ndarray,
    K: np.ndarray,
    board_layout,
    min_tags: int,
) -> Optional[np.ndarray]:
    from apriltag_board import estimate_board_pose_bundle_pnp

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detections = detector.detect(gray, False)
    try:
        X_CamBoard, _ = estimate_board_pose_bundle_pnp(
            detections,
            board_layout,
            K,
            D=None,
            min_tags=min_tags,
        )
    except (RuntimeError, ValueError, cv2.error):
        return None
    return X_CamBoard


def detect_target_pose(
    calibrator,
    rgb: np.ndarray,
    args: argparse.Namespace,
    grid_board: Optional[AprilTagGridBoard],
    board_layout=None,
) -> Optional[np.ndarray]:
    if board_layout is not None:
        return detect_yaml_board_cam_target(
            calibrator.tag_detector,
            rgb,
            calibrator.config.K,
            board_layout,
            args.board_min_tags,
        )
    if grid_board is None:
        raise RuntimeError("Grid board detection requested, but the simulator has no grid board.")
    return detect_grid_board_cam_target(
        calibrator.tag_detector,
        rgb,
        calibrator.config.K,
        calibrator.config.tag_size,
        grid_board,
        args.board_min_tags,
    )


def add_viser_frame(server, name: str, X: np.ndarray, axes_length: float = 0.16, axes_radius: float = 0.006) -> None:
    server.scene.add_frame(
        name=name,
        axes_length=axes_length,
        axes_radius=axes_radius,
        wxyz=R.from_matrix(X[:3, :3]).as_quat()[[3, 0, 1, 2]],
        position=X[:3, 3],
    )


def depth_buffer_to_meters(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    return far * near / (far - (far - near) * depth)


def update_point_cloud(
    server,
    name: str,
    rgb: np.ndarray,
    depth_buffer: np.ndarray,
    K: np.ndarray,
    X_WorldCam: np.ndarray,
    near: float,
    far: float,
    stride: int,
    max_points: int,
) -> None:
    z = depth_buffer_to_meters(depth_buffer, near, far)
    mask = np.isfinite(z) & (z > near) & (z < far * 0.98)
    rows, cols = np.nonzero(mask[::stride, ::stride])
    if rows.size == 0:
        return
    v = rows * stride
    u = cols * stride
    z_sample = z[v, u]
    x = (u.astype(float) - K[0, 2]) * z_sample / K[0, 0]
    y = (v.astype(float) - K[1, 2]) * z_sample / K[1, 1]
    points_cam = np.column_stack([x, y, z_sample])
    points_world = (X_WorldCam[:3, :3] @ points_cam.T).T + X_WorldCam[:3, 3]
    colors = rgb[v, u]

    if points_world.shape[0] > max_points:
        keep = np.linspace(0, points_world.shape[0] - 1, max_points).astype(int)
        points_world = points_world[keep]
        colors = colors[keep]

    try:
        server.scene.add_point_cloud(
            name=name,
            points=points_world.astype(np.float32),
            colors=colors.astype(np.uint8),
            point_size=0.004,
        )
    except TypeError:
        server.scene.add_point_cloud(
            name=name,
            points=points_world.astype(np.float32),
            colors=(colors.astype(np.float32) / 255.0),
            point_size=0.004,
        )


def compute_errors(tools: RuntimeTools, calibrator, gt: GroundTruth, sample_count: int) -> ErrorSample:
    x_rot, x_trans = tools.pose_err_deg_m(calibrator.X_CammountCam, gt.X_CammountCam)
    y_rot, y_trans = tools.pose_err_deg_m(calibrator.X_TagmountTag, gt.X_TagmountTag)
    return ErrorSample(
        sample_count=sample_count,
        cam_rot_deg=float(x_rot),
        cam_trans_m=float(x_trans),
        tag_rot_deg=float(y_rot),
        tag_trans_m=float(y_trans),
    )


def report_errors(error: ErrorSample, prefix: str) -> None:
    print(
        f"{prefix} "
        f"X_CammountCam err: {error.cam_rot_deg:.3f} deg, {error.cam_trans_m:.5f} m | "
        f"X_TagmountTag err: {error.tag_rot_deg:.3f} deg, {error.tag_trans_m:.5f} m"
    )


def write_error_csv(path: Path, history: List[ErrorSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_count", "cam_rot_deg", "cam_trans_m", "tag_rot_deg", "tag_trans_m"])
        for row in history:
            writer.writerow(
                [
                    row.sample_count,
                    row.cam_rot_deg,
                    row.cam_trans_m,
                    row.tag_rot_deg,
                    row.tag_trans_m,
                ]
            )


def save_error_plot(path: Path, history: List[ErrorSample], mode: str) -> None:
    if not history:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array([row.sample_count for row in history], dtype=float)
    cam_rot = np.array([row.cam_rot_deg for row in history], dtype=float)
    tag_rot = np.array([row.tag_rot_deg for row in history], dtype=float)
    cam_trans = np.array([row.cam_trans_m for row in history], dtype=float)
    tag_trans = np.array([row.tag_trans_m for row in history], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x, cam_rot, marker="o", label="X_CammountCam")
    axes[0].plot(x, tag_rot, marker="o", label="X_TagmountTag")
    axes[0].set_ylabel("Rotation error (deg)")
    axes[0].set_title(f"{mode} calibration error vs number of samples")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, cam_trans, marker="o", label="X_CammountCam")
    axes[1].plot(x, tag_trans, marker="o", label="X_TagmountTag")
    axes[1].set_xlabel("Number of accepted samples")
    axes[1].set_ylabel("Translation error (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def mode_output_path(path: Optional[Path], mode: str) -> Optional[Path]:
    if path is None:
        return None
    if "{mode}" in str(path):
        return Path(str(path).format(mode=mode))
    if path.suffix:
        return path.with_name(f"{path.stem}_{mode}{path.suffix}")
    return path


def estimated_world_camera_pose(calibrator) -> np.ndarray:
    return calibrator.X_WorldCammount @ calibrator.X_CammountCam


def detector_world_tag_pose(calibrator, detected_X_CamTag: np.ndarray) -> np.ndarray:
    return estimated_world_camera_pose(calibrator) @ detected_X_CamTag


def update_sim_visuals(
    calibrator,
    gt: GroundTruth,
    frame: RenderFrame,
    K: np.ndarray,
    near: float,
    far: float,
    pointcloud_stride: int,
    max_pointcloud_points: int,
    detected_X_CamTag: Optional[np.ndarray] = None,
) -> None:
    add_viser_frame(calibrator.server, f"sim/{gt.mode}/gt_camera", frame.X_WorldCam_gt)
    add_viser_frame(calibrator.server, f"sim/{gt.mode}/gt_tag", frame.X_WorldTag_gt)

    X_WorldCam_est = estimated_world_camera_pose(calibrator)
    add_viser_frame(
        calibrator.server,
        f"sim/{gt.mode}/pointcloud_X_WorldCam_est",
        X_WorldCam_est,
        axes_length=0.12,
        axes_radius=0.004,
    )
    if detected_X_CamTag is not None:
        add_viser_frame(
            calibrator.server,
            f"sim/{gt.mode}/detected_tag_from_apriltag",
            detector_world_tag_pose(calibrator, detected_X_CamTag),
            axes_length=0.12,
            axes_radius=0.004,
        )
    update_point_cloud(
        calibrator.server,
        f"sim/{gt.mode}/current_estimated_pointcloud",
        frame.rgb,
        frame.depth,
        K,
        X_WorldCam_est,
        near,
        far,
        pointcloud_stride,
        max_pointcloud_points,
    )


def make_calibrator(tools: RuntimeTools, args: argparse.Namespace, mode: str, gt: GroundTruth, K: np.ndarray):
    config = tools.ExtrinsicsCalibConfig(
        robot_urdf_path=DEFAULT_ROBOT_URDF,
        cammount_link_name=gt.cammount_link_name,
        tagmount_link_name=gt.tagmount_link_name,
        K=K,
        output_file_path=SIM_OUTPUTS_DIR / f"sim_extrinsics_{mode}.yaml",
        tag_size=args.tag_size,
        tag_family="tag36h11",
        ransac_sample_size=args.ransac_sample_size,
    )
    return tools.CamTagCalibrator(config)


def stop_viser_server(calibrator) -> None:
    stop = getattr(calibrator.server, "stop", None)
    if callable(stop):
        stop()


def run_batch_for_mode(
    tools: RuntimeTools,
    args: argparse.Namespace,
    mode: str,
    hold: bool,
) -> None:
    K, width, height = load_intrinsics(args.intrinsics, args.width, args.height)
    gt = build_ground_truth(mode)
    rng = np.random.default_rng(args.seed + (0 if mode == "thirdview" else 1000))

    with tempfile.TemporaryDirectory(prefix="robotcamcalib_sim_") as tmp:
        sim = PyBulletXarmSim(
            robot_urdf_path=DEFAULT_ROBOT_URDF,
            K=K,
            width=width,
            height=height,
            tag_size=args.tag_size,
            tag_id=args.tag_id,
            tag_marker_fraction=args.tag_marker_fraction,
            tag_rows=args.tag_rows,
            tag_cols=args.tag_cols,
            tag_gap_ratio=args.tag_gap_ratio,
            tag_board_yaml=args.tag_board_yaml,
            tag_board_image=args.tag_board_image,
            tag_board_image_width_m=args.tag_board_image_width_m,
            tag_board_image_height_m=args.tag_board_image_height_m,
            near=args.near,
            far=args.far,
            use_gui=args.pybullet_gui,
            renderer_name=args.renderer,
            tmpdir=Path(tmp),
        )
        calibrator = make_calibrator(tools, args, mode, gt, K)
        rest_qpos = np.zeros(len(sim.joint_indices))
        accepted = 0
        attempts = 0
        error_history: List[ErrorSample] = []
        max_attempts = max(args.samples * args.max_attempts_per_sample, args.samples)

        print(f"[{mode}] Viser is started by CamTagCalibrator; open the printed localhost URL.")
        try:
            while accepted < args.samples and attempts < max_attempts:
                attempts += 1
                qpos = sample_candidate_qpos(sim, gt, rng, rest_qpos)
                frame = sim.render(gt, qpos)
                calibrator.vis_step(frame.rgb, qpos, input_joint_names=sim.joint_names)
                detected_X_CamTag = detect_target_pose(calibrator, frame.rgb, args, sim.grid_board, sim.board_layout)
                update_sim_visuals(
                    calibrator,
                    gt,
                    frame,
                    K,
                    args.near,
                    args.far,
                    args.pointcloud_stride,
                    args.max_pointcloud_points,
                    detected_X_CamTag=detected_X_CamTag,
                )

                if detected_X_CamTag is None:
                    if attempts % 20 == 0:
                        print(f"[{mode}] {attempts} attempts, {accepted} valid AprilTag detections.")
                    continue

                calibrator.X_CamTag = detected_X_CamTag
                calibrator.append_and_solve()
                accepted += 1
                rest_qpos = qpos

                if accepted >= args.ransac_sample_size:
                    error = compute_errors(tools, calibrator, gt, accepted)
                    error_history.append(error)
                    report_errors(error, f"[{mode}] sample {accepted:03d}/{args.samples:03d}:")
                else:
                    print(f"[{mode}] sample {accepted:03d}/{args.samples:03d}: appended, waiting for solve threshold.")

                if args.step_delay > 0.0:
                    time.sleep(args.step_delay)

            if accepted < args.samples:
                raise RuntimeError(
                    f"[{mode}] Only collected {accepted}/{args.samples} valid samples after {attempts} attempts. "
                    "Try increasing --max-attempts-per-sample, using --renderer opengl, or lowering --samples."
                )

            print(f"[{mode}] completed {accepted} valid simulated calibration samples.")
            error_csv_path = mode_output_path(args.error_csv, mode)
            error_plot_path = mode_output_path(args.error_plot, mode)
            if error_csv_path is not None:
                write_error_csv(error_csv_path, error_history)
                print(f"[{mode}] saved error CSV to {error_csv_path}")
            if error_plot_path is not None:
                save_error_plot(error_plot_path, error_history, mode)
                print(f"[{mode}] saved error plot to {error_plot_path}")
            if hold:
                print(f"[{mode}] holding final Viser scene. Press Ctrl+C to exit.")
                while True:
                    time.sleep(1.0)
        finally:
            if not hold:
                stop_viser_server(calibrator)
            sim.disconnect()


def run_interactive_for_mode(tools: RuntimeTools, args: argparse.Namespace, mode: str) -> None:
    K, width, height = load_intrinsics(args.intrinsics, args.width, args.height)
    gt = build_ground_truth(mode)
    rng = np.random.default_rng(args.seed + (0 if mode == "thirdview" else 1000))

    with tempfile.TemporaryDirectory(prefix="robotcamcalib_sim_") as tmp:
        sim = PyBulletXarmSim(
            robot_urdf_path=DEFAULT_ROBOT_URDF,
            K=K,
            width=width,
            height=height,
            tag_size=args.tag_size,
            tag_id=args.tag_id,
            tag_marker_fraction=args.tag_marker_fraction,
            tag_rows=args.tag_rows,
            tag_cols=args.tag_cols,
            tag_gap_ratio=args.tag_gap_ratio,
            tag_board_yaml=args.tag_board_yaml,
            tag_board_image=args.tag_board_image,
            tag_board_image_width_m=args.tag_board_image_width_m,
            tag_board_image_height_m=args.tag_board_image_height_m,
            near=args.near,
            far=args.far,
            use_gui=args.pybullet_gui,
            renderer_name=args.renderer,
            tmpdir=Path(tmp),
        )
        calibrator = make_calibrator(tools, args, mode, gt, K)
        state = {
            "qpos": np.zeros(len(sim.joint_indices)),
            "frame": None,
            "detected_X_CamTag": None,
            "accepted": 0,
        }

        def render_next_sample() -> None:
            state["qpos"] = sample_candidate_qpos(sim, gt, rng, state["qpos"])
            frame = sim.render(gt, state["qpos"])
            calibrator.vis_step(frame.rgb, state["qpos"], input_joint_names=sim.joint_names)
            state["detected_X_CamTag"] = detect_target_pose(calibrator, frame.rgb, args, sim.grid_board, sim.board_layout)
            state["frame"] = frame
            update_sim_visuals(
                calibrator,
                gt,
                frame,
                K,
                args.near,
                args.far,
                args.pointcloud_stride,
                args.max_pointcloud_points,
                detected_X_CamTag=state["detected_X_CamTag"],
            )
            status = "valid" if state["detected_X_CamTag"] is not None else "no tag"
            print(f"[{mode}] rendered sample candidate ({status}).")

        def append_current_sample() -> None:
            if state["frame"] is None or state["detected_X_CamTag"] is None:
                print(f"[{mode}] current frame has no valid AprilTag detection; rendering another sample.")
                render_next_sample()
                return
            calibrator.X_CamTag = state["detected_X_CamTag"]
            base_append_and_solve()
            state["accepted"] += 1
            if state["accepted"] >= args.ransac_sample_size:
                error = compute_errors(tools, calibrator, gt, state["accepted"])
                report_errors(error, f"[{mode}] sample {state['accepted']:03d}:")
            else:
                print(f"[{mode}] sample {state['accepted']:03d}: appended, waiting for solve threshold.")
            render_next_sample()

        base_append_and_solve = calibrator.append_and_solve
        calibrator.append_and_solve = append_current_sample

        sample_button = calibrator.server.gui.add_button("sim_next_sample")
        sample_button.on_click(lambda _: render_next_sample())

        print(f"[{mode}] interactive mode. click_and_append appends, solves, then advances to a new sample.")
        print(f"[{mode}] Use sim_next_sample to skip the current sample without appending.")
        try:
            render_next_sample()
            while True:
                time.sleep(1.0)
        finally:
            sim.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["thirdview", "wrist", "both"], default="both")
    parser.add_argument("--run", choices=["batch", "interactive"], default="batch")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size", type=float, default=0.048)
    parser.add_argument("--tag-marker-fraction", type=float, default=0.92)
    parser.add_argument("--tag-rows", type=int, default=4)
    parser.add_argument("--tag-cols", type=int, default=4)
    parser.add_argument("--tag-gap-ratio", type=float, default=0.0)
    parser.add_argument("--board-min-tags", type=int, default=4)
    parser.add_argument("--tag-board-yaml", type=Path, default=DEFAULT_COMPACT_BOARD_YAML, help="AprilTag board YAML layout.")
    parser.add_argument("--tag-board-image", type=Path, default=DEFAULT_COMPACT_BOARD_IMAGE, help="Board image raster to render at physical scale in sim.")
    parser.add_argument("--tag-board-image-width-m", type=float, default=None, help="Override board image physical width in meters.")
    parser.add_argument("--tag-board-image-height-m", type=float, default=None, help="Override board image physical height in meters.")
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS_PATH)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--near", type=float, default=0.02)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument("--ransac-sample-size", type=int, default=8)
    parser.add_argument("--step-delay", type=float, default=0.03)
    parser.add_argument("--max-attempts-per-sample", type=int, default=80)
    parser.add_argument("--pointcloud-stride", type=int, default=4)
    parser.add_argument("--max-pointcloud-points", type=int, default=25000)
    parser.add_argument("--renderer", choices=["tiny", "opengl"], default="tiny")
    parser.add_argument("--pybullet-gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="Keep the final batch Viser scene alive until Ctrl+C.")
    parser.add_argument(
        "--error-plot",
        type=Path,
        default=SIM_OUTPUTS_DIR / "sim_error_plot_{mode}.png",
        help="Save batch rotation/translation error plot.",
    )
    parser.add_argument(
        "--error-csv",
        type=Path,
        default=SIM_OUTPUTS_DIR / "sim_error_history_{mode}.csv",
        help="Save batch rotation/translation error values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive.")
    if args.tag_size <= 0:
        raise SystemExit("--tag-size must be positive.")
    if args.tag_marker_fraction <= 0 or args.tag_marker_fraction > 1:
        raise SystemExit("--tag-marker-fraction must be in (0, 1].")
    if args.tag_rows <= 0 or args.tag_cols <= 0:
        raise SystemExit("--tag-rows and --tag-cols must be positive.")
    if args.tag_gap_ratio < 0:
        raise SystemExit("--tag-gap-ratio must be non-negative.")
    if args.board_min_tags <= 0:
        raise SystemExit("--board-min-tags must be positive.")
    if args.board_min_tags > args.tag_rows * args.tag_cols:
        raise SystemExit("--board-min-tags cannot exceed the number of grid tags.")
    if (args.tag_board_yaml is None) != (args.tag_board_image is None):
        raise SystemExit("--tag-board-yaml and --tag-board-image must be provided together.")
    if args.tag_board_yaml is not None:
        if not args.tag_board_yaml.exists():
            raise SystemExit(f"--tag-board-yaml does not exist: {args.tag_board_yaml}")
        if not args.tag_board_image.exists():
            raise SystemExit(f"--tag-board-image does not exist: {args.tag_board_image}")
    if args.tag_board_image_width_m is not None and args.tag_board_image_width_m <= 0:
        raise SystemExit("--tag-board-image-width-m must be positive.")
    if args.tag_board_image_height_m is not None and args.tag_board_image_height_m <= 0:
        raise SystemExit("--tag-board-image-height-m must be positive.")
    if args.ransac_sample_size <= 1:
        raise SystemExit("--ransac-sample-size must be greater than 1.")
    if args.pointcloud_stride <= 0:
        raise SystemExit("--pointcloud-stride must be positive.")
    if args.run == "interactive" and args.mode == "both":
        raise SystemExit("--run interactive requires --mode thirdview or --mode wrist.")

    tools = load_runtime_tools()
    modes = ["thirdview", "wrist"] if args.mode == "both" else [args.mode]

    if args.run == "batch":
        for index, mode in enumerate(modes):
            run_batch_for_mode(tools, args, mode, hold=args.hold and index == len(modes) - 1)
    else:
        run_interactive_for_mode(tools, args, modes[0])


if __name__ == "__main__":
    main()
