#!/usr/bin/env python3
"""Visualize ^hand_back_palm T_g305_raw_left_rgb in Viser."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


DEFAULT_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/"
    "extrinsics_hand_back_palm_g305_raw_left_rgb_0801_182056.yaml"
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080


def load_yaml(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    return resolved, payload


def load_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains non-finite values")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not +1")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} has an invalid homogeneous row")
    return transform


def matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def load_scene_data(
    payload: dict[str, Any],
) -> tuple[np.ndarray, float, float, np.ndarray, dict[str, Any]]:
    expected_schema = "robot_cam_calib.hand_back_palm_g305_raw_left_rgb.v1"
    if payload.get("schema") != expected_schema:
        raise ValueError(
            f"Expected schema {expected_schema}, got {payload.get('schema')}"
        )
    solution = payload.get("solution")
    if not isinstance(solution, dict):
        raise ValueError("YAML has no solution mapping")
    transform = load_transform(
        solution["T_hand_back_palm_g305_raw_left_rgb"],
        "T_hand_back_palm_g305_raw_left_rgb",
    )
    inverse = load_transform(
        solution["T_g305_raw_left_rgb_hand_back_palm"],
        "T_g305_raw_left_rgb_hand_back_palm",
    )
    inverse_error = float(
        np.max(np.abs(transform @ inverse - np.eye(4)))
    )
    if inverse_error > 1e-6:
        raise ValueError(
            f"Stored forward/inverse transforms disagree: {inverse_error}"
        )

    camera = payload["inputs"]["g305_raw_left_rgb"]
    width, height = [float(value) for value in camera["resolution"]]
    intrinsic = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    vertical_fov = 2.0 * np.arctan2(height, 2.0 * intrinsic[1, 1])
    aspect = width / height

    cube_dimensions = np.full(3, 0.0625, dtype=np.float64)
    config_path = Path(
        payload["inputs"]["targets"][
            "hand_back_palm_aprilcube_config"
        ]
    ).expanduser()
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        dimensions_mm = config.get("box_dims")
        if dimensions_mm is not None:
            cube_dimensions = (
                np.asarray(dimensions_mm, dtype=np.float64).reshape(3)
                / 1000.0
            )
    return transform, float(vertical_fov), float(aspect), cube_dimensions, solution


def format_matrix(transform: np.ndarray) -> str:
    rows = [
        "[" + ", ".join(f"{value: .8f}" for value in row) + "]"
        for row in transform
    ]
    return "\n".join(rows)


def build_scene(
    server: Any,
    payload: dict[str, Any],
    source_path: Path,
    axis_length: float,
    frustum_scale: float,
) -> None:
    transform, fov, aspect, cube_dimensions, solution = load_scene_data(
        payload
    )
    camera_position = transform[:3, 3]
    camera_wxyz = matrix_to_wxyz(transform[:3, :3])
    distance = float(np.linalg.norm(camera_position))
    euler_xyz_deg = Rotation.from_matrix(transform[:3, :3]).as_euler(
        "xyz", degrees=True
    )

    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    server.scene.add_grid(
        "/reference_grid",
        width=0.4,
        height=0.4,
        cell_size=0.01,
        section_size=0.05,
        plane="xy",
        cell_color=(190, 190, 190),
        section_color=(120, 120, 120),
    )
    server.scene.add_frame(
        "/frames/hand_back_palm",
        axes_length=axis_length,
        axes_radius=axis_length * 0.035,
        origin_radius=axis_length * 0.07,
    )
    cube = server.scene.add_box(
        "/targets/hand_back_aprilcube",
        dimensions=cube_dimensions,
        color=(70, 120, 255),
        opacity=0.28,
    )
    camera_frame = server.scene.add_frame(
        "/frames/g305_raw_left_rgb",
        axes_length=axis_length,
        axes_radius=axis_length * 0.035,
        origin_radius=axis_length * 0.07,
        wxyz=camera_wxyz,
        position=camera_position,
    )
    frustum = server.scene.add_camera_frustum(
        "/cameras/g305_raw_left_rgb",
        fov=fov,
        aspect=aspect,
        scale=frustum_scale,
        color=(255, 120, 40),
        line_width=2.5,
        wxyz=camera_wxyz,
        position=camera_position,
    )
    link = server.scene.add_line_segments(
        "/links/hand_back_to_g305",
        points=np.asarray([[[0.0, 0.0, 0.0], camera_position]]),
        colors=(255, 210, 60),
        line_width=3.0,
    )
    optical_endpoint = (
        camera_position
        + transform[:3, :3] @ np.asarray([0.0, 0.0, axis_length * 1.8])
    )
    optical_axis = server.scene.add_line_segments(
        "/cameras/g305_positive_z_optical_axis",
        points=np.asarray([[camera_position, optical_endpoint]]),
        colors=(255, 80, 255),
        line_width=3.0,
    )
    hand_label = server.scene.add_label(
        "/labels/hand_back_palm",
        "hand_back_palm / AprilCube origin",
        position=(0.0, 0.0, axis_length * 0.75),
        font_size_mode="scene",
        font_scene_height=axis_length * 0.18,
    )
    camera_label = server.scene.add_label(
        "/labels/g305",
        "G305 raw-left RGB optical frame",
        position=camera_position + np.asarray([0.0, 0.0, axis_length * 0.5]),
        font_size_mode="scene",
        font_scene_height=axis_length * 0.18,
    )

    with server.gui.add_folder("Visibility"):
        show_cube = server.gui.add_checkbox("AprilCube", initial_value=True)
        show_frame = server.gui.add_checkbox(
            "G305 coordinate frame", initial_value=True
        )
        show_frustum = server.gui.add_checkbox(
            "G305 camera frustum", initial_value=True
        )
        show_link = server.gui.add_checkbox(
            "Origin-to-camera link", initial_value=True
        )
        show_optical = server.gui.add_checkbox(
            "+Z optical axis", initial_value=True
        )
        show_labels = server.gui.add_checkbox("Labels", initial_value=True)

    @show_cube.on_update
    def _(_) -> None:
        cube.visible = bool(show_cube.value)

    @show_frame.on_update
    def _(_) -> None:
        camera_frame.visible = bool(show_frame.value)

    @show_frustum.on_update
    def _(_) -> None:
        frustum.visible = bool(show_frustum.value)

    @show_link.on_update
    def _(_) -> None:
        link.visible = bool(show_link.value)

    @show_optical.on_update
    def _(_) -> None:
        optical_axis.visible = bool(show_optical.value)

    @show_labels.on_update
    def _(_) -> None:
        visible = bool(show_labels.value)
        hand_label.visible = visible
        camera_label.visible = visible

    residual_rot = solution["residual_rot_deg"]
    residual_trans = solution["residual_trans_m"]
    with server.gui.add_folder("Extrinsic summary"):
        server.gui.add_markdown(
            f"""
**Source:** `{source_path}`

**Transform:** `T_hand_back_palm_g305_raw_left_rgb`

- Camera center `[x, y, z]`: `{(camera_position * 1000.0).round(3).tolist()} mm`
- Origin distance: `{distance * 1000.0:.3f} mm`
- Euler XYZ: `{euler_xyz_deg.round(3).tolist()} deg`
- Vertical FOV: `{np.degrees(fov):.3f} deg`
- Aspect: `{aspect:.3f}`
- Inliers: `{len(solution['inlier_indices'])}`
- Residual mean: `{residual_rot['mean']:.3f} deg / {residual_trans['mean'] * 1000.0:.3f} mm`
- Residual P95: `{residual_rot['p95']:.3f} deg / {residual_trans['p95'] * 1000.0:.3f} mm`
"""
        )
    with server.gui.add_folder("4x4 matrix"):
        server.gui.add_markdown(
            "```text\n" + format_matrix(transform) + "\n```"
        )

    @server.on_client_connect
    def _(client: Any) -> None:
        client.camera.position = np.asarray([0.24, -0.25, 0.18])
        client.camera.look_at = 0.5 * camera_position
        client.camera.up = np.asarray([0.0, 0.0, 1.0])

    print(f"[INFO] source: {source_path}")
    print("[INFO] visualized: T_hand_back_palm_g305_raw_left_rgb")
    print(f"[INFO] camera center: {(camera_position * 1000.0).tolist()} mm")
    print(f"[INFO] origin distance: {distance * 1000.0:.3f} mm")
    print(f"[INFO] vertical FOV/aspect: {np.degrees(fov):.3f} deg/{aspect:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--axis-length", type=float, default=0.05)
    parser.add_argument("--frustum-scale", type=float, default=0.06)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    source_path, payload = load_yaml(args.yaml)
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "viser is required; run this script in the pyroki environment"
        ) from exc

    server = viser.ViserServer(host=args.host, port=args.port)
    build_scene(
        server,
        payload,
        source_path,
        axis_length=float(args.axis_length),
        frustum_scale=float(args.frustum_scale),
    )
    print(f"[INFO] Viser URL: http://localhost:{args.port}")
    print("[INFO] Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] stopping Viser")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
