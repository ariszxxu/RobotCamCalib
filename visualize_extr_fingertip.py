from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_EXTRINSICS_YAML = Path(
    "/home/ps/RobotCamCalib1/outputs/extrinsics_wrist_Q_thumb_web_cam_middle_finger_cam_apriltag_grid_offline_2samples_0712_030212_0712_031300.yaml"
)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_AXIS_LENGTH_M = 0.05
DEFAULT_AXIS_RADIUS_M = 0.002
DEFAULT_FRUSTUM_SCALE_M = 0.04
DEFAULT_CUBE_SIZE_M = 0.0125
LABELS_VISIBLE_BY_DEFAULT = False


def load_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_transform(data: dict[str, Any], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing transform key '{key}' in extrinsics YAML")
    T = np.asarray(data[key], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{key} must be 4x4, got {T.shape}")
    return T


def rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def pose_to_wxyz_position(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return rotation_matrix_to_wxyz(T[:3, :3]), T[:3, 3].astype(np.float64)


def load_camera_model(path_str: str | None) -> tuple[float, float]:
    if not path_str:
        return np.deg2rad(80.0), 4.0 / 3.0

    path = Path(path_str).expanduser()
    if not path.exists():
        return np.deg2rad(80.0), 4.0 / 3.0

    data = load_yaml(path)
    K_key = "K" if "K" in data else "camera_matrix"
    K = np.asarray(data[K_key], dtype=np.float64)
    width, height = [float(v) for v in data["image_size"]]
    fy = float(K[1, 1])
    vertical_fov = 2.0 * np.arctan2(height, 2.0 * fy)
    aspect = width / height
    return float(vertical_fov), float(aspect)


def resolve_camera_entries(extrinsics: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = extrinsics.get("inputs", {})
    entries: list[dict[str, Any]] = []

    camera_names = inputs.get("camera_names")
    if isinstance(camera_names, list):
        for camera_name in camera_names:
            key = f"Q_T_{camera_name}"
            if key not in extrinsics:
                continue
            intrinsics_yaml = inputs.get(f"{camera_name}_intrinsics_yaml")
            intrinsics_yaml = intrinsics_yaml or inputs.get("cv2_camera_to_intrinsics_yaml", {}).get(camera_name)
            entries.append(
                {
                    "name": str(camera_name),
                    "transform": load_transform(extrinsics, key),
                    "intrinsics_yaml": intrinsics_yaml,
                }
            )

    if len(entries) >= 2:
        return entries[:2]

    if "Q_T_root" in extrinsics and "Q_T_tip" in extrinsics:
        return [
            {
                "name": "root_cam",
                "transform": load_transform(extrinsics, "Q_T_root"),
                "intrinsics_yaml": inputs.get("root_intrinsics_yaml"),
            },
            {
                "name": "tip_cam",
                "transform": load_transform(extrinsics, "Q_T_tip"),
                "intrinsics_yaml": inputs.get("tip_intrinsics_yaml"),
            },
        ]

    transform_keys = sorted(k for k in extrinsics if k.startswith("Q_T_"))
    for key in transform_keys[:2]:
        camera_name = key.removeprefix("Q_T_")
        entries.append(
            {
                "name": camera_name,
                "transform": load_transform(extrinsics, key),
                "intrinsics_yaml": inputs.get(f"{camera_name}_intrinsics_yaml"),
            }
        )

    if len(entries) < 2:
        raise KeyError(
            "Extrinsics YAML must contain two Q_T_* camera transforms, "
            "or legacy Q_T_root/Q_T_tip keys."
        )
    return entries


def add_pose_frame(
    server: Any,
    name: str,
    T_Q_frame: np.ndarray,
    *,
    axes_length: float,
    axes_radius: float,
    label_offset: np.ndarray,
    label_handles: list[Any],
) -> None:
    wxyz, position = pose_to_wxyz_position(T_Q_frame)
    server.scene.add_frame(
        name=f"/frames/{name}",
        axes_length=axes_length,
        axes_radius=axes_radius,
        origin_radius=axes_radius * 2.0,
        wxyz=wxyz,
        position=position,
    )
    label_handles.append(
        server.scene.add_label(
            name=f"/labels/{name}",
            text=name,
            position=position + label_offset,
            font_size_mode="scene",
            font_scene_height=axes_length * 0.22,
            visible=LABELS_VISIBLE_BY_DEFAULT,
        )
    )


def build_scene(
    server: Any,
    extrinsics: dict[str, Any],
    *,
    axis_length: float,
    axis_radius: float,
    frustum_scale: float,
    cube_size: float,
) -> list[Any]:
    camera_entries = resolve_camera_entries(extrinsics)
    label_handles: list[Any] = []

    server.scene.set_up_direction("+z")

    server.scene.add_frame(
        name="/frames/Q_aprilcube_origin",
        axes_length=axis_length,
        axes_radius=axis_radius,
        origin_radius=axis_radius * 2.5,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_box(
        name="/aprilcube_Q",
        dimensions=(cube_size, cube_size, cube_size),
        color=(70, 120, 255),
        opacity=0.25,
        position=(0.0, 0.0, 0.0),
    )
    label_handles.append(
        server.scene.add_label(
            name="/labels/Q",
            text="Q / AprilCube frame",
            position=(0.0, 0.0, axis_length * 1.25),
            font_size_mode="scene",
            font_scene_height=axis_length * 0.2,
            visible=LABELS_VISIBLE_BY_DEFAULT,
        )
    )

    label_offset = np.asarray([0.0, 0.0, axis_length * 0.55], dtype=np.float64)
    colors = [(255, 80, 80), (80, 180, 255)]
    camera_positions: list[np.ndarray] = []
    camera_names: list[str] = []
    camera_model_info: list[tuple[str, float, float]] = []

    for entry, color in zip(camera_entries, colors):
        camera_name = entry["name"]
        T_Q_camera = entry["transform"]
        add_pose_frame(
            server,
            camera_name,
            T_Q_camera,
            axes_length=axis_length,
            axes_radius=axis_radius,
            label_offset=label_offset,
            label_handles=label_handles,
        )
        camera_wxyz, camera_position = pose_to_wxyz_position(T_Q_camera)
        fov, aspect = load_camera_model(entry.get("intrinsics_yaml"))
        server.scene.add_camera_frustum(
            name=f"/frustums/{camera_name}",
            fov=fov,
            aspect=aspect,
            scale=frustum_scale,
            color=color,
            line_width=2.0,
            wxyz=camera_wxyz,
            position=camera_position,
        )
        camera_positions.append(camera_position)
        camera_names.append(camera_name)
        camera_model_info.append((camera_name, fov, aspect))

    line_points = np.asarray(
        [
            [[0.0, 0.0, 0.0], camera_position]
            for camera_position in camera_positions
        ],
        dtype=np.float32,
    )
    line_colors = np.asarray(
        [
            [color, color]
            for color in colors[: len(camera_positions)]
        ],
        dtype=np.uint8,
    )
    server.scene.add_line_segments(
        name="/links/Q_to_cameras",
        points=line_points,
        colors=line_colors,
        line_width=2.0,
    )

    baseline = float(np.linalg.norm(camera_positions[0] - camera_positions[1]))
    baseline_name = f"{camera_names[0]}-{camera_names[1]}"
    label_handles.append(
        server.scene.add_label(
            name="/labels/baseline",
            text=f"{baseline_name} baseline: {baseline * 1000.0:.1f} mm",
            position=(0.0, -axis_length * 0.9, axis_length * 0.65),
            font_size_mode="scene",
            font_scene_height=axis_length * 0.16,
            visible=LABELS_VISIBLE_BY_DEFAULT,
        )
    )

    print("Loaded fingertip extrinsics in Q / AprilCube frame")
    for camera_name, camera_position in zip(camera_names, camera_positions):
        print(f"  Q_T_{camera_name} translation: {camera_position.tolist()} m")
    print(f"  {baseline_name} baseline: {baseline * 1000.0:.2f} mm")
    for camera_name, fov, aspect in camera_model_info:
        print(f"  {camera_name} frustum fov/aspect: {np.degrees(fov):.1f} deg / {aspect:.3f}")
    print(f"  labels visible by default: {LABELS_VISIBLE_BY_DEFAULT}")
    return label_handles


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize Q-frame camera extrinsics in viser."
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=DEFAULT_EXTRINSICS_YAML,
        help="Filtered fingertip extrinsics YAML.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="viser server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="viser server port.")
    parser.add_argument("--axis-length", type=float, default=DEFAULT_AXIS_LENGTH_M)
    parser.add_argument("--axis-radius", type=float, default=DEFAULT_AXIS_RADIUS_M)
    parser.add_argument("--frustum-scale", type=float, default=DEFAULT_FRUSTUM_SCALE_M)
    parser.add_argument("--cube-size", type=float, default=DEFAULT_CUBE_SIZE_M)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    extrinsics = load_yaml(args.yaml)

    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "viser is required. Run this script in your pyroki environment."
        ) from exc

    server = viser.ViserServer(host=args.host, port=args.port)
    build_scene(
        server,
        extrinsics,
        axis_length=args.axis_length,
        axis_radius=args.axis_radius,
        frustum_scale=args.frustum_scale,
        cube_size=args.cube_size,
    )

    print(f"viser is running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping viser server.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
