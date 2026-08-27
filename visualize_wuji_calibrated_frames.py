#!/usr/bin/env python3
"""Visualize calibrated Wuji G305 and thumb-mesh frames on a posed URDF."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation


FINGEREYE_ROOT = Path("/home/CNF2025915223/桌面/FingerEyeV2")
DEFAULT_URDF = FINGEREYE_ROOT / (
    "assets/thirdparty/xarm7_wuji_left_description/"
    "wuji_left_w_fingereye_6dof_floating_joint.urdf"
)
DEFAULT_EXTRINSICS = Path(__file__).resolve().parent / (
    "outputs/extrinsics/wuji_g305_thumb_fingertip/"
    "extrinsics_0824_012533.yaml"
)
CAMERA_KEY = "T_left_palm_link_wuji_g305_raw_left_optical"
THUMB_KEY = "T_left_finger1_link4_thumb_fingertip_mesh_frame"


@dataclass(frozen=True)
class SceneData:
    source: Path
    urdf_path: Path
    mesh_path: Path
    configuration: np.ndarray
    base_T_palm: np.ndarray
    base_T_thumb_link4: np.ndarray
    base_T_camera: np.ndarray
    base_T_thumb_mesh: np.ndarray
    camera_fov: float
    camera_aspect: float
    sample_index: int


def _load_yaml(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    return resolved, payload


def _transform(value: Any, name: str) -> np.ndarray:
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not +1")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} homogeneous row is invalid")
    return T


def _wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return xyzw[[3, 0, 1, 2]]


def _pose_kwargs(T: np.ndarray) -> dict[str, np.ndarray]:
    return {"position": T[:3, 3], "wxyz": _wxyz(T[:3, :3])}


def load_scene_data(
    extrinsics_path: Path, urdf_path: Path, sample_index: int
) -> SceneData:
    import yourdfpy

    source, payload = _load_yaml(extrinsics_path)
    resolved_urdf = urdf_path.expanduser().resolve()
    if not resolved_urdf.is_file():
        raise FileNotFoundError(resolved_urdf)
    known = payload.get("known_camera_extrinsic")
    if not isinstance(known, dict) or CAMERA_KEY not in known:
        raise KeyError(f"Missing known_camera_extrinsic.{CAMERA_KEY}")
    if THUMB_KEY not in payload:
        raise KeyError(f"Missing {THUMB_KEY}")
    palm_T_camera = _transform(known[CAMERA_KEY], CAMERA_KEY)
    thumb_link_T_mesh = _transform(payload[THUMB_KEY], THUMB_KEY)

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Extrinsics YAML has no captured samples")
    selected = sample_index if sample_index >= 0 else len(samples) + sample_index
    if not 0 <= selected < len(samples):
        raise IndexError(f"Sample index {sample_index} is outside 0..{len(samples)-1}")
    qpos20 = np.asarray(samples[selected]["qpos20_rad"], dtype=np.float64)
    if qpos20.shape != (20,) or not np.all(np.isfinite(qpos20)):
        raise ValueError("Selected sample qpos20_rad is invalid")

    model = yourdfpy.URDF.load(
        str(resolved_urdf),
        filename_handler=partial(
            yourdfpy.filename_handler_magic, dir=resolved_urdf.parent
        ),
        build_scene_graph=True,
        build_collision_scene_graph=False,
        load_meshes=False,
        load_collision_meshes=False,
    )
    expected = tuple(
        [
            "left_palm_floating_x_joint",
            "left_palm_floating_y_joint",
            "left_palm_floating_z_joint",
            "left_palm_floating_roll_joint",
            "left_palm_floating_pitch_joint",
            "left_palm_floating_yaw_joint",
        ]
        + [
            f"left_finger{finger}_joint{joint}"
            for finger in range(1, 6)
            for joint in range(1, 5)
        ]
    )
    if tuple(model.actuated_joint_names) != expected:
        raise ValueError("Floating Wuji URDF actuated-joint order is unexpected")
    configuration = np.concatenate((np.zeros(6, dtype=np.float64), qpos20))
    model.update_cfg(dict(zip(model.actuated_joint_names, configuration, strict=True)))
    base_T_palm = _transform(
        model.get_transform("left_palm_link", model.base_link), "base_T_palm"
    )
    base_T_thumb_link4 = _transform(
        model.get_transform("left_finger1_link4", model.base_link),
        "base_T_thumb_link4",
    )

    target = payload.get("metadata", {}).get("target", {})
    mesh_path = Path(str(target.get("mesh", ""))).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Thumb mesh not found: {mesh_path}")
    camera = payload.get("metadata", {}).get("camera", {})
    K = np.asarray(camera.get("K"), dtype=np.float64).reshape(3, 3)
    profile = str(camera.get("profile", "1280x800"))
    resolution = profile.split("@", 1)[0].split("x")
    width, height = float(resolution[0]), float(resolution[1])
    fov = 2.0 * np.arctan2(height, 2.0 * float(K[1, 1]))

    return SceneData(
        source=source,
        urdf_path=resolved_urdf,
        mesh_path=mesh_path,
        configuration=configuration,
        base_T_palm=base_T_palm,
        base_T_thumb_link4=base_T_thumb_link4,
        base_T_camera=base_T_palm @ palm_T_camera,
        base_T_thumb_mesh=base_T_thumb_link4 @ thumb_link_T_mesh,
        camera_fov=float(fov),
        camera_aspect=float(width / height),
        sample_index=selected,
    )


def _format_matrix(T: np.ndarray) -> str:
    return "\n".join(
        "[" + ", ".join(f"{value: .8f}" for value in row) + "]" for row in T
    )


def build_scene(server: Any, data: SceneData) -> None:
    from viser.extras import ViserUrdf

    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True
    server.scene.add_grid(
        "/ground",
        width=0.35,
        height=0.35,
        plane="xy",
        cell_size=0.01,
        section_size=0.05,
    )
    robot = ViserUrdf(
        server,
        data.urdf_path,
        root_node_name="/wuji_floating_urdf",
        load_meshes=True,
        load_collision_meshes=False,
    )
    robot.update_cfg(data.configuration)

    axis_length = 0.025
    frames = {
        "palm": server.scene.add_frame(
            "/frames/left_palm_link",
            axes_length=axis_length,
            axes_radius=0.0012,
            **_pose_kwargs(data.base_T_palm),
        ),
        "camera": server.scene.add_frame(
            "/frames/wuji_g305_raw_left_optical",
            axes_length=axis_length,
            axes_radius=0.0012,
            **_pose_kwargs(data.base_T_camera),
        ),
        "thumb_link": server.scene.add_frame(
            "/frames/left_finger1_link4",
            axes_length=axis_length,
            axes_radius=0.0012,
            **_pose_kwargs(data.base_T_thumb_link4),
        ),
        "thumb_mesh": server.scene.add_frame(
            "/frames/thumb_fingertip_mesh_frame",
            axes_length=axis_length,
            axes_radius=0.0012,
            **_pose_kwargs(data.base_T_thumb_mesh),
        ),
    }
    frustum = server.scene.add_camera_frustum(
        "/camera/g305_raw_left_optical",
        fov=data.camera_fov,
        aspect=data.camera_aspect,
        scale=0.035,
        color=(255, 120, 30),
        line_width=2.5,
        **_pose_kwargs(data.base_T_camera),
    )
    camera_ray_end = data.base_T_camera[:3, 3] + data.base_T_camera[:3, 2] * 0.08
    camera_ray = server.scene.add_line_segments(
        "/links/g305_positive_z",
        points=np.asarray([[data.base_T_camera[:3, 3], camera_ray_end]]),
        colors=(40, 100, 255),
        line_width=3.0,
    )
    links = server.scene.add_line_segments(
        "/links/calibrated_transforms",
        points=np.asarray(
            [
                [data.base_T_palm[:3, 3], data.base_T_camera[:3, 3]],
                [data.base_T_thumb_link4[:3, 3], data.base_T_thumb_mesh[:3, 3]],
            ]
        ),
        colors=np.asarray([[(255, 150, 30)] * 2, [(180, 70, 255)] * 2]),
        line_width=3.0,
    )

    thumb_mesh = trimesh.load(data.mesh_path, force="mesh", process=False)
    if not isinstance(thumb_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected one triangle mesh: {data.mesh_path}")
    thumb_mesh.apply_scale(0.001)  # thumb.obj stores millimetres.
    thumb_mesh.apply_transform(data.base_T_thumb_mesh)
    mesh_handle = server.scene.add_mesh_trimesh(
        "/calibrated_thumb_obj",
        thumb_mesh,
    )

    labels = [
        server.scene.add_label(
            "/labels/palm", "left_palm_link", position=data.base_T_palm[:3, 3]
        ),
        server.scene.add_label(
            "/labels/camera",
            "wuji_g305_raw_left_optical",
            position=data.base_T_camera[:3, 3],
        ),
        server.scene.add_label(
            "/labels/thumb_link",
            "left_finger1_link4",
            position=data.base_T_thumb_link4[:3, 3],
        ),
        server.scene.add_label(
            "/labels/thumb_mesh",
            "thumb_fingertip_mesh_frame",
            position=data.base_T_thumb_mesh[:3, 3],
        ),
    ]

    with server.gui.add_folder("Visibility"):
        show_robot = server.gui.add_checkbox("Floating URDF", initial_value=True)
        show_frames = server.gui.add_checkbox("Calibrated frames", initial_value=True)
        show_mesh = server.gui.add_checkbox("Calibrated thumb.obj", initial_value=True)
        show_camera = server.gui.add_checkbox("G305 frustum/+Z", initial_value=True)
        show_labels = server.gui.add_checkbox("Labels", initial_value=True)

    @show_robot.on_update
    def _(_) -> None:
        robot.show_visual = bool(show_robot.value)

    @show_frames.on_update
    def _(_) -> None:
        visible = bool(show_frames.value)
        for handle in frames.values():
            handle.visible = visible
        links.visible = visible

    @show_mesh.on_update
    def _(_) -> None:
        mesh_handle.visible = bool(show_mesh.value)

    @show_camera.on_update
    def _(_) -> None:
        frustum.visible = bool(show_camera.value)
        camera_ray.visible = bool(show_camera.value)

    @show_labels.on_update
    def _(_) -> None:
        for handle in labels:
            handle.visible = bool(show_labels.value)

    with server.gui.add_folder("Transforms"):
        server.gui.add_markdown(
            f"**URDF:** `{data.urdf_path}`  \n"
            f"**Sample:** `{data.sample_index}`  \n"
            "Orange: `left_palm_link → G305 optical`  \n"
            "Purple: `left_finger1_link4 → thumb mesh frame`"
        )
        server.gui.add_markdown(
            "`T_left_palm_link_wuji_g305_raw_left_optical`\n```text\n"
            + _format_matrix(np.linalg.inv(data.base_T_palm) @ data.base_T_camera)
            + "\n```"
        )
        server.gui.add_markdown(
            "`T_left_finger1_link4_thumb_fingertip_mesh_frame`\n```text\n"
            + _format_matrix(
                np.linalg.inv(data.base_T_thumb_link4) @ data.base_T_thumb_mesh
            )
            + "\n```"
        )

    center = 0.5 * (data.base_T_palm[:3, 3] + data.base_T_thumb_mesh[:3, 3])
    server.initial_camera.position = tuple(center + np.asarray([0.25, -0.25, 0.18]))
    server.initial_camera.look_at = tuple(center)
    server.initial_camera.up = (0.0, 0.0, 1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Visualization server must listen on loopback")
    data = load_scene_data(args.yaml, args.urdf, args.sample_index)
    print(f"[OK] URDF: {data.urdf_path}")
    print(f"[OK] extrinsics: {data.source}")
    print(f"[OK] qpos sample: {data.sample_index}")
    print(f"[OK] thumb mesh: {data.mesh_path}")
    print(f"[OK] base_T_camera translation: {data.base_T_camera[:3, 3].tolist()} m")
    print(f"[OK] base_T_thumb_mesh translation: {data.base_T_thumb_mesh[:3, 3].tolist()} m")
    if args.check:
        return

    import viser

    server = viser.ViserServer(host=args.host, port=args.port, label="Wuji calibrated frames")
    build_scene(server, data)
    print(f"[INFO] Viser URL: http://127.0.0.1:{server.get_port()}")
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
