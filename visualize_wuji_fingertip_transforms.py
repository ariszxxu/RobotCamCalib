#!/usr/bin/env python3
"""Visualize calibrated Wuji thumb/index mesh frames on the floating-hand URDF."""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from robot_cam_calib.fingertip_extrinsics import PalmTipKinematics, _origin_transform
from robot_cam_calib.geometry import inv_T


FINGEREYE_ROOT = Path("/home/CNF2025915223/桌面/FingerEyeV2")
DEFAULT_URDF = FINGEREYE_ROOT / (
    "assets/thirdparty/xarm7_wuji_left_description/"
    "wuji_left_w_fingereye_6dof_floating_joint.urdf"
)
DEFAULT_THUMB_YAML = Path(
    "outputs/extrinsics/wuji_g305_thumb_fingertip/thumb_extrinsics.yaml"
)
DEFAULT_INDEX_YAML = Path(
    "outputs/extrinsics/wuji_g305_fingertip/index_extrinsics.yaml"
)
DEFAULT_0820_CAMERA_YAML = Path(
    "outputs/extrinsics/xarm7_g305_eye_in_hand/extrinsics_0820_174322.yaml"
)
DEFAULT_LINK7_PALM_URDF = FINGEREYE_ROOT / (
    "assets/thirdparty/xarm7_wuji_left_description/"
    "xarm7_wuji_left_w_fingereye_v4_XS130507J56A10.urdf"
)
DEFAULT_THUMB_MESH = FINGEREYE_ROOT / (
    "assets/thirdparty/xarm7_wuji_left_description/fingereye_mesh/thumb.obj"
)
DEFAULT_INDEX_MESH = FINGEREYE_ROOT / (
    "assets/thirdparty/xarm7_wuji_left_description/fingereye_mesh/"
    "index_wuji_w_cube.stl"
)
THUMB_LINK = "left_finger1_link4"
INDEX_LINK = "left_finger2_link4"
THUMB_FRAME = "thumb_fingertip_mesh_frame"
INDEX_FRAME = "index_wuji_w_cube_update"
CAMERA_FRAME = "wuji_g305_raw_left_optical"
CAMERA_KEY = f"T_left_palm_link_{CAMERA_FRAME}"
RECTIFIED_CAMERA_FRAME = "wuji_g305_rectified_left_optical"
RECTIFIED_CAMERA_KEY = f"T_left_palm_link_{RECTIFIED_CAMERA_FRAME}"
LINK7_CAMERA_KEY = f"T_link7_{CAMERA_FRAME}"
DEFAULT_DISPLAY_QPOS20 = np.asarray(
    [
        0.9968166503816333,
        -0.06143662576209986,
        0.1396353317742144,
        0.6799425553389793,
        0.008934570290589722,
        0.00016236635185731124,
        0.007067401131773041,
        -0.026781138045795565,
        0.8370800400648373,
        0.21213943605494326,
        -0.4106338021544563,
        0.3472791737560977,
        1.4006342841778583,
        0.188966322128086,
        1.3929626949532306,
        1.4136077694044906,
        1.4317781014116422,
        0.2407392504510414,
        1.4582142716708413,
        1.5505700455596787,
    ],
    dtype=np.float64,
)
HAND_JOINT_NAMES = tuple(
    f"left_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)


def _load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")
    return payload


def _load_transform(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Missing transform {key}")
    transform = np.asarray(payload[key], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{key} must be a finite 4x4 transform")
    return transform


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _load_display_qpos(thumb_payload: dict[str, Any]) -> np.ndarray:
    samples = thumb_payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return DEFAULT_DISPLAY_QPOS20.copy()
    qpos = np.asarray(samples[-1]["qpos20_rad"], dtype=np.float64)
    if qpos.shape != (20,) or not np.all(np.isfinite(qpos)):
        raise ValueError("Thumb display qpos must contain 20 finite radians")
    return qpos


def _load_metric_mesh(path: Path, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh = trimesh.load(path.expanduser().resolve(), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load triangle mesh: {path}")
    # The historical cube-frame OBJ/STL files are authored in millimetres.
    if float(np.max(mesh.extents)) > 1.0:
        mesh.apply_scale(0.001)
    mesh.visual.face_colors = color
    return mesh


def _scene_data(args: argparse.Namespace) -> dict[str, Any]:
    urdf = args.urdf.expanduser().resolve()
    thumb_yaml = args.thumb_yaml.expanduser().resolve()
    index_yaml = args.index_yaml.expanduser().resolve()
    camera_0820_yaml = args.camera_0820_yaml.expanduser().resolve()
    link7_palm_urdf = args.link7_palm_urdf.expanduser().resolve()
    for path in (
        urdf,
        thumb_yaml,
        index_yaml,
        camera_0820_yaml,
        link7_palm_urdf,
        args.thumb_mesh,
        args.index_mesh,
    ):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)

    thumb_payload = _load_yaml(thumb_yaml)
    index_payload = _load_yaml(index_yaml)
    T_thumb_link_mesh = _load_transform(
        thumb_payload, f"T_{THUMB_LINK}_{THUMB_FRAME}"
    )
    aprilcube_key = f"T_{THUMB_LINK}_thumb_aprilcube_detection_frame"
    T_thumb_link_aprilcube = (
        _load_transform(thumb_payload, aprilcube_key)
        if aprilcube_key in thumb_payload
        else T_thumb_link_mesh.copy()
    )
    T_index_link_mesh = _load_transform(
        index_payload, f"T_{INDEX_LINK}_{INDEX_FRAME}"
    )
    T_palm_camera_thumb = _load_transform(thumb_payload, CAMERA_KEY)
    T_palm_camera_index = _load_transform(index_payload, CAMERA_KEY)
    if not np.allclose(T_palm_camera_thumb, T_palm_camera_index, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("Thumb/index YAMLs do not contain the same raw camera pose")
    T_palm_camera_rectified_thumb = _load_transform(
        thumb_payload, RECTIFIED_CAMERA_KEY
    )
    T_palm_camera_rectified_index = _load_transform(
        index_payload, RECTIFIED_CAMERA_KEY
    )
    if not np.allclose(
        T_palm_camera_rectified_thumb,
        T_palm_camera_rectified_index,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Thumb/index YAMLs do not contain the same rectified camera pose")
    camera_0820_payload = _load_yaml(camera_0820_yaml)
    T_link7_camera = _load_transform(camera_0820_payload, LINK7_CAMERA_KEY)
    root = ET.parse(link7_palm_urdf).getroot()
    link7_palm_joint = next(
        (
            joint
            for joint in root.findall("joint")
            if joint.find("parent") is not None
            and joint.find("parent").get("link") == "link7"
            and joint.find("child") is not None
            and joint.find("child").get("link") == "left_palm_link"
        ),
        None,
    )
    if link7_palm_joint is None:
        raise RuntimeError("Comparison URDF has no link7 -> left_palm_link joint")
    T_palm_camera_offset = inv_T(_origin_transform(link7_palm_joint)) @ T_link7_camera
    qpos20 = _load_display_qpos(thumb_payload)
    thumb_fk = PalmTipKinematics(urdf, THUMB_LINK)
    index_fk = PalmTipKinematics(urdf, INDEX_LINK)
    T_palm_thumb_link = thumb_fk.forward(qpos20)
    T_palm_index_link = index_fk.forward(qpos20)
    return {
        "urdf": urdf,
        "thumb_yaml": thumb_yaml,
        "index_yaml": index_yaml,
        "qpos20": qpos20,
        "T_thumb_link_mesh": T_thumb_link_mesh,
        "T_thumb_link_aprilcube": T_thumb_link_aprilcube,
        "T_index_link_mesh": T_index_link_mesh,
        "T_palm_camera_final": T_palm_camera_thumb,
        "T_palm_camera_rectified": T_palm_camera_rectified_thumb,
        "T_palm_camera_offset": T_palm_camera_offset,
        "T_camera_offset_final": inv_T(T_palm_camera_offset) @ T_palm_camera_thumb,
        "T_palm_thumb_link": T_palm_thumb_link,
        "T_palm_index_link": T_palm_index_link,
        "T_palm_thumb_mesh": T_palm_thumb_link @ T_thumb_link_mesh,
        "T_palm_thumb_aprilcube": T_palm_thumb_link @ T_thumb_link_aprilcube,
        "T_palm_index_mesh": T_palm_index_link @ T_index_link_mesh,
    }


def _add_frame(server: Any, name: str, transform: np.ndarray, length: float) -> Any:
    return server.scene.add_frame(
        name,
        axes_length=length,
        axes_radius=0.0008,
        origin_radius=0.0014,
        wxyz=_matrix_to_wxyz(transform[:3, :3]),
        position=transform[:3, 3],
    )


def _format_matrix(name: str, transform: np.ndarray) -> str:
    rows = ["[" + ", ".join(f"{value: .6f}" for value in row) + "]" for row in transform]
    return f"**{name}**\n\n```text\n" + "\n".join(rows) + "\n```"


def run(args: argparse.Namespace) -> None:
    data = _scene_data(args)
    print(f"[OK] URDF: {data['urdf']}")
    print(f"[OK] display qpos20: {data['qpos20'].tolist()}")
    print(
        f"[OK] T_{THUMB_LINK}_{THUMB_FRAME}:\n"
        f"{data['T_thumb_link_mesh']}"
    )
    print(f"[OK] T_{INDEX_LINK}_{INDEX_FRAME}:\n{data['T_index_link_mesh']}")
    print(f"[OK] final raw palm camera:\n{data['T_palm_camera_final']}")
    print(f"[OK] final rectified palm camera:\n{data['T_palm_camera_rectified']}")
    print(f"[OK] 0820 + URDF calculated raw palm camera:\n{data['T_palm_camera_offset']}")
    print(f"[OK] inverse(0820+URDF) @ final raw:\n{data['T_camera_offset_final']}")
    if args.check:
        return

    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(
        host=args.host, port=args.port, label="Wuji thumb + index calibrated frames"
    )
    server.gui.set_panel_label("WujiHand · thumb/index mesh-frame extrinsics")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True
    server.initial_camera.position = (0.24, -0.24, 0.22)
    server.initial_camera.look_at = (0.0, 0.0, 0.10)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.initial_camera.near = 0.001
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
        data["urdf"],
        root_node_name="/wuji_floating_urdf",
        mesh_color_override=(0.72, 0.76, 0.84, 0.72),
    )
    joint_names = tuple(robot.get_actuated_joint_names())
    if len(joint_names) != 26 or joint_names[6:] != HAND_JOINT_NAMES:
        raise RuntimeError(f"Unexpected floating URDF joint order: {joint_names}")
    robot.update_cfg(np.concatenate((np.zeros(6), data["qpos20"])))

    _add_frame(server, "/frames/left_palm_link", np.eye(4), 0.035)
    thumb_link_handle = _add_frame(
        server, f"/frames/{THUMB_LINK}", data["T_palm_thumb_link"], 0.022
    )
    index_link_handle = _add_frame(
        server, f"/frames/{INDEX_LINK}", data["T_palm_index_link"], 0.022
    )
    thumb_mesh_frame_handle = _add_frame(
        server, f"/frames/{THUMB_FRAME}", data["T_palm_thumb_mesh"], 0.028
    )
    camera_offset_handle = _add_frame(
        server,
        f"/frames/{CAMERA_FRAME}_0820_urdf_offset",
        data["T_palm_camera_offset"],
        0.032,
    )
    camera_final_handle = _add_frame(
        server,
        f"/frames/{CAMERA_FRAME}_final",
        data["T_palm_camera_final"],
        0.032,
    )
    camera_rectified_handle = _add_frame(
        server,
        f"/frames/{RECTIFIED_CAMERA_FRAME}_final",
        data["T_palm_camera_rectified"],
        0.032,
    )
    server.scene.add_label(
        "/labels/camera_link7_derived",
        "camera raw · 0820 + URDF calculated",
        position=data["T_palm_camera_offset"][:3, 3],
    )
    server.scene.add_label(
        "/labels/camera_final",
        "camera raw · final (coincident with 0820+URDF)",
        position=data["T_palm_camera_final"][:3, 3],
    )
    server.scene.add_label(
        "/labels/camera_rectified",
        "camera rectified · live stereo calibration",
        position=data["T_palm_camera_rectified"][:3, 3],
    )
    server.scene.add_label(
        "/labels/thumb_mesh",
        "thumb mesh frame",
        position=data["T_palm_thumb_mesh"][:3, 3],
    )
    index_mesh_frame_handle = _add_frame(
        server, f"/frames/{INDEX_FRAME}", data["T_palm_index_mesh"], 0.028
    )
    server.scene.add_label(
        "/labels/index_mesh",
        "index mesh frame",
        position=data["T_palm_index_mesh"][:3, 3],
    )

    camera_points = np.asarray(
        [[data["T_palm_camera_offset"][:3, 3], data["T_palm_camera_final"][:3, 3]]],
        dtype=np.float64,
    )
    camera_delta_handle = server.scene.add_line_segments(
        "/comparisons/camera_origin_delta",
        points=camera_points,
        colors=(255, 40, 200),
        line_width=5.0,
    )
    camera_delta_rotation = float(
        np.degrees(
            Rotation.from_matrix(
                data["T_palm_camera_offset"][:3, :3].T
                @ data["T_palm_camera_final"][:3, :3]
            ).magnitude()
        )
    )
    camera_delta_translation = float(
        np.linalg.norm(
            data["T_palm_camera_final"][:3, 3]
            - data["T_palm_camera_offset"][:3, 3]
        )
    )
    server.scene.add_label(
        "/labels/camera_delta",
        f"camera delta: {camera_delta_translation * 1000.0:.2f} mm / "
        f"{camera_delta_rotation:.2f} deg · xyz="
        f"{np.round(data['T_camera_offset_final'][:3, 3] * 1000.0, 2).tolist()} mm",
        position=np.mean(camera_points[0], axis=0),
    )

    thumb_mesh = _load_metric_mesh(args.thumb_mesh, (255, 145, 70, 105))
    index_mesh = _load_metric_mesh(args.index_mesh, (70, 185, 255, 105))
    thumb_ghost = server.scene.add_mesh_trimesh(
        "/calibrated_meshes/thumb_final",
        thumb_mesh,
        wxyz=_matrix_to_wxyz(data["T_palm_thumb_mesh"][:3, :3]),
        position=data["T_palm_thumb_mesh"][:3, 3],
    )
    index_ghost = server.scene.add_mesh_trimesh(
        "/calibrated_meshes/index_wuji_w_cube",
        index_mesh,
        wxyz=_matrix_to_wxyz(data["T_palm_index_mesh"][:3, :3]),
        position=data["T_palm_index_mesh"][:3, 3],
    )
    thumb_cube = server.scene.add_box(
        "/diagnostics/thumb_aprilcube_detection_ids_12_17",
        dimensions=(0.01875, 0.01875, 0.01875),
        color=(255, 125, 35),
        opacity=0.35,
        wireframe=True,
        wxyz=_matrix_to_wxyz(data["T_palm_thumb_aprilcube"][:3, :3]),
        position=data["T_palm_thumb_aprilcube"][:3, 3],
    )
    index_cube = server.scene.add_box(
        "/aprilcubes/index_ids_6_11",
        dimensions=(0.01875, 0.01875, 0.01875),
        color=(35, 150, 255),
        opacity=0.35,
        wireframe=True,
        wxyz=_matrix_to_wxyz(data["T_palm_index_mesh"][:3, :3]),
        position=data["T_palm_index_mesh"][:3, 3],
    )

    with server.gui.add_folder("Visibility", expand_by_default=True):
        show_urdf = server.gui.add_checkbox("Floating Wuji URDF", initial_value=True)
        show_thumb = server.gui.add_checkbox(
            "Orange: final thumb", initial_value=True
        )
        show_index = server.gui.add_checkbox("Blue: index", initial_value=True)
        show_cubes = server.gui.add_checkbox(
            "AprilCube detection diagnostics", initial_value=False
        )
        show_link_frames = server.gui.add_checkbox("link4 frames", initial_value=True)
        show_target_frames = server.gui.add_checkbox(
            "thumb/index target frames", initial_value=True
        )
        show_camera_frames = server.gui.add_checkbox(
            "Raw + rectified camera frames", initial_value=True
        )
        show_camera_delta = server.gui.add_checkbox(
            "Magenta: camera delta", initial_value=True
        )

    @show_urdf.on_update
    def _(_: Any) -> None:
        robot.show_visual = bool(show_urdf.value)

    @show_thumb.on_update
    def _(_: Any) -> None:
        thumb_ghost.visible = bool(show_thumb.value)

    @show_index.on_update
    def _(_: Any) -> None:
        index_ghost.visible = bool(show_index.value)

    @show_cubes.on_update
    def _(_: Any) -> None:
        thumb_cube.visible = bool(show_cubes.value)
        index_cube.visible = bool(show_cubes.value)

    @show_link_frames.on_update
    def _(_: Any) -> None:
        thumb_link_handle.visible = bool(show_link_frames.value)
        index_link_handle.visible = bool(show_link_frames.value)

    @show_target_frames.on_update
    def _(_: Any) -> None:
        thumb_mesh_frame_handle.visible = bool(show_target_frames.value)
        index_mesh_frame_handle.visible = bool(show_target_frames.value)

    @show_camera_frames.on_update
    def _(_: Any) -> None:
        camera_offset_handle.visible = bool(show_camera_frames.value)
        camera_final_handle.visible = bool(show_camera_frames.value)
        camera_rectified_handle.visible = bool(show_camera_frames.value)

    @show_camera_delta.on_update
    def _(_: Any) -> None:
        camera_delta_handle.visible = bool(show_camera_delta.value)

    with server.gui.add_folder("Calibrated transforms", expand_by_default=False):
        server.gui.add_markdown(
            _format_matrix(
                f"T_{THUMB_LINK}_{THUMB_FRAME}",
                data["T_thumb_link_mesh"],
            )
        )
        server.gui.add_markdown(
            _format_matrix(
                f"0820 + URDF calculated T_left_palm_link_{CAMERA_FRAME}",
                data["T_palm_camera_offset"],
            )
        )
        server.gui.add_markdown(
            _format_matrix(
                f"Final T_left_palm_link_{CAMERA_FRAME}",
                data["T_palm_camera_final"],
            )
        )
        server.gui.add_markdown(
            _format_matrix(
                f"Final T_left_palm_link_{RECTIFIED_CAMERA_FRAME}",
                data["T_palm_camera_rectified"],
            )
        )
        server.gui.add_markdown(
            _format_matrix(
                "inverse(0820+URDF raw) @ final raw",
                data["T_camera_offset_final"],
            )
        )
        server.gui.add_markdown(
            _format_matrix(
                f"T_{INDEX_LINK}_{INDEX_FRAME}", data["T_index_link_mesh"]
            )
        )
    server.gui.add_markdown(
        "**Color key:** orange = final thumb; blue = index IDs 6–11; "
        "magenta = final raw versus 0820+URDF raw (should be zero).  "
        "The thumb AprilCube and mesh use the same photographed target frame.  "
        "Frame axes: red +X, green +Y, blue +Z."
    )
    print(f"[VISER] http://127.0.0.1:{server.get_port()}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--thumb-yaml", type=Path, default=DEFAULT_THUMB_YAML)
    parser.add_argument("--index-yaml", type=Path, default=DEFAULT_INDEX_YAML)
    parser.add_argument(
        "--camera-0820-yaml", type=Path, default=DEFAULT_0820_CAMERA_YAML
    )
    parser.add_argument("--link7-palm-urdf", type=Path, default=DEFAULT_LINK7_PALM_URDF)
    parser.add_argument("--thumb-mesh", type=Path, default=DEFAULT_THUMB_MESH)
    parser.add_argument("--index-mesh", type=Path, default=DEFAULT_INDEX_MESH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--check", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
