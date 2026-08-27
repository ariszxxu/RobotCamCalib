#!/usr/bin/env python3
"""Reframe the Wuji index fingertip mesh without changing its physical geometry."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh
import yaml

from robot_cam_calib.io import atomic_yaml_dump


FINGEREYE_ROOT = Path("/home/CNF2025915223/桌面/FingerEyeV2")
MESH_ROOT = FINGEREYE_ROOT / (
    "assets/thirdparty/xarm7_wuji_left_description/fingereye_mesh"
)
DEFAULT_INPUT = MESH_ROOT / "index_wuji_w_cube.stl"
DEFAULT_OUTPUT = MESH_ROOT / "index_wuji_w_cube_update.stl"
DEFAULT_EXTRINSICS = Path(
    "outputs/extrinsics/wuji_g305_fingertip/"
    "extrinsics_0824_010826_raw_cube_frame.yaml"
)
SOURCE_TRANSFORM_KEY = "T_left_finger2_link4_index_wuji_w_cube_update"
REFRAMED_CANONICAL_SHA256 = (
    "059428fea14587059893a72494907fe7c33d98d876447497a968fa3e92194e36"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected one triangle mesh: {path}")
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError(f"Mesh is empty: {path}")
    return loaded


def run(args: argparse.Namespace) -> None:
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    extrinsics = args.extrinsics.expanduser().resolve()
    reframed_yaml = args.reframed_yaml.expanduser().resolve()
    sidecar = args.sidecar.expanduser().resolve()
    required_inputs = (source,) if args.mesh_only else (source, extrinsics)
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(source) == REFRAMED_CANONICAL_SHA256:
        raise RuntimeError(
            f"Input is already the canonical Rz(-90deg) mesh; refusing to rotate twice: {source}"
        )
    outputs = (output,) if args.mesh_only else (output, reframed_yaml, sidecar)
    for path in outputs:
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite without --force: {path}")

    # User-confirmed frame correction: bake -90 degrees about Z into the mesh
    # coordinates. The link pose is composed with the inverse (+90 degrees),
    # so the physical geometry remains unchanged while the frame itself moves.
    # Therefore p_new = Rz(-90 deg) @ p_old = [old_y, -old_x, old_z].
    R_new_old = np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T_new_old = np.eye(4, dtype=np.float64)
    T_new_old[:3, :3] = R_new_old
    T_old_new = np.linalg.inv(T_new_old)

    source_mesh = _load_mesh(source)
    reframed_mesh = source_mesh.copy()
    reframed_mesh.apply_transform(T_new_old)
    output.parent.mkdir(parents=True, exist_ok=True)
    reframed_mesh.export(output)
    verified = _load_mesh(output)
    expected_vertices = np.asarray(source_mesh.vertices) @ R_new_old.T
    expected_bounds = np.stack(
        (expected_vertices.min(axis=0), expected_vertices.max(axis=0))
    )
    if not np.allclose(verified.bounds, expected_bounds, rtol=0.0, atol=1.0e-5):
        raise RuntimeError("Exported mesh bounds do not match the requested reframe")
    if len(verified.faces) != len(source_mesh.faces):
        raise RuntimeError("Exported mesh face count changed")

    print(f"[SOURCE] {source} sha256={_sha256(source)}")
    print(f"[OUTPUT] {output} sha256={_sha256(output)}")
    print("[AXES] mesh vertices baked with Rz(-90 deg): p_new = [old_y, -old_x, old_z]")
    print(f"[EXTENTS] {source_mesh.extents.tolist()} -> {verified.extents.tolist()} mm")
    if args.mesh_only:
        return

    payload = yaml.safe_load(extrinsics.read_text(encoding="utf-8"))
    if SOURCE_TRANSFORM_KEY not in payload:
        raise KeyError(f"Missing {SOURCE_TRANSFORM_KEY} in {extrinsics}")
    T_link_target = np.asarray(payload[SOURCE_TRANSFORM_KEY], dtype=np.float64)
    # p_new = T_new_old @ p_old, hence p_old = T_old_new @ p_new and
    # T_link_new = T_link_old @ T_old_new.
    T_link_new = T_link_target @ T_old_new
    payload[SOURCE_TRANSFORM_KEY] = T_link_new.tolist()
    samples = payload.get("samples", [])
    if samples:
        raise ValueError(
            "Use the compact final YAML without embedded camera-target samples"
        )
    target_metadata = payload.setdefault("metadata", {}).setdefault("target", {})
    target_metadata["mesh"] = str(output)
    target_metadata["frame"] = "index_wuji_w_cube_update"
    target_metadata["mesh_frame_contract"] = (
        "Frame redefined by p_new=Rz(-90deg)@p_old and "
        "T_link_new=T_link_old@Rz(+90deg)"
    )
    payload["mesh_frame_reorientation"] = {
        "source_extrinsics": str(extrinsics),
        "source_mesh": str(source),
        "reframed_mesh": str(output),
        "axis_contract": {
            "new_+X": "old_+Y",
            "new_+Y": "old_-X",
            "new_+Z": "old_+Z",
            "coordinate_formula": "p_new = [old_y, -old_x, old_z]",
        },
        "T_new_old": T_new_old.tolist(),
        "T_old_new": T_old_new.tolist(),
        "transform_update": "T_link_new = T_link_old @ T_old_new",
        "embedded_sample_update": "not applicable; compact final YAML has no samples",
    }
    atomic_yaml_dump(reframed_yaml, payload)

    atomic_yaml_dump(
        sidecar,
        {
            "schema": "robot_cam_calib.mesh_frame_reorientation.v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "source_mesh": str(source),
            "source_sha256": _sha256(source),
            "output_mesh": str(output),
            "output_sha256": _sha256(output),
            "source_units": "mm",
            "source_extents": np.asarray(source_mesh.extents).tolist(),
            "output_extents": np.asarray(verified.extents).tolist(),
            "face_count": int(len(verified.faces)),
            "axis_contract": {
                "new_+X": "old_+Y",
                "new_+Y": "old_-X",
                "new_+Z": "old_+Z",
            },
            "T_new_old": T_new_old.tolist(),
            "T_old_new": T_old_new.tolist(),
            "source_extrinsics": str(extrinsics),
            "reframed_extrinsics": str(reframed_yaml),
            "extrinsics_composition": "T_link_new = T_link_old @ T_old_new",
            SOURCE_TRANSFORM_KEY: T_link_new.tolist(),
        },
    )
    print(f"[RESULT] {SOURCE_TRANSFORM_KEY} (reframed):\n{T_link_new}")
    print(f"[YAML] {reframed_yaml}")
    print(f"[SIDECAR] {sidecar}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    parser.add_argument(
        "--reframed-yaml",
        type=Path,
        default=Path(
            "outputs/extrinsics/wuji_g305_fingertip/"
            "extrinsics_0824_010826_reframed.yaml"
        ),
    )
    parser.add_argument(
        "--sidecar", type=Path, default=DEFAULT_OUTPUT.with_suffix(".frame.yaml")
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Bake Rz(-90 deg) into the output mesh without writing YAML sidecars",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
