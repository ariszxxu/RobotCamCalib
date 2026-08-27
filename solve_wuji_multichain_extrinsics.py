#!/usr/bin/env python3
"""Jointly solve the Wuji palm camera, thumb target, and index target poses."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from calibrate_wuji_g305_fingertip import _load_manifest, _serializable
from robot_cam_calib.fingertip_extrinsics import (
    MultiChainObservation,
    _origin_transform,
    solve_multichain_fingertip_extrinsics,
)
from robot_cam_calib.geometry import inv_T, transform_delta
from robot_cam_calib.io import atomic_yaml_dump


CAMERA_KEY = "T_left_palm_link_wuji_g305_raw_left_optical"
THUMB_KEY = "T_left_finger1_link4_thumb_fingertip_mesh_frame"
INDEX_KEY = "T_left_finger2_link4_index_wuji_w_cube_update"
LINK7_CAMERA_KEY = "T_link7_wuji_g305_raw_left_optical"


def _load_group(chain: str, paths: list[Path]) -> tuple[list[MultiChainObservation], list[dict[str, Any]]]:
    observations: list[MultiChainObservation] = []
    sources: list[dict[str, Any]] = []
    next_index = 0
    for path in paths:
        samples, metadata = _load_manifest(path)
        batch_start = next_index
        for sample in samples:
            observations.append(
                MultiChainObservation(
                    chain,
                    next_index,
                    sample.T_palm_tip,
                    sample.T_camera_cube,
                )
            )
            next_index += 1
        sources.append(
            {
                "manifest": str(path.expanduser().resolve()),
                "sample_count": len(samples),
                "global_index_start": batch_start,
                "global_index_stop": next_index,
                "metadata": metadata,
            }
        )
    return observations, sources


def _delta(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    rotation_deg, translation_m = transform_delta(reference, candidate)
    return {
        "rotation_deg": float(rotation_deg),
        "translation_m": float(translation_m),
        "translation_mm": float(1000.0 * translation_m),
    }


def _solve(
    thumb: list[MultiChainObservation],
    index: list[MultiChainObservation],
    *,
    starts: int,
    seed: int,
) -> dict[str, Any]:
    return solve_multichain_fingertip_extrinsics(
        thumb + index,
        min_samples_per_chain=12,
        starts=starts,
        seed=seed,
    )


def run(args: argparse.Namespace) -> None:
    thumb, thumb_sources = _load_group("thumb", args.thumb_manifest)
    index, index_sources = _load_group("index", args.index_manifest)
    result = _solve(thumb, index, starts=args.starts, seed=args.seed)

    validations: dict[str, Any] = {}
    if len(args.thumb_manifest) >= 2 and len(args.index_manifest) >= 2:
        thumb_batches = []
        for source in thumb_sources:
            thumb_batches.append(
                thumb[source["global_index_start"] : source["global_index_stop"]]
            )
        index_batches = []
        for source in index_sources:
            index_batches.append(
                index[source["global_index_start"] : source["global_index_stop"]]
            )
        pair_a = _solve(
            thumb_batches[0], index_batches[0], starts=args.validation_starts, seed=args.seed + 10
        )
        pair_b = _solve(
            thumb_batches[1], index_batches[1], starts=args.validation_starts, seed=args.seed + 20
        )
        validations["independent_pair_a_vs_b"] = {
            "camera": _delta(pair_a["T_palm_camera"], pair_b["T_palm_camera"]),
            "thumb": _delta(
                pair_a["chains"]["thumb"]["T_tip_target"],
                pair_b["chains"]["thumb"]["T_tip_target"],
            ),
            "index": _delta(
                pair_a["chains"]["index"]["T_tip_target"],
                pair_b["chains"]["index"]["T_tip_target"],
            ),
            "pair_a_condition": pair_a["jacobian_condition"],
            "pair_b_condition": pair_b["jacobian_condition"],
        }

    compare_0820 = None
    if args.compare_camera_yaml is not None:
        payload = yaml.safe_load(args.compare_camera_yaml.read_text(encoding="utf-8"))
        reference_camera = None
        conversion = "direct palm-camera key"
        if CAMERA_KEY in payload:
            reference_camera = np.asarray(payload[CAMERA_KEY], dtype=np.float64)
        elif LINK7_CAMERA_KEY in payload:
            root = ET.parse(args.link7_palm_urdf.expanduser().resolve()).getroot()
            joint = next(
                (
                    item
                    for item in root.findall("joint")
                    if item.find("parent") is not None
                    and item.find("parent").get("link") == "link7"
                    and item.find("child") is not None
                    and item.find("child").get("link") == "left_palm_link"
                ),
                None,
            )
            if joint is None:
                raise RuntimeError("URDF has no fixed link7 -> left_palm_link joint")
            T_link7_palm = _origin_transform(joint)
            reference_camera = inv_T(T_link7_palm) @ np.asarray(
                payload[LINK7_CAMERA_KEY], dtype=np.float64
            )
            conversion = "inverse(T_link7_left_palm_link from URDF) @ T_link7_camera"
        if reference_camera is not None:
            compare_0820 = {
                "source": str(args.compare_camera_yaml.expanduser().resolve()),
                "conversion": conversion,
                "T_left_palm_link_wuji_g305_raw_left_optical": reference_camera.tolist(),
                "delta": _delta(
                    reference_camera,
                    np.asarray(result["T_palm_camera"], dtype=np.float64),
                ),
            }

    payload = {
        "schema": "robot_cam_calib.wuji_multichain_extrinsics.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "convention": "T_A_B maps coordinates from frame B into frame A",
        "method": (
            "one joint robust least-squares solve; palm-camera, thumb target, "
            "and index target are all free variables"
        ),
        "fixed_calibration_used": False,
        CAMERA_KEY: result["T_palm_camera"],
        THUMB_KEY: result["chains"]["thumb"]["T_tip_target"],
        INDEX_KEY: result["chains"]["index"]["T_tip_target"],
        "solver": result,
        "validation": validations,
        "comparison_to_0820_not_used_in_solve": compare_0820,
        "sources": {"thumb": thumb_sources, "index": index_sources},
    }
    atomic_yaml_dump(args.output, _serializable(payload))

    print(f"[RESULT] {CAMERA_KEY}:\n{np.asarray(payload[CAMERA_KEY])}")
    print(f"[RESULT] {THUMB_KEY}:\n{np.asarray(payload[THUMB_KEY])}")
    print(f"[RESULT] {INDEX_KEY}:\n{np.asarray(payload[INDEX_KEY])}")
    print(
        "[DIAGNOSTICS] rank={}/{} cond={:.1f} samples={} thumb={:.3f}deg/{:.2f}mm "
        "index={:.3f}deg/{:.2f}mm".format(
            result["jacobian_rank"],
            result["parameter_count"],
            result["jacobian_condition"],
            result["sample_count"],
            result["chains"]["thumb"]["rotation_stats_deg"]["median"],
            1000.0 * result["chains"]["thumb"]["translation_stats_m"]["median"],
            result["chains"]["index"]["rotation_stats_deg"]["median"],
            1000.0 * result["chains"]["index"]["translation_stats_m"]["median"],
        )
    )
    if validations:
        print(f"[VALIDATION] {validations}")
    if compare_0820 is not None:
        print(f"[COMPARE 0820 only] {compare_0820}")
    print(f"[INFO] Saved candidate {args.output.expanduser().resolve()}")

    if args.write_final:
        solver_summary = {
            "all_transforms_free": True,
            "sample_count": result["sample_count"],
            "parameter_count": result["parameter_count"],
            "jacobian_rank": result["jacobian_rank"],
            "jacobian_condition": result["jacobian_condition"],
            "optimizer_success": result["optimizer_success"],
            "thumb": {
                key: result["chains"]["thumb"][key]
                for key in (
                    "sample_count",
                    "inlier_count",
                    "rotation_stats_deg",
                    "translation_stats_m",
                )
            },
            "index": {
                key: result["chains"]["index"][key]
                for key in (
                    "sample_count",
                    "inlier_count",
                    "rotation_stats_deg",
                    "translation_stats_m",
                )
            },
            "independent_batch_validation": validations,
            "comparison_to_0820_not_used_in_solve": compare_0820,
        }
        common = {
            "convention": payload["convention"],
            CAMERA_KEY: payload[CAMERA_KEY],
            "method": payload["method"],
            "solver_summary": solver_summary,
        }
        thumb_final = dict(common)
        thumb_final[THUMB_KEY] = payload[THUMB_KEY]
        index_final = dict(common)
        index_final[INDEX_KEY] = payload[INDEX_KEY]
        atomic_yaml_dump(args.thumb_final, _serializable(thumb_final))
        atomic_yaml_dump(args.index_final, _serializable(index_final))
        print(f"[FINAL] wrote {args.thumb_final.expanduser().resolve()}")
        print(f"[FINAL] wrote {args.index_final.expanduser().resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thumb-manifest", type=Path, action="append", required=True)
    parser.add_argument("--index-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--validation-starts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--compare-camera-yaml", type=Path)
    parser.add_argument(
        "--link7-palm-urdf",
        type=Path,
        default=Path(
            "/home/CNF2025915223/桌面/FingerEyeV2/assets/thirdparty/"
            "xarm7_wuji_left_description/"
            "xarm7_wuji_left_w_fingereye_v4_XS130507J56A10.urdf"
        ),
    )
    parser.add_argument("--write-final", action="store_true")
    parser.add_argument(
        "--thumb-final",
        type=Path,
        default=Path("outputs/extrinsics/wuji_g305_thumb_fingertip/thumb_extrinsics.yaml"),
    )
    parser.add_argument(
        "--index-final",
        type=Path,
        default=Path("outputs/extrinsics/wuji_g305_fingertip/index_extrinsics.yaml"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
