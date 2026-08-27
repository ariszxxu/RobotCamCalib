#!/usr/bin/env python3
"""Solve Wuji thumb link-to-cube extrinsics from multi-pose 2D tag corners.

The palm-to-camera transform is fixed.  A single constant
``T_left_finger1_link4_thumb_fingertip_mesh_frame`` is optimized across all
captured poses, so planar single-tag pose ambiguity is not solved independently
in every image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot_cam_calib.geometry import (
    inv_T,
    params_to_transform,
    transform_delta,
    transform_to_params,
)
from robot_cam_calib.io import atomic_yaml_dump


FINGEREYE_ROOT = Path("/home/CNF2025915223/桌面/FingerEyeV2")
CAMERA_KEY = "T_left_palm_link_wuji_g305_raw_left_optical"
TARGET_KEY = "T_left_finger1_link4_thumb_fingertip_mesh_frame"
DEFAULT_CUBE_CONFIG = FINGEREYE_ROOT / (
    "assets/cubes/cube_april_36h11_12_17_1x1x1_15mm/config.json"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _transform(payload: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(payload[key], dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must be a finite 4x4 matrix")
    return value


def _tag_geometry(config_path: Path, tag_id: int) -> tuple[np.ndarray, int]:
    root = str(FINGEREYE_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from FingereyeData.image2cube_pose._vendor.aprilcube.detect import (  # noqa: PLC0415
        build_tag_corner_map,
        load_cube_config,
    )

    config, _faces = load_cube_config(str(config_path.expanduser().resolve()))
    corners = build_tag_corner_map(config)
    if tag_id not in corners:
        raise KeyError(f"Tag {tag_id} is absent from {config_path}")
    return np.asarray(corners[tag_id], dtype=np.float64) * 0.001, config.dict_id


def _observations(
    manifest: dict[str, Any], tag_id: int, dictionary_id: int
) -> list[dict[str, Any]]:
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary_id)
    )
    result: list[dict[str, Any]] = []
    for sample in manifest.get("samples", []):
        image_path = Path(str(sample["raw_image_path"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        corners, identifiers, _rejected = detector.detectMarkers(image)
        ids = [] if identifiers is None else identifiers.reshape(-1).tolist()
        if tag_id not in ids:
            print(f"[SKIP] sample={sample['index']} has tags={ids}, not {tag_id}")
            continue
        result.append(
            {
                "index": int(sample["index"]),
                "T_palm_tip": np.asarray(
                    sample["T_left_palm_link_tip"], dtype=np.float64
                ),
                "T_camera_cube_pnp": np.asarray(
                    sample[
                        "T_wuji_g305_raw_left_optical_"
                        "thumb_fingertip_mesh_frame"
                    ],
                    dtype=np.float64,
                ),
                "corners_px": np.asarray(
                    corners[ids.index(tag_id)], dtype=np.float64
                ).reshape(4, 2),
            }
        )
    if len(result) < 12:
        raise RuntimeError(f"Need at least 12 valid tag observations, got {len(result)}")
    return result


def _mean_pnp_initial(
    observations: list[dict[str, Any]], T_palm_camera: np.ndarray
) -> np.ndarray:
    candidates = [
        inv_T(item["T_palm_tip"])
        @ T_palm_camera
        @ item["T_camera_cube_pnp"]
        for item in observations
    ]
    rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in candidates]))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotations.mean().as_matrix()
    result[:3, 3] = np.median(
        np.stack([item[:3, 3] for item in candidates]), axis=0
    )
    return result


def _project(
    T_tip_cube: np.ndarray,
    observation: dict[str, Any],
    T_camera_palm: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    T_camera_cube = T_camera_palm @ observation["T_palm_tip"] @ T_tip_cube
    rvec = Rotation.from_matrix(T_camera_cube[:3, :3]).as_rotvec().reshape(3, 1)
    projected, _jacobian = cv2.projectPoints(
        object_points,
        rvec,
        T_camera_cube[:3, 3].reshape(3, 1),
        K,
        distortion,
    )
    return projected.reshape(4, 2)


def _solve(
    observations: list[dict[str, Any]],
    selected: list[int],
    initial: np.ndarray,
    T_camera_palm: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    def residual(parameters: np.ndarray) -> np.ndarray:
        T_tip_cube = params_to_transform(parameters)
        return np.concatenate(
            [
                (
                    _project(
                        T_tip_cube,
                        observations[index],
                        T_camera_palm,
                        object_points,
                        K,
                        distortion,
                    )
                    - observations[index]["corners_px"]
                ).reshape(-1)
                for index in selected
            ]
        )

    solution = least_squares(
        residual,
        transform_to_params(initial),
        method="trf",
        loss="cauchy",
        f_scale=1.5,
        max_nfev=3000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Reprojection solve failed: {solution.message}")
    return params_to_transform(solution.x)


def _per_image_rms(
    transform: np.ndarray,
    observations: list[dict[str, Any]],
    T_camera_palm: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            _project(
                                transform,
                                item,
                                T_camera_palm,
                                object_points,
                                K,
                                distortion,
                            )
                            - item["corners_px"]
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
            for item in observations
        ],
        dtype=np.float64,
    )


def solve(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_yaml(manifest_path)
    camera_payload = _load_yaml(args.camera_yaml)
    initial_payload = _load_yaml(args.initial_yaml)
    T_palm_camera = _transform(camera_payload, CAMERA_KEY)
    T_camera_palm = inv_T(T_palm_camera)
    T_cad = _transform(initial_payload, TARGET_KEY)
    object_points, dictionary_id = _tag_geometry(args.cube_config, args.tag_id)
    observations = _observations(manifest, args.tag_id, dictionary_id)
    camera = manifest["metadata"]["camera"]
    K = np.asarray(camera["K"], dtype=np.float64)
    distortion = np.asarray(camera["dist"], dtype=np.float64)
    pnp_initial = _mean_pnp_initial(observations, T_palm_camera)
    all_indices = list(range(len(observations)))

    candidates = [
        _solve(
            observations,
            all_indices,
            initial,
            T_camera_palm,
            object_points,
            K,
            distortion,
        )
        for initial in (pnp_initial, T_cad)
    ]
    errors = [
        _per_image_rms(
            item, observations, T_camera_palm, object_points, K, distortion
        )
        for item in candidates
    ]
    best_index = int(np.argmin([np.median(item) for item in errors]))
    estimate = candidates[best_index]
    per_image = errors[best_index]

    median = float(np.median(per_image))
    mad = float(1.4826 * np.median(np.abs(per_image - median)))
    limit = max(2.0, median + 3.0 * mad)
    inliers = [index for index, error in enumerate(per_image) if error <= limit]
    if len(inliers) < max(12, len(observations) * 3 // 4):
        raise RuntimeError(f"Only {len(inliers)}/{len(observations)} images are inliers")
    estimate = _solve(
        observations,
        inliers,
        estimate,
        T_camera_palm,
        object_points,
        K,
        distortion,
    )
    per_image = _per_image_rms(
        estimate, observations, T_camera_palm, object_points, K, distortion
    )

    odd = inliers[::2]
    even = inliers[1::2]
    odd_estimate = _solve(
        observations, odd, estimate, T_camera_palm, object_points, K, distortion
    )
    even_estimate = _solve(
        observations, even, estimate, T_camera_palm, object_points, K, distortion
    )
    split_rotation, split_translation = transform_delta(odd_estimate, even_estimate)

    loo_deltas: list[tuple[float, float]] = []
    held_out_rms: list[float] = []
    for held_out in inliers:
        training = [index for index in inliers if index != held_out]
        loo = _solve(
            observations,
            training,
            estimate,
            T_camera_palm,
            object_points,
            K,
            distortion,
        )
        loo_deltas.append(transform_delta(estimate, loo))
        held_out_rms.append(
            float(
                _per_image_rms(
                    loo,
                    [observations[held_out]],
                    T_camera_palm,
                    object_points,
                    K,
                    distortion,
                )[0]
            )
        )

    cad_rotation, cad_translation = transform_delta(T_cad, estimate)
    payload: dict[str, Any] = {
        "schema": "robot_cam_calib.wuji_thumb_global_reprojection.v1",
        "convention": "T_A_B maps coordinates from frame B into frame A",
        CAMERA_KEY: T_palm_camera.tolist(),
        TARGET_KEY: estimate.tolist(),
        "diagnostics": {
            "samples": len(observations),
            "inlier_sample_indices": [observations[i]["index"] for i in inliers],
            "outlier_sample_indices": [
                item["index"]
                for index, item in enumerate(observations)
                if index not in inliers
            ],
            "per_image_rms_px": per_image.tolist(),
            "reprojection_median_px": float(np.median(per_image[inliers])),
            "reprojection_p95_px": float(np.percentile(per_image[inliers], 95)),
            "reprojection_max_px": float(np.max(per_image[inliers])),
            "odd_even_rotation_deg": split_rotation,
            "odd_even_translation_m": split_translation,
            "leave_one_out_rotation_max_deg": float(
                max(item[0] for item in loo_deltas)
            ),
            "leave_one_out_translation_max_m": float(
                max(item[1] for item in loo_deltas)
            ),
            "leave_one_out_held_out_rms_p95_px": float(
                np.percentile(held_out_rms, 95)
            ),
            "delta_from_cad_rotation_deg": cad_rotation,
            "delta_from_cad_translation_m": cad_translation,
            "tag_id": args.tag_id,
            "manifest": str(manifest_path),
            "camera_source": str(args.camera_yaml.expanduser().resolve()),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera-yaml", type=Path, required=True)
    parser.add_argument("--initial-yaml", type=Path, required=True)
    parser.add_argument("--cube-config", type=Path, default=DEFAULT_CUBE_CONFIG)
    parser.add_argument("--tag-id", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = solve(args)
    atomic_yaml_dump(args.output.expanduser().resolve(), payload)
    print(f"[RESULT] {TARGET_KEY}:")
    print(np.asarray(payload[TARGET_KEY]))
    diagnostics = payload["diagnostics"]
    print(
        "[DIAGNOSTICS] reprojection median/p95/max="
        f"{diagnostics['reprojection_median_px']:.3f}/"
        f"{diagnostics['reprojection_p95_px']:.3f}/"
        f"{diagnostics['reprojection_max_px']:.3f}px"
    )
    print(
        "[STABILITY] odd-even="
        f"{diagnostics['odd_even_rotation_deg']:.3f}deg/"
        f"{diagnostics['odd_even_translation_m'] * 1000.0:.2f}mm; "
        "LOO max="
        f"{diagnostics['leave_one_out_rotation_max_deg']:.3f}deg/"
        f"{diagnostics['leave_one_out_translation_max_m'] * 1000.0:.2f}mm"
    )
    print(f"[INFO] Saved {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
