"""Convention-explicit eye-in-hand calibration and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import inv_T, make_T, residual_stats, robust_limit, transform_delta


@dataclass(frozen=True)
class HandEyeObservation:
    """One stationary ``base_T_gripper`` / ``camera_T_target`` pair."""

    index: int
    T_base_gripper: np.ndarray
    T_camera_target: np.ndarray


_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("Cannot average no transforms")
    rotations = Rotation.from_matrix(
        np.stack([np.asarray(T, dtype=np.float64)[:3, :3] for T in transforms])
    )
    translation = np.mean(
        np.stack([np.asarray(T, dtype=np.float64)[:3, 3] for T in transforms]),
        axis=0,
    )
    return make_T(rotations.mean().as_matrix(), translation)


def _check_rigid(T: np.ndarray) -> None:
    transform = np.asarray(T, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(transform)):
        raise ValueError("Hand-eye solver returned non-finite values")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("Hand-eye solver returned a non-orthonormal rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("Hand-eye solver returned an improper rotation")


def solve_method(
    observations: list[HandEyeObservation],
    method_name: str,
) -> np.ndarray:
    if len(observations) < 3:
        raise ValueError("OpenCV hand-eye calibration needs at least 3 poses")
    method = _METHODS[method_name]
    gripper_R = [item.T_base_gripper[:3, :3] for item in observations]
    gripper_t = [item.T_base_gripper[:3, 3] for item in observations]
    target_R = [item.T_camera_target[:3, :3] for item in observations]
    target_t = [item.T_camera_target[:3, 3] for item in observations]
    camera_R, camera_t = cv2.calibrateHandEye(
        gripper_R,
        gripper_t,
        target_R,
        target_t,
        method=method,
    )
    result = make_T(camera_R, np.asarray(camera_t).reshape(3))
    _check_rigid(result)
    return result


def board_pose_residuals(
    observations: list[HandEyeObservation],
    T_gripper_camera: np.ndarray,
) -> tuple[np.ndarray, list[float], list[float]]:
    board_poses = [
        item.T_base_gripper @ T_gripper_camera @ item.T_camera_target
        for item in observations
    ]
    center = mean_transform(board_poses)
    deltas = [transform_delta(center, pose) for pose in board_poses]
    return center, [item[0] for item in deltas], [item[1] for item in deltas]


def _candidate(
    observations: list[HandEyeObservation],
    method_name: str,
) -> dict[str, Any]:
    transform = solve_method(observations, method_name)
    board_pose, rotations, translations = board_pose_residuals(
        observations, transform
    )
    rotation_stats = residual_stats(rotations)
    translation_stats = residual_stats(translations)
    return {
        "method": method_name,
        "T_gripper_camera": transform,
        "T_base_target_mean": board_pose,
        "rotation_residual_deg": rotations,
        "translation_residual_m": translations,
        "rotation_stats_deg": rotation_stats,
        "translation_stats_m": translation_stats,
        "score": rotation_stats["median"]
        + 100.0 * translation_stats["median"],
    }


def best_candidate(observations: list[HandEyeObservation]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for method_name in _METHODS:
        try:
            candidates.append(_candidate(observations, method_name))
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            failures[method_name] = f"{type(exc).__name__}: {exc}"
    if not candidates:
        raise RuntimeError(f"All hand-eye methods failed: {failures}")
    winner = min(candidates, key=lambda item: float(item["score"]))
    winner["method_candidates"] = [
        {
            "method": item["method"],
            "score": float(item["score"]),
            "rotation_stats_deg": item["rotation_stats_deg"],
            "translation_stats_m": item["translation_stats_m"],
        }
        for item in candidates
    ]
    winner["method_failures"] = failures
    return winner


def excitation_diagnostics(
    observations: list[HandEyeObservation],
) -> dict[str, Any]:
    rotation_vectors: list[np.ndarray] = []
    for first_index, first in enumerate(observations):
        for second in observations[first_index + 1 :]:
            relative = inv_T(first.T_base_gripper) @ second.T_base_gripper
            vector = Rotation.from_matrix(relative[:3, :3]).as_rotvec()
            if np.linalg.norm(vector) > np.deg2rad(0.5):
                rotation_vectors.append(vector)
    if rotation_vectors:
        singular = np.linalg.svd(np.stack(rotation_vectors), compute_uv=False)
        rank = int(np.sum(singular > max(float(singular[0]) * 0.05, 1e-8)))
    else:
        singular = np.zeros(3, dtype=np.float64)
        rank = 0
    translations = np.stack(
        [item.T_base_gripper[:3, 3] for item in observations]
    )
    return {
        "relative_rotation_singular_values_rad": singular.tolist(),
        "relative_rotation_rank": rank,
        "translation_span_m": np.ptp(translations, axis=0).tolist(),
        "translation_span_norm_m": float(
            np.linalg.norm(np.ptp(translations, axis=0))
        ),
    }


def solve_hand_eye_robust(
    observations: list[HandEyeObservation],
    *,
    min_samples: int = 10,
    max_iterations: int = 5,
) -> dict[str, Any]:
    if len(observations) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} hand-eye samples, got {len(observations)}"
        )
    active = list(observations)
    rejected: list[int] = []
    iterations: list[dict[str, Any]] = []
    for iteration in range(max_iterations):
        result = best_candidate(active)
        rotations = result["rotation_residual_deg"]
        translations = result["translation_residual_m"]
        rotation_limit = robust_limit(rotations, 0.5, 8.0)
        translation_limit = robust_limit(translations, 0.002, 0.030)
        bad = [
            item.index
            for item, rotation, translation in zip(
                active, rotations, translations
            )
            if rotation > rotation_limit or translation > translation_limit
        ]
        iterations.append(
            {
                "iteration": iteration,
                "active_indices": [item.index for item in active],
                "method": result["method"],
                "rotation_limit_deg": rotation_limit,
                "translation_limit_m": translation_limit,
                "new_rejected_indices": bad,
            }
        )
        if not bad or len(active) - len(bad) < min_samples:
            break
        bad_set = set(bad)
        rejected.extend(bad)
        active = [item for item in active if item.index not in bad_set]

    result = best_candidate(active)
    result["inlier_indices"] = [item.index for item in active]
    result["outlier_indices"] = sorted(set(rejected))
    result["robust_iterations"] = iterations
    result["excitation"] = excitation_diagnostics(active)

    even = active[::2]
    odd = active[1::2]
    cross_validation: dict[str, Any] = {"available": False}
    if len(even) >= 5 and len(odd) >= 5:
        try:
            even_T = solve_method(even, str(result["method"]))
            odd_T = solve_method(odd, str(result["method"]))
            rotation, translation = transform_delta(even_T, odd_T)
            cross_validation = {
                "available": True,
                "even_count": len(even),
                "odd_count": len(odd),
                "rotation_delta_deg": rotation,
                "translation_delta_m": translation,
            }
        except (ValueError, cv2.error) as exc:
            cross_validation = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    result["cross_validation"] = cross_validation
    return result
