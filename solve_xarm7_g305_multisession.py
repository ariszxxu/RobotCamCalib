#!/usr/bin/env python3
"""Jointly solve one xArm7-to-G305 transform from multiple capture sessions.

Each session gets its own nuisance ``T_base_charuco``.  This avoids treating a
small movement of the stationary ChArUco board between sessions as movement of
the eye-in-hand camera extrinsic.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot_cam_calib.geometry import (
    inv_T,
    params_to_transform,
    residual_stats,
    robust_limit,
    transform_delta,
    transform_to_params,
)
from robot_cam_calib.hand_eye import mean_transform


def _transform(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(4, 4)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _atomic_yaml_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(_plain(payload), sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _stats(values: list[float], scale: float = 1.0) -> dict[str, float]:
    return {key: scale * value for key, value in residual_stats(values).items()}


def _sample_groups(documents: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "index": int(sample["index"]),
                "T_base_link7": _transform(sample["T_base_link7"]),
                "T_camera_charuco": _transform(
                    sample["T_wuji_g305_raw_left_optical_charuco"]
                ),
            }
            for sample in document["samples"]
        ]
        for document in documents
    ]


def _validate_sources(
    paths: list[Path], documents: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("At least two calibration result files are required")
    expected_schema = "robot_cam_calib.xarm7_g305_eye_in_hand.v1"
    for path, document in zip(paths, documents):
        if document.get("schema") != expected_schema:
            raise ValueError(f"Unexpected schema in {path}: {document.get('schema')}")
        if int(document.get("sample_count", -1)) != len(document.get("samples", [])):
            raise ValueError(f"sample_count does not match samples in {path}")
        indices = [int(sample["index"]) for sample in document["samples"]]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate sample indices in {path}")

    cameras = [document["metadata"]["camera"] for document in documents]
    serials = {camera["serial"] for camera in cameras}
    if len(serials) != 1:
        raise ValueError(f"Camera serial mismatch: {sorted(serials)}")
    reference_K = np.asarray(cameras[0]["K"], dtype=np.float64)
    reference_dist = np.asarray(cameras[0]["dist"], dtype=np.float64)
    for path, camera in zip(paths[1:], cameras[1:]):
        if not np.array_equal(reference_K, np.asarray(camera["K"], dtype=np.float64)):
            raise ValueError(f"Camera intrinsics K mismatch in {path}")
        if not np.array_equal(
            reference_dist, np.asarray(camera["dist"], dtype=np.float64)
        ):
            raise ValueError(f"Camera distortion mismatch in {path}")

    targets = [document["metadata"]["target"] for document in documents]
    target_keys = (
        "squares_x",
        "squares_y",
        "square_length",
        "marker_length",
        "dictionary",
        "legacy_pattern",
    )
    reference_target = {key: targets[0][key] for key in target_keys}
    for path, target in zip(paths[1:], targets[1:]):
        current = {key: target[key] for key in target_keys}
        if current != reference_target:
            raise ValueError(f"ChArUco target mismatch in {path}")
    return {
        "camera_serial": next(iter(serials)),
        "K": reference_K,
        "dist": reference_dist,
        "intrinsics_source": cameras[0]["intrinsics_source"],
        "target": reference_target,
    }


def _fit(
    groups: list[list[dict[str, Any]]],
    initial_X: np.ndarray,
    rotation_scale_deg: float,
    translation_scale_m: float,
) -> tuple[np.ndarray, list[np.ndarray], Any]:
    initial_boards = [
        mean_transform(
            [
                sample["T_base_link7"]
                @ initial_X
                @ sample["T_camera_charuco"]
                for sample in group
            ]
        )
        for group in groups
    ]
    params0 = np.hstack(
        [transform_to_params(initial_X)]
        + [transform_to_params(board) for board in initial_boards]
    )
    rotation_scale_rad = np.deg2rad(rotation_scale_deg)

    def objective(params: np.ndarray) -> np.ndarray:
        X = params_to_transform(params[:6])
        boards = [
            params_to_transform(params[6 + 6 * index : 12 + 6 * index])
            for index in range(len(groups))
        ]
        residual: list[float] = []
        for group, board in zip(groups, boards):
            for sample in group:
                delta = (
                    inv_T(board)
                    @ sample["T_base_link7"]
                    @ X
                    @ sample["T_camera_charuco"]
                )
                residual.extend(
                    Rotation.from_matrix(delta[:3, :3]).as_rotvec()
                    / rotation_scale_rad
                )
                residual.extend(delta[:3, 3] / translation_scale_m)
        return np.asarray(residual, dtype=np.float64)

    result = least_squares(
        objective,
        params0,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=3000,
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
    )
    X = params_to_transform(result.x[:6])
    boards = [
        params_to_transform(result.x[6 + 6 * index : 12 + 6 * index])
        for index in range(len(groups))
    ]
    return X, boards, result


def _pose_residual(
    sample: dict[str, Any], X: np.ndarray, board: np.ndarray
) -> tuple[float, float]:
    delta = (
        inv_T(board)
        @ sample["T_base_link7"]
        @ X
        @ sample["T_camera_charuco"]
    )
    rotation_deg = float(
        np.degrees(
            np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
        )
    )
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    return rotation_deg, translation_m


def _session_diagnostics(
    groups: list[list[dict[str, Any]]], X: np.ndarray, boards: list[np.ndarray]
) -> list[dict[str, Any]]:
    diagnostics = []
    for group, board in zip(groups, boards):
        rotations: list[float] = []
        translations: list[float] = []
        samples = []
        for sample in group:
            rotation, translation = _pose_residual(sample, X, board)
            rotations.append(rotation)
            translations.append(translation)
            samples.append(
                {
                    "index": sample["index"],
                    "rotation_residual_deg": rotation,
                    "translation_residual_m": translation,
                }
            )
        diagnostics.append(
            {
                "sample_count": len(group),
                "T_base_charuco": board,
                "rotation_stats_deg": _stats(rotations),
                "translation_stats_m": _stats(translations),
                "samples": samples,
            }
        )
    return diagnostics


def _fit_robust(
    groups: list[list[dict[str, Any]]],
    initial_X: np.ndarray,
    rotation_scale_deg: float,
    translation_scale_m: float,
    *,
    max_iterations: int = 5,
    min_samples_per_session: int = 10,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    Any,
    list[list[dict[str, Any]]],
    dict[str, Any],
]:
    """Jointly reject session-local pose outliers and refit the shared transform."""
    active = [list(group) for group in groups]
    rejected: list[list[int]] = [[] for _group in groups]
    iterations: list[dict[str, Any]] = []
    current_X = initial_X

    for iteration in range(max_iterations + 1):
        X, boards, result = _fit(
            active,
            current_X,
            rotation_scale_deg,
            translation_scale_m,
        )
        diagnostics = _session_diagnostics(active, X, boards)
        session_iterations = []
        applied_rejections: list[set[int]] = []

        for session_index, (group, diagnostic) in enumerate(
            zip(active, diagnostics)
        ):
            rotations = [
                sample["rotation_residual_deg"]
                for sample in diagnostic["samples"]
            ]
            translations = [
                sample["translation_residual_m"]
                for sample in diagnostic["samples"]
            ]
            rotation_limit_deg = robust_limit(rotations, 0.5, 8.0)
            translation_limit_m = robust_limit(translations, 0.002, 0.030)
            proposed = {
                sample["index"]
                for sample, rotation, translation in zip(
                    group, rotations, translations
                )
                if rotation > rotation_limit_deg
                or translation > translation_limit_m
            }
            can_apply = (
                iteration < max_iterations
                and len(group) - len(proposed) >= min_samples_per_session
            )
            applied = proposed if can_apply else set()
            applied_rejections.append(applied)
            session_iterations.append(
                {
                    "session": session_index,
                    "active_indices": [sample["index"] for sample in group],
                    "rotation_limit_deg": rotation_limit_deg,
                    "translation_limit_m": translation_limit_m,
                    "proposed_rejected_indices": sorted(proposed),
                    "applied_rejected_indices": sorted(applied),
                    "minimum_sample_guard_triggered": bool(
                        proposed and not can_apply and iteration < max_iterations
                    ),
                }
            )
        iterations.append(
            {
                "iteration": iteration,
                "sessions": session_iterations,
            }
        )

        if not any(applied_rejections):
            return X, boards, result, active, {
                "enabled": True,
                "max_iterations": max_iterations,
                "min_samples_per_session": min_samples_per_session,
                "iterations": iterations,
                "rejected_indices_by_session": [
                    sorted(set(indices)) for indices in rejected
                ],
            }

        for session_index, bad in enumerate(applied_rejections):
            rejected[session_index].extend(sorted(bad))
            active[session_index] = [
                sample
                for sample in active[session_index]
                if sample["index"] not in bad
            ]
        current_X = X

    raise RuntimeError("Robust multi-session solve did not terminate")


def _cross_validate(
    groups: list[list[dict[str, Any]]],
    full_X: np.ndarray,
    rotation_scale_deg: float,
    translation_scale_m: float,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    fold_indices = []
    for group in groups:
        indices = np.arange(len(group))
        rng.shuffle(indices)
        fold_indices.append(np.array_split(indices, folds))

    fold_results = []
    all_rotation: list[float] = []
    all_translation: list[float] = []
    for fold in range(folds):
        train_groups = []
        validation_groups = []
        for session, group in enumerate(groups):
            validation_set = set(fold_indices[session][fold].tolist())
            train_groups.append(
                [sample for index, sample in enumerate(group) if index not in validation_set]
            )
            validation_groups.append(
                [sample for index, sample in enumerate(group) if index in validation_set]
            )
        fold_X, train_boards, _result = _fit(
            train_groups,
            full_X,
            rotation_scale_deg,
            translation_scale_m,
        )
        fold_rotation: list[float] = []
        fold_translation: list[float] = []
        for group, board in zip(validation_groups, train_boards):
            for sample in group:
                rotation, translation = _pose_residual(sample, fold_X, board)
                fold_rotation.append(rotation)
                fold_translation.append(translation)
        all_rotation.extend(fold_rotation)
        all_translation.extend(fold_translation)
        model_rotation, model_translation = transform_delta(full_X, fold_X)
        fold_results.append(
            {
                "fold": fold,
                "training_sample_count": sum(map(len, train_groups)),
                "validation_sample_count": sum(map(len, validation_groups)),
                "model_delta_from_full_deg": model_rotation,
                "model_delta_from_full_m": model_translation,
                "validation_rotation_stats_deg": _stats(fold_rotation),
                "validation_translation_stats_m": _stats(fold_translation),
            }
        )
    return {
        "folds": folds,
        "seed": seed,
        "stratified_by_session": True,
        "rotation_stats_deg": _stats(all_rotation),
        "translation_stats_m": _stats(all_translation),
        "max_model_delta_from_full_deg": max(
            item["model_delta_from_full_deg"] for item in fold_results
        ),
        "max_model_delta_from_full_m": max(
            item["model_delta_from_full_m"] for item in fold_results
        ),
        "fold_results": fold_results,
    }


def _leave_one_session_out(
    groups: list[list[dict[str, Any]]],
    full_X: np.ndarray,
    rotation_scale_deg: float,
    translation_scale_m: float,
) -> list[dict[str, Any]]:
    output = []
    for held_out in range(len(groups)):
        training = [group for index, group in enumerate(groups) if index != held_out]
        X, _boards, _result = _fit(
            training, full_X, rotation_scale_deg, translation_scale_m
        )
        held_group = groups[held_out]
        held_board = mean_transform(
            [
                sample["T_base_link7"] @ X @ sample["T_camera_charuco"]
                for sample in held_group
            ]
        )
        rotations = []
        translations = []
        for sample in held_group:
            rotation, translation = _pose_residual(sample, X, held_board)
            rotations.append(rotation)
            translations.append(translation)
        model_rotation, model_translation = transform_delta(full_X, X)
        output.append(
            {
                "held_out_session": held_out,
                "training_sample_count": sum(map(len, training)),
                "held_out_sample_count": len(held_group),
                "model_delta_from_full_deg": model_rotation,
                "model_delta_from_full_m": model_translation,
                "held_out_rotation_stats_deg": _stats(rotations),
                "held_out_translation_stats_m": _stats(translations),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rotation-scale-deg", type=float, default=0.25)
    parser.add_argument("--translation-scale-mm", type=float, default=2.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Iteratively reject joint residual outliers within each session",
    )
    parser.add_argument("--robust-max-iterations", type=int, default=5)
    parser.add_argument("--robust-min-samples-per-session", type=int, default=10)
    args = parser.parse_args()

    paths = [path.expanduser().resolve() for path in args.inputs]
    documents = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    shared = _validate_sources(paths, documents)
    input_groups = _sample_groups(documents)
    initial_X = mean_transform(
        [_transform(document["T_link7_wuji_g305_raw_left_optical"]) for document in documents]
    )
    translation_scale_m = args.translation_scale_mm / 1000.0
    robust_filter: dict[str, Any] = {"enabled": False}
    if args.robust:
        X, boards, result, groups, robust_filter = _fit_robust(
            input_groups,
            initial_X,
            args.rotation_scale_deg,
            translation_scale_m,
            max_iterations=args.robust_max_iterations,
            min_samples_per_session=args.robust_min_samples_per_session,
        )
    else:
        groups = input_groups
        X, boards, result = _fit(
            groups, initial_X, args.rotation_scale_deg, translation_scale_m
        )
    if min(map(len, groups)) < args.folds:
        parser.error(
            "--folds cannot exceed the number of retained samples in any session"
        )
    diagnostics = _session_diagnostics(groups, X, boards)
    cross_validation = _cross_validate(
        groups,
        X,
        args.rotation_scale_deg,
        translation_scale_m,
        args.folds,
        args.seed,
    )
    leave_one_session_out = _leave_one_session_out(
        groups, X, args.rotation_scale_deg, translation_scale_m
    )
    source_records = []
    rejected_by_session = robust_filter.get(
        "rejected_indices_by_session", [[] for _path in paths]
    )
    for session_index, (path, document, group) in enumerate(
        zip(paths, documents, groups)
    ):
        source_X = _transform(document["T_link7_wuji_g305_raw_left_optical"])
        rotation, translation = transform_delta(X, source_X)
        source_records.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sample_count": int(document["sample_count"]),
                "used_sample_count": len(group),
                "used_indices": [sample["index"] for sample in group],
                "joint_rejected_indices": rejected_by_session[session_index],
                "capture_manifest": document["capture_manifest"],
                "delta_from_joint_result_deg": rotation,
                "delta_from_joint_result_m": translation,
            }
        )
    payload = {
        "schema": "robot_cam_calib.xarm7_g305_multi_session_hand_eye.v1",
        "status": "candidate_requires_physical_validation",
        "created_at": datetime.now().astimezone().isoformat(),
        "conventions": {
            "transform": "T_A_B maps B-frame points into A",
            "output": "T_link7_wuji_g305_raw_left_optical",
            "equation_per_session": (
                "T_base_link7_i @ T_link7_wuji_g305_raw_left_optical @ "
                "T_wuji_g305_raw_left_optical_charuco_i = T_base_charuco_session"
            ),
        },
        "T_link7_wuji_g305_raw_left_optical": X,
        "solver": {
            "method": (
                "multi_session_nonlinear_se3_soft_l1_robust_mad"
                if args.robust
                else "multi_session_nonlinear_se3_soft_l1"
            ),
            "shared_extrinsic": True,
            "separate_board_pose_per_session": True,
            "rotation_scale_deg": args.rotation_scale_deg,
            "translation_scale_m": translation_scale_m,
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "function_evaluations": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
        },
        "robust_filter": robust_filter,
        "input_sample_count": sum(map(len, input_groups)),
        "sample_count": sum(map(len, groups)),
        "session_count": len(groups),
        "shared_hardware_and_target": shared,
        "session_diagnostics": diagnostics,
        "cross_validation": cross_validation,
        "leave_one_session_out": leave_one_session_out,
        "sources": source_records,
    }
    output = args.output.expanduser().resolve()
    _atomic_yaml_dump(output, payload)
    print(f"[RESULT] method={payload['solver']['method']}")
    print("[RESULT] T_link7_wuji_g305_raw_left_optical:")
    print(X)
    print(
        "[CROSS_VALIDATION] 5fold median={:.3f}deg/{:.2f}mm "
        "p95={:.3f}deg/{:.2f}mm model_max={:.3f}deg/{:.2f}mm".format(
            cross_validation["rotation_stats_deg"]["median"],
            1000.0 * cross_validation["translation_stats_m"]["median"],
            cross_validation["rotation_stats_deg"]["p95"],
            1000.0 * cross_validation["translation_stats_m"]["p95"],
            cross_validation["max_model_delta_from_full_deg"],
            1000.0 * cross_validation["max_model_delta_from_full_m"],
        )
    )
    if args.robust:
        print(
            "[ROBUST] retained={}/{} rejected_by_session={}".format(
                sum(map(len, groups)),
                sum(map(len, input_groups)),
                rejected_by_session,
            )
        )
    print(f"[INFO] Saved {output}")


if __name__ == "__main__":
    main()
