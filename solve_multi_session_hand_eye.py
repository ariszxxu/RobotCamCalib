#!/usr/bin/env python3
"""Configuration-driven multi-session eye-in-hand solve with image QA."""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from robot_cam_calib.geometry import transform_delta
from robot_cam_calib.hand_eye import (
    HandEyeObservation,
    mean_transform,
    solve_hand_eye_robust,
)
from robot_cam_calib.image_quality import (
    PlanarTargetQualityConfig,
    planar_target_sharpness,
    robust_low_sharpness_indices,
)
from solve_xarm7_g305_multisession import (
    _cross_validate,
    _fit,
    _leave_one_session_out,
    _plain,
    _session_diagnostics,
)


def dotted_get(value: Any, path: str, *, default: Any = ...) -> Any:
    current = value
    for part in str(path).split("."):
        if isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        elif default is not ...:
            return default
        else:
            raise KeyError(f"Missing configured field: {path}")
    return current


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or int(config.get("version", 0)) != 1:
        raise ValueError(f"Expected version: 1 configuration: {path}")
    for key in ("input", "output", "solver", "image_quality"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Configuration is missing mapping '{key}'")
    return config


def _as_transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} is not a finite 4x4 transform")
    return transform


def _equal_values(first: Any, second: Any) -> bool:
    try:
        left = np.asarray(first, dtype=np.float64)
        right = np.asarray(second, dtype=np.float64)
        return left.shape == right.shape and np.array_equal(left, right)
    except (TypeError, ValueError):
        return first == second


def validate_documents(
    paths: list[Path], documents: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    input_config = config["input"]
    allowed = set(input_config.get("allowed_schemas", []))
    samples_path = input_config["samples_path"]
    consistency_paths = input_config.get("consistency_paths", [])
    reference_values: dict[str, Any] = {}
    for source_index, (path, document) in enumerate(zip(paths, documents)):
        schema = document.get("schema")
        if allowed and schema not in allowed:
            raise ValueError(f"Unsupported schema in {path}: {schema}")
        samples = dotted_get(document, samples_path)
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"No samples at '{samples_path}' in {path}")
        for field in consistency_paths:
            current = dotted_get(document, field)
            if source_index == 0:
                reference_values[field] = current
            elif not _equal_values(reference_values[field], current):
                raise ValueError(f"Cross-session mismatch at '{field}' in {path}")
    return reference_values


def build_groups(
    documents: list[dict[str, Any]], config: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    input_config = config["input"]
    fields = input_config["sample_fields"]
    groups = []
    for session, document in enumerate(documents):
        samples = dotted_get(document, input_config["samples_path"])
        group = []
        for position, source in enumerate(samples):
            index = dotted_get(source, fields["index"], default=position)
            group.append(
                {
                    "index": int(index),
                    "session": session,
                    "T_base_link7": _as_transform(
                        dotted_get(source, fields["base_gripper_transform"]),
                        f"session {session} sample {index} base/gripper",
                    ),
                    "T_camera_charuco": _as_transform(
                        dotted_get(source, fields["camera_target_transform"]),
                        f"session {session} sample {index} camera/target",
                    ),
                    "image_path": str(dotted_get(source, fields["image_path"])),
                    "reprojection_error_px": float(
                        dotted_get(
                            source,
                            fields["reprojection_error_px"],
                            default=float("nan"),
                        )
                    ),
                }
            )
        groups.append(group)
    return groups


def initial_extrinsic(
    groups: list[list[dict[str, Any]]],
    documents: list[dict[str, Any]],
    config: dict[str, Any],
) -> np.ndarray:
    source_field = config["input"].get("source_extrinsic_path")
    candidates = []
    if source_field:
        for document in documents:
            value = dotted_get(document, source_field, default=None)
            if value is not None:
                candidates.append(_as_transform(value, source_field))
    if not candidates:
        for session, group in enumerate(groups):
            observations = [
                HandEyeObservation(
                    index=sample["index"],
                    T_base_gripper=sample["T_base_link7"],
                    T_camera_target=sample["T_camera_charuco"],
                )
                for sample in group
            ]
            solution = solve_hand_eye_robust(
                observations,
                min_samples=int(config["solver"].get("min_samples_per_session", 10)),
            )
            candidates.append(solution["T_gripper_camera"])
            print(
                f"[INIT] session={session} method={solution['method']} "
                f"inliers={len(solution['inlier_indices'])}/{len(group)}"
            )
    return mean_transform(candidates)


def filter_image_quality(
    groups: list[list[dict[str, Any]]],
    documents: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    quality = config["image_quality"]
    if not bool(quality.get("enabled", True)):
        return groups, {"enabled": False, "reason": "disabled_by_configuration"}
    target = PlanarTargetQualityConfig(
        width_m=float(quality["target_size_m"][0]),
        height_m=float(quality["target_size_m"][1]),
        canonical_width_px=int(quality.get("canonical_size_px", [700, 500])[0]),
        canonical_height_px=int(quality.get("canonical_size_px", [700, 500])[1]),
        robust_z_limit=float(quality.get("robust_z_limit", -2.5)),
        max_reject_fraction=float(quality.get("max_reject_fraction", 0.10)),
    )
    K_path = quality["intrinsics_K_path"]
    dist_path = quality["distortion_path"]
    camera_model_path = quality.get("camera_model_path")
    camera_model_default = quality.get("camera_model", "pinhole")
    max_reprojection = float(quality.get("max_reprojection_error_px", 1.0))
    fail_on_image_error = bool(quality.get("fail_on_image_error", True))
    minimum = int(config["solver"].get("min_samples_per_session", 10))
    filtered = []
    reports = []
    for session, (group, document) in enumerate(zip(groups, documents)):
        K = np.asarray(dotted_get(document, K_path), dtype=np.float64).reshape(3, 3)
        dist = np.asarray(dotted_get(document, dist_path), dtype=np.float64).reshape(-1)
        camera_model = (
            str(dotted_get(document, camera_model_path, default=camera_model_default))
            if camera_model_path
            else str(camera_model_default)
        )
        records = []
        scored = []
        hard_rejected = set()
        for sample in group:
            key = sample["index"]
            record = {
                "index": key,
                "image_path": sample["image_path"],
                "reprojection_error_px": sample["reprojection_error_px"],
            }
            reason = None
            if (
                not np.isfinite(sample["reprojection_error_px"])
                or sample["reprojection_error_px"] > max_reprojection
            ):
                reason = "reprojection_error"
            image = None if reason else cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
            if reason is None and image is None:
                reason = "image_read_error"
            if reason is None:
                try:
                    score = planar_target_sharpness(
                        image,
                        sample["T_camera_charuco"],
                        K,
                        dist,
                        target,
                        camera_model=camera_model,
                    )
                    sample["image_sharpness"] = score
                    record["image_sharpness"] = score
                    scored.append((key, score))
                except (ValueError, cv2.error) as exc:
                    reason = f"sharpness_error:{type(exc).__name__}"
            if reason is not None:
                record["status"] = "rejected"
                record["reason"] = reason
                if fail_on_image_error or reason == "reprojection_error":
                    hard_rejected.add(key)
                else:
                    record["status"] = "usable_unscored"
            else:
                record["status"] = "scored"
            records.append(record)
        blur_rejected, robust_report = robust_low_sharpness_indices(
            scored,
            robust_z_limit=target.robust_z_limit,
            max_reject_fraction=target.max_reject_fraction,
        )
        rejected = hard_rejected | blur_rejected
        for record in records:
            if record["index"] in blur_rejected:
                record["status"] = "rejected"
                record["reason"] = "low_sharpness_outlier"
            elif record["status"] == "scored":
                record["status"] = "usable"
        usable = [sample for sample in group if sample["index"] not in rejected]
        if len(usable) < minimum:
            raise ValueError(
                f"Session {session} has only {len(usable)} usable samples after "
                f"image QA; need {minimum}"
            )
        filtered.append(usable)
        reports.append(
            {
                "session": session,
                "input_count": len(group),
                "usable_count": len(usable),
                "rejected_count": len(rejected),
                "robust_sharpness": robust_report,
                "samples": records,
            }
        )
        print(
            f"[IMAGE_QA] session={session} usable={len(usable)}/{len(group)} "
            f"rejected={sorted(rejected)}"
        )
    return filtered, {
        "enabled": True,
        "method": "canonical_planar_ROI_Tenengrad_plus_Laplacian",
        "target_size_m": [target.width_m, target.height_m],
        "canonical_size_px": [target.canonical_width_px, target.canonical_height_px],
        "max_reprojection_error_px": max_reprojection,
        "sessions": reports,
    }


def _atomic_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(_plain(payload), sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = [path.expanduser().resolve() for path in args.inputs]
    documents = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    config = load_config(args.config.expanduser().resolve())
    consistency = validate_documents(paths, documents, config)
    raw_groups = build_groups(documents, config)
    initial_X = initial_extrinsic(raw_groups, documents, config)
    groups, image_quality = filter_image_quality(raw_groups, documents, config)
    solver = config["solver"]
    rotation_scale_deg = float(solver.get("rotation_scale_deg", 0.25))
    translation_scale_m = float(solver.get("translation_scale_mm", 2.0)) / 1000.0
    X, boards, result = _fit(
        groups, initial_X, rotation_scale_deg, translation_scale_m
    )
    folds = int(solver.get("cross_validation_folds", 5))
    seed = int(solver.get("random_seed", 20260827))
    cross_validation = _cross_validate(
        groups, X, rotation_scale_deg, translation_scale_m, folds, seed
    )
    leave_one_out = _leave_one_session_out(
        groups, X, rotation_scale_deg, translation_scale_m
    )
    source_records = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema": document.get("schema"),
            "input_sample_count": len(group),
        }
        for path, document, group in zip(paths, documents, raw_groups)
    ]
    output_config = config["output"]
    session_diagnostics = _session_diagnostics(groups, X, boards)
    target_transform_key = output_config.get(
        "session_target_transform_key", "T_base_target"
    )
    for diagnostic in session_diagnostics:
        diagnostic[target_transform_key] = diagnostic.pop("T_base_charuco")
    payload = {
        "schema": output_config["schema"],
        "status": "candidate_requires_physical_validation",
        "created_at": datetime.now().astimezone().isoformat(),
        "conventions": output_config.get("conventions", {}),
        output_config["transform_key"]: X,
        "solver": {
            "method": "multi_session_nonlinear_se3_soft_l1",
            "shared_extrinsic": True,
            "separate_target_pose_per_session": True,
            "rotation_scale_deg": rotation_scale_deg,
            "translation_scale_m": translation_scale_m,
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "function_evaluations": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
        },
        "input_sample_count": sum(map(len, raw_groups)),
        "usable_sample_count": sum(map(len, groups)),
        "session_count": len(groups),
        "consistency_checks": consistency,
        "image_quality": image_quality,
        "session_diagnostics": session_diagnostics,
        "cross_validation": cross_validation,
        "leave_one_session_out": leave_one_out,
        "sources": source_records,
        "adapter_config": str(args.config.expanduser().resolve()),
    }
    output = args.output.expanduser().resolve()
    _atomic_dump(output, payload)
    print("[RESULT] T_shared_mount_camera:")
    print(X)
    print(
        "[CROSS_VALIDATION] median={:.3f}deg/{:.2f}mm "
        "p95={:.3f}deg/{:.2f}mm model_max={:.3f}deg/{:.2f}mm".format(
            cross_validation["rotation_stats_deg"]["median"],
            1000.0 * cross_validation["translation_stats_m"]["median"],
            cross_validation["rotation_stats_deg"]["p95"],
            1000.0 * cross_validation["translation_stats_m"]["p95"],
            cross_validation["max_model_delta_from_full_deg"],
            1000.0 * cross_validation["max_model_delta_from_full_m"],
        )
    )
    print(f"[INFO] Saved {output}")


if __name__ == "__main__":
    main()
