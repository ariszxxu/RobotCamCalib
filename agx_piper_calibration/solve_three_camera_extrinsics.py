"""Solve a local three-camera extrinsic set without changing control config."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


H = Path(__file__).resolve().parent
CORE_ROOT = H.parent if (H.parent / "robot_cam_calib").is_dir() else H / "RobotCamCalib"
sys.path.insert(0, str(CORE_ROOT))

from piper_charuco import clean  # noqa: E402
from robot_cam_calib.geometry import (  # noqa: E402
    inv_T,
    params_to_transform,
    residual_stats,
    robust_limit,
    transform_delta,
    transform_to_params,
)
from robot_cam_calib.hand_eye import (  # noqa: E402
    HandEyeObservation,
    excitation_diagnostics,
    mean_transform,
)
from robot_cam_calib.image_quality import (  # noqa: E402
    PlanarTargetQualityConfig,
    planar_target_sharpness,
    robust_low_sharpness_indices,
)
from solve_xarm7_g305_multisession import (  # noqa: E402
    _cross_validate,
    _fit,
    _fit_robust,
    _pose_residual,
)


OLD_RIGHT = H / "sessions/manual-20261019-right-v1"
NEW_FRONT = H / "sessions/manual-20261019-front-from-right-relaxed-v1"
RIGHT_REFERENCE = H / "exports/agx_right_wrist_extri/agx_right_wrist_extri.json"
LEFT_REFERENCE = H / "exports/agx_left_wrist_extri_best.json"
EXPORTS = H / "exports"
TARGET = PlanarTargetQualityConfig(0.28, 0.20)
ROTATION_SCALES_DEG = (0.25, 0.5, 1.0)
TRANSLATION_SCALES_M = (0.002, 0.004, 0.008)
SEED = 20261019
WORK_DISTANCE_M = 0.15


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(slot: Path, sample: dict) -> None:
    for relative, expected in sample["source_files_sha256"].items():
        source = (slot / relative).resolve()
        if not source.is_relative_to(slot.resolve()) or sha(source) != expected:
            raise ValueError("Captured evidence changed: " + str(source))


def sharpness(slot: Path, role: str, view: dict) -> float:
    raw = slot / "raw" / role
    image = cv2.imread(str(raw / "rgb.jpg"), cv2.IMREAD_COLOR)
    metadata = read(raw / "metadata.json")["color"]
    if image is None:
        raise ValueError("Unreadable image: " + str(raw / "rgb.jpg"))
    return planar_target_sharpness(
        image,
        np.asarray(view["T_camera_board_m"], dtype=float),
        np.asarray(metadata["K"], dtype=float).reshape(3, 3),
        np.asarray(metadata["D"], dtype=float),
        TARGET,
        camera_model="pinhole",
    )


def quality(view: dict) -> tuple[int, float, bool]:
    return (
        int(view["corner_count"]),
        float(view["normalized_ray_reprojection_rmse_pixel_equivalent"]),
        bool(view["metric_pose_eligible"]),
    )


def load_old_right() -> tuple[list[dict], dict]:
    identity = read(OLD_RIGHT / "identity.json")
    rows = []
    for index in range(7, 69):
        slot = OLD_RIGHT / ("right-%04d" % index)
        sample = read(slot / "sample.json")
        expected_split = "holdout" if index % 4 == 3 else "train"
        if (
            sample["identity"] != identity
            or sample["active_arm"] != "right"
            or sample["index"] != index
            or sample["split"] != expected_split
        ):
            raise ValueError("Old right sample identity/split changed: " + str(slot))
        verify_sources(slot, sample)
        view = sample["cameras"]["right_wrist"]
        corners, rmse, eligible = quality(view)
        rows.append(
            {
                "index": index,
                "key": "old-right-%04d" % index,
                "session": "old_right",
                "blur_batch": "old_first" if index < 38 else "old_second",
                "split": expected_split,
                "T_base_link7": np.asarray(
                    sample["T_base_link6_m"]["right"], dtype=float
                ),
                "T_camera_charuco": np.asarray(view["T_camera_board_m"], dtype=float),
                "right_sharpness": sharpness(slot, "right_wrist", view),
                "right_corners": corners,
                "right_rmse_px": rmse,
                "right_eligible": eligible,
                "sample_sha256": sha(slot / "sample.json"),
            }
        )
    return rows, identity


def load_new_front() -> tuple[list[dict], dict]:
    identity = read(NEW_FRONT / "identity.json")
    reference_sha = sha(RIGHT_REFERENCE)
    if identity["right_reference_sha256"] != reference_sha:
        raise ValueError("Front session right-reference hash changed")
    rows = []
    paths = sorted(NEW_FRONT.glob("right-*/sample.json"))
    if len(paths) != 30:
        raise ValueError("Expected 30 new dual-camera samples, got %d" % len(paths))
    for path in paths:
        slot = path.parent
        sample = read(path)
        index = int(sample["index"])
        expected_split = "holdout" if index % 4 == 3 else "train"
        if (
            sample["identity"] != identity
            or sample["active_arm"] != "right"
            or sample["index"] != index
            or sample["split"] != expected_split
            or sample["target_transform"] != "T_base_front_color_optical"
            or not sample["board_fixed_during_collection"]
            or not sample["front_camera_fixed_during_collection"]
        ):
            raise ValueError("New front sample identity/split changed: " + str(slot))
        verify_sources(slot, sample)
        right = sample["cameras"]["right_wrist"]
        front = sample["cameras"]["front"]
        rc, rr, re = quality(right)
        fc, fr, fe = quality(front)
        rows.append(
            {
                "index": index,
                "key": "new-front-%04d" % index,
                "session": "new_front",
                "blur_batch": "new_right",
                "split": expected_split,
                "T_base_link7": np.asarray(
                    sample["T_base_link6_m"]["right"], dtype=float
                ),
                "T_camera_charuco": np.asarray(right["T_camera_board_m"], dtype=float),
                "T_front_board": np.asarray(front["T_camera_board_m"], dtype=float),
                "right_sharpness": sharpness(slot, "right_wrist", right),
                "front_sharpness": sharpness(slot, "front", front),
                "right_corners": rc,
                "right_rmse_px": rr,
                "right_eligible": re,
                "front_corners": fc,
                "front_rmse_px": fr,
                "front_eligible": fe,
                "camera_stamp_span_s": float(sample["camera_stamp_span_s"]),
                "sample_sha256": sha(path),
            }
        )
    if [row["index"] for row in rows] != list(range(30)):
        raise ValueError("New front indices are not contiguous 0..29")
    return rows, identity


def image_filter(rows: list[dict], role: str, batch_key: str) -> tuple[set[str], dict]:
    rejected: set[str] = set()
    report = {}
    batches = sorted({row[batch_key] for row in rows})
    for batch in batches:
        batch_rows = [row for row in rows if row[batch_key] == batch]
        score_key = role + "_sharpness"
        low, robust = robust_low_sharpness_indices(
            [(row["key"], row[score_key]) for row in batch_rows],
            robust_z_limit=-2.5,
            max_reject_fraction=0.10,
        )
        hard = {
            row["key"]
            for row in batch_rows
            if not row[role + "_eligible"]
            or row[role + "_corners"] < 8
            or row[role + "_rmse_px"] > 1.0
        }
        rejected |= low | hard
        report[batch] = {
            "robust_sharpness": robust,
            "low_sharpness_keys": sorted(low),
            "hard_quality_rejected_keys": sorted(hard),
            "scores": [
                {
                    "key": row["key"],
                    "sharpness": row[score_key],
                    "corner_count": row[role + "_corners"],
                    "reprojection_rmse_px": row[role + "_rmse_px"],
                }
                for row in batch_rows
            ],
        }
    return rejected, report


def stat(values: list[float], scale: float = 1.0) -> dict:
    return {key: value * scale for key, value in residual_stats(values).items()}


def residual_report(rows: list[dict], X: np.ndarray, board: np.ndarray) -> dict:
    residuals = [_pose_residual(row, X, board) for row in rows]
    return {
        "count": len(rows),
        "rotation_stats_deg": stat([x[0] for x in residuals]),
        "translation_stats_mm": stat([x[1] for x in residuals], 1000.0),
        "samples": [
            {
                "key": row["key"],
                "rotation_deg": value[0],
                "translation_mm": 1000.0 * value[1],
            }
            for row, value in zip(rows, residuals)
        ],
    }


def cv_score(cv: dict) -> float:
    return 1000.0 * (
        cv["translation_stats_m"]["p95"]
        + WORK_DISTANCE_M * math.radians(cv["rotation_stats_deg"]["p95"])
        + cv["max_model_delta_from_full_m"]
        + WORK_DISTANCE_M * math.radians(cv["max_model_delta_from_full_deg"])
    )


def fit_pose_mean(rows: list[dict], key: str) -> tuple[np.ndarray, list[dict], dict]:
    active = list(rows)
    rejected = []
    iterations = []
    pose = mean_transform([row[key] for row in active])
    for iteration in range(5):
        start = transform_to_params(pose)

        def objective(params):
            center = params_to_transform(params)
            output = []
            for row in active:
                delta = inv_T(center) @ row[key]
                output.extend(
                    Rotation.from_matrix(delta[:3, :3]).as_rotvec()
                    / np.deg2rad(0.25)
                )
                output.extend(delta[:3, 3] / 0.001)
            return np.asarray(output)

        result = least_squares(
            objective,
            start,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=2000,
            xtol=2e-12,
            ftol=2e-12,
            gtol=2e-12,
        )
        pose = params_to_transform(result.x)
        values = [transform_delta(pose, row[key]) for row in active]
        rlim = robust_limit([x[0] for x in values], 0.15, 2.0)
        tlim = robust_limit([x[1] for x in values], 0.0005, 0.010)
        bad = {
            row["key"]
            for row, value in zip(active, values)
            if value[0] > rlim or value[1] > tlim
        }
        iterations.append(
            {
                "iteration": iteration,
                "active_keys": [row["key"] for row in active],
                "rotation_limit_deg": rlim,
                "translation_limit_m": tlim,
                "rejected_keys": sorted(bad),
            }
        )
        if not bad or len(active) - len(bad) < 10:
            break
        rejected.extend(sorted(bad))
        active = [row for row in active if row["key"] not in bad]
    return pose, active, {"rejected_keys": sorted(set(rejected)), "iterations": iterations}


def direct_front_report(
    rows: list[dict], X: np.ndarray, T_base_front: np.ndarray
) -> dict:
    values = []
    for row in rows:
        predicted_front_board = (
            inv_T(T_base_front)
            @ row["T_base_link7"]
            @ X
            @ row["T_camera_charuco"]
        )
        values.append(transform_delta(row["T_front_board"], predicted_front_board))
    return {
        "count": len(rows),
        "rotation_stats_deg": stat([x[0] for x in values]),
        "translation_stats_mm": stat([x[1] for x in values], 1000.0),
        "samples": [
            {
                "key": row["key"],
                "rotation_deg": value[0],
                "translation_mm": 1000.0 * value[1],
            }
            for row, value in zip(rows, values)
        ],
    }


def front_cross_validation(
    old_rows: list[dict],
    new_rows: list[dict],
    full_X: np.ndarray,
    full_front: np.ndarray,
    rotation_scale: float,
    translation_scale: float,
) -> dict:
    rng = np.random.default_rng(SEED)
    old_order = np.arange(len(old_rows))
    new_order = np.arange(len(new_rows))
    rng.shuffle(old_order)
    rng.shuffle(new_order)
    old_folds = np.array_split(old_order, 5)
    new_folds = np.array_split(new_order, 5)
    fold_results = []
    all_values = []
    for fold in range(5):
        old_val = set(old_folds[fold].tolist())
        new_val = set(new_folds[fold].tolist())
        old_train = [row for i, row in enumerate(old_rows) if i not in old_val]
        new_train = [row for i, row in enumerate(new_rows) if i not in new_val]
        new_validation = [row for i, row in enumerate(new_rows) if i in new_val]
        fold_X, boards, _ = _fit(
            [old_train, new_train], full_X, rotation_scale, translation_scale
        )
        # Estimate the requested transform directly from each synchronized pair.
        # This preserves the correlation between the two camera PnP estimates and
        # the arm pose instead of separately averaging board poses.
        fold_front = mean_transform(
            [
                row["T_base_link7"]
                @ fold_X
                @ row["T_camera_charuco"]
                @ inv_T(row["T_front_board"])
                for row in new_train
            ]
        )
        report = direct_front_report(new_validation, fold_X, fold_front)
        all_values.extend(
            (sample["rotation_deg"], sample["translation_mm"] / 1000.0)
            for sample in report["samples"]
        )
        model_r, model_t = transform_delta(full_front, fold_front)
        fold_results.append(
            {
                "fold": fold,
                "validation": report,
                "model_delta_from_full_deg": model_r,
                "model_delta_from_full_mm": model_t * 1000.0,
            }
        )
    return {
        "folds": 5,
        "seed": SEED,
        "rotation_stats_deg": stat([x[0] for x in all_values]),
        "translation_stats_mm": stat([x[1] for x in all_values], 1000.0),
        "max_model_delta_from_full_deg": max(
            row["model_delta_from_full_deg"] for row in fold_results
        ),
        "max_model_delta_from_full_mm": max(
            row["model_delta_from_full_mm"] for row in fold_results
        ),
        "fold_results": fold_results,
    }


def main() -> None:
    EXPORTS.mkdir(exist_ok=True)
    generated_schemas = {
        "agx.right_wrist_multisession_training_cv.v1",
        "agx.front_from_moving_right_wrist_training_holdout.v1",
        "agx.three_camera_extrinsic_bundle.v1",
    }
    for path in (
        EXPORTS / "agx_right_wrist_extri_combined_best.json",
        EXPORTS / "agx_front_camera_extri_best.json",
        EXPORTS / "agx_three_camera_extrinsics_20261019.json",
    ):
        if path.exists() and read(path).get("schema") not in generated_schemas:
            raise FileExistsError("Refuse to replace unrelated result: " + str(path))

    old_rows, old_identity = load_old_right()
    new_rows, new_identity = load_new_front()
    all_right = old_rows + new_rows
    right_rejected, right_quality = image_filter(all_right, "right", "blur_batch")
    front_rejected, front_quality = image_filter(new_rows, "front", "session")
    old_train = [
        row for row in old_rows if row["split"] == "train" and row["key"] not in right_rejected
    ]
    new_train = [
        row for row in new_rows if row["split"] == "train" and row["key"] not in right_rejected
    ]
    old_holdout = [
        row for row in old_rows if row["split"] == "holdout" and row["key"] not in right_rejected
    ]
    new_holdout = [
        row for row in new_rows if row["split"] == "holdout" and row["key"] not in right_rejected
    ]
    initial_X = np.asarray(read(RIGHT_REFERENCE)["T_link6_camera_m"], dtype=float)

    ranked = []
    fits = {}
    for rotation_scale in ROTATION_SCALES_DEG:
        for translation_scale in TRANSLATION_SCALES_M:
            X, boards, result, active, robust = _fit_robust(
                [old_train, new_train],
                initial_X,
                rotation_scale,
                translation_scale,
                min_samples_per_session=14,
            )
            cv = _cross_validate(
                active, X, rotation_scale, translation_scale, 5, SEED
            )
            summary = {
                "rotation_scale_deg": rotation_scale,
                "translation_scale_mm": translation_scale * 1000.0,
                "score_mm_at_150mm": cv_score(cv),
                "used_counts": [len(group) for group in active],
                "rejected_indices_by_session": robust["rejected_indices_by_session"],
                "cv_rotation_p95_deg": cv["rotation_stats_deg"]["p95"],
                "cv_translation_p95_mm": cv["translation_stats_m"]["p95"] * 1000.0,
                "max_model_delta_rotation_deg": cv["max_model_delta_from_full_deg"],
                "max_model_delta_translation_mm": cv["max_model_delta_from_full_m"] * 1000.0,
            }
            ranked.append(summary)
            fits[(rotation_scale, translation_scale)] = (
                X,
                boards,
                result,
                active,
                robust,
                cv,
            )
    ranked.sort(key=lambda row: row["score_mm_at_150mm"])
    selected = ranked[0]
    selected_key = (
        selected["rotation_scale_deg"],
        selected["translation_scale_mm"] / 1000.0,
    )
    X, boards, result, active, robust, right_cv = fits[selected_key]
    old_active, new_active = active

    right_training = {
        "old_right": residual_report(old_active, X, boards[0]),
        "new_front_session": residual_report(new_active, X, boards[1]),
    }
    right_holdout = {
        "old_right": residual_report(old_holdout, X, boards[0]),
        "new_front_session": residual_report(new_holdout, X, boards[1]),
    }
    old_X_delta = transform_delta(initial_X, X)
    excitation = excitation_diagnostics(
        [
            HandEyeObservation(i, row["T_base_link7"], row["T_camera_charuco"])
            for i, row in enumerate(new_active)
        ]
    )

    new_active_keys = {row["key"] for row in new_active}
    front_train = [
        row
        for row in new_rows
        if row["split"] == "train"
        and row["key"] not in front_rejected
        and row["key"] in new_active_keys
    ]
    front_holdout = [
        row for row in new_rows if row["split"] == "holdout" and row["key"] not in front_rejected
    ]
    for row in front_train:
        row["T_base_front_candidate"] = (
            row["T_base_link7"]
            @ X
            @ row["T_camera_charuco"]
            @ inv_T(row["T_front_board"])
        )
    T_base_front, front_active, front_robust = fit_pose_mean(
        front_train, "T_base_front_candidate"
    )
    mean_front_board = mean_transform(
        [row["T_front_board"] for row in front_active]
    )
    front_training_report = direct_front_report(front_active, X, T_base_front)
    front_holdout_report = direct_front_report(front_holdout, X, T_base_front)
    front_cv = front_cross_validation(
        old_active,
        [row for row in new_active if row["key"] not in front_rejected],
        X,
        T_base_front,
        selected_key[0],
        selected_key[1],
    )

    now = time.time()
    right_output = EXPORTS / "agx_right_wrist_extri_combined_best.json"
    front_output = EXPORTS / "agx_front_camera_extri_best.json"
    bundle_output = EXPORTS / "agx_three_camera_extrinsics_20261019.json"
    left = read(LEFT_REFERENCE)
    right_checks = {
        "predetermined_holdout_count_at_least_20": len(old_holdout) + len(new_holdout) >= 20,
        "new_session_rotation_excitation_rank3": excitation["relative_rotation_rank"] == 3,
        "holdout_translation_max_le15mm": max(
            right_holdout["old_right"]["translation_stats_mm"]["max"],
            right_holdout["new_front_session"]["translation_stats_mm"]["max"],
        ) <= 15.0,
        "holdout_rotation_max_le3deg": max(
            right_holdout["old_right"]["rotation_stats_deg"]["max"],
            right_holdout["new_front_session"]["rotation_stats_deg"]["max"],
        ) <= 3.0,
        "camera_mount_translation_norm_lt300mm": np.linalg.norm(X[:3, 3]) < 0.3,
    }
    right_payload = {
        "schema": "agx.right_wrist_multisession_training_cv.v1",
        "name": "agx_right_wrist_extri_combined_best",
        "status": "new_candidate_requires_physical_validation",
        "created_unix": now,
        "frame": "T_right_link6_right_wrist_color_optical",
        "meaning": "maps right-wrist D435 color optical points into right link6",
        "units": {"translation_matrix": "metres", "reported_translation": "millimetres"},
        "T_link6_camera_m": X,
        "translation_mm": X[:3, 3] * 1000.0,
        "scipy_as_euler_xyz_deg": Rotation.from_matrix(X[:3, :3]).as_euler("xyz", degrees=True),
        "T_base_board_by_session_m": {"old_right": boards[0], "new_front_session": boards[1]},
        "selection": {"selected": selected, "candidates_ranked": ranked, "seed": SEED},
        "image_quality": right_quality,
        "image_rejected_keys": sorted(right_rejected),
        "geometric_rejected_indices_by_session": robust["rejected_indices_by_session"],
        "training_residuals": right_training,
        "holdout_residuals": right_holdout,
        "cross_validation": right_cv,
        "excitation_new_session": excitation,
        "delta_from_previous_accepted": {
            "reference_path": str(RIGHT_REFERENCE),
            "reference_sha256": sha(RIGHT_REFERENCE),
            "rotation_deg": old_X_delta[0],
            "translation_mm": old_X_delta[1] * 1000.0,
        },
        "checks": right_checks,
        "qualified": all(right_checks.values()),
        "accepted_for_control": False,
        "control_config_changed": False,
        "source_sessions": [str(OLD_RIGHT), str(NEW_FRONT)],
        "session_identity": {"old_right": old_identity, "new_front": new_identity},
        "robot_motion_commands_sent_by_solver": 0,
    }
    front_checks = {
        "predetermined_holdout_count_at_least_7": len(front_holdout) >= 7,
        "holdout_translation_max_le15mm": front_holdout_report["translation_stats_mm"]["max"] <= 15.0,
        "holdout_rotation_max_le3deg": front_holdout_report["rotation_stats_deg"]["max"] <= 3.0,
        "front_camera_translation_norm_lt2m": np.linalg.norm(T_base_front[:3, 3]) < 2.0,
        "all_camera_stamp_spans_le80ms": max(row["camera_stamp_span_s"] for row in new_rows) <= 0.08,
    }
    front_payload = {
        "schema": "agx.front_from_moving_right_wrist_training_holdout.v1",
        "name": "agx_front_camera_extri_best",
        "status": "new_candidate_requires_physical_validation",
        "created_unix": now,
        "frame": "T_base_front_color_optical",
        "meaning": "maps front D435 color optical points into robot base",
        "units": {"translation_matrix": "metres", "reported_translation": "millimetres"},
        "T_base_camera_m": T_base_front,
        "translation_mm": T_base_front[:3, 3] * 1000.0,
        "scipy_as_euler_xyz_deg": Rotation.from_matrix(T_base_front[:3, :3]).as_euler("xyz", degrees=True),
        "T_base_board_m": boards[1],
        "T_front_camera_board_mean_m": mean_front_board,
        "right_wrist_reference": {"path": str(right_output), "T_link6_camera_m": X},
        "image_quality": front_quality,
        "image_rejected_keys": sorted(front_rejected),
        "front_pose_geometric_filter": front_robust,
        "training_end_to_end_residuals": front_training_report,
        "holdout_end_to_end_residuals": front_holdout_report,
        "cross_validation": front_cv,
        "checks": front_checks,
        "qualified": all(front_checks.values()) and all(right_checks.values()),
        "accepted_for_control": False,
        "control_config_changed": False,
        "source_session": str(NEW_FRONT),
        "session_identity": new_identity,
        "old_observations_used": False,
        "simulator_information_used": False,
        "robot_motion_commands_sent_by_solver": 0,
    }
    right_output.write_text(json.dumps(clean(right_payload), indent=2) + "\n")
    front_output.write_text(json.dumps(clean(front_payload), indent=2) + "\n")
    bundle = {
        "schema": "agx.three_camera_extrinsic_bundle.v1",
        "name": "agx_three_camera_extrinsics_20261019",
        "created_unix": now,
        "units": "metres",
        "control_config_changed": False,
        "accepted_for_control": False,
        "cameras": {
            "front": {
                "serial": "239622301704",
                "frame": front_payload["frame"],
                "T_base_camera_m": T_base_front,
                "source": str(front_output),
            },
            "left_wrist": {
                "serial": left["camera_serial"],
                "frame": left["frame"],
                "T_link6_camera_m": left["T_link6_camera_m"],
                "source": str(LEFT_REFERENCE),
                "qualified": left["qualified"],
            },
            "right_wrist": {
                "serial": "346522074314",
                "frame": right_payload["frame"],
                "T_link6_camera_m": X,
                "source": str(right_output),
                "qualified": right_payload["qualified"],
            },
        },
        "source_sha256": {
            "front": sha(front_output),
            "left_wrist": sha(LEFT_REFERENCE),
            "right_wrist": sha(right_output),
        },
        "limitations": [
            "All three results remain local candidates; no control configuration was changed.",
            "The front transform inherits right-arm kinematics and right-wrist hand-eye uncertainty.",
            "The left-wrist source did not pass every original qualification threshold.",
        ],
    }
    bundle_output.write_text(json.dumps(clean(bundle), indent=2) + "\n")
    print(
        json.dumps(
            clean(
                {
                    "right": {
                        "path": str(right_output),
                        "T": X,
                        "qualified": right_payload["qualified"],
                        "holdout": right_holdout,
                        "delta_from_previous": right_payload["delta_from_previous_accepted"],
                    },
                    "front": {
                        "path": str(front_output),
                        "T": T_base_front,
                        "qualified": front_payload["qualified"],
                        "holdout": front_holdout_report,
                    },
                    "left": {
                        "path": str(LEFT_REFERENCE),
                        "T": left["T_link6_camera_m"],
                        "qualified": left["qualified"],
                    },
                    "bundle": str(bundle_output),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
