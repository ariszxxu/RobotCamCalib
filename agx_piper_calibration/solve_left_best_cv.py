"""Select and save the best left-wrist model using training CV only."""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from piper_charuco import clean
from robot_cam_calib.geometry import transform_delta
from robot_cam_calib.hand_eye import excitation_diagnostics, solve_hand_eye_robust
from solve_xarm7_g305_multisession import _cross_validate, _fit_robust
from solve_left_combined_filtered import (
    BATCHES,
    H,
    PROFILE_FILE,
    SERIAL,
    SESSION,
    image_filter,
    load_rows,
    observation,
    residual_report,
)

ROTATION_SCALES_DEG = (0.15, 0.25, 0.5, 1.0)
TRANSLATION_SCALES_M = (0.001, 0.002, 0.004, 0.008)
WORK_DISTANCE_M = 0.15
SEED = 20261019


def fit_group(rows, rotation_scale_deg, translation_scale_m):
    initial = solve_hand_eye_robust(
        [observation(row) for row in rows], min_samples=14
    )
    X, boards, result, groups, robust = _fit_robust(
        [rows],
        initial["T_gripper_camera"],
        rotation_scale_deg,
        translation_scale_m,
        min_samples_per_session=14,
    )
    return X, boards[0], result, groups[0], robust, initial


def score_cv(cv):
    """Equivalent p95 plus model instability at a 150 mm work distance."""
    return 1000.0 * (
        cv["translation_stats_m"]["p95"]
        + WORK_DISTANCE_M * math.radians(cv["rotation_stats_deg"]["p95"])
        + cv["max_model_delta_from_full_m"]
        + WORK_DISTANCE_M
        * math.radians(cv["max_model_delta_from_full_deg"])
    )


def main():
    rows, identity, profile = load_rows()
    image_rejected, image_quality = image_filter(rows)
    usable = [row for row in rows if row["index"] not in image_rejected]
    training = [row for row in usable if row["split"] == "train"]
    holdout = [row for row in usable if row["split"] == "holdout"]
    initial = solve_hand_eye_robust(
        [observation(row) for row in training], min_samples=14
    )

    candidates = []
    fits = {}
    for rotation_scale in ROTATION_SCALES_DEG:
        for translation_scale in TRANSLATION_SCALES_M:
            X, boards, result, groups, robust = _fit_robust(
                [training],
                initial["T_gripper_camera"],
                rotation_scale,
                translation_scale,
                min_samples_per_session=14,
            )
            active = groups[0]
            cv = _cross_validate(
                [active], X, rotation_scale, translation_scale, 5, SEED
            )
            summary = {
                "rotation_scale_deg": rotation_scale,
                "translation_scale_mm": translation_scale * 1000.0,
                "score_mm_at_150mm": score_cv(cv),
                "used_training_count": len(active),
                "rejected_training_indices": robust[
                    "rejected_indices_by_session"
                ][0],
                "cv_rotation_p95_deg": cv["rotation_stats_deg"]["p95"],
                "cv_translation_p95_mm": cv["translation_stats_m"]["p95"]
                * 1000.0,
                "max_model_delta_rotation_deg": cv[
                    "max_model_delta_from_full_deg"
                ],
                "max_model_delta_translation_mm": cv[
                    "max_model_delta_from_full_m"
                ]
                * 1000.0,
            }
            key = (rotation_scale, translation_scale)
            fits[key] = (X, boards[0], result, active, robust, cv)
            candidates.append(summary)

    candidates.sort(key=lambda row: row["score_mm_at_150mm"])
    selected = candidates[0]
    selected_key = (
        selected["rotation_scale_deg"],
        selected["translation_scale_mm"] / 1000.0,
    )
    X, board, result, active, robust, cross_validation = fits[selected_key]
    training_report = residual_report(active, X, board)
    holdout_report = residual_report(holdout, X, board)
    holdout_by_batch = {
        batch: residual_report(
            [row for row in holdout if row["batch"] == batch], X, board
        )
        for batch in BATCHES
    }

    batch_solutions = {}
    batch_transforms = []
    for batch in BATCHES:
        batch_training = [row for row in training if row["batch"] == batch]
        batch_X, _batch_board, _batch_result, batch_active, batch_robust, batch_initial = fit_group(
            batch_training, selected_key[0], selected_key[1]
        )
        batch_transforms.append(batch_X)
        batch_solutions[batch] = {
            "T_link6_camera_m": batch_X,
            "input_training_count": len(batch_training),
            "used_training_count": len(batch_active),
            "initial_hand_eye_method": batch_initial["method"],
            "initial_hand_eye_outliers": batch_initial["outlier_indices"],
            "nonlinear_outliers": batch_robust[
                "rejected_indices_by_session"
            ][0],
        }
    batch_rotation, batch_translation = transform_delta(*batch_transforms)
    excitation = excitation_diagnostics([observation(row) for row in active])
    hold_rot = holdout_report["rotation_stats_deg"]
    hold_mm = holdout_report["translation_stats_mm"]
    checks = {
        "predetermined_holdout_at_least_10": len(holdout) >= 10,
        "rotation_excitation_rank3": excitation["relative_rotation_rank"] == 3,
        "holdout_translation_max_le4mm": hold_mm["max"] <= 4.0,
        "holdout_rotation_max_le1_5deg": hold_rot["max"] <= 1.5,
        "batch_repeatability_translation_le5mm": batch_translation <= 0.005,
        "batch_repeatability_rotation_le1_5deg": batch_rotation <= 1.5,
        "cross_validation_model_translation_max_le5mm": (
            cross_validation["max_model_delta_from_full_m"] <= 0.005
        ),
        "cross_validation_model_rotation_max_le1_5deg": (
            cross_validation["max_model_delta_from_full_deg"] <= 1.5
        ),
        "camera_mount_translation_norm_lt300mm": np.linalg.norm(X[:3, 3]) < 0.3,
    }

    session_output = SESSION / ("candidate-left-best-cv-%d.json" % time.time_ns())
    export_dir = H / "exports"
    export_dir.mkdir(exist_ok=True)
    local_output = export_dir / "agx_left_wrist_extri_best.json"
    if local_output.exists():
        raise FileExistsError("Preserve existing local best file: " + str(local_output))
    payload = {
        "schema": "agx.left_wrist_training_cv_selected_hand_eye.v1",
        "name": "agx_left_wrist_extri_best",
        "status": "best_within_frozen_dataset_requires_physical_validation",
        "created_unix": time.time(),
        "arm": "left",
        "camera_serial": SERIAL,
        "frame": "T_left_link6_left_wrist_color_optical",
        "meaning": "maps left-wrist D435 color optical points into left link6",
        "units": {
            "translation_matrix": "metres",
            "reported_translation": "millimetres",
        },
        "T_link6_camera_m": X,
        "translation_mm": X[:3, 3] * 1000.0,
        "scipy_as_euler_xyz_deg": Rotation.from_matrix(X[:3, :3]).as_euler(
            "xyz", degrees=True
        ),
        "T_base_board_m": board,
        "selection_policy": {
            "holdout_use": "15 predetermined holdouts were excluded from model and hyperparameter selection",
            "image_blur": "per-half canonical-board sharpness robust z below -2.5, capped at 10 percent",
            "model_grid": {
                "rotation_scales_deg": ROTATION_SCALES_DEG,
                "translation_scales_mm": [x * 1000.0 for x in TRANSLATION_SCALES_M],
            },
            "selection_metric": "5-fold training CV p95 pose error plus model instability, converted at 150 mm work distance",
            "work_distance_m": WORK_DISTANCE_M,
            "seed": SEED,
        },
        "model_selection": {
            "selected": selected,
            "candidates_ranked": candidates,
        },
        "image_quality": image_quality,
        "image_rejected_indices": sorted(image_rejected),
        "input_training_indices": [row["index"] for row in training],
        "used_training_indices": [row["index"] for row in active],
        "geometric_training_rejected_indices": robust[
            "rejected_indices_by_session"
        ][0],
        "holdout_indices": [row["index"] for row in holdout],
        "initial_hand_eye": {
            "method": initial["method"],
            "inlier_indices": initial["inlier_indices"],
            "outlier_indices": initial["outlier_indices"],
        },
        "solver": {
            "method": "training_cv_selected_shared_board_nonlinear_se3_soft_l1",
            "rotation_scale_deg": selected_key[0],
            "translation_scale_m": selected_key[1],
            "cost": result.cost,
            "optimality": result.optimality,
            "function_evaluations": result.nfev,
            "robust_filter": robust,
        },
        "training_residuals": training_report,
        "holdout_residuals": holdout_report,
        "holdout_by_batch": holdout_by_batch,
        "cross_validation": cross_validation,
        "excitation": excitation,
        "batch_solutions": batch_solutions,
        "batch_repeatability": {
            "rotation_deg": batch_rotation,
            "translation_mm": batch_translation * 1000.0,
        },
        "checks": checks,
        "qualified": all(checks.values()),
        "accepted_for_control": False,
        "control_config_changed": False,
        "session_identity": identity,
        "camera_profile": profile,
        "source_session": str(SESSION),
        "session_candidate_path": str(session_output),
        "local_best_path": str(local_output),
        "robot_motion_commands_sent": 0,
    }
    encoded = json.dumps(clean(payload), indent=2) + "\n"
    with session_output.open("x") as handle:
        handle.write(encoded)
    with local_output.open("x") as handle:
        handle.write(encoded)
    print(
        json.dumps(
            {
                "local_best": str(local_output),
                "session_candidate": str(session_output),
                "T_link6_camera_m": clean(X),
                "translation_mm": clean(payload["translation_mm"]),
                "scipy_as_euler_xyz_deg": clean(
                    payload["scipy_as_euler_xyz_deg"]
                ),
                "image_rejected_indices": sorted(image_rejected),
                "geometric_training_rejected_indices": payload[
                    "geometric_training_rejected_indices"
                ],
                "used_training_count": len(active),
                "holdout_count": len(holdout),
                "holdout_rotation_stats_deg": hold_rot,
                "holdout_translation_stats_mm": hold_mm,
                "batch_repeatability": payload["batch_repeatability"],
                "qualified": payload["qualified"],
                "accepted_for_control": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
