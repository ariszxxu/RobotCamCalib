"""Combine the two buffered right-wrist captures without publishing a result."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from piper_charuco import clean, read, sha
from robot_cam_calib.geometry import residual_stats, transform_delta
from robot_cam_calib.hand_eye import (
    HandEyeObservation,
    excitation_diagnostics,
    solve_hand_eye_robust,
)
from robot_cam_calib.image_quality import (
    PlanarTargetQualityConfig,
    planar_target_sharpness,
    robust_low_sharpness_indices,
)
from solve_xarm7_g305_multisession import (
    _cross_validate,
    _fit_robust,
    _pose_residual,
)

H = Path(__file__).resolve().parent
SESSION = H / "sessions/manual-20261019-right-v1"
PROFILE_FILE = Path(
    "/home/agilex/yichen/rsi_sim2real/releases/"
    "agx-stack-cube-20261018/calibration-v2/factory-current-color-intrinsics.json"
)
SERIAL = "346522074314"
BATCHES = {"first": range(7, 38), "second": range(38, 69)}


def stats(values, scale=1.0):
    return {key: value * scale for key, value in residual_stats(values).items()}


def load_rows():
    identity = read(SESSION / "identity.json")
    profiles = [
        item for item in read(PROFILE_FILE)["profiles"] if item["serial"] == SERIAL
    ]
    if len(profiles) != 1:
        raise ValueError("Expected exactly one current right-wrist camera profile")
    target = PlanarTargetQualityConfig(0.28, 0.20)
    rows = []
    for batch, indices in BATCHES.items():
        for index in indices:
            slot = SESSION / ("right-%04d" % index)
            sample_file = slot / "sample.json"
            sample = read(sample_file)
            if sample["identity"] != identity or sample["active_arm"] != "right":
                raise ValueError("Sample identity or arm changed: " + str(sample_file))
            expected_split = "holdout" if index % 4 == 3 else "train"
            if sample["index"] != index or sample["split"] != expected_split:
                raise ValueError("Predetermined split changed: " + str(sample_file))
            for relative, digest in sample["source_files_sha256"].items():
                source = (slot / relative).resolve()
                if not source.is_relative_to(slot.resolve()) or sha(source) != digest:
                    raise ValueError("Captured evidence changed: " + str(source))
            view = sample["cameras"]["right_wrist"]
            raw = slot / "raw/right_wrist"
            image = cv2.imread(str(raw / "rgb.jpg"), cv2.IMREAD_COLOR)
            metadata = read(raw / "metadata.json")["color"]
            if image is None:
                raise ValueError("Image cannot be read: " + str(raw / "rgb.jpg"))
            score = planar_target_sharpness(
                image,
                np.asarray(view["T_camera_board_m"], dtype=float),
                np.asarray(metadata["K"], dtype=float).reshape(3, 3),
                np.asarray(metadata["D"], dtype=float),
                target,
                camera_model="pinhole",
            )
            row = {
                "index": index,
                "batch": batch,
                "split": expected_split,
                "T_base_link7": np.asarray(sample["T_base_link6_m"]["right"], dtype=float),
                "T_camera_charuco": np.asarray(view["T_camera_board_m"], dtype=float),
                "sharpness": score,
                "corner_count": int(view["corner_count"]),
                "reprojection_rmse_px": float(
                    view["normalized_ray_reprojection_rmse_pixel_equivalent"]
                ),
                "metric_pose_eligible": bool(view["metric_pose_eligible"]),
                "capture_sync": sample.get("capture_sync"),
                "sample_sha256": hashlib.sha256(sample_file.read_bytes()).hexdigest(),
            }
            rows.append(row)
    return rows, identity, profiles[0]


def image_filter(rows):
    rejected = set()
    reports = {}
    for batch in BATCHES:
        batch_rows = [row for row in rows if row["batch"] == batch]
        low, report = robust_low_sharpness_indices(
            [(row["index"], row["sharpness"]) for row in batch_rows],
            robust_z_limit=-2.5,
            max_reject_fraction=0.10,
        )
        hard = {
            row["index"]
            for row in batch_rows
            if not row["metric_pose_eligible"]
            or row["corner_count"] < 8
            or row["reprojection_rmse_px"] > 1.0
        }
        rejected |= low | hard
        reports[batch] = {
            "robust_report": report,
            "low_sharpness_indices": sorted(low),
            "hard_quality_rejected_indices": sorted(hard),
            "scores": [
                {
                    "index": row["index"],
                    "sharpness": row["sharpness"],
                    "corner_count": row["corner_count"],
                    "reprojection_rmse_px": row["reprojection_rmse_px"],
                }
                for row in batch_rows
            ],
        }
    return rejected, reports


def observation(row):
    return HandEyeObservation(
        row["index"], row["T_base_link7"], row["T_camera_charuco"]
    )


def fit(rows):
    initial = solve_hand_eye_robust(
        [observation(row) for row in rows], min_samples=14
    )
    X, boards, result, groups, robust = _fit_robust(
        [rows],
        initial["T_gripper_camera"],
        0.25,
        0.002,
        min_samples_per_session=14,
    )
    return X, boards[0], result, groups[0], robust, initial


def residual_report(rows, X, board):
    values = [_pose_residual(row, X, board) for row in rows]
    return {
        "count": len(rows),
        "rotation_stats_deg": stats([value[0] for value in values]),
        "translation_stats_mm": stats([value[1] for value in values], 1000.0),
        "samples": [
            {
                "index": row["index"],
                "batch": row["batch"],
                "rotation_deg": value[0],
                "translation_mm": value[1] * 1000.0,
            }
            for row, value in zip(rows, values)
        ],
    }


def main():
    rows, identity, profile = load_rows()
    image_rejected, image_quality = image_filter(rows)
    usable = [row for row in rows if row["index"] not in image_rejected]
    training = [row for row in usable if row["split"] == "train"]
    holdout = [row for row in usable if row["split"] == "holdout"]
    X, board, result, active, robust, initial = fit(training)
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
        batch_X, _batch_board, _result, batch_active, batch_robust, batch_initial = fit(
            batch_training
        )
        batch_transforms.append(batch_X)
        batch_solutions[batch] = {
            "T_link6_camera_m": batch_X,
            "input_training_count": len(batch_training),
            "used_training_count": len(batch_active),
            "initial_hand_eye_method": batch_initial["method"],
            "initial_hand_eye_outliers": batch_initial["outlier_indices"],
            "nonlinear_outliers": batch_robust["rejected_indices_by_session"][0],
        }
    batch_rotation, batch_translation = transform_delta(*batch_transforms)
    cross_validation = _cross_validate([active], X, 0.25, 0.002, 5, 20261019)
    excitation = excitation_diagnostics([observation(row) for row in active])
    old_file = SESSION / "candidate-right-buffered-1792396888396569639.json"
    old_X = np.asarray(read(old_file)["T_link6_camera_m"], dtype=float)
    old_rotation, old_translation = transform_delta(old_X, X)
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
    payload = {
        "schema": "agx.right_wrist_combined_filtered_hand_eye.v1",
        "status": "candidate_requires_independent_physical_validation",
        "created_unix": time.time(),
        "arm": "right",
        "camera_serial": SERIAL,
        "frame": "T_right_link6_right_wrist_color_optical",
        "meaning": "maps right-wrist D435 color optical points into right link6",
        "units": {"translation_matrix": "metres", "reported_translation": "millimetres"},
        "T_link6_camera_m": X,
        "translation_mm": X[:3, 3] * 1000.0,
        "scipy_as_euler_xyz_deg": Rotation.from_matrix(X[:3, :3]).as_euler(
            "xyz", degrees=True
        ),
        "T_base_board_m": board,
        "input_batches": {
            "first": {"indices": [7, 37], "count": 31},
            "second": {"indices": [38, 68], "count": 31},
        },
        "selection_policy": {
            "split": "index modulo 4 equals 3 is holdout; holdouts never fit or tune",
            "image_blur": "per-batch canonical-board Tenengrad plus Laplacian, robust z < -2.5, maximum 10 percent",
            "pose_outlier": "training only; joint rotation/translation MAD filter followed by nonlinear SE3 soft-L1 refit",
            "board_constraint": "one shared T_base_board because operator confirmed board stayed fixed across both captures",
        },
        "image_quality": image_quality,
        "image_rejected_indices": sorted(image_rejected),
        "input_training_indices": [row["index"] for row in training],
        "used_training_indices": [row["index"] for row in active],
        "geometric_training_rejected_indices": robust["rejected_indices_by_session"][0],
        "holdout_indices": [row["index"] for row in holdout],
        "initial_hand_eye": {
            "method": initial["method"],
            "inlier_indices": initial["inlier_indices"],
            "outlier_indices": initial["outlier_indices"],
        },
        "solver": {
            "method": "shared_board_nonlinear_se3_soft_l1_robust_mad",
            "rotation_scale_deg": 0.25,
            "translation_scale_m": 0.002,
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
        "delta_from_first_candidate": {
            "source": str(old_file),
            "rotation_deg": old_rotation,
            "translation_mm": old_translation * 1000.0,
        },
        "checks": checks,
        "qualified": all(checks.values()),
        "accepted_for_control": False,
        "session_identity": identity,
        "camera_profile": profile,
        "robot_motion_commands_sent": 0,
        "control_config_changed": False,
    }
    output = SESSION / ("candidate-right-combined-filtered-%d.json" % time.time_ns())
    with output.open("x") as handle:
        json.dump(clean(payload), handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "candidate": str(output),
        "T_link6_camera_m": clean(X),
        "translation_mm": clean(X[:3, 3] * 1000.0),
        "scipy_as_euler_xyz_deg": clean(payload["scipy_as_euler_xyz_deg"]),
        "image_rejected_indices": sorted(image_rejected),
        "geometric_training_rejected_indices": payload["geometric_training_rejected_indices"],
        "used_training_count": len(active),
        "holdout_count": len(holdout),
        "holdout_rotation_stats_deg": hold_rot,
        "holdout_translation_stats_mm": hold_mm,
        "batch_repeatability": payload["batch_repeatability"],
        "qualified": payload["qualified"],
        "accepted_for_control": False,
    }, indent=2))


if __name__ == "__main__":
    main()
