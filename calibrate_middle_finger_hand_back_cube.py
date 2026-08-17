#!/usr/bin/env python3
"""Calibrate ^hand_back_cube T_middle_finger_cam from paired target views.

The third-view camera observes the hand-back AprilCube while the middle-finger
camera observes the fixed ChArUco board.  For T_A_B mapping B coordinates into
A coordinates, every stationary pair obeys

    T_middle_finger_cam_charuco_i
    @ T_charuco_third_view_cam
    @ T_third_view_cam_hand_back_cube_i
    = T_middle_finger_cam_hand_back_cube.

Both constant transforms are estimated jointly.  The requested result is
the inverse, T_hand_back_cube_middle_finger_cam.

Capture stops automatically at --max-samples.  Postprocessing re-detects the
saved images in isolated CPU processes, uses OpenCV/OpenCL for GPU sharpness
scoring when available, performs a parallel robust multi-start SE(3) solve,
checks split-sample stability, and writes one timestamped YAML report.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

import calibrate_g305_left_hand_back_palm as core
from intr_calib_charuco import find_camera_nodes_by_usb_port


REPO_ROOT = Path(__file__).resolve().parent
MIDDLE_FINGER_CAMERA_NAME = "middle_finger_cam"
MIDDLE_FINGER_PORT = "5-4:1.0"
MIDDLE_FINGER_INTRINSICS_YAML = REPO_ROOT / (
    "outputs/intrinsics_None_charuco_2592x1944_0801_185224.yaml"
)
MIDDLE_FINGER_FPS = 50
MIDDLE_FINGER_FOURCC = "MJPG"
THIRD_VIEW_PORT = "7-4:1.0"
THIRD_VIEW_FPS = 20
THIRD_VIEW_FOURCC = "MJPG"

OUTPUT_PATH = REPO_ROOT / (
    "outputs/extrinsics_hand_back_cube_middle_finger_cam.yaml"
)
SAMPLE_IMAGE_ROOT = REPO_ROOT / (
    "outputs/extrinsics_hand_back_cube_middle_finger_cam_samples"
)

DEFAULT_MAX_SAMPLES = 60
MIN_SAMPLES_TO_SOLVE = 16
DISPLAY_SCALE_MIDDLE = 0.35
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))


class ProgressBar:
    """Small dependency-free terminal progress bar."""

    def __init__(self, label: str, total: int, enabled: bool) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.enabled = enabled
        self.current = 0
        self.started = time.perf_counter()

    def update(self, amount: int = 1) -> None:
        self.current = min(self.total, self.current + amount)
        if not self.enabled:
            return
        width = 28
        fraction = self.current / self.total
        filled = int(round(width * fraction))
        elapsed = time.perf_counter() - self.started
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{self.label:<20} [{bar}] "
            f"{self.current:>3}/{self.total:<3} "
            f"{fraction * 100:5.1f}% {elapsed:6.1f}s",
            end="",
            flush=True,
        )
        if self.current == self.total:
            print()


def sample_to_dict(sample: core.CalibrationSample) -> dict[str, Any]:
    return {
        "index": int(sample.index),
        "timestamp": float(sample.timestamp),
        "pair_skew_s": float(sample.pair_skew_s),
        "third_view_frame_index": int(sample.third_view_frame_index),
        "middle_finger_frame_index": int(sample.g305_frame_index),
        "T_third_view_cam_hand_back_cube": (
            sample.T_third_view_hand_back_palm.tolist()
        ),
        "T_middle_finger_cam_charuco": (
            sample.T_g305_left_charuco.tolist()
        ),
        "cube_tags": int(sample.cube_tags),
        "cube_reproj_error_px": float(sample.cube_reproj_error_px),
        "charuco_corners": int(sample.charuco_corners),
        "charuco_reproj_error_px": float(
            sample.charuco_reproj_error_px
        ),
        "third_view_image_path": sample.third_view_image_path,
        "middle_finger_image_path": sample.g305_image_path,
        "capture_mode": sample.capture_mode,
        "third_view_sharpness": (
            None
            if not np.isfinite(sample.third_view_sharpness)
            else float(sample.third_view_sharpness)
        ),
        "middle_finger_sharpness": (
            None
            if not np.isfinite(sample.g305_sharpness)
            else float(sample.g305_sharpness)
        ),
        "offline_status": sample.offline_status,
        "offline_error": sample.offline_error,
    }


def write_capture_manifest(
    sample_dir: Path,
    samples: list[core.CalibrationSample],
) -> None:
    core.atomic_yaml_dump(
        sample_dir / "capture_manifest.yaml",
        {
            "schema": (
                "robot_cam_calib.middle_finger_hand_back_capture.v1"
            ),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "frame_convention": (
                "T_A_B maps coordinates from frame B into frame A"
            ),
            "num_samples": len(samples),
            "samples": [sample_to_dict(sample) for sample in samples],
        },
    )


def load_samples_from_manifest(
    path: Path,
) -> list[core.CalibrationSample]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if (
        not isinstance(payload, dict)
        or payload.get("schema")
        != "robot_cam_calib.middle_finger_hand_back_capture.v1"
    ):
        raise ValueError(f"Unsupported capture manifest: {resolved}")
    records = payload.get("samples")
    if not isinstance(records, list):
        raise ValueError(f"Manifest has no sample list: {resolved}")

    samples: list[core.CalibrationSample] = []
    for record in records:
        third_path = Path(record["third_view_image_path"]).resolve()
        middle_path = Path(record["middle_finger_image_path"]).resolve()
        if not third_path.is_file() or not middle_path.is_file():
            raise FileNotFoundError(
                f"Missing images for sample {record.get('index')}"
            )
        samples.append(
            core.CalibrationSample(
                index=int(record["index"]),
                timestamp=float(record["timestamp"]),
                pair_skew_s=float(record["pair_skew_s"]),
                third_view_frame_index=int(
                    record["third_view_frame_index"]
                ),
                g305_frame_index=int(
                    record["middle_finger_frame_index"]
                ),
                T_third_view_hand_back_palm=np.asarray(
                    record["T_third_view_cam_hand_back_cube"],
                    dtype=np.float64,
                ).reshape(4, 4),
                T_g305_left_charuco=np.asarray(
                    record["T_middle_finger_cam_charuco"],
                    dtype=np.float64,
                ).reshape(4, 4),
                cube_tags=int(record["cube_tags"]),
                cube_reproj_error_px=float(
                    record["cube_reproj_error_px"]
                ),
                charuco_corners=int(record["charuco_corners"]),
                charuco_reproj_error_px=float(
                    record["charuco_reproj_error_px"]
                ),
                third_view_image_path=str(third_path),
                g305_image_path=str(middle_path),
                capture_mode=str(record.get("capture_mode", "offline")),
            )
        )
    if len(samples) != int(payload.get("num_samples", len(samples))):
        raise ValueError("Manifest sample count mismatch")
    return sorted(samples, key=lambda sample: sample.index)


def create_sample_dir() -> Path:
    path = SAMPLE_IMAGE_ROOT / datetime.now().strftime("%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=False)
    return path


def store_sample(
    samples: list[core.CalibrationSample],
    sample_dir: Path,
    third_frame: core.TimedFrame,
    middle_frame: core.TimedFrame,
    skew_s: float,
    cube: core.PoseDetection,
    charuco: core.PoseDetection,
    mode: str,
) -> core.CalibrationSample:
    assert cube.T is not None and charuco.T is not None
    index = len(samples)
    third_path = (
        sample_dir
        / f"sample_{index:04d}_third_view_hand_back_cube.png"
    )
    middle_path = (
        sample_dir
        / f"sample_{index:04d}_middle_finger_charuco.png"
    )
    if not cv2.imwrite(str(third_path), third_frame.frame_bgr):
        raise RuntimeError(f"Failed to save {third_path}")
    if not cv2.imwrite(str(middle_path), middle_frame.frame_bgr):
        raise RuntimeError(f"Failed to save {middle_path}")
    sample = core.CalibrationSample(
        index=index,
        timestamp=0.5 * (third_frame.timestamp + middle_frame.timestamp),
        pair_skew_s=float(skew_s),
        third_view_frame_index=third_frame.index,
        g305_frame_index=middle_frame.index,
        T_third_view_hand_back_palm=cube.T.copy(),
        T_g305_left_charuco=charuco.T.copy(),
        cube_tags=int(cube.n_points),
        cube_reproj_error_px=float(cube.reproj_error),
        charuco_corners=int(charuco.n_points),
        charuco_reproj_error_px=float(charuco.reproj_error),
        third_view_image_path=str(third_path),
        g305_image_path=str(middle_path),
        capture_mode=mode,
    )
    samples.append(sample)
    write_capture_manifest(sample_dir, samples)
    return sample


def _redetect_worker(
    sample: core.CalibrationSample,
    third_intr: core.Intrinsics,
    middle_intr: core.Intrinsics,
) -> core.CalibrationSample:
    cv2.setNumThreads(1)
    return core.offline_redetect_one(sample, third_intr, middle_intr)


def parallel_redetect(
    samples: list[core.CalibrationSample],
    third_intr: core.Intrinsics,
    middle_intr: core.Intrinsics,
    workers: int,
    show_progress: bool,
) -> list[core.CalibrationSample]:
    workers = max(1, min(int(workers), len(samples)))
    progress = ProgressBar(
        "1/4 image detection", len(samples), show_progress
    )
    completed: list[core.CalibrationSample] = []
    if workers == 1:
        for sample in samples:
            completed.append(
                _redetect_worker(sample, third_intr, middle_intr)
            )
            progress.update()
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    _redetect_worker,
                    sample,
                    third_intr,
                    middle_intr,
                )
                for sample in samples
            ]
            for future in as_completed(futures):
                completed.append(future.result())
                progress.update()
    return sorted(completed, key=lambda sample: sample.index)


def gpu_sharpness_filter(
    samples: list[core.CalibrationSample],
    third_intr: core.Intrinsics,
    middle_intr: core.Intrinsics,
    charuco_config: dict[str, Any],
    cube_points_m: np.ndarray,
    show_progress: bool,
) -> tuple[list[core.CalibrationSample], dict[str, Any]]:
    backend = core.configure_opencl_backend()
    use_opencl = backend["backend"] == "opencl_umat"
    progress = ProgressBar(
        "2/4 GPU sharpness", len(samples), show_progress
    )
    scored: list[core.CalibrationSample] = []
    for sample in samples:
        if sample.offline_status != "redetected":
            scored.append(sample)
            progress.update()
            continue
        try:
            third_patch = core.canonical_cube_patch(
                sample.third_view_image_path,
                sample.T_third_view_hand_back_palm,
                third_intr,
                cube_points_m,
                use_opencl,
            )
            middle_patch = core.canonical_charuco_patch(
                sample.g305_image_path,
                sample.T_g305_left_charuco,
                middle_intr,
                charuco_config,
                use_opencl,
            )
            scored.append(
                replace(
                    sample,
                    third_view_sharpness=core.focus_metric(
                        third_patch, use_opencl
                    ),
                    g305_sharpness=core.focus_metric(
                        middle_patch, use_opencl
                    ),
                )
            )
        except Exception as exc:
            scored.append(
                replace(
                    sample,
                    offline_status="rejected_sharpness_exception",
                    offline_error=f"{type(exc).__name__}: {exc}",
                )
            )
        progress.update()

    third_rejected, third_stats = core.robust_low_score_indices(
        [
            (sample.index, sample.third_view_sharpness)
            for sample in scored
            if sample.offline_status == "redetected"
        ]
    )
    middle_rejected, middle_stats = core.robust_low_score_indices(
        [
            (sample.index, sample.g305_sharpness)
            for sample in scored
            if sample.offline_status == "redetected"
        ]
    )
    rejected = third_rejected | middle_rejected
    checked: list[core.CalibrationSample] = []
    for sample in scored:
        if (
            sample.offline_status == "redetected"
            and sample.index in rejected
        ):
            checked.append(
                replace(
                    sample,
                    offline_status="rejected_blur",
                    offline_error=(
                        "robust target sharpness outlier: "
                        f"third={sample.index in third_rejected}, "
                        f"middle={sample.index in middle_rejected}"
                    ),
                )
            )
        elif sample.offline_status == "redetected":
            checked.append(replace(sample, offline_status="usable"))
        else:
            checked.append(sample)
    return checked, {
        **backend,
        "third_view": third_stats,
        "middle_finger_cam": middle_stats,
        "rejected_indices": sorted(rejected),
    }


def _seeded_solve_once(
    samples: list[core.CalibrationSample],
    seed: dict[str, Any],
) -> dict[str, Any]:
    cube = [
        sample.T_third_view_hand_back_palm for sample in samples
    ]
    charuco = [
        sample.T_g305_left_charuco for sample in samples
    ]
    params0 = np.hstack(
        [
            core.transform_to_params(
                seed["T_charuco_third_view_cam"]
            ),
            core.transform_to_params(
                seed["T_g305_raw_left_rgb_hand_back_palm"]
            ),
        ]
    )
    result = core.run_joint_least_squares(params0, cube, charuco)
    X = core.params_to_transform(result.x[:6])
    Y = core.params_to_transform(result.x[6:])
    residuals: list[dict[str, Any]] = []
    for sample in samples:
        closure = (
            core.inv_T(Y)
            @ sample.T_g305_left_charuco
            @ X
            @ sample.T_third_view_hand_back_palm
        )
        residuals.append(
            {
                "index": int(sample.index),
                "rot_deg": float(
                    np.degrees(
                        np.linalg.norm(
                            core.so3_log(closure[:3, :3])
                        )
                    )
                ),
                "trans_m": float(
                    np.linalg.norm(closure[:3, 3])
                ),
            }
        )
    singular = np.linalg.svd(result.jac, compute_uv=False)
    positive = singular[singular > 1e-10]
    return {
        "T_charuco_third_view_cam": X,
        "T_g305_raw_left_rgb_hand_back_palm": Y,
        "T_hand_back_palm_g305_raw_left_rgb": core.inv_T(Y),
        "optimizer_success": bool(result.success),
        "jacobian_rank": int(
            np.linalg.matrix_rank(result.jac, tol=1e-8)
        ),
        "jacobian_condition": (
            float(positive[0] / positive[-1])
            if positive.size
            else float("inf")
        ),
        "per_sample_residuals": residuals,
    }


def _seeded_solve_with_outliers(
    samples: list[core.CalibrationSample],
    seed: dict[str, Any],
) -> dict[str, Any]:
    active = list(samples)
    for _iteration in range(core.OUTLIER_MAX_ITERATIONS):
        solution = _seeded_solve_once(active, seed)
        residuals = solution["per_sample_residuals"]
        rot_limit = core.robust_limit(
            [item["rot_deg"] for item in residuals],
            core.OUTLIER_MIN_ROT_DEG,
            core.OUTLIER_MAX_ROT_DEG,
        )
        trans_limit = core.robust_limit(
            [item["trans_m"] for item in residuals],
            core.OUTLIER_MIN_TRANS_M,
            core.OUTLIER_MAX_TRANS_M,
        )
        kept = {
            item["index"]
            for item in residuals
            if item["rot_deg"] <= rot_limit
            and item["trans_m"] <= trans_limit
        }
        next_active = [
            sample for sample in active if sample.index in kept
        ]
        if (
            len(next_active) < MIN_SAMPLES_TO_SOLVE
            or len(next_active) == len(active)
        ):
            break
        active = next_active

    solution = _seeded_solve_once(active, seed)
    inliers = [sample.index for sample in active]
    inlier_set = set(inliers)
    solution["inlier_indices"] = inliers
    solution["outlier_indices"] = [
        sample.index
        for sample in samples
        if sample.index not in inlier_set
    ]
    solution["residual_rot_deg"] = core.residual_stats(
        [
            item["rot_deg"]
            for item in solution["per_sample_residuals"]
        ]
    )
    solution["residual_trans_m"] = core.residual_stats(
        [
            item["trans_m"]
            for item in solution["per_sample_residuals"]
        ]
    )
    return solution


def _stability_worker(
    name: str,
    samples: list[core.CalibrationSample],
    full_solution: dict[str, Any],
) -> dict[str, Any]:
    solution = _seeded_solve_with_outliers(samples, full_solution)
    mount_rot, mount_trans = core.transform_delta(
        full_solution["T_hand_back_palm_g305_raw_left_rgb"],
        solution["T_hand_back_palm_g305_raw_left_rgb"],
    )
    board_rot, board_trans = core.transform_delta(
        full_solution["T_charuco_third_view_cam"],
        solution["T_charuco_third_view_cam"],
    )
    return {
        "name": name,
        "initialization": "full_solution",
        "input_count": len(samples),
        "input_indices": [sample.index for sample in samples],
        "inlier_count": len(solution["inlier_indices"]),
        "inlier_indices": solution["inlier_indices"],
        "outlier_indices": solution["outlier_indices"],
        "mount_delta_from_full_rot_deg": float(mount_rot),
        "mount_delta_from_full_trans_m": float(mount_trans),
        "board_delta_from_full_rot_deg": float(board_rot),
        "board_delta_from_full_trans_m": float(board_trans),
        "jacobian_rank": solution["jacobian_rank"],
        "jacobian_condition": solution["jacobian_condition"],
        "residual_rot_deg": solution["residual_rot_deg"],
        "residual_trans_m": solution["residual_trans_m"],
    }


def stability_check(
    usable: list[core.CalibrationSample],
    solution: dict[str, Any],
    workers: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    half = len(usable) // 2
    groups = [
        ("first_half", usable[:half]),
        ("second_half", usable[half:]),
        ("even_positions", usable[::2]),
        ("odd_positions", usable[1::2]),
    ]
    progress = ProgressBar(
        "4/4 stability", len(groups), show_progress
    )
    results: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(groups))),
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(
                _stability_worker, name, subset, solution
            )
            for name, subset in groups
        ]
        for future in as_completed(futures):
            results.append(future.result())
            progress.update()
    order = {name: position for position, (name, _items) in enumerate(groups)}
    return sorted(results, key=lambda item: order[item["name"]])


def quality_assessment(
    solution: dict[str, Any],
    stability: list[dict[str, Any]],
) -> dict[str, Any]:
    by_name = {item["name"]: item for item in stability}
    interleaved = [
        by_name["even_positions"],
        by_name["odd_positions"],
    ]
    checks = {
        "optimizer_success": bool(solution["optimizer_success"]),
        "full_rank_12": int(solution["jacobian_rank"]) == 12,
        "jacobian_condition_le_1000": (
            float(solution["jacobian_condition"]) <= 1000.0
        ),
        "inliers_ge_30": len(solution["inlier_indices"]) >= 30,
        "mean_rotation_residual_le_0_7_deg": (
            solution["residual_rot_deg"]["mean"] <= 0.7
        ),
        "mean_translation_residual_le_0_002_m": (
            solution["residual_trans_m"]["mean"] <= 0.002
        ),
        "interleaved_rotation_delta_le_0_5_deg": all(
            item["mount_delta_from_full_rot_deg"] <= 0.5
            for item in interleaved
        ),
        "interleaved_translation_delta_le_0_003_m": all(
            item["mount_delta_from_full_trans_m"] <= 0.003
            for item in interleaved
        ),
    }
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "note": (
            "Interleaved subsets preserve the full pose range and are the "
            "primary repeatability check. Temporal halves diagnose pose "
            "coverage and are not acceptance gates."
        ),
    }


def postprocess(
    samples: list[core.CalibrationSample],
    sample_dir: Path,
    third_intr: core.Intrinsics,
    middle_intr: core.Intrinsics,
    charuco_config: dict[str, Any],
    cube_points_m: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    list[core.CalibrationSample],
    list[core.CalibrationSample],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    started = time.perf_counter()
    redetected = parallel_redetect(
        samples,
        third_intr,
        middle_intr,
        args.process_workers,
        args.progress,
    )
    checked, sharpness = gpu_sharpness_filter(
        redetected,
        third_intr,
        middle_intr,
        charuco_config,
        cube_points_m,
        args.progress,
    )
    usable = [
        sample for sample in checked if sample.offline_status == "usable"
    ]
    write_capture_manifest(sample_dir, checked)
    if len(usable) < MIN_SAMPLES_TO_SOLVE:
        raise RuntimeError(
            f"Only {len(usable)} usable samples after postprocessing; "
            f"need {MIN_SAMPLES_TO_SOLVE}"
        )

    solve_progress = ProgressBar("3/4 joint solve", 1, args.progress)
    solve_started = time.perf_counter()
    solution = core.solve_with_outlier_rejection(
        usable, args.solver_workers
    )
    solve_elapsed = time.perf_counter() - solve_started
    solve_progress.update()
    stability = stability_check(
        usable,
        solution,
        args.process_workers,
        args.progress,
    )
    assessment = quality_assessment(solution, stability)
    if not assessment["accepted"] and not args.allow_low_quality:
        failed = [
            name
            for name, passed in assessment["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            "Calibration failed quality gates: "
            f"{failed}. Images remain in {sample_dir}. "
            "Use --allow-low-quality only for diagnostics."
        )

    report = {
        "method": "free_joint_all_usable_samples",
        "elapsed_seconds": float(time.perf_counter() - started),
        "precise_redetection": {
            "backend": "multiprocessing_spawn",
            "workers": int(args.process_workers),
            "input_count": len(samples),
            "usable_count": len(usable),
        },
        "sharpness": sharpness,
        "solver": {
            "backend": "parallel_multistart_least_squares",
            "workers_requested": int(args.solver_workers),
            "workers_used": int(solution["optimizer_workers"]),
            "elapsed_seconds": float(solve_elapsed),
        },
        "sample_status_counts": {
            status: sum(
                sample.offline_status == status for sample in checked
            )
            for status in sorted(
                {sample.offline_status for sample in checked}
            )
        },
        "quality_assessment": assessment,
    }
    return checked, usable, solution, stability, report


def serialize_solution(solution: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "optimizer_success",
        "optimizer_message",
        "optimizer_nfev",
        "optimizer_num_starts",
        "optimizer_workers",
        "optimizer_selected_start",
        "optimizer_candidate_costs",
        "jacobian_rank",
        "jacobian_condition",
        "jacobian_singular_values",
        "residual_rot_deg",
        "residual_trans_m",
        "inlier_indices",
        "outlier_indices",
        "outlier_rejection_iterations",
        "per_sample_residuals",
    )
    X = solution["T_charuco_third_view_cam"]
    Y = solution["T_g305_raw_left_rgb_hand_back_palm"]
    requested = solution["T_hand_back_palm_g305_raw_left_rgb"]
    return {
        "method": "free_joint_all_usable_samples",
        "T_charuco_third_view_cam": X.tolist(),
        "T_third_view_cam_charuco": core.inv_T(X).tolist(),
        "T_middle_finger_cam_hand_back_cube": Y.tolist(),
        "T_hand_back_cube_middle_finger_cam": requested.tolist(),
        "requested_output": {
            "name": "T_hand_back_cube_middle_finger_cam",
            "meaning": (
                "maps middle-finger camera optical-frame coordinates into "
                "the hand-back/wrist-Q AprilCube object frame"
            ),
            "units": "meters",
        },
        **{key: solution[key] for key in keys},
    }


def save_results(
    output_path: Path,
    checked: list[core.CalibrationSample],
    usable: list[core.CalibrationSample],
    solution: dict[str, Any],
    stability: list[dict[str, Any]],
    report: dict[str, Any],
    sample_dir: Path,
    third_device: int | str,
    middle_device: int | str,
    third_intr: core.Intrinsics,
    middle_intr: core.Intrinsics,
    charuco_config: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    output = core.append_timestamp(output_path)
    payload = {
        "schema": (
            "robot_cam_calib.hand_back_cube_middle_finger_cam."
            "free_joint.v1"
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frame_convention": (
            "T_A_B maps coordinates from frame B into frame A; "
            "translation is meters"
        ),
        "measurement_equation": (
            "T_middle_finger_cam_charuco_i @ "
            "T_charuco_third_view_cam @ "
            "T_third_view_cam_hand_back_cube_i = "
            "T_middle_finger_cam_hand_back_cube"
        ),
        "selection": {
            "selected_method": "free_joint_all_usable_samples",
            "quality_assessment": report["quality_assessment"],
        },
        "inputs": {
            "third_view": {
                "usb_port": args.third_view_port,
                "active_device": str(third_device),
                "resolution": list(third_intr.image_size),
                "fps": int(args.third_view_fps),
                "fourcc": THIRD_VIEW_FOURCC,
                "intrinsics_yaml": str(third_intr.path),
            },
            "middle_finger_cam": {
                "usb_port": args.middle_finger_port,
                "active_device": str(middle_device),
                "resolution": list(middle_intr.image_size),
                "fps": int(args.middle_finger_fps),
                "fourcc": MIDDLE_FINGER_FOURCC,
                "intrinsics_yaml": str(middle_intr.path),
            },
            "targets": {
                "hand_back_cube_aprilcube_config": str(
                    core.HAND_BACK_PALM_APRILCUBE_CONFIG.resolve()
                ),
                "charuco_board_yaml": str(
                    core.CHARUCO_BOARD_YAML.resolve()
                ),
                "charuco": charuco_config,
            },
        },
        "capture": {
            "sample_image_dir": str(sample_dir),
            "num_raw_samples": len(checked),
            "num_usable_after_postprocess": len(usable),
            "automatic_capture": True,
            "automatic_stop_sample_count": int(args.max_samples),
            "software_timestamp_pairing": True,
            "max_pair_skew_s": core.MAX_PAIR_SKEW_S,
            "stable_required_pairs": core.STABLE_REQUIRED_PAIRS,
            "min_sample_rot_delta_deg": core.MIN_SAMPLE_ROT_DELTA_DEG,
            "min_sample_trans_delta_m": (
                core.MIN_SAMPLE_TRANS_DELTA_M
            ),
        },
        "postprocess": report,
        "solution": serialize_solution(solution),
        "stability": stability,
        "samples": [sample_to_dict(sample) for sample in checked],
    }
    core.atomic_yaml_dump(output, payload)
    return output


def validate_configuration(
    third_intr: core.Intrinsics,
    middle_intr: core.Intrinsics,
) -> None:
    required = (
        core.THIRD_VIEW_INTRINSICS_YAML,
        MIDDLE_FINGER_INTRINSICS_YAML,
        core.HAND_BACK_PALM_APRILCUBE_CONFIG,
        core.CHARUCO_BOARD_YAML,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing calibration inputs: {missing}")
    expected = (2592, 1944)
    if third_intr.image_size != expected:
        raise ValueError(
            f"third-view size mismatch: {third_intr.image_size}"
        )
    if middle_intr.image_size != expected:
        raise ValueError(
            f"middle-finger size mismatch: {middle_intr.image_size}"
        )
    if middle_intr.camera_model != "pinhole":
        raise ValueError(
            f"middle-finger model must be pinhole, got "
            f"{middle_intr.camera_model}"
        )


def open_camera(
    port: str,
    intr: core.Intrinsics,
    fps: int,
    fourcc: str,
    label: str,
) -> tuple[Any, int | str]:
    if not port.startswith("/dev/video"):
        candidates = find_camera_nodes_by_usb_port(port)
        if not candidates:
            raise RuntimeError(
                f"No V4L2 camera found at USB port {port} for {label}"
            )
    capture, device = core.start_capture(
        port,
        intr.image_size[0],
        intr.image_size[1],
        fps,
        fourcc,
    )
    actual = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    if actual != intr.image_size:
        capture.release()
        raise RuntimeError(
            f"{label} opened at {actual}, expected {intr.image_size}"
        )
    return capture, device


def print_result(
    output: Path,
    checked: list[core.CalibrationSample],
    usable: list[core.CalibrationSample],
    solution: dict[str, Any],
    report: dict[str, Any],
) -> None:
    print(f"[INFO] Saved {output}")
    print("[RESULT] T_hand_back_cube_middle_finger_cam:")
    print(solution["T_hand_back_palm_g305_raw_left_rgb"])
    print(
        f"[DIAGNOSTICS] usable={len(usable)}/{len(checked)} "
        f"inliers={len(solution['inlier_indices'])} "
        f"rank={solution['jacobian_rank']} "
        f"condition={solution['jacobian_condition']:.2f}"
    )
    print(
        f"[DIAGNOSTICS] residual mean="
        f"{solution['residual_rot_deg']['mean']:.3f}deg/"
        f"{solution['residual_trans_m']['mean'] * 1000.0:.2f}mm "
        f"accepted={report['quality_assessment']['accepted']}"
    )


def run_offline(
    args: argparse.Namespace,
    manifest: Path,
    third_intr: core.Intrinsics,
    middle_intr: core.Intrinsics,
    charuco_config: dict[str, Any],
    cube_points_m: np.ndarray,
) -> None:
    samples = load_samples_from_manifest(manifest)
    checked, usable, solution, stability, report = postprocess(
        samples,
        manifest.parent,
        third_intr,
        middle_intr,
        charuco_config,
        cube_points_m,
        args,
    )
    report["offline_source_manifest"] = str(manifest)
    output = save_results(
        args.output,
        checked,
        usable,
        solution,
        stability,
        report,
        manifest.parent,
        "offline_saved_images",
        "offline_saved_images",
        third_intr,
        middle_intr,
        charuco_config,
        args,
    )
    print_result(output, checked, usable, solution, report)


def capture_samples(
    args: argparse.Namespace,
    third_intr: core.Intrinsics,
    middle_intr: core.Intrinsics,
    board: Any,
    charuco_detector: Any,
    cube_context: Any,
) -> tuple[
    list[core.CalibrationSample],
    Path,
    int | str,
    int | str,
]:
    middle_capture: Any = None
    third_capture: Any = None
    middle_worker: Optional[core.FrameWorker] = None
    third_worker: Optional[core.FrameWorker] = None
    samples: list[core.CalibrationSample] = []
    sample_dir: Optional[Path] = None
    middle_device: int | str = ""
    third_device: int | str = ""
    try:
        middle_capture, middle_device = open_camera(
            args.middle_finger_port,
            middle_intr,
            args.middle_finger_fps,
            MIDDLE_FINGER_FOURCC,
            MIDDLE_FINGER_CAMERA_NAME,
        )
        third_capture, third_device = open_camera(
            args.third_view_port,
            third_intr,
            args.third_view_fps,
            THIRD_VIEW_FOURCC,
            "third_view_cam",
        )

        def read_middle():
            ok, frame = middle_capture.read()
            if not ok or frame is None:
                raise RuntimeError("middle-finger read failed")
            return frame, None, None

        def read_third():
            ok, frame = third_capture.read()
            if not ok or frame is None:
                raise RuntimeError("third-view read failed")
            return frame, None, None

        middle_worker = core.FrameWorker(
            "middle-finger-capture", read_middle
        )
        third_worker = core.FrameWorker("third-view-capture", read_third)
        middle_worker.start()
        third_worker.start()

        if args.check_hardware:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                if middle_worker.snapshot() and third_worker.snapshot():
                    print("[INFO] Hardware check OK.")
                    return samples, Path(), third_device, middle_device
                time.sleep(0.05)
            raise RuntimeError("Hardware check timed out")

        sample_dir = create_sample_dir()
        last_pair: Optional[tuple[int, int]] = None
        previous_valid: Optional[tuple[np.ndarray, np.ndarray]] = None
        stable_count = 0
        last_capture_time = -float("inf")
        last_status = "waiting for synchronized frames"
        print(f"[INFO] Samples: {sample_dir}")
        print("[INFO] Move to a new pose and hold still; capture is automatic.")
        print("[INFO] [s] manual store  [q/esc] solve early")

        while len(samples) < args.max_samples:
            pair = core.select_synchronized_pair(
                third_worker.snapshot(),
                middle_worker.snapshot(),
                last_pair,
            )
            if pair is None:
                key = cv2.waitKey(5) & 0xFF if args.preview else 255
                if key in (ord("q"), 27):
                    break
                continue
            third_frame, middle_frame, skew = pair
            last_pair = (third_frame.index, middle_frame.index)
            cube = core.detect_hand_back_palm_pose(
                third_frame.frame_bgr, cube_context, third_intr
            )
            charuco = core.detect_charuco_pose(
                middle_frame.frame_bgr,
                charuco_detector,
                board,
                middle_intr,
                "middle-finger/ChArUco",
            )
            valid, reason = core.detection_quality(cube, charuco, skew)
            stable_reason = "detections invalid"
            if valid and cube.T is not None and charuco.T is not None:
                stable, stable_reason = core.is_stable_pair(
                    previous_valid, cube.T, charuco.T
                )
                stable_count = stable_count + 1 if stable else 0
                previous_valid = (cube.T.copy(), charuco.T.copy())
            else:
                stable_count = 0
                previous_valid = None

            diverse = False
            diversity_reason = "waiting for valid pose"
            if valid and cube.T is not None:
                diverse, diversity_reason = core.is_diverse_from_all(
                    samples, cube.T
                )
            now = time.monotonic()
            auto_store = (
                valid
                and stable_count >= core.STABLE_REQUIRED_PAIRS
                and diverse
                and now - last_capture_time
                >= core.AUTO_CAPTURE_COOLDOWN_S
            )
            stored = False
            if auto_store:
                sample = store_sample(
                    samples,
                    sample_dir,
                    third_frame,
                    middle_frame,
                    skew,
                    cube,
                    charuco,
                    "auto",
                )
                stored = True
                stable_count = 0
                last_capture_time = now
                last_status = f"stored sample {len(samples)}"
                print(
                    f"[CAPTURE] {last_status}: "
                    f"skew={sample.pair_skew_s * 1000.0:.1f}ms "
                    f"cube={sample.cube_tags}tags "
                    f"charuco={sample.charuco_corners}corners"
                )
            else:
                last_status = (
                    f"{reason}; stable={stable_count}/"
                    f"{core.STABLE_REQUIRED_PAIRS} {stable_reason}; "
                    f"{diversity_reason}"
                )

            key = 255
            if args.preview:
                lines = [
                    f"samples={len(samples)}/{args.max_samples} "
                    f"skew={skew * 1000.0:.1f}ms",
                    last_status,
                    cube.message,
                    charuco.message,
                    "Move, hold | [s] manual | [q] solve early",
                ]
                third_vis = core.put_lines(
                    cube.vis
                    if cube.vis is not None
                    else third_frame.frame_bgr,
                    lines,
                )
                middle_vis = core.put_lines(
                    charuco.vis
                    if charuco.vis is not None
                    else middle_frame.frame_bgr,
                    lines,
                )
                cv2.imshow(
                    "third_view / hand-back cube",
                    core.resize_for_display(
                        third_vis, core.DISPLAY_SCALE_THIRD_VIEW
                    ),
                )
                cv2.imshow(
                    "middle_finger_cam / ChArUco",
                    core.resize_for_display(
                        middle_vis, DISPLAY_SCALE_MIDDLE
                    ),
                )
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and not stored:
                if not valid:
                    print(f"[WARN] Manual sample rejected: {reason}")
                else:
                    store_sample(
                        samples,
                        sample_dir,
                        third_frame,
                        middle_frame,
                        skew,
                        cube,
                        charuco,
                        "manual",
                    )
                    stable_count = 0
                    last_capture_time = now
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted; solving saved samples if sufficient.")
    finally:
        if middle_worker is not None:
            middle_worker.stop()
        if third_worker is not None:
            third_worker.stop()
        if middle_capture is not None:
            middle_capture.release()
        if third_capture is not None:
            third_capture.release()
        cv2.destroyAllWindows()

    if sample_dir is None:
        if args.check_hardware:
            return samples, Path(), third_device, middle_device
        raise RuntimeError("No capture session was created")
    return samples, sample_dir, third_device, middle_device


def main(args: argparse.Namespace) -> None:
    global MIDDLE_FINGER_INTRINSICS_YAML, SAMPLE_IMAGE_ROOT
    MIDDLE_FINGER_INTRINSICS_YAML = Path(args.middle_finger_intrinsics)
    SAMPLE_IMAGE_ROOT = Path(args.sample_root)
    core.THIRD_VIEW_INTRINSICS_YAML = Path(args.third_view_intrinsics)
    core.HAND_BACK_PALM_APRILCUBE_CONFIG = Path(
        args.aprilcube_config
    )
    core.CHARUCO_BOARD_YAML = Path(args.charuco_board)

    third_intr = core.load_intrinsics(core.THIRD_VIEW_INTRINSICS_YAML)
    middle_intr = core.load_intrinsics(MIDDLE_FINGER_INTRINSICS_YAML)
    validate_configuration(third_intr, middle_intr)
    board, charuco_detector, charuco_config = core.load_charuco_target(
        core.CHARUCO_BOARD_YAML
    )
    cube_context = core.create_aprilcube_context(third_intr, fast=True)
    cube_points_m = np.asarray(
        cube_context.detector.box_corners_3d, dtype=np.float64
    ).reshape(-1, 3) / 1000.0

    print("[INFO] Free-joint equation:")
    print(
        "  T_middle_finger_cam_charuco @ "
        "T_charuco_third_view_cam @"
    )
    print(
        "  T_third_view_cam_hand_back_cube = "
        "T_middle_finger_cam_hand_back_cube"
    )
    print("[INFO] Requested: T_hand_back_cube_middle_finger_cam")
    if args.check_config:
        print("[INFO] Configuration and targets OK.")
        return
    if args.offline_manifest is not None:
        run_offline(
            args,
            args.offline_manifest.expanduser().resolve(),
            third_intr,
            middle_intr,
            charuco_config,
            cube_points_m,
        )
        return

    samples, sample_dir, third_device, middle_device = capture_samples(
        args,
        third_intr,
        middle_intr,
        board,
        charuco_detector,
        cube_context,
    )
    if args.check_hardware:
        return
    if len(samples) < MIN_SAMPLES_TO_SOLVE:
        print(
            f"[WARN] Only {len(samples)} samples; need "
            f"{MIN_SAMPLES_TO_SOLVE}. Images remain at {sample_dir}."
        )
        return

    checked, usable, solution, stability, report = postprocess(
        samples,
        sample_dir,
        third_intr,
        middle_intr,
        charuco_config,
        cube_points_m,
        args,
    )
    output = save_results(
        args.output,
        checked,
        usable,
        solution,
        stability,
        report,
        sample_dir,
        third_device,
        middle_device,
        third_intr,
        middle_intr,
        charuco_config,
        args,
    )
    print_result(output, checked, usable, solution, report)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--middle-finger-port", default=MIDDLE_FINGER_PORT
    )
    parser.add_argument(
        "--middle-finger-fps", type=int, default=MIDDLE_FINGER_FPS
    )
    parser.add_argument(
        "--middle-finger-intrinsics",
        type=Path,
        default=MIDDLE_FINGER_INTRINSICS_YAML,
    )
    parser.add_argument("--third-view-port", default=THIRD_VIEW_PORT)
    parser.add_argument(
        "--third-view-fps", type=int, default=THIRD_VIEW_FPS
    )
    parser.add_argument(
        "--third-view-intrinsics",
        type=Path,
        default=core.THIRD_VIEW_INTRINSICS_YAML,
    )
    parser.add_argument(
        "--aprilcube-config",
        type=Path,
        default=core.HAND_BACK_PALM_APRILCUBE_CONFIG,
    )
    parser.add_argument(
        "--charuco-board",
        type=Path,
        default=core.CHARUCO_BOARD_YAML,
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=SAMPLE_IMAGE_ROOT,
    )
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES
    )
    parser.add_argument(
        "--process-workers",
        "--offline-workers",
        dest="process_workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="spawned processes for image detection and stability checks",
    )
    parser.add_argument(
        "--solver-workers",
        type=int,
        default=min(7, DEFAULT_WORKERS),
        help="parallel multi-start optimizer workers",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show postprocessing progress bars",
    )
    parser.add_argument(
        "--allow-low-quality",
        action="store_true",
        help="write a diagnostic YAML even when quality gates fail",
    )
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-hardware", action="store_true")
    parser.add_argument("--offline-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    if cli_args.max_samples < MIN_SAMPLES_TO_SOLVE:
        raise SystemExit(
            f"--max-samples must be >= {MIN_SAMPLES_TO_SOLVE}"
        )
    if cli_args.process_workers < 1 or cli_args.solver_workers < 1:
        raise SystemExit("worker counts must be >= 1")
    if cli_args.self_test:
        core.run_self_test(cli_args.solver_workers)
    else:
        main(cli_args)
