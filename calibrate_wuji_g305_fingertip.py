#!/usr/bin/env python3
"""Calibrate a palm camera and a moving Wuji fingertip AprilCube jointly.

For every stationary sample, this program records the raw G305 left image,
the measured Wuji qpos20, URDF forward kinematics, and AprilCube PnP pose.  It
solves the two constants in

    T_left_palm_link_tip(q_i) @ T_tip_cube
      = T_left_palm_link_g305_raw_left_optical @ T_g305_raw_left_optical_cube_i.

Motion is locked unless ``--execute-motion`` is supplied.  Executed targets go
through FingerEyeV2's live factory/replay limits, collision check, waypoint
planner, fresh readback checks, and motor-disable cleanup after every pose.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from calibrate_g305_left_hand_back_palm import (
    G305RawLeftCamera,
)
from calibrate_xarm7_g305_eye_in_hand import read_complete_left_frame
from robot_cam_calib.fingertip_extrinsics import (
    FingertipObservation,
    PalmTipKinematics,
    solve_fingertip_extrinsics,
)
from robot_cam_calib.geometry import inv_T, make_T, residual_stats, transform_delta
from robot_cam_calib.io import append_timestamp, atomic_yaml_dump
from robot_cam_calib.targets import (
    Intrinsics,
    PoseDetection,
)


REPO_ROOT = Path(__file__).resolve().parent
FINGEREYE_ROOT = Path("/home/CNF2025915223/桌面/FingerEyeV2")
DEFAULT_URDF = FINGEREYE_ROOT / (
    "assets/thirdparty/xarm7_wuji_left_description/wuji_left_w_fingereye.urdf"
)
DEFAULT_CUBE_CONFIG = FINGEREYE_ROOT / (
    "assets/cubes/cube_april_36h11_6_11_1x1x1_15mm/config.json"
)
DEFAULT_CUBE_MESH = FINGEREYE_ROOT / (
    "assets/thirdparty/xarm7_wuji_left_description/fingereye_mesh/"
    "index_wuji_w_cube.stl"
)
DEFAULT_IMAGE2CUBE_PACKAGE = FINGEREYE_ROOT / "FingereyeData/image2cube_pose"
DEFAULT_WUJI_CONFIG = FINGEREYE_ROOT / "FingerEyeRW/PolicyReview/configs/v2.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/extrinsics/wuji_g305_fingertip/extrinsics.yaml"
DEFAULT_SAMPLE_ROOT = REPO_ROOT / "outputs/extrinsics/wuji_g305_fingertip/samples"
DEFAULT_TIP_LINK = "left_finger2_link4"
CAMERA_FRAME = "wuji_g305_raw_left_optical"
DEFAULT_TARGET_FRAME = "index_wuji_w_cube_update"
MIN_SAMPLES = 12


@dataclass(frozen=True)
class CapturedSample:
    index: int
    captured_at: str
    raw_image_path: Path
    annotated_image_path: Path
    qpos20_rad: np.ndarray
    T_palm_tip: np.ndarray
    T_camera_cube: np.ndarray
    cube_tags: int
    reprojection_error_px: float
    g305_device_timestamp_ms: Optional[float]
    g305_system_timestamp_us: Optional[int]
    requested_qpos8_rad: Optional[np.ndarray]
    effective_qpos8_rad: Optional[np.ndarray]

    def observation(self) -> FingertipObservation:
        return FingertipObservation(
            self.index, self.T_palm_tip, self.T_camera_cube
        )


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


@dataclass
class Image2CubePoseDetector:
    """Live adapter around image2cube_pose's formal Strict estimator."""

    estimator: Any

    def process(self, frame_bgr: np.ndarray) -> PoseDetection:
        # Calibration samples are intentionally independent.  Do not let a
        # previous hand pose seed optical flow across a commanded movement.
        self.estimator.reset_temporal_state()
        record = {"image_bgr": np.asarray(frame_bgr, dtype=np.uint8)}
        packed = self.estimator.process_record(record)
        result = dict(packed.get("result", {}))
        try:
            visual = self.estimator.overlay_image(record, result)
        except (cv2.error, OverflowError, ValueError):
            visual = frame_bgr.copy()
        if not bool(packed.get("success", False)) or result.get("T") is None:
            reason = str(result.get("failure_reason", "unknown"))
            return PoseDetection(
                ok=False,
                T=None,
                n_points=int(result.get("n_tags", 0) or 0),
                reproj_error=float(result.get("reproj_error", float("inf"))),
                message=f"image2cube_pose Strict rejected: {reason}",
                vis=visual,
            )
        transform = np.asarray(result["T"], dtype=np.float64).reshape(4, 4).copy()
        # image2cube_pose/AprilCube uses millimetres; RobotCamCalib stores SI.
        transform[:3, 3] *= 0.001
        source = str(result.get("pose_source", "unknown"))
        ids = [int(value) for value in result.get("tag_ids", [])]
        return PoseDetection(
            ok=True,
            T=transform,
            n_points=int(result.get("n_tags", len(ids)) or len(ids)),
            reproj_error=float(result.get("reproj_error", float("inf"))),
            message=(
                f"image2cube_pose Strict ok source={source} ids={ids} "
                f"err={float(result.get('reproj_error', float('inf'))):.2f}px"
            ),
            vis=visual,
        )


def _create_cube_detector(
    config_path: Path,
    package_path: Path,
    intrinsics: Intrinsics,
) -> Image2CubePoseDetector:
    package = package_path.expanduser().resolve()
    if package.name != "image2cube_pose" or not (package / "core/kernels.py").is_file():
        raise ValueError(f"Invalid image2cube_pose package path: {package}")
    source = str(FINGEREYE_ROOT.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from FingereyeData.image2cube_pose.core.kernels import (
        StrictAprilCubeEstimationConfig,
        StrictAprilCubeEstimator,
    )

    metadata = {
        "image_size": list(intrinsics.image_size),
        "raw_camera_matrix": intrinsics.K.tolist(),
        "raw_dist_coeffs": intrinsics.dist.tolist(),
        # Keep the exact SDK raw-left pixel geometry.  The packaged G305
        # calibration belongs to a different physical serial/profile.
        "undistort_for_detection": False,
    }
    estimator = StrictAprilCubeEstimator(
        metadata,
        StrictAprilCubeEstimationConfig(
            pkl_path="live_g305_not_a_pkl",
            output_pkl=Path("live_g305_not_written.pkl"),
            cube_cfg=config_path.expanduser().resolve(),
            no_undistort=True,
            slow=True,
            no_filter=True,
            fallback_layout="cfg",
            fallback_max_reproj=2.0,
            fallback_ransac_reproj=2.0,
            no_fill_missing_pose=True,
            precompute_only=True,
            show_viser=False,
        ),
    )
    return Image2CubePoseDetector(estimator)


def _detect(frame: np.ndarray, detector: Image2CubePoseDetector, intr: Intrinsics) -> PoseDetection:
    del intr
    try:
        return detector.process(frame)
    except Exception as exc:
        return PoseDetection(
            ok=False,
            T=None,
            message=f"AprilCube exception: {type(exc).__name__}: {exc}",
            vis=frame.copy(),
        )


def _sample_record(sample: CapturedSample, target_frame: str) -> dict[str, Any]:
    return {
        "index": sample.index,
        "captured_at": sample.captured_at,
        "raw_image_path": str(sample.raw_image_path.resolve()),
        "annotated_image_path": str(sample.annotated_image_path.resolve()),
        "qpos20_rad": sample.qpos20_rad.tolist(),
        "T_left_palm_link_tip": sample.T_palm_tip.tolist(),
        f"T_{CAMERA_FRAME}_{target_frame}": sample.T_camera_cube.tolist(),
        "cube_tags": sample.cube_tags,
        "reprojection_error_px": sample.reprojection_error_px,
        "g305_device_timestamp_ms": sample.g305_device_timestamp_ms,
        "g305_system_timestamp_us": sample.g305_system_timestamp_us,
        "requested_qpos8_rad": (
            None
            if sample.requested_qpos8_rad is None
            else sample.requested_qpos8_rad.tolist()
        ),
        "effective_qpos8_rad": (
            None
            if sample.effective_qpos8_rad is None
            else sample.effective_qpos8_rad.tolist()
        ),
    }


def _write_manifest(
    path: Path,
    samples: list[CapturedSample],
    metadata: dict[str, Any],
) -> None:
    target_frame = str(metadata["target"]["frame"])
    atomic_yaml_dump(
        path,
        _serializable(
            {
                "schema": "robot_cam_calib.wuji_g305_fingertip_capture.v1",
                "updated_at": datetime.now().astimezone().isoformat(),
                "conventions": {
                    "transform": "T_A_B maps B-frame points into A",
                    "equation": (
                        "T_left_palm_link_tip(q_i) @ T_tip_cube = "
                        f"T_left_palm_link_{CAMERA_FRAME} @ "
                        f"T_{CAMERA_FRAME}_{target_frame}_i"
                    ),
                },
                "metadata": metadata,
                "num_samples": len(samples),
                "samples": [
                    _sample_record(sample, target_frame) for sample in samples
                ],
            }
        ),
    )


def _load_manifest(path: Path) -> tuple[list[CapturedSample], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != "robot_cam_calib.wuji_g305_fingertip_capture.v1":
        raise ValueError(f"Unsupported manifest schema: {resolved}")
    metadata = dict(payload["metadata"])
    target_frame = str(
        metadata.get("target", {}).get("frame", DEFAULT_TARGET_FRAME)
    )
    camera_cube_key = f"T_{CAMERA_FRAME}_{target_frame}"
    samples: list[CapturedSample] = []
    for record in payload["samples"]:
        raw_path = Path(record["raw_image_path"])
        annotated_path = Path(record["annotated_image_path"])
        if not raw_path.is_file() or not annotated_path.is_file():
            raise FileNotFoundError(f"Missing captured image for sample {record['index']}")
        samples.append(
            CapturedSample(
                index=int(record["index"]),
                captured_at=str(record["captured_at"]),
                raw_image_path=raw_path,
                annotated_image_path=annotated_path,
                qpos20_rad=np.asarray(record["qpos20_rad"], dtype=np.float64),
                T_palm_tip=np.asarray(
                    record["T_left_palm_link_tip"], dtype=np.float64
                ),
                T_camera_cube=np.asarray(record[camera_cube_key], dtype=np.float64),
                cube_tags=int(record["cube_tags"]),
                reprojection_error_px=float(record["reprojection_error_px"]),
                g305_device_timestamp_ms=record.get("g305_device_timestamp_ms"),
                g305_system_timestamp_us=record.get("g305_system_timestamp_us"),
                requested_qpos8_rad=(
                    None
                    if record.get("requested_qpos8_rad") is None
                    else np.asarray(record["requested_qpos8_rad"], dtype=np.float64)
                ),
                effective_qpos8_rad=(
                    None
                    if record.get("effective_qpos8_rad") is None
                    else np.asarray(record["effective_qpos8_rad"], dtype=np.float64)
                ),
            )
        )
    return samples, metadata


def _solve_and_save(
    samples: list[CapturedSample],
    metadata: dict[str, Any],
    output: Path,
    manifest: Path,
    *,
    starts: int,
) -> Path:
    solution = solve_fingertip_extrinsics(
        [sample.observation() for sample in samples],
        min_samples=MIN_SAMPLES,
        starts=starts,
    )
    tip_link = str(metadata["kinematics"]["tip_link"])
    target_frame = str(metadata["target"]["frame"])
    palm_camera_key = f"T_left_palm_link_{CAMERA_FRAME}"
    tip_cube_key = f"T_{tip_link}_{target_frame}"
    actual_output = append_timestamp(output.expanduser().resolve())
    payload = {
        "schema": "robot_cam_calib.wuji_g305_fingertip_extrinsics.v1",
        "status": "candidate_requires_physical_validation",
        "conventions": {
            "transform": "T_A_B maps B-frame points into A",
            "outputs": [palm_camera_key, tip_cube_key],
            "equation": (
                f"T_left_palm_link_{tip_link}(q_i) @ {tip_cube_key} = "
                f"{palm_camera_key} @ T_{CAMERA_FRAME}_{target_frame}_i"
            ),
        },
        palm_camera_key: solution.pop("T_palm_camera"),
        tip_cube_key: solution.pop("T_tip_cube"),
        "solver": solution,
        "capture_manifest": str(manifest.expanduser().resolve()),
        "metadata": metadata,
        "sample_count": len(samples),
        "samples": [_sample_record(sample, target_frame) for sample in samples],
    }
    atomic_yaml_dump(actual_output, _serializable(payload))
    print(f"[RESULT] {palm_camera_key}:")
    print(np.asarray(payload[palm_camera_key]))
    print(f"[RESULT] {tip_cube_key}:")
    print(np.asarray(payload[tip_cube_key]))
    stats = payload["solver"]
    print(
        "[DIAGNOSTICS] inliers={}/{} median={:.3f}deg/{:.2f}mm rank={} cond={:.3g}".format(
            stats["num_inliers"],
            stats["num_samples"],
            stats["rotation_stats_deg"]["median"],
            1000.0 * stats["translation_stats_m"]["median"],
            stats["jacobian_rank"],
            stats["jacobian_condition"],
        )
    )
    print(f"[INFO] Saved {actual_output}")
    return actual_output


def _solve_target_with_known_camera_and_save(
    samples: list[CapturedSample],
    metadata: dict[str, Any],
    output: Path,
    manifest: Path,
    known_palm_camera_yaml: Path,
) -> Path:
    """Solve only the photographed fingertip target while keeping camera fixed."""
    T_palm_camera, camera_path = _load_palm_camera(known_palm_camera_yaml)
    solution = _solve_target_with_known_camera(samples, T_palm_camera)
    tip_link = str(metadata["kinematics"]["tip_link"])
    target_frame = str(metadata["target"]["frame"])
    palm_camera_key = f"T_left_palm_link_{CAMERA_FRAME}"
    tip_target_key = f"T_{tip_link}_{target_frame}"
    T_tip_target = solution.pop("T_tip_target")
    actual_output = append_timestamp(output.expanduser().resolve())
    payload = {
        "schema": "robot_cam_calib.wuji_fingertip_from_photos_fixed_camera.v1",
        "status": "candidate_requires_independent_batch_validation",
        "conventions": {
            "transform": "T_A_B maps B-frame points into A",
            "equation": (
                f"{tip_target_key} = inv(T_left_palm_link_{tip_link}(q_i)) "
                f"@ {palm_camera_key} @ T_{CAMERA_FRAME}_{target_frame}_i"
            ),
        },
        palm_camera_key: T_palm_camera,
        tip_target_key: T_tip_target,
        "solver": solution,
        "known_camera_source": str(camera_path),
        "capture_manifest": str(manifest.expanduser().resolve()),
        "metadata": metadata,
        "sample_count": len(samples),
        "samples": [_sample_record(sample, target_frame) for sample in samples],
    }
    atomic_yaml_dump(actual_output, _serializable(payload))
    stats = payload["solver"]
    print(f"[FIXED CAMERA] {palm_camera_key}:\n{T_palm_camera}")
    print(f"[PHOTO RESULT] {tip_target_key}:\n{T_tip_target}")
    print(
        "[DIAGNOSTICS] inliers={}/{} median={:.3f}deg/{:.2f}mm".format(
            stats["num_inliers"],
            stats["num_samples"],
            stats["rotation_stats_deg"]["median"],
            1000.0 * stats["translation_stats_m"]["median"],
        )
    )
    print(f"[INFO] Saved {actual_output}")
    return actual_output


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("Cannot average an empty transform list")
    items = [np.asarray(item, dtype=np.float64).reshape(4, 4) for item in transforms]
    rotation = Rotation.from_matrix(
        np.stack([item[:3, :3] for item in items])
    ).mean()
    translation = np.mean([item[:3, 3] for item in items], axis=0)
    return make_T(rotation.as_matrix(), translation)


def _solve_target_with_known_camera(
    samples: list[CapturedSample], T_palm_camera: np.ndarray
) -> dict[str, Any]:
    """Estimate one link-to-target constant when palm-to-camera is known."""
    candidates = [
        inv_T(sample.T_palm_tip) @ T_palm_camera @ sample.T_camera_cube
        for sample in samples
    ]
    initial = _mean_transform(candidates)
    initial_delta = [transform_delta(initial, item) for item in candidates]
    rotations = np.asarray([item[0] for item in initial_delta], dtype=np.float64)
    translations = np.asarray([item[1] for item in initial_delta], dtype=np.float64)
    rotation_limit = float(
        np.clip(
            np.median(rotations)
            + 3.0 * 1.4826 * np.median(np.abs(rotations - np.median(rotations))),
            0.25,
            3.0,
        )
    )
    translation_limit = float(
        np.clip(
            np.median(translations)
            + 3.0
            * 1.4826
            * np.median(np.abs(translations - np.median(translations))),
            0.0005,
            0.010,
        )
    )
    inliers = [
        index
        for index, (rotation, translation) in enumerate(initial_delta)
        if rotation <= rotation_limit and translation <= translation_limit
    ]
    if len(inliers) < max(6, len(samples) // 2):
        raise RuntimeError(
            f"Only {len(inliers)}/{len(samples)} stationary observations agree"
        )
    estimate = _mean_transform([candidates[index] for index in inliers])
    final_delta = [transform_delta(estimate, item) for item in candidates]
    final_rotations = [item[0] for item in final_delta]
    final_translations = [item[1] for item in final_delta]
    return {
        "T_tip_target": estimate,
        "candidate_T_tip_target": candidates,
        "inlier_indices": [samples[index].index for index in inliers],
        "outlier_indices": [
            sample.index for index, sample in enumerate(samples) if index not in inliers
        ],
        "rotation_residual_deg": final_rotations,
        "translation_residual_m": final_translations,
        "rotation_stats_deg": residual_stats(final_rotations),
        "translation_stats_m": residual_stats(final_translations),
        "rotation_limit_deg": rotation_limit,
        "translation_limit_m": translation_limit,
        "num_samples": len(samples),
        "num_inliers": len(inliers),
    }


def _load_palm_camera(path: Path) -> tuple[np.ndarray, Path]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    key = f"T_left_palm_link_{CAMERA_FRAME}"
    if key not in payload:
        raise KeyError(f"{resolved} has no {key}")
    return np.asarray(payload[key], dtype=np.float64).reshape(4, 4), resolved


def _load_known_tip_target(
    path: Path, tip_link: str, target_frame: str
) -> tuple[np.ndarray, Path]:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    key = f"T_{tip_link}_{target_frame}"
    if key not in payload:
        raise KeyError(f"{resolved} has no {key}")
    return np.asarray(payload[key], dtype=np.float64).reshape(4, 4), resolved


def _solve_camera_with_known_target(
    samples: list[CapturedSample], T_tip_target: np.ndarray
) -> dict[str, Any]:
    """Estimate palm-to-camera while enforcing the CAD link-to-mesh transform."""
    candidates = [
        sample.T_palm_tip @ T_tip_target @ inv_T(sample.T_camera_cube)
        for sample in samples
    ]
    initial = _mean_transform(candidates)
    initial_delta = [transform_delta(initial, item) for item in candidates]
    rotations = np.asarray([item[0] for item in initial_delta], dtype=np.float64)
    translations = np.asarray([item[1] for item in initial_delta], dtype=np.float64)
    rotation_limit = float(
        np.clip(
            np.median(rotations)
            + 3.0 * 1.4826 * np.median(np.abs(rotations - np.median(rotations))),
            0.25,
            3.0,
        )
    )
    translation_limit = float(
        np.clip(
            np.median(translations)
            + 3.0
            * 1.4826
            * np.median(np.abs(translations - np.median(translations))),
            0.0005,
            0.010,
        )
    )
    inliers = [
        index
        for index, (rotation, translation) in enumerate(initial_delta)
        if rotation <= rotation_limit and translation <= translation_limit
    ]
    if len(inliers) < max(6, len(samples) // 2):
        raise RuntimeError(
            f"Only {len(inliers)}/{len(samples)} known-mesh camera observations agree"
        )
    estimate = _mean_transform([candidates[index] for index in inliers])
    final_delta = [transform_delta(estimate, item) for item in candidates]
    return {
        "T_palm_camera": estimate,
        "candidate_T_palm_camera": candidates,
        "inlier_indices": [samples[index].index for index in inliers],
        "outlier_indices": [
            sample.index for index, sample in enumerate(samples) if index not in inliers
        ],
        "rotation_residual_deg": [item[0] for item in final_delta],
        "translation_residual_m": [item[1] for item in final_delta],
        "rotation_stats_deg": residual_stats([item[0] for item in final_delta]),
        "translation_stats_m": residual_stats([item[1] for item in final_delta]),
        "rotation_limit_deg": rotation_limit,
        "translation_limit_m": translation_limit,
        "num_samples": len(samples),
        "num_inliers": len(inliers),
    }


def _solve_known_tip_and_save(
    samples: list[CapturedSample],
    metadata: dict[str, Any],
    output: Path,
    manifest: Path,
    known_tip_target_yaml: Path,
) -> Path:
    tip_link = str(metadata["kinematics"]["tip_link"])
    target_frame = str(metadata["target"]["frame"])
    T_tip_target, known_path = _load_known_tip_target(
        known_tip_target_yaml, tip_link, target_frame
    )
    solution = _solve_camera_with_known_target(samples, T_tip_target)
    palm_camera_key = f"T_left_palm_link_{CAMERA_FRAME}"
    tip_target_key = f"T_{tip_link}_{target_frame}"
    actual_output = append_timestamp(output.expanduser().resolve())
    payload = {
        "schema": "robot_cam_calib.wuji_g305_camera_from_known_tip_mesh.v1",
        "status": "candidate_requires_repeatability_validation",
        "conventions": {
            "transform": "T_A_B maps B-frame points into A",
            "equation": (
                f"T_left_palm_link_{tip_link}(q_i) @ {tip_target_key} = "
                f"{palm_camera_key} @ T_{CAMERA_FRAME}_{target_frame}_i"
            ),
        },
        palm_camera_key: solution.pop("T_palm_camera"),
        tip_target_key: T_tip_target,
        "known_tip_target_source": str(known_path),
        "solver": solution,
        "capture_manifest": str(manifest.expanduser().resolve()),
        "metadata": metadata,
        "sample_count": len(samples),
        "samples": [_sample_record(sample, target_frame) for sample in samples],
    }
    atomic_yaml_dump(actual_output, _serializable(payload))
    print(f"[RESULT] {palm_camera_key}:")
    print(np.asarray(payload[palm_camera_key]))
    print(f"[FIXED] {tip_target_key}:")
    print(T_tip_target)
    stats = payload["solver"]
    print(
        "[DIAGNOSTICS] inliers={}/{} median={:.3f}deg/{:.2f}mm".format(
            stats["num_inliers"],
            stats["num_samples"],
            stats["rotation_stats_deg"]["median"],
            1000.0 * stats["translation_stats_m"]["median"],
        )
    )
    print(f"[INFO] Saved {actual_output}")
    return actual_output


def _load_wuji_tools(config_path: Path) -> tuple[dict[str, Any], Any]:
    root = str(FINGEREYE_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from FingerEyeRW.PolicyReview.hand_qpos8_preview import isolated_wuji_operation

    config = yaml.safe_load(config_path.expanduser().resolve().read_text(encoding="utf-8"))
    return config, isolated_wuji_operation


def _read_hand(
    config: dict[str, Any], operation: Any, requested_qpos8: np.ndarray
) -> dict[str, Any]:
    return dict(operation("preview", config, requested_qpos8))


def _execute_hand(
    config: dict[str, Any], operation: Any, requested_qpos8: np.ndarray
) -> dict[str, Any]:
    return dict(operation("execute", config, requested_qpos8))


@contextmanager
def _hold_hand_at_target(
    config: dict[str, Any], requested_qpos8: np.ndarray
) -> Any:
    """Execute one guarded target and keep its controller enabled while yielding."""
    root = str(FINGEREYE_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from FingerEyeRW.PolicyReview import init_hand
    from FingerEyeRW.PolicyReview import hand_qpos8_preview as preview
    from FingerEyeRW.PolicyReview.robot import SafetyLimits

    qpos8 = preview.validate_qpos8(requested_qpos8)
    settings = preview._targeted_settings(config)
    if not settings.enabled:
        raise RuntimeError("initialization.enabled=false; held capture is locked")
    safety = SafetyLimits(
        init_hand._section(config, "safety"), command_arm=False, command_hand=True
    )
    hand = init_hand._make_wuji(config)
    stage = "readonly_connect"
    target_write_count = 0
    try:
        initial = hand.connect_readonly()
        preview._validate_snapshot_target(initial, qpos8, safety, settings)

        def _pre_enable_validate(snapshot: Any) -> None:
            preview._validate_snapshot_target(snapshot, qpos8, safety, settings)

        stage = "controller_initialization_before_enable"
        prepared = hand.prepare_motion(startup_validator=_pre_enable_validate)
        if prepared is None:
            raise RuntimeError("Wuji held preparation returned no fresh snapshot")
        stage = "controller_enabled_seeded"
        target, validation, plan = preview._validate_snapshot_target(
            prepared, qpos8, safety, settings
        )
        held_anchor = np.asarray(prepared.qpos20_rad, dtype=np.float64)[8:].copy()
        last = prepared
        if not plan.current_safe:
            stage = "recovery_target_write"
            written = hand.command_full_target20(plan.entry_target20_rad)
            target_write_count += 1
            if not np.array_equal(written, plan.entry_target20_rad):
                raise RuntimeError("Wuji client changed held recovery target")
            last = init_hand._wait_for_waypoint(
                hand, plan.entry_target20_rad, safety, settings
            )
        for waypoint_index, planned_waypoint in enumerate(plan.final_waypoints20_rad):
            waypoint = (
                target.copy()
                if waypoint_index == len(plan.final_waypoints20_rad) - 1
                else planned_waypoint
            )
            if not np.array_equal(waypoint[8:], held_anchor):
                raise RuntimeError("Held waypoint changed a preserved joint")
            stage = f"target_waypoint_write_{waypoint_index}"
            written = hand.command_full_target20(waypoint)
            target_write_count += 1
            if not np.array_equal(written, waypoint):
                raise RuntimeError("Wuji client changed held target")
            last = init_hand._wait_for_waypoint(hand, waypoint, safety, settings)
        yield {
            "hand": hand,
            "target": target,
            "effective_qpos8_rad": target[:8].copy(),
            "actual": np.asarray(last.qpos20_rad, dtype=np.float64).copy(),
            "validation": validation,
            "target_write_count": target_write_count,
        }
    except BaseException as exc:
        raise RuntimeError(
            f"Wuji held execution failed at stage={stage}; "
            f"target_writes={target_write_count}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        cleanup = hand.close()
        cleanup_log = (
            cleanup.as_log_dict()
            if hasattr(cleanup, "as_log_dict")
            else {"success": cleanup is None, "steps": []}
        )
        if not cleanup_log.get("success", False):
            print(f"[WARNING] held-controller cleanup={cleanup_log}")


def _candidate_targets(
    baseline8: np.ndarray,
    lower8: np.ndarray,
    upper8: np.ndarray,
    active_indices: tuple[int, ...],
    attempts: int,
    seed: int,
    max_amplitude_rad: float,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    targets: list[np.ndarray] = []
    for index in range(attempts):
        amplitude = min(max_amplitude_rad, 0.10 + 0.025 * (index // 4))
        target = baseline8.copy()
        delta = rng.uniform(-amplitude, amplitude, size=len(active_indices))
        for local, joint_index in enumerate(active_indices):
            target[joint_index] = np.clip(
                baseline8[joint_index] + delta[local],
                lower8[joint_index],
                upper8[joint_index],
            )
        targets.append(target)
    return targets


def _load_motion_targets(
    path: Path,
    baseline8: np.ndarray,
    active_indices: tuple[int, ...],
) -> list[np.ndarray]:
    """Load explicit active-chain targets while preserving the other fingers."""
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    rows = payload.get("targets", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Motion target file has no targets: {path}")
    targets: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        values = np.asarray(row, dtype=np.float64).reshape(-1)
        if values.size == len(active_indices):
            target = baseline8.copy()
            target[list(active_indices)] = values
        elif values.size == 8:
            target = values.copy()
        else:
            raise ValueError(
                f"Motion target {row_index} must have {len(active_indices)} or 8 values"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError(f"Motion target {row_index} contains non-finite values")
        targets.append(target)
    return targets


def _is_diverse(samples: list[CapturedSample], candidate: np.ndarray) -> bool:
    return all(
        (
            lambda delta: delta[0] >= 2.0 or delta[1] >= 0.002
        )(transform_delta(sample.T_camera_cube, candidate))
        for sample in samples
    )


def _capture_one(
    camera: G305RawLeftCamera,
    context: Image2CubePoseDetector,
    intrinsics: Intrinsics,
    kinematics: PalmTipKinematics,
    qpos20: np.ndarray,
    sample_dir: Path,
    samples: list[CapturedSample],
    max_reprojection_error: float,
    min_cube_tags: int,
    requested8: Optional[np.ndarray],
    effective8: Optional[np.ndarray],
    frames_per_pose: int,
) -> Optional[CapturedSample]:
    requested_frames = max(1, int(frames_per_pose))
    accepted_frames: list[tuple[np.ndarray, PoseDetection, Optional[float], Optional[int]]] = []
    last_frame: Optional[np.ndarray] = None
    last_detection: Optional[PoseDetection] = None
    for _frame_index in range(requested_frames):
        frame, device_timestamp, system_timestamp = read_complete_left_frame(camera)
        detection = _detect(frame, context, intrinsics)
        last_frame = frame
        last_detection = detection
        if (
            detection.ok
            and detection.T is not None
            and detection.n_points >= min_cube_tags
            and detection.reproj_error <= max_reprojection_error
        ):
            accepted_frames.append(
                (frame, detection, device_timestamp, system_timestamp)
            )
    required_frames = (requested_frames + 1) // 2
    if len(accepted_frames) < required_frames:
        assert last_frame is not None and last_detection is not None
        rejected = sample_dir / "rejected" / (
            f"attempt_{int(time.time() * 1000)}.png"
        )
        rejected.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(rejected),
            last_detection.vis if last_detection.vis is not None else last_frame,
        )
        print(
            f"[REJECT] valid pose frames={len(accepted_frames)}/{requested_frames}; "
            f"last={last_detection.message}; required_tags>={min_cube_tags}; "
            f"saved {rejected}"
        )
        return None
    transforms = [np.asarray(item[1].T, dtype=np.float64) for item in accepted_frames]
    averaged = np.eye(4, dtype=np.float64)
    averaged[:3, :3] = Rotation.from_matrix(
        np.stack([item[:3, :3] for item in transforms])
    ).mean().as_matrix()
    averaged[:3, 3] = np.median(
        np.stack([item[:3, 3] for item in transforms]), axis=0
    )
    representative_index = int(
        np.argmin(
            [
                transform_delta(averaged, item)[0]
                + 1000.0 * transform_delta(averaged, item)[1]
                for item in transforms
            ]
        )
    )
    frame, detection, device_timestamp, system_timestamp = accepted_frames[
        representative_index
    ]
    if not _is_diverse(samples, averaged):
        print("[REJECT] detected cube pose is too similar to an accepted sample")
        return None
    index = len(samples)
    raw_path = sample_dir / f"sample_{index:04d}_g305_raw_left.png"
    annotated_path = sample_dir / f"sample_{index:04d}_detected.png"
    if not cv2.imwrite(str(raw_path), frame):
        raise RuntimeError(f"Failed to save {raw_path}")
    visual = detection.vis if detection.vis is not None else frame.copy()
    if not cv2.imwrite(str(annotated_path), visual):
        raise RuntimeError(f"Failed to save {annotated_path}")
    sample = CapturedSample(
        index=index,
        captured_at=datetime.now().astimezone().isoformat(),
        raw_image_path=raw_path,
        annotated_image_path=annotated_path,
        qpos20_rad=np.asarray(qpos20, dtype=np.float64).copy(),
        T_palm_tip=kinematics.forward(qpos20),
        T_camera_cube=averaged,
        cube_tags=min(item[1].n_points for item in accepted_frames),
        reprojection_error_px=float(
            np.median([item[1].reproj_error for item in accepted_frames])
        ),
        g305_device_timestamp_ms=device_timestamp,
        g305_system_timestamp_us=system_timestamp,
        requested_qpos8_rad=(None if requested8 is None else requested8.copy()),
        effective_qpos8_rad=(None if effective8 is None else effective8.copy()),
    )
    samples.append(sample)
    print(
        f"[ACCEPT] sample={index} tags={sample.cube_tags} "
        f"err={sample.reprojection_error_px:.3f}px "
        f"pose_frames={len(accepted_frames)}/{requested_frames} "
        f"image={raw_path.name}"
    )
    return sample


def _apply_bracketed_qpos(
    samples: list[CapturedSample],
    sample: CapturedSample,
    qpos_before: np.ndarray,
    qpos_after: np.ndarray,
    kinematics: PalmTipKinematics,
) -> CapturedSample:
    """Associate an image with the midpoint of its bracketing joint reads."""
    before = np.asarray(qpos_before, dtype=np.float64).reshape(20)
    after = np.asarray(qpos_after, dtype=np.float64).reshape(20)
    midpoint = 0.5 * (before + after)
    updated = replace(
        sample,
        qpos20_rad=midpoint,
        T_palm_tip=kinematics.forward(midpoint),
    )
    if not samples or samples[-1] is not sample:
        raise RuntimeError("Bracketed qpos update must target the latest sample")
    samples[-1] = updated
    drift_deg = np.degrees(after[:4] - before[:4])
    print(
        "[TIMING] bracketing thumb qpos drift deg="
        f"{np.array2string(drift_deg, precision=3)}"
    )
    return updated


def _camera(args: argparse.Namespace) -> G305RawLeftCamera:
    return G305RawLeftCamera(
        args.g305_serial,
        args.g305_width,
        args.g305_height,
        args.g305_fps,
        args.g305_format,
        args.g305_work_mode,
        args.frame_timeout_ms,
    )


def check_configuration(args: argparse.Namespace) -> None:
    for path in (
        args.urdf,
        args.cube_config,
        args.cube_mesh,
        args.image2cube_package,
        args.wuji_config,
    ):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    kinematics = PalmTipKinematics(args.urdf, args.tip_link)
    cube = json.loads(args.cube_config.read_text(encoding="utf-8"))
    print(f"[OK] URDF: {kinematics.urdf_path}")
    print(
        f"[OK] chain {kinematics.palm_link} -> {kinematics.tip_link}: "
        f"{kinematics.chain_joint_names}, hardware indices={kinematics.chain_joint_indices}"
    )
    print(
        f"[OK] AprilCube IDs={cube['tag_ids']} size={cube['tag_size_mm']}mm "
        f"box={cube['box_dims']}mm"
    )
    print(f"[OK] cube-frame mesh: {args.cube_mesh.expanduser().resolve()}")


def check_hardware(args: argparse.Namespace) -> None:
    check_configuration(args)
    config, operation = _load_wuji_tools(args.wuji_config)
    hand = _read_hand(config, operation, np.zeros(8, dtype=np.float64))
    actual = np.asarray(hand["actual"], dtype=np.float64)
    camera = _camera(args)
    try:
        profile = camera.open()
        frame, _device_timestamp, _system_timestamp = read_complete_left_frame(camera)
        context = _create_cube_detector(
            args.cube_config, args.image2cube_package, profile.as_intrinsics()
        )
        detection = _detect(frame, context, profile.as_intrinsics())
        print(f"[OK] Wuji qpos20={actual.tolist()}")
        clipping = hand["clipping"]
        print(
            "[OK] Wuji execution qpos8 bounds="
            f"{clipping['execution_lower_qpos8_rad']} .. "
            f"{clipping['execution_upper_qpos8_rad']}"
        )
        print(f"[OK] G305 serial={profile.serial} profile={profile.width}x{profile.height}@{profile.fps}")
        print(f"[{'OK' if detection.ok else 'FAIL'}] {detection.message}")
        if not detection.ok:
            raise RuntimeError("AprilCube is not visible in the G305 raw-left image")
    finally:
        camera.close()


def stationary_annotation(args: argparse.Namespace) -> None:
    """Annotate a rigid tip mesh frame without commanding any hand motion."""
    check_configuration(args)
    if args.known_palm_camera_yaml is None:
        raise RuntimeError(
            "Stationary annotation requires --known-palm-camera-yaml"
        )
    T_palm_camera, camera_extrinsics_path = _load_palm_camera(
        args.known_palm_camera_yaml
    )
    kinematics = PalmTipKinematics(args.urdf, args.tip_link)
    config, operation = _load_wuji_tools(args.wuji_config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = args.sample_root.expanduser().resolve() / stamp
    sample_dir.mkdir(parents=True, exist_ok=False)
    manifest = sample_dir / "capture_manifest.yaml"
    samples: list[CapturedSample] = []
    metadata: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "stationary_annotation_with_known_palm_camera",
        "kinematics": {
            "urdf": str(args.urdf.expanduser().resolve()),
            "palm_link": kinematics.palm_link,
            "tip_link": kinematics.tip_link,
            "chain_joint_names": list(kinematics.chain_joint_names),
            "chain_joint_indices": list(kinematics.chain_joint_indices),
        },
        "target": {
            "config": str(args.cube_config.expanduser().resolve()),
            "mesh": str(args.cube_mesh.expanduser().resolve()),
            "frame": args.target_frame,
            "mesh_frame_contract": (
                f"AprilCube frame equals the source frame of {args.cube_mesh.name}"
            ),
        },
        "known_palm_camera_extrinsics": str(camera_extrinsics_path),
        "motion": {
            "commanded": False,
            "note": "Read-only stationary multi-frame annotation",
        },
    }
    camera = _camera(args)
    try:
        profile = camera.open()
        intrinsics = profile.as_intrinsics()
        detector = _create_cube_detector(
            args.cube_config, args.image2cube_package, intrinsics
        )
        metadata["camera"] = {
            "frame": CAMERA_FRAME,
            "serial": profile.serial,
            "profile": (
                f"{profile.width}x{profile.height}@{profile.fps} "
                f"{profile.format_name}"
            ),
            "K": intrinsics.K.tolist(),
            "dist": intrinsics.dist.tolist(),
            "previous_work_mode": profile.previous_work_mode,
            "capture_work_mode": profile.active_work_mode,
        }
        for attempt in range(args.max_motion_attempts):
            if len(samples) >= args.samples:
                break
            frame, device_timestamp, system_timestamp = read_complete_left_frame(
                camera
            )
            detection = _detect(frame, detector, intrinsics)
            if (
                not detection.ok
                or detection.T is None
                or detection.reproj_error > args.max_reprojection_error
            ):
                rejected = sample_dir / "rejected" / f"attempt_{attempt:04d}.png"
                rejected.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(rejected),
                    detection.vis if detection.vis is not None else frame,
                )
                print(f"[REJECT] attempt={attempt} {detection.message}")
                continue
            hand = _read_hand(config, operation, np.zeros(8, dtype=np.float64))
            qpos20 = np.asarray(hand["actual"], dtype=np.float64)
            index = len(samples)
            raw_path = sample_dir / f"sample_{index:04d}_g305_raw_left.png"
            annotated_path = sample_dir / f"sample_{index:04d}_detected.png"
            cv2.imwrite(str(raw_path), frame)
            cv2.imwrite(
                str(annotated_path),
                detection.vis if detection.vis is not None else frame,
            )
            samples.append(
                CapturedSample(
                    index=index,
                    captured_at=datetime.now().astimezone().isoformat(),
                    raw_image_path=raw_path,
                    annotated_image_path=annotated_path,
                    qpos20_rad=qpos20,
                    T_palm_tip=kinematics.forward(qpos20),
                    T_camera_cube=detection.T.copy(),
                    cube_tags=detection.n_points,
                    reprojection_error_px=detection.reproj_error,
                    g305_device_timestamp_ms=device_timestamp,
                    g305_system_timestamp_us=system_timestamp,
                    requested_qpos8_rad=None,
                    effective_qpos8_rad=qpos20[:8].copy(),
                )
            )
            print(
                f"[ACCEPT] sample={index} tags={detection.n_points} "
                f"err={detection.reproj_error:.3f}px"
            )
            _write_manifest(manifest, samples, metadata)
        if len(samples) < max(6, args.samples // 2):
            raise RuntimeError(
                f"Captured only {len(samples)}/{args.samples} stationary samples"
            )
        solution = _solve_target_with_known_camera(samples, T_palm_camera)
        target_frame = str(args.target_frame)
        tip_target_key = f"T_{args.tip_link}_{target_frame}"
        palm_camera_key = f"T_left_palm_link_{CAMERA_FRAME}"
        actual_output = append_timestamp(args.output.expanduser().resolve())
        T_tip_target = solution.pop("T_tip_target")
        payload = {
            "schema": "robot_cam_calib.wuji_fingertip_mesh_annotation.v1",
            "status": "candidate_requires_physical_validation",
            "conventions": {
                "transform": "T_A_B maps B-frame points into A",
                "output": tip_target_key,
                "equation": (
                    f"{tip_target_key} = inv(T_left_palm_link_{args.tip_link}(q_i)) "
                    f"@ {palm_camera_key} @ T_{CAMERA_FRAME}_{target_frame}_i"
                ),
            },
            tip_target_key: T_tip_target,
            "known_camera_extrinsic": {
                "source": str(camera_extrinsics_path),
                palm_camera_key: T_palm_camera,
            },
            "solver": solution,
            "capture_manifest": str(manifest),
            "metadata": metadata,
            "sample_count": len(samples),
            "samples": [
                _sample_record(sample, target_frame) for sample in samples
            ],
        }
        atomic_yaml_dump(actual_output, _serializable(payload))
        print(f"[RESULT] {tip_target_key}:")
        print(T_tip_target)
        stats = payload["solver"]
        print(
            "[DIAGNOSTICS] inliers={}/{} median={:.3f}deg/{:.2f}mm".format(
                stats["num_inliers"],
                stats["num_samples"],
                stats["rotation_stats_deg"]["median"],
                1000.0 * stats["translation_stats_m"]["median"],
            )
        )
        print(f"[INFO] Saved {actual_output}")
    finally:
        camera.close()


def probe_tip_motion(args: argparse.Namespace) -> None:
    """Make one guarded small move and verify that the observed cube follows it."""
    if not args.execute_motion:
        raise RuntimeError("Tip-motion probing is motion-locked; pass --execute-motion")
    check_configuration(args)
    kinematics = PalmTipKinematics(args.urdf, args.tip_link)
    if not set(kinematics.chain_joint_indices).issubset(set(range(8))):
        raise RuntimeError("The guarded targeted mover supports only Wuji qpos20[:8]")
    config, operation = _load_wuji_tools(args.wuji_config)
    initial = _read_hand(config, operation, np.zeros(8, dtype=np.float64))
    actual20 = np.asarray(initial["actual"], dtype=np.float64)
    baseline8 = actual20[:8].copy()
    clipping = initial["clipping"]
    lower8 = np.asarray(clipping["execution_lower_qpos8_rad"], dtype=np.float64)
    upper8 = np.asarray(clipping["execution_upper_qpos8_rad"], dtype=np.float64)
    joint_index = kinematics.chain_joint_indices[0]
    positive_room = upper8[joint_index] - baseline8[joint_index]
    negative_room = baseline8[joint_index] - lower8[joint_index]
    direction = 1.0 if positive_room >= negative_room else -1.0
    target8 = baseline8.copy()
    target8[joint_index] = np.clip(
        baseline8[joint_index] + direction * args.probe_delta_rad,
        lower8[joint_index],
        upper8[joint_index],
    )
    actual_delta = abs(float(target8[joint_index] - baseline8[joint_index]))
    if actual_delta < 0.5 * args.probe_delta_rad:
        raise RuntimeError("Insufficient safe range for the requested probe delta")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "outputs/diagnostics/wuji_tip_probe" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    camera = _camera(args)
    moved = False
    try:
        profile = camera.open()
        intrinsics = profile.as_intrinsics()
        context = _create_cube_detector(
            args.cube_config, args.image2cube_package, intrinsics
        )

        def observe(label: str) -> PoseDetection:
            frame, _device, _system = read_complete_left_frame(camera)
            detection = _detect(frame, context, intrinsics)
            raw_path = output_dir / f"{label}_raw.png"
            vis_path = output_dir / f"{label}_detected.png"
            cv2.imwrite(str(raw_path), frame)
            cv2.imwrite(
                str(vis_path),
                detection.vis if detection.vis is not None else frame,
            )
            if not detection.ok or detection.T is None:
                raise RuntimeError(f"{label}: {detection.message}")
            print(f"[PROBE] {label}: {detection.message}")
            return detection

        before = observe("before")
        preview = _read_hand(config, operation, target8)
        if not preview.get("eligible", False):
            raise RuntimeError(f"Probe target rejected: {preview.get('reason', 'unknown')}")
        print(
            f"[PROBE] moving hardware joint {joint_index} by "
            f"{target8[joint_index] - baseline8[joint_index]:+.4f} rad"
        )
        executed = _execute_hand(config, operation, target8)
        moved = True
        time.sleep(args.capture_settle_seconds)
        after = observe("after")
        rotation_deg, translation_m = transform_delta(before.T, after.T)
        linked = rotation_deg >= 1.0 or translation_m >= 0.0015
        report = {
            "schema": "robot_cam_calib.wuji_tip_motion_probe.v1",
            "tip_link": args.tip_link,
            "joint_index": joint_index,
            "baseline_qpos8_rad": baseline8.tolist(),
            "target_qpos8_rad": target8.tolist(),
            "executed_actual_qpos20_rad": executed["actual"],
            f"T_{CAMERA_FRAME}_{args.target_frame}_before": before.T.tolist(),
            f"T_{CAMERA_FRAME}_{args.target_frame}_after": after.T.tolist(),
            "cube_delta_rotation_deg": rotation_deg,
            "cube_delta_translation_m": translation_m,
            "cube_follows_selected_chain": linked,
        }
        atomic_yaml_dump(output_dir / "probe.yaml", report)
        print(
            f"[PROBE] cube delta={rotation_deg:.3f}deg/"
            f"{translation_m * 1000.0:.2f}mm linked={linked}"
        )
        print(f"[PROBE] Saved {output_dir}")
        if not linked:
            raise RuntimeError(
                f"Observed cube does not follow {args.tip_link}; do not solve with this chain"
            )
    finally:
        camera.close()
        if moved:
            print("[PROBE] returning to the exact initial qpos8")
            _execute_hand(config, operation, baseline8)


def live_capture(args: argparse.Namespace) -> None:
    if not args.execute_motion:
        raise RuntimeError("Live exploration is motion-locked; pass --execute-motion")
    kinematics = PalmTipKinematics(args.urdf, args.tip_link)
    if not set(kinematics.chain_joint_indices).issubset(set(range(8))):
        raise RuntimeError("The guarded targeted mover supports only Wuji qpos20[:8]")
    config, operation = _load_wuji_tools(args.wuji_config)
    initial_report = _read_hand(config, operation, np.zeros(8, dtype=np.float64))
    initial_qpos20 = np.asarray(initial_report["actual"], dtype=np.float64)
    baseline8 = initial_qpos20[:8].copy()
    clipping = initial_report["clipping"]
    lower8 = np.asarray(clipping["execution_lower_qpos8_rad"], dtype=np.float64)
    upper8 = np.asarray(clipping["execution_upper_qpos8_rad"], dtype=np.float64)
    if args.motion_targets_yaml is None:
        targets = _candidate_targets(
            baseline8,
            lower8,
            upper8,
            kinematics.chain_joint_indices,
            args.max_motion_attempts,
            args.seed,
            args.max_target_amplitude_rad,
        )
    else:
        targets = _load_motion_targets(
            args.motion_targets_yaml,
            baseline8,
            kinematics.chain_joint_indices,
        )[: args.max_motion_attempts]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = args.sample_root.expanduser().resolve() / stamp
    sample_dir.mkdir(parents=True, exist_ok=False)
    manifest = sample_dir / "capture_manifest.yaml"
    samples: list[CapturedSample] = []
    metadata: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "kinematics": {
            "urdf": str(args.urdf.expanduser().resolve()),
            "palm_link": kinematics.palm_link,
            "tip_link": kinematics.tip_link,
            "chain_joint_names": list(kinematics.chain_joint_names),
            "chain_joint_indices": list(kinematics.chain_joint_indices),
        },
        "target": {
            "config": str(args.cube_config.expanduser().resolve()),
            "mesh": str(args.cube_mesh.expanduser().resolve()),
            "frame": args.target_frame,
            "mesh_frame_contract": (
                f"AprilCube frame equals the source frame of {args.cube_mesh.name}"
            ),
        },
        "motion": {
            "guard": "FingerEyeV2 hand_qpos8 targeted executor",
            "seed": args.seed,
            "maximum_attempts": args.max_motion_attempts,
            "return_to_start": args.return_to_start,
            "controller_held_during_capture": args.hold_during_capture,
            "baseline_qpos8_rad": baseline8.tolist(),
        },
    }
    camera = _camera(args)
    last_good8 = baseline8.copy()
    profile = None
    try:
        profile = camera.open()
        intrinsics = profile.as_intrinsics()
        context = _create_cube_detector(
            args.cube_config, args.image2cube_package, intrinsics
        )
        metadata["camera"] = {
            "frame": CAMERA_FRAME,
            "serial": profile.serial,
            "profile": f"{profile.width}x{profile.height}@{profile.fps} {profile.format_name}",
            "K": intrinsics.K.tolist(),
            "dist": intrinsics.dist.tolist(),
            "previous_work_mode": profile.previous_work_mode,
            "capture_work_mode": profile.active_work_mode,
        }
        if args.hold_during_capture:
            with _hold_hand_at_target(config, baseline8) as held:
                time.sleep(args.capture_settle_seconds)
                before = held["hand"].read_fresh_snapshot()
                initial_sample = _capture_one(
                    camera,
                    context,
                    intrinsics,
                    kinematics,
                    np.asarray(before.qpos20_rad, dtype=np.float64),
                    sample_dir,
                    samples,
                    args.max_reprojection_error,
                    args.min_cube_tags,
                    None,
                    np.asarray(held["effective_qpos8_rad"], dtype=np.float64),
                    args.frames_per_pose,
                )
                after = held["hand"].read_fresh_snapshot()
            if initial_sample is not None:
                _apply_bracketed_qpos(
                    samples,
                    initial_sample,
                    np.asarray(before.qpos20_rad, dtype=np.float64),
                    np.asarray(after.qpos20_rad, dtype=np.float64),
                    kinematics,
                )
        else:
            initial_sample = _capture_one(
                camera,
                context,
                intrinsics,
                kinematics,
                initial_qpos20,
                sample_dir,
                samples,
                args.max_reprojection_error,
                args.min_cube_tags,
                None,
                baseline8,
                args.frames_per_pose,
            )
            if initial_sample is not None:
                initial_after = _read_hand(config, operation, baseline8)
                _apply_bracketed_qpos(
                    samples,
                    initial_sample,
                    initial_qpos20,
                    np.asarray(initial_after["actual"], dtype=np.float64),
                    kinematics,
                )
        _write_manifest(manifest, samples, metadata)
        for attempt, target8 in enumerate(targets, start=1):
            if len(samples) >= args.samples:
                break
            print(f"[MOVE] attempt={attempt}/{len(targets)} qpos8={target8.tolist()}")
            preview = _read_hand(config, operation, target8)
            if not preview.get("eligible", False):
                print(f"[REJECT] safety preview: {preview.get('reason', 'unknown')}")
                continue
            try:
                if args.hold_during_capture:
                    with _hold_hand_at_target(config, target8) as held:
                        effective8 = np.asarray(
                            held["effective_qpos8_rad"], dtype=np.float64
                        )
                        time.sleep(args.capture_settle_seconds)
                        before = held["hand"].read_fresh_snapshot()
                        actual20 = np.asarray(before.qpos20_rad, dtype=np.float64)
                        accepted = _capture_one(
                            camera,
                            context,
                            intrinsics,
                            kinematics,
                            actual20,
                            sample_dir,
                            samples,
                            args.max_reprojection_error,
                            args.min_cube_tags,
                            target8,
                            effective8,
                            args.frames_per_pose,
                        )
                        after = held["hand"].read_fresh_snapshot()
                    if accepted is not None:
                        accepted = _apply_bracketed_qpos(
                            samples,
                            accepted,
                            actual20,
                            np.asarray(after.qpos20_rad, dtype=np.float64),
                            kinematics,
                        )
                else:
                    executed = _execute_hand(config, operation, target8)
                    effective8 = np.asarray(
                        executed["effective_qpos8_rad"], dtype=np.float64
                    )
                    time.sleep(args.capture_settle_seconds)
                    # Associate the image with bracketing reads after the
                    # short-lived executor has released its controller.
                    settled = _read_hand(config, operation, effective8)
                    actual20 = np.asarray(settled["actual"], dtype=np.float64)
                    accepted = _capture_one(
                        camera,
                        context,
                        intrinsics,
                        kinematics,
                        actual20,
                        sample_dir,
                        samples,
                        args.max_reprojection_error,
                        args.min_cube_tags,
                        target8,
                        effective8,
                        args.frames_per_pose,
                    )
                    if accepted is not None:
                        settled_after = _read_hand(config, operation, effective8)
                        accepted = _apply_bracketed_qpos(
                            samples,
                            accepted,
                            actual20,
                            np.asarray(settled_after["actual"], dtype=np.float64),
                            kinematics,
                        )
            except Exception as exc:
                # Hardware state can drift slightly between the isolated
                # preview and execute processes.  Treat an execute-time safety
                # rejection as a rejected candidate, not as a session abort.
                print(
                    f"[REJECT] execute-time safety check: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if accepted is not None:
                last_good8 = effective8.copy()
                _write_manifest(manifest, samples, metadata)
            else:
                print("[MOVE] restoring last pose with a valid cube detection")
                _execute_hand(config, operation, last_good8)
        if len(samples) < MIN_SAMPLES:
            print(
                f"[INFO] Captured {len(samples)} valid samples; need {MIN_SAMPLES}. "
                f"Manifest remains at {manifest}"
            )
            return
        if args.known_palm_camera_yaml is not None:
            _solve_target_with_known_camera_and_save(
                samples,
                metadata,
                args.output,
                manifest,
                args.known_palm_camera_yaml,
            )
        elif args.known_tip_target_yaml is None:
            _solve_and_save(
                samples, metadata, args.output, manifest, starts=args.solver_starts
            )
        else:
            _solve_known_tip_and_save(
                samples,
                metadata,
                args.output,
                manifest,
                args.known_tip_target_yaml,
            )
    finally:
        camera.close()
        if args.return_to_start:
            try:
                print("[MOVE] returning Wuji thumb/index to the initial qpos8")
                _execute_hand(config, operation, baseline8)
            except Exception as exc:
                print(f"[WARNING] return-to-start failed: {type(exc).__name__}: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--tip-link", default=DEFAULT_TIP_LINK)
    parser.add_argument("--cube-config", type=Path, default=DEFAULT_CUBE_CONFIG)
    parser.add_argument("--cube-mesh", type=Path, default=DEFAULT_CUBE_MESH)
    parser.add_argument("--target-frame", default=DEFAULT_TARGET_FRAME)
    parser.add_argument(
        "--image2cube-package", type=Path, default=DEFAULT_IMAGE2CUBE_PACKAGE
    )
    parser.add_argument("--wuji-config", type=Path, default=DEFAULT_WUJI_CONFIG)
    parser.add_argument("--g305-serial", default="CV2T661000NC")
    parser.add_argument("--g305-work-mode", default="Dual Color Streams")
    parser.add_argument("--g305-width", type=int, default=1280)
    parser.add_argument("--g305-height", type=int, default=800)
    parser.add_argument("--g305-fps", type=int, default=20)
    parser.add_argument("--g305-format", default="RGB")
    parser.add_argument("--frame-timeout-ms", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-motion-attempts", type=int, default=60)
    parser.add_argument("--capture-settle-seconds", type=float, default=0.5)
    parser.add_argument("--max-reprojection-error", type=float, default=2.0)
    parser.add_argument("--min-cube-tags", type=int, default=1)
    parser.add_argument("--frames-per-pose", type=int, default=1)
    parser.add_argument("--hold-during-capture", action="store_true")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-target-amplitude-rad", type=float, default=0.45)
    parser.add_argument("--motion-targets-yaml", type=Path)
    parser.add_argument("--probe-delta-rad", type=float, default=0.08)
    parser.add_argument("--solver-starts", type=int, default=24)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute-motion", action="store_true")
    parser.add_argument(
        "--return-to-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--offline-manifest", type=Path)
    parser.add_argument("--known-palm-camera-yaml", type=Path)
    parser.add_argument("--known-tip-target-yaml", type=Path)
    parser.add_argument("--stationary-annotation", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-hardware", action="store_true")
    parser.add_argument("--probe-tip-motion", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if (
        args.known_palm_camera_yaml is not None
        and args.known_tip_target_yaml is not None
    ):
        raise SystemExit(
            "Use only one of --known-palm-camera-yaml and --known-tip-target-yaml"
        )
    if args.samples < MIN_SAMPLES:
        raise SystemExit(f"--samples must be >= {MIN_SAMPLES}")
    if args.max_motion_attempts < args.samples - 1:
        raise SystemExit("--max-motion-attempts must allow all requested samples")
    if args.check_config:
        check_configuration(args)
        return
    if args.check_hardware:
        check_hardware(args)
        return
    if args.stationary_annotation:
        stationary_annotation(args)
        return
    if args.probe_tip_motion:
        probe_tip_motion(args)
        return
    if args.offline_manifest is not None:
        samples, metadata = _load_manifest(args.offline_manifest)
        if args.known_palm_camera_yaml is not None:
            _solve_target_with_known_camera_and_save(
                samples,
                metadata,
                args.output,
                args.offline_manifest,
                args.known_palm_camera_yaml,
            )
        elif args.known_tip_target_yaml is None:
            _solve_and_save(
                samples,
                metadata,
                args.output,
                args.offline_manifest,
                starts=args.solver_starts,
            )
        else:
            _solve_known_tip_and_save(
                samples,
                metadata,
                args.output,
                args.offline_manifest,
                args.known_tip_target_yaml,
            )
        return
    check_configuration(args)
    live_capture(args)


if __name__ == "__main__":
    main()
