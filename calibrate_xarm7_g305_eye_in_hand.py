#!/usr/bin/env python3
"""Calibrate ``link7_T_wuji_g305_raw_left_optical`` without robot writes.

The G305 is rigidly mounted on xArm7 link7 and observes one stationary
ChArUco board.  The operator moves the robot using the xArm web UI.  Once the
robot has remained still for 0.5 seconds, the program automatically captures
one sample and waits for a deliberate move before re-arming.  Each accepted
sample stores the raw-left image, the seven measured joint angles, controller
forward kinematics ``base_T_link7``, and PnP ``camera_T_charuco``.

No xArm ``set_*`` or motion API is called.  The selected G305 work mode is
temporarily enabled and restored on exit.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from calibrate_g305_left_hand_back_palm import (
    G305RawLeftCamera,
    detect_charuco_pose,
    load_charuco_target,
)
from robot_cam_calib.geometry import make_T, transform_delta
from robot_cam_calib.hand_eye import (
    HandEyeObservation,
    solve_hand_eye_robust,
)
from robot_cam_calib.io import append_timestamp, atomic_yaml_dump


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BOARD = REPO_ROOT / (
    "assets/targets/charuco/"
    "charuco_7x5_40mm_marker30mm_DICT_5X5_50.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "outputs/extrinsics/xarm7_g305_eye_in_hand/extrinsics.yaml"
)
DEFAULT_SAMPLE_ROOT = REPO_ROOT / (
    "outputs/extrinsics/xarm7_g305_eye_in_hand/samples"
)
MIN_SAMPLES = 12
MAX_REPROJECTION_ERROR_PX = 2.0
MIN_POSE_ROTATION_DELTA_DEG = 3.0
MIN_POSE_TRANSLATION_DELTA_M = 0.010
QPOS_STABILITY_READS = 5
QPOS_STABILITY_INTERVAL_S = 0.04
QPOS_STABILITY_MAX_DEG = 0.12
AUTO_STABLE_SECONDS = 0.5
AUTO_REARM_JOINT_DELTA_DEG = 2.0
COMPLETE_FRAME_ATTEMPTS = 20


@dataclass(frozen=True)
class RobotMeasurement:
    qpos_rad: np.ndarray
    T_base_link7: np.ndarray
    state: int
    mode: int
    max_joint_range_deg: float


@dataclass(frozen=True)
class StabilityStatus:
    armed: bool
    ready: bool
    stable_for_s: float
    max_joint_delta_deg: float
    rearmed: bool = False


def max_joint_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the largest wrap-safe absolute joint delta in degrees."""
    delta = np.arctan2(np.sin(first - second), np.cos(first - second))
    return float(np.degrees(np.max(np.abs(delta))))


class StabilityGate:
    """Require continuous stability and a deliberate move between captures."""

    def __init__(
        self,
        *,
        stable_seconds: float,
        stable_joint_range_deg: float,
        rearm_joint_delta_deg: float,
    ) -> None:
        self.stable_seconds = stable_seconds
        self.stable_joint_range_deg = stable_joint_range_deg
        self.rearm_joint_delta_deg = rearm_joint_delta_deg
        self.armed = True
        self._anchor_qpos: Optional[np.ndarray] = None
        self._stable_since: Optional[float] = None
        self._captured_qpos: Optional[np.ndarray] = None

    def update(self, qpos_rad: np.ndarray, now: float) -> StabilityStatus:
        qpos = np.asarray(qpos_rad, dtype=np.float64)
        rearmed = False
        if not self.armed:
            assert self._captured_qpos is not None
            moved = max_joint_delta_deg(qpos, self._captured_qpos)
            if moved < self.rearm_joint_delta_deg:
                return StabilityStatus(False, False, 0.0, moved)
            self.armed = True
            self._anchor_qpos = qpos.copy()
            self._stable_since = now
            rearmed = True

        if self._anchor_qpos is None or self._stable_since is None:
            self._anchor_qpos = qpos.copy()
            self._stable_since = now

        delta = max_joint_delta_deg(qpos, self._anchor_qpos)
        if delta > self.stable_joint_range_deg:
            self._anchor_qpos = qpos.copy()
            self._stable_since = now
            delta = 0.0
        stable_for = max(0.0, now - self._stable_since)
        return StabilityStatus(
            True,
            stable_for >= self.stable_seconds,
            stable_for,
            delta,
            rearmed,
        )

    def mark_captured(self, qpos_rad: np.ndarray) -> None:
        self.armed = False
        self._captured_qpos = np.asarray(qpos_rad, dtype=np.float64).copy()
        self._anchor_qpos = None
        self._stable_since = None


@dataclass(frozen=True)
class CapturedSample:
    index: int
    timestamp: float
    image_path: Path
    qpos_rad: np.ndarray
    T_base_link7: np.ndarray
    T_camera_charuco: np.ndarray
    charuco_corners: int
    reprojection_error_px: float
    g305_device_timestamp_ms: Optional[float]
    g305_system_timestamp_us: Optional[int]
    robot_state: int
    robot_mode: int
    max_joint_range_deg: float

    def observation(self) -> HandEyeObservation:
        return HandEyeObservation(
            index=self.index,
            T_base_gripper=self.T_base_link7,
            T_camera_target=self.T_camera_charuco,
        )


def sdk_pose_to_transform(pose: list[float]) -> np.ndarray:
    """Convert xArm ``[mm, mm, mm, roll, pitch, yaw]`` to ``base_T_link7``."""
    if len(pose) != 6:
        raise ValueError(f"Expected six FK values, got {pose}")
    values = np.asarray(pose, dtype=np.float64)
    rotation = Rotation.from_euler("xyz", values[3:]).as_matrix()
    return make_T(rotation, values[:3] / 1000.0)


class ReadOnlyXArm7:
    """Read qpos and request FK; never invoke robot mutation APIs."""

    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.arm: Any = None

    def open(self) -> None:
        from xarm.wrapper import XArmAPI

        self.arm = XArmAPI(self.ip, is_radian=True)
        if not self.arm.connected:
            raise RuntimeError(f"xArm is not connected at {self.ip}")
        if int(self.arm.axis) != 7:
            raise RuntimeError(f"Expected xArm7, controller reports axis={self.arm.axis}")
        tcp_offset = np.asarray(self.arm.tcp_offset, dtype=np.float64)
        if tcp_offset.shape != (6,) or not np.allclose(tcp_offset, 0.0, atol=1e-7):
            raise RuntimeError(
                "Controller TCP offset must be zero so FK is the link7 frame; "
                f"got {tcp_offset.tolist()}"
            )

    def close(self) -> None:
        if self.arm is not None:
            self.arm.disconnect()
            self.arm = None

    def read_qpos_once(self) -> np.ndarray:
        if self.arm is None:
            raise RuntimeError("xArm connection is not open")
        code, angles = self.arm.get_servo_angle(is_radian=True)
        if code != 0 or len(angles) != 7:
            raise RuntimeError(
                f"get_servo_angle failed: code={code}, values={angles}"
            )
        return np.asarray(angles, dtype=np.float64)

    def measure_stationary(
        self, max_joint_range_deg: float = QPOS_STABILITY_MAX_DEG
    ) -> RobotMeasurement:
        readings: list[np.ndarray] = []
        for read_index in range(QPOS_STABILITY_READS):
            readings.append(self.read_qpos_once())
            if read_index + 1 < QPOS_STABILITY_READS:
                time.sleep(QPOS_STABILITY_INTERVAL_S)
        stacked = np.stack(readings)
        max_range_deg = max(
            max_joint_delta_deg(reading, readings[0]) for reading in readings
        )
        if max_range_deg > max_joint_range_deg:
            raise RuntimeError(
                f"Robot is still moving: max joint range {max_range_deg:.3f} deg "
                f"> {max_joint_range_deg:.3f} deg"
            )
        qpos = np.median(stacked, axis=0)
        code, pose = self.arm.get_forward_kinematics(
            qpos.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        if code != 0 or len(pose) != 6:
            raise RuntimeError(
                f"get_forward_kinematics failed: code={code}, values={pose}"
            )
        state_code, state = self.arm.get_state()
        if state_code != 0:
            raise RuntimeError(f"get_state failed: code={state_code}")
        return RobotMeasurement(
            qpos_rad=qpos,
            T_base_link7=sdk_pose_to_transform(pose),
            state=int(state),
            mode=int(self.arm.mode),
            max_joint_range_deg=max_range_deg,
        )


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def sample_record(sample: CapturedSample) -> dict[str, Any]:
    return {
        "index": sample.index,
        "timestamp": sample.timestamp,
        "image_path": str(sample.image_path.resolve()),
        "qpos_rad": sample.qpos_rad.tolist(),
        "T_base_link7": sample.T_base_link7.tolist(),
        "T_wuji_g305_raw_left_optical_charuco": (
            sample.T_camera_charuco.tolist()
        ),
        "charuco_corners": sample.charuco_corners,
        "reprojection_error_px": sample.reprojection_error_px,
        "g305_device_timestamp_ms": sample.g305_device_timestamp_ms,
        "g305_system_timestamp_us": sample.g305_system_timestamp_us,
        "robot_state": sample.robot_state,
        "robot_mode": sample.robot_mode,
        "max_joint_range_deg": sample.max_joint_range_deg,
    }


def write_manifest(
    path: Path,
    samples: list[CapturedSample],
    metadata: dict[str, Any],
) -> None:
    atomic_yaml_dump(
        path,
        _serializable(
            {
                "schema": "robot_cam_calib.xarm7_g305_hand_eye_capture.v1",
                "conventions": {
                    "transform": "T_A_B maps B-frame points into A",
                    "robot_pose": "T_base_link7 from qpos and xArm controller FK",
                    "camera_pose": (
                        "T_wuji_g305_raw_left_optical_charuco from ChArUco PnP"
                    ),
                    "requested_output": (
                        "T_link7_wuji_g305_raw_left_optical"
                    ),
                },
                "metadata": metadata,
                "samples": [sample_record(item) for item in samples],
            }
        ),
    )


def load_manifest(path: Path) -> tuple[list[CapturedSample], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if payload.get("schema") != "robot_cam_calib.xarm7_g305_hand_eye_capture.v1":
        raise ValueError(f"Unsupported capture manifest: {resolved}")
    samples: list[CapturedSample] = []
    for record in payload.get("samples", []):
        samples.append(
            CapturedSample(
                index=int(record["index"]),
                timestamp=float(record["timestamp"]),
                image_path=Path(record["image_path"]),
                qpos_rad=np.asarray(record["qpos_rad"], dtype=np.float64),
                T_base_link7=np.asarray(
                    record["T_base_link7"], dtype=np.float64
                ),
                T_camera_charuco=np.asarray(
                    record["T_wuji_g305_raw_left_optical_charuco"],
                    dtype=np.float64,
                ),
                charuco_corners=int(record["charuco_corners"]),
                reprojection_error_px=float(record["reprojection_error_px"]),
                g305_device_timestamp_ms=record.get("g305_device_timestamp_ms"),
                g305_system_timestamp_us=record.get("g305_system_timestamp_us"),
                robot_state=int(record.get("robot_state", -1)),
                robot_mode=int(record.get("robot_mode", -1)),
                max_joint_range_deg=float(record.get("max_joint_range_deg", 0.0)),
            )
        )
    return samples, dict(payload.get("metadata", {}))


def is_diverse(
    samples: list[CapturedSample],
    candidate: np.ndarray,
) -> tuple[bool, str]:
    if not samples:
        return True, "first pose"
    deltas = [transform_delta(item.T_base_link7, candidate) for item in samples]
    nearest_rotation, nearest_translation = min(
        deltas, key=lambda item: item[0] + 100.0 * item[1]
    )
    if (
        nearest_rotation < MIN_POSE_ROTATION_DELTA_DEG
        and nearest_translation < MIN_POSE_TRANSLATION_DELTA_M
    ):
        return False, (
            f"pose too similar: nearest delta {nearest_rotation:.2f} deg, "
            f"{nearest_translation * 1000.0:.1f} mm"
        )
    return True, (
        f"nearest delta {nearest_rotation:.2f} deg, "
        f"{nearest_translation * 1000.0:.1f} mm"
    )


def read_complete_left_frame(
    camera: G305RawLeftCamera,
    *,
    attempts: int = COMPLETE_FRAME_ATTEMPTS,
) -> tuple[np.ndarray, Optional[float], Optional[int]]:
    """Skip startup/incomplete stereo framesets, but preserve a hard bound."""
    last_error = "no frames received"
    for _attempt in range(attempts):
        try:
            return camera.read_bgr()
        except RuntimeError as exc:
            last_error = str(exc)
            if "no raw left color frame" not in last_error:
                raise
    raise RuntimeError(
        f"No complete G305 raw-left frameset in {attempts} attempts: "
        f"{last_error}"
    )


def solve_and_save(
    samples: list[CapturedSample],
    metadata: dict[str, Any],
    output_path: Path,
    source_manifest: Path,
) -> Path:
    solution = solve_hand_eye_robust(
        [item.observation() for item in samples], min_samples=MIN_SAMPLES
    )
    actual_output = append_timestamp(output_path.expanduser().resolve())
    payload = {
        "schema": "robot_cam_calib.xarm7_g305_eye_in_hand.v1",
        "status": "candidate_requires_physical_validation",
        "conventions": {
            "transform": "T_A_B maps B-frame points into A",
            "output": "T_link7_wuji_g305_raw_left_optical",
            "equation": (
                "T_base_link7_i @ T_link7_wuji_g305_raw_left_optical @ "
                "T_wuji_g305_raw_left_optical_charuco_i = T_base_charuco"
            ),
        },
        "T_link7_wuji_g305_raw_left_optical": solution[
            "T_gripper_camera"
        ],
        "T_base_charuco_mean": solution["T_base_target_mean"],
        "solver": {
            key: value
            for key, value in solution.items()
            if key not in {"T_gripper_camera", "T_base_target_mean"}
        },
        "capture_manifest": str(source_manifest.resolve()),
        "metadata": metadata,
        "sample_count": len(samples),
        "samples": [sample_record(item) for item in samples],
    }
    atomic_yaml_dump(actual_output, _serializable(payload))
    print(f"[RESULT] method={solution['method']}")
    print("[RESULT] T_link7_wuji_g305_raw_left_optical:")
    print(solution["T_gripper_camera"])
    print(
        "[DIAGNOSTICS] inliers={} outliers={} median={:.3f}deg/{:.2f}mm rank={}".format(
            len(solution["inlier_indices"]),
            solution["outlier_indices"],
            solution["rotation_stats_deg"]["median"],
            1000.0 * solution["translation_stats_m"]["median"],
            solution["excitation"]["relative_rotation_rank"],
        )
    )
    print(f"[INFO] Saved {actual_output}")
    return actual_output


def profile_metadata(profile: Any, board_path: Path, board: dict[str, Any], robot: ReadOnlyXArm7) -> dict[str, Any]:
    arm = robot.arm
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "robot": {
            "ip": robot.ip,
            "axis": int(arm.axis),
            "version": str(arm.version),
            "tcp_offset": list(arm.tcp_offset),
            "hardware_writes": 0,
            "qpos_units": "rad",
            "fk_translation_units": "m",
            "fk_source": "xArm get_forward_kinematics(qpos)",
        },
        "camera": {
            "frame": "wuji_g305_raw_left_optical",
            "serial": profile.serial,
            "device_name": profile.device_name,
            "firmware": profile.firmware,
            "connection_type": profile.connection_type,
            "previous_work_mode": profile.previous_work_mode,
            "capture_work_mode": profile.active_work_mode,
            "profile": (
                f"{profile.width}x{profile.height}@{profile.fps} "
                f"{profile.format_name}"
            ),
            "K": profile.K.tolist(),
            "dist": profile.dist.tolist(),
            "intrinsics_source": profile.intrinsics_source,
        },
        "target": {
            "path": str(board_path.resolve()),
            **board,
            "stationary_relative_to": "xarm_base",
        },
    }


def live_capture(args: argparse.Namespace) -> None:
    board, detector, board_config = load_charuco_target(args.charuco_board)
    camera = G305RawLeftCamera(
        serial=args.g305_serial,
        width=args.g305_width,
        height=args.g305_height,
        fps=args.g305_fps,
        format_name=args.g305_format,
        work_mode=args.g305_work_mode,
        timeout_ms=args.frame_timeout_ms,
    )
    robot = ReadOnlyXArm7(args.robot_ip)
    samples: list[CapturedSample] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = args.sample_root.expanduser().resolve() / stamp
    sample_dir.mkdir(parents=True, exist_ok=False)
    manifest = sample_dir / "capture_manifest.yaml"
    profile = None
    gate = StabilityGate(
        stable_seconds=args.stable_seconds,
        stable_joint_range_deg=args.stable_joint_range_deg,
        rearm_joint_delta_deg=args.rearm_joint_delta_deg,
    )
    try:
        robot.open()
        profile = camera.open()
        intrinsics = profile.as_intrinsics()
        metadata = profile_metadata(
            profile, args.charuco_board, board_config, robot
        )
        metadata["capture"] = {
            "automatic": args.auto_capture,
            "stable_seconds": args.stable_seconds,
            "stable_joint_range_deg": args.stable_joint_range_deg,
            "rearm_joint_delta_deg": args.rearm_joint_delta_deg,
            "manual_key": "s",
        }
        write_manifest(manifest, samples, metadata)
        print(
            "[INFO] Robot is read-only. Move it in the xArm web UI; after "
            f"{args.stable_seconds:.2f}s stable, capture is automatic. "
            "Press s for a manual capture or q/esc to solve."
        )
        print(
            f"[INFO] Need at least {MIN_SAMPLES}, target={args.samples}; "
            "use rotations around multiple axes."
        )
        while True:
            frame, device_timestamp, system_timestamp = read_complete_left_frame(
                camera
            )
            detection = detect_charuco_pose(
                frame,
                detector,
                board,
                intrinsics,
                "G305 raw-left/ChArUco",
            )
            qpos = robot.read_qpos_once()
            stability = gate.update(qpos, time.monotonic())
            visual = detection.vis if detection.vis is not None else frame.copy()
            color = (0, 255, 0) if detection.ok else (0, 0, 255)
            if stability.armed:
                stability_text = (
                    f"AUTO stable {min(stability.stable_for_s, args.stable_seconds):.2f}/"
                    f"{args.stable_seconds:.2f}s "
                    f"(range {stability.max_joint_delta_deg:.3f}deg)"
                )
            else:
                stability_text = (
                    "AUTO waiting for move "
                    f"{stability.max_joint_delta_deg:.2f}/"
                    f"{args.rearm_joint_delta_deg:.2f}deg"
                )
            lines = [
                f"samples {len(samples)}/{args.samples}",
                detection.message,
                stability_text if args.auto_capture else "manual capture mode",
                "move in web UI, hold | auto capture | s manual | q solve",
            ]
            for row, line in enumerate(lines):
                cv2.putText(
                    visual,
                    line,
                    (12, 30 + 30 * row),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color if row == 1 else (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if args.preview:
                display = cv2.resize(
                    visual,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow("xArm7 G305 eye-in-hand", display)
                key = cv2.waitKey(1) & 0xFF
            else:
                command = input("Enter=capture, q=solve: ").strip().lower()
                key = ord("q") if command == "q" else ord("s")
            if key in (ord("q"), 27):
                break
            manual_requested = key == ord("s")
            automatic_requested = args.auto_capture and stability.ready
            if not manual_requested and not automatic_requested:
                continue
            if manual_requested and args.auto_capture and not stability.ready:
                print(
                    "[REJECT] Wait until the stability indicator reaches "
                    f"{args.stable_seconds:.2f}s"
                )
                continue
            if not detection.ok or detection.T is None:
                if manual_requested:
                    print(f"[REJECT] {detection.message}")
                continue
            if detection.reproj_error > args.max_reprojection_error:
                if manual_requested:
                    print(
                        f"[REJECT] reprojection {detection.reproj_error:.3f}px > "
                        f"{args.max_reprojection_error:.3f}px"
                    )
                continue
            try:
                measurement = robot.measure_stationary(
                    args.stable_joint_range_deg
                )
            except RuntimeError as exc:
                print(f"[REJECT] {exc}")
                continue
            trigger_delta = max_joint_delta_deg(measurement.qpos_rad, qpos)
            if trigger_delta > args.stable_joint_range_deg:
                print(
                    f"[REJECT] Robot moved {trigger_delta:.3f} deg during "
                    "capture verification"
                )
                continue
            diverse, diversity_message = is_diverse(
                samples, measurement.T_base_link7
            )
            if not diverse:
                print(f"[REJECT] {diversity_message}")
                gate.mark_captured(measurement.qpos_rad)
                continue
            index = len(samples)
            image_path = sample_dir / f"sample_{index:04d}_g305_raw_left.png"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Failed to save {image_path}")
            sample = CapturedSample(
                index=index,
                timestamp=time.time(),
                image_path=image_path,
                qpos_rad=measurement.qpos_rad,
                T_base_link7=measurement.T_base_link7,
                T_camera_charuco=detection.T,
                charuco_corners=detection.n_points,
                reprojection_error_px=detection.reproj_error,
                g305_device_timestamp_ms=device_timestamp,
                g305_system_timestamp_us=system_timestamp,
                robot_state=measurement.state,
                robot_mode=measurement.mode,
                max_joint_range_deg=measurement.max_joint_range_deg,
            )
            samples.append(sample)
            gate.mark_captured(measurement.qpos_rad)
            write_manifest(manifest, samples, metadata)
            print(
                f"[ACCEPT] sample={index} corners={detection.n_points} "
                f"err={detection.reproj_error:.3f}px "
                f"stable={measurement.max_joint_range_deg:.3f}deg "
                f"{diversity_message}"
            )
            if len(samples) >= args.samples:
                break
        if len(samples) < MIN_SAMPLES:
            print(
                f"[INFO] Only {len(samples)} samples; manifest saved at {manifest}. "
                f"Need at least {MIN_SAMPLES} to solve."
            )
            return
        solve_and_save(samples, metadata, args.output, manifest)
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        camera.close()
        robot.close()


def check_hardware(args: argparse.Namespace) -> None:
    board, detector, board_config = load_charuco_target(args.charuco_board)
    del board_config
    camera = G305RawLeftCamera(
        serial=args.g305_serial,
        width=args.g305_width,
        height=args.g305_height,
        fps=args.g305_fps,
        format_name=args.g305_format,
        work_mode=args.g305_work_mode,
        timeout_ms=args.frame_timeout_ms,
    )
    robot = ReadOnlyXArm7(args.robot_ip)
    try:
        robot.open()
        profile = camera.open()
        frame, device_timestamp, _system_timestamp = read_complete_left_frame(
            camera
        )
        measurement = robot.measure_stationary()
        detection = detect_charuco_pose(
            frame,
            detector,
            board,
            profile.as_intrinsics(),
            "G305 raw-left/ChArUco",
        )
        print(
            f"[OK] xArm7 qpos/FK read-only: state={measurement.state} "
            f"mode={measurement.mode} stable={measurement.max_joint_range_deg:.3f}deg"
        )
        print(f"[OK] qpos_rad={measurement.qpos_rad.tolist()}")
        print(f"[OK] T_base_link7=\n{measurement.T_base_link7}")
        print(
            f"[OK] G305 serial={profile.serial} profile="
            f"{profile.width}x{profile.height}@{profile.fps} {profile.format_name} "
            f"device_timestamp_ms={device_timestamp}"
        )
        print(f"[OK] G305 raw-left K={profile.K.tolist()}")
        print(f"[OK] G305 raw-left dist={profile.dist.tolist()}")
        print(f"[INFO] board detection: {detection.message}")
        print("[OK] robot hardware writes: 0")
    finally:
        camera.close()
        robot.close()


def run_self_test() -> None:
    rng = np.random.default_rng(20260817)
    expected = make_T(
        Rotation.from_rotvec([0.18, -0.12, 0.07]).as_matrix(),
        [0.035, -0.022, 0.081],
    )
    base_target = make_T(
        Rotation.from_rotvec([-0.1, 0.05, 0.2]).as_matrix(),
        [0.55, 0.04, 0.12],
    )
    observations: list[HandEyeObservation] = []
    for index in range(24):
        base_gripper = make_T(
            Rotation.from_rotvec(rng.uniform(-0.8, 0.8, 3)).as_matrix(),
            rng.uniform([-0.2, -0.2, 0.25], [0.55, 0.35, 0.75]),
        )
        camera_target = (
            np.linalg.inv(expected) @ np.linalg.inv(base_gripper) @ base_target
        )
        observations.append(
            HandEyeObservation(index, base_gripper, camera_target)
        )
    solution = solve_hand_eye_robust(observations, min_samples=12)
    rotation, translation = transform_delta(
        expected, solution["T_gripper_camera"]
    )
    if rotation > 1e-4 or translation > 1e-6:
        raise AssertionError(
            f"Synthetic hand-eye mismatch: {rotation} deg, {translation} m"
        )
    print(
        f"[SELF-TEST] PASS method={solution['method']} "
        f"delta={rotation:.3e}deg/{translation:.3e}m"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.2.213")
    parser.add_argument("--g305-serial", default="CV2T661000NC")
    parser.add_argument("--g305-work-mode", default="Dual Color Streams")
    parser.add_argument("--g305-width", type=int, default=1280)
    parser.add_argument("--g305-height", type=int, default=800)
    parser.add_argument("--g305-fps", type=int, default=20)
    parser.add_argument("--g305-format", default="RGB")
    parser.add_argument("--frame-timeout-ms", type=int, default=1500)
    parser.add_argument("--charuco-board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--auto-capture", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--stable-seconds", type=float, default=AUTO_STABLE_SECONDS
    )
    parser.add_argument(
        "--stable-joint-range-deg", type=float, default=QPOS_STABILITY_MAX_DEG
    )
    parser.add_argument(
        "--rearm-joint-delta-deg",
        type=float,
        default=AUTO_REARM_JOINT_DELTA_DEG,
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=MAX_REPROJECTION_ERROR_PX,
    )
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--offline-manifest", type=Path)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--check-hardware", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.samples < MIN_SAMPLES:
        raise SystemExit(f"--samples must be >= {MIN_SAMPLES}")
    if args.stable_seconds <= 0:
        raise SystemExit("--stable-seconds must be > 0")
    if args.stable_joint_range_deg <= 0:
        raise SystemExit("--stable-joint-range-deg must be > 0")
    if args.rearm_joint_delta_deg <= args.stable_joint_range_deg:
        raise SystemExit(
            "--rearm-joint-delta-deg must be greater than "
            "--stable-joint-range-deg"
        )
    board, _detector, config = load_charuco_target(args.charuco_board)
    del board
    if args.self_test:
        run_self_test()
        return
    if args.check_config:
        print(f"[OK] ChArUco {args.charuco_board.resolve()}: {config}")
        print("[OK] xArm path is read-only: qpos + FK, no motion commands")
        return
    if args.check_hardware:
        check_hardware(args)
        return
    if args.offline_manifest is not None:
        samples, metadata = load_manifest(args.offline_manifest)
        solve_and_save(
            samples,
            metadata,
            args.output,
            args.offline_manifest,
        )
        return
    if args.auto_capture and not args.preview:
        raise SystemExit("automatic capture requires the preview window")
    live_capture(args)


if __name__ == "__main__":
    main()
