"""Local OpenCV capture for a fixed board and a manually moved right arm.

The front D435 and right-wrist D435 must see the same fixed ChArUco board.
Stable, novel right-arm poses are buffered in memory. Press q to save them.
No actuator command is sent and no camera device is opened by this process.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PYREALSENSE_SITE = Path(
    "/home/agilex/miniconda3/envs/evorl-ljy/lib/python3.11/site-packages"
)
if PYREALSENSE_SITE.is_dir():
    sys.path.append(str(PYREALSENSE_SITE))

from manual_calibration_gui import next_index
from piper_charuco import clean, detect, sha
from right_wrist_feedback import FeedbackStream
from robot_cam_calib.geometry import transform_delta

H = Path(__file__).resolve().parent
CONFIG = Path(
    "/home/agilex/yichen/rsi_sim2real/releases/"
    "agx-stack-cube-20261018/existing-hardware-binding.json"
)
FACTORY = CONFIG.parent / "calibration-v2/factory-current-color-intrinsics.json"
RIGHT_REFERENCE = H / "exports/agx_right_wrist_extri/agx_right_wrist_extri.json"
WINDOW = "Front Extrinsic from Moving Right Wrist"
CAMERAS = {
    "front": {
        "serial": "239622301704",
        "namespace": "camera_f",
        "info": Path("/dev/shm/front-d435-color-info.json"),
        "shm": Path("/dev/shm/front-d435-color.bin"),
        "magic": b"FSD435\0\0",
    },
    "right_wrist": {
        "serial": "346522074314",
        "namespace": "camera_r",
        "info": Path("/dev/shm/right-d435-color-info.json"),
        "shm": Path("/dev/shm/right-d435-color.bin"),
        "magic": b"RSD435\0\0",
    },
}


class RelaxedStationaryGate:
    """Practical manual gate while preserving health and pose diversity."""

    def __init__(self):
        self.anchor = None
        self.since = 0.0

    def check(self, state, view, pose, known, drift, now):
        joints = np.asarray(state.get("joint_deg", []), dtype=float)
        healthy = (
            state.get("complete")
            and not state.get("stale")
            and state.get("error_code") == 0
            and state.get("arm_status") == 0
            and joints.shape == (6,)
            and np.all(np.isfinite(joints))
        )
        if not healthy:
            self.anchor = None
            return False, "等待右臂反馈正常"
        if drift > 0.15:
            self.anchor = None
            return False, "正在移动，等待停稳"
        if not view.get("metric_pose_eligible"):
            self.anchor = None
            return False, "等待右腕看清标定板（至少 8 个角点且质量合格）"
        if self.anchor is None or np.max(np.abs(joints - self.anchor)) > 0.08:
            self.anchor = joints.copy()
            self.since = now
            return False, "检测到新姿态，等待持续停稳"
        if now - self.since < 0.5:
            return False, "等待停稳至少 0.5 秒"
        if any(
            rotation < 3.0 and translation < 0.006
            for rotation, translation in (
                transform_delta(old, pose) for old in known
            )
        ):
            return False, "此姿态已采集，请换一个位置或角度"
        return True, "姿态稳定且图像合格，自动保存中"


def healthy_right(states):
    state = states.get("right", {})
    joints = np.asarray(state.get("joint_deg", []), dtype=float)
    return bool(
        state.get("complete")
        and not state.get("stale")
        and state.get("error_code") == 0
        and state.get("arm_status") == 0
        and joints.shape == (6,)
        and np.all(np.isfinite(joints))
    )


def relaxed_frame_feedback(history, stamp, now, since=0.0):
    """Pair a camera frame with feedback using manual-calibration tolerances."""
    if not history or not np.isfinite(stamp) or not -0.08 <= now - stamp <= 0.35:
        return None, float("inf"), None
    latest = history[-1]
    if not 0.0 <= now - latest["epoch"] <= 0.25:
        return None, float("inf"), None
    start = max(
        0,
        next(
            (i for i, row in enumerate(history) if row["epoch"] >= since),
            len(history),
        )
        - 1,
    )
    recent = history[start:]
    continuous = all(
        0.0 <= after["epoch"] - before["epoch"] <= 0.25
        for before, after in zip(recent, recent[1:])
    )
    drift = float("inf")
    if len(recent) >= 2 and continuous and all(
        healthy_right(row["states"]) for row in recent
    ):
        joints = np.asarray(
            [row["states"]["right"]["joint_deg"] for row in recent]
        )
        drift = float(np.max(np.ptp(joints, axis=0)))
    before = next(
        (row for row in reversed(history) if row["epoch"] <= stamp), None
    )
    after = next((row for row in history if row["epoch"] >= stamp), None)
    pair = None
    if before is not None and after is not None:
        rows = (before, after)
        if all(
            abs(row["epoch"] - stamp) <= 0.20
            and healthy_right(row["states"])
            and all(abs(epoch - stamp) <= 0.20 for epoch in row["joint_epochs"])
            for row in rows
        ):
            joints = [
                np.asarray(row["states"]["right"]["joint_deg"])
                for row in rows
            ]
            joint_drift = float(np.max(np.abs(joints[1] - joints[0])))
            if joint_drift <= 0.15:
                pair = {
                    "states": after["states"],
                    "feedback_before_epoch": before["epoch"],
                    "feedback_after_epoch": after["epoch"],
                    "joint_feedback_epochs": [
                        row["joint_epochs"] for row in rows
                    ],
                    "joint_drift_deg": joint_drift,
                    "thresholds": {
                        "feedback_pair_window_s": 0.20,
                        "joint_pair_drift_deg": 0.15,
                        "stability_drift_deg": 0.15,
                        "hold_seconds": 0.5,
                    },
                }
    return latest, drift, pair


def read_frame(spec, after_seq=None):
    header_size = struct.calcsize("<8sQdIII")
    with spec["shm"].open("rb") as handle:
        header = handle.read(header_size)
        if len(header) != header_size:
            raise ValueError("Partial frame header for " + spec["namespace"])
        magic, seq, stamp, height, width, channels = struct.unpack(
            "<8sQdIII", header
        )
        if magic != spec["magic"] or channels != 3:
            raise ValueError("Unexpected frame header for " + spec["namespace"])
        if seq == after_seq:
            return int(seq), float(stamp), None
        payload = handle.read()
    image = np.frombuffer(payload, dtype=np.uint8)
    if image.size != height * width * channels:
        raise ValueError("Partial frame for " + spec["namespace"])
    return int(seq), float(stamp), image.reshape(height, width, channels).copy()


def link_pose(kinematics, joints):
    pose = kinematics.transform(np.asarray(joints, dtype=float))
    pose[:3, 3] /= 1000.0
    return pose


def disk_poses(root):
    poses = []
    for path in sorted(root.glob("right-*/sample.json")):
        sample = json.loads(path.read_text())
        if sample.get("active_arm") == "right":
            poses.append(
                np.asarray(sample["T_base_link6_m"]["right"], dtype=float)
            )
    return poses


def session_identity():
    reference = json.loads(RIGHT_REFERENCE.read_text())
    if reference.get("accepted_for_control") is not True:
        raise ValueError("Accepted right-wrist reference is required")
    return {
        "schema": "agx.front_from_right_fixed_board_capture.v1",
        "board_yaml_sha256": sha(H / "board.yaml"),
        "config_sha256": sha(CONFIG),
        "factory_profiles_sha256": sha(FACTORY),
        "capture_gui_sha256": sha(Path(__file__)),
        "frame_server_sha256": sha(H / "front_right_frame_server.py"),
        "right_reference_path": str(RIGHT_REFERENCE),
        "right_reference_sha256": sha(RIGHT_REFERENCE),
        "right_reference_frame": reference["frame"],
        "right_reference_name": reference["name"],
        "front_camera_serial": CAMERAS["front"]["serial"],
        "right_camera_serial": CAMERAS["right_wrist"]["serial"],
        "units": "metres",
        "split_rule": "index modulo 4 equals 3 -> holdout; otherwise train",
        "board_fixed_during_collection": True,
        "front_camera_fixed_during_collection": True,
        "capture_thresholds": {
            "camera_stamp_span_s": 0.08,
            "feedback_pair_window_s": 0.20,
            "joint_pair_drift_deg": 0.15,
            "stability_drift_deg": 0.15,
            "hold_seconds": 0.5,
            "pose_novelty_rotation_deg": 3.0,
            "pose_novelty_translation_m": 0.006,
            "minimum_charuco_corners_each_camera": 8,
        },
    }


def ensure_identity(root):
    expected = session_identity()
    path = root / "identity.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("Session source, board, camera, or reference changed")
    else:
        with path.open("x") as handle:
            json.dump(expected, handle, indent=2)
            handle.write("\n")
    return expected


def metadata(spec, info, stamp):
    return {
        "serial": spec["serial"],
        "role": "front" if spec["namespace"] == "camera_f" else "right_wrist",
        "ros_namespace": spec["namespace"],
        "color": {
            "width": info["width"],
            "height": info["height"],
            "frame_id": info["frame_id"],
            "stamp": stamp,
            "distortion_model": info["distortion_model"],
            "D": info["D"],
            "K": info["K"],
        },
        "color_stamp": stamp,
        "depth_registered_to_rgb": False,
        "capture": "synchronized_memory_buffer_flushed_on_q",
        "robot_camera_extrinsic": None,
        "old_extrinsics_reused_for_detection": False,
    }


def write_group(root, kinematics, profiles, identity, infos, item):
    index = next_index(root, "right")
    slot = root / ("right-%04d" % index)
    slot.mkdir()
    files = {}
    views = {}
    for role in ("front", "right_wrist"):
        raw = slot / "raw" / role
        raw.mkdir(parents=True)
        image_path = raw / "rgb.jpg"
        metadata_path = raw / "metadata.json"
        if not cv2.imwrite(str(image_path), item["images"][role]):
            raise RuntimeError("Could not write " + role + " RGB")
        metadata_path.write_text(
            json.dumps(
                metadata(CAMERAS[role], infos[role], item["stamps"][role]),
                indent=2,
            )
            + "\n"
        )
        views[role] = detect(raw, profiles)
        files[str(image_path.relative_to(slot))] = sha(image_path)
        files[str(metadata_path.relative_to(slot))] = sha(metadata_path)

    joints_path = slot / "joints.json"
    joints_path.write_text(json.dumps(clean(item["states"]), indent=2) + "\n")
    files["joints.json"] = sha(joints_path)
    poses = {}
    for arm in ("left", "right"):
        state = item["states"].get(arm)
        if state and state.get("joint_deg"):
            poses[arm] = link_pose(kinematics, state["joint_deg"]).tolist()
    split = "holdout" if index % 4 == 3 else "train"
    sample = {
        "index": index,
        "split": split,
        "active_arm": "right",
        "source": "fixed_board_front_from_moving_right_wrist_buffered",
        "identity": identity,
        "T_base_link6_m": poses,
        "cameras": views,
        "joint_deg_right": item["states"]["right"]["joint_deg"],
        "image_stamps": item["stamps"],
        "camera_stamp_span_s": abs(
            item["stamps"]["front"] - item["stamps"]["right_wrist"]
        ),
        "buffer_epoch": item["epoch"],
        "capture_sync": item["sync"],
        "source_files_sha256": files,
        "right_reference_extrinsic": {
            "path": str(RIGHT_REFERENCE),
            "sha256": sha(RIGHT_REFERENCE),
        },
        "target_transform": "T_base_front_color_optical",
        "board_fixed_during_collection": True,
        "front_camera_fixed_during_collection": True,
        "robot_motion_commands_sent_by_collector": 0,
        "old_observations_used": False,
        "simulator_information_used": False,
        "physical_board_scale_source": "user confirmed 100 percent print",
    }
    sample_path = slot / "sample.json"
    sample_path.write_text(json.dumps(clean(sample), indent=2) + "\n")
    return {
        "index": index,
        "split": split,
        "front_corners": views["front"].get("corner_count", 0),
        "right_corners": views["right_wrist"].get("corner_count", 0),
        "front_eligible": bool(views["front"].get("metric_pose_eligible")),
        "right_eligible": bool(
            views["right_wrist"].get("metric_pose_eligible")
        ),
    }


def board_overlay(detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    vis = image.copy()
    if marker_corners is not None and marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    count = 0 if ids is None else int(len(ids))
    if corners is not None and ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
    return vis, count, 0 if marker_ids is None else int(len(marker_ids))


def panel(image, width=640, height=480):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def put_lines(image, lines, x, color=(0, 255, 255)):
    for row, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, 28 + row * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color if row else (40, 230, 40),
            2,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session", default="manual-20261019-front-from-right-v1"
    )
    args = parser.parse_args()
    root = H / "sessions" / args.session
    root.mkdir(parents=True, exist_ok=True)
    identity = ensure_identity(root)
    infos = {
        role: json.loads(spec["info"].read_text())
        for role, spec in CAMERAS.items()
    }
    profiles = json.loads(FACTORY.read_text())["profiles"]
    config = json.loads(CONFIG.read_text())
    from agilex_control.kinematics import Kinematics

    kinematics = Kinematics(config)
    board = cv2.aruco.CharucoBoard(
        (7, 5),
        0.04,
        0.03,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    )
    board.setLegacyPattern(False)
    detector = cv2.aruco.CharucoDetector(board)
    cv2.setNumThreads(4)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WINDOW, 1360, 620)

    feedback = FeedbackStream(config)
    feedback.start()
    gate = RelaxedStationaryGate()
    known = disk_poses(root)
    buffer = []
    latest = {
        role: {"seq": -1, "stamp": 0.0, "image": None}
        for role in CAMERAS
    }
    feedback_since = 0.0
    save_on_exit = False
    notice = "move right arm, hold still; q saves, Esc cancels"
    try:
        while True:
            changed = False
            for role, spec in CAMERAS.items():
                try:
                    seq, stamp, image = read_frame(spec, latest[role]["seq"])
                except (FileNotFoundError, ValueError):
                    continue
                if image is not None:
                    latest[role] = {"seq": seq, "stamp": stamp, "image": image}
                    changed = True
            if not changed or any(latest[role]["image"] is None for role in CAMERAS):
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    save_on_exit = True
                    break
                if key == 27:
                    break
                continue

            now = time.time()
            stamps = {role: latest[role]["stamp"] for role in CAMERAS}
            fresh = all(-0.08 <= now - stamp <= 0.35 for stamp in stamps.values())
            camera_delta = abs(stamps["front"] - stamps["right_wrist"])
            rendered = {}
            corners = {}
            markers = {}
            for role in CAMERAS:
                rendered[role], corners[role], markers[role] = board_overlay(
                    detector, latest[role]["image"]
                )

            history, feedback_error = feedback.snapshot()
            measured, drift, pair = relaxed_frame_feedback(
                history, stamps["right_wrist"], now, feedback_since
            )
            states = None if measured is None else measured["states"]
            if measured is not None:
                feedback_since = measured["epoch"]
            right = None if states is None else states.get("right")
            joints = None
            pose = None
            if right is not None and right.get("joint_deg"):
                joints = np.asarray(right["joint_deg"], dtype=float)
                pose = link_pose(kinematics, joints)

            ready = False
            if not fresh:
                gate.anchor = None
                status = "waiting for fresh camera frames"
            elif camera_delta > 0.08:
                gate.anchor = None
                status = "waiting for synchronized cameras"
            elif corners["front"] < 8:
                gate.anchor = None
                status = "front camera needs a clear board"
            elif feedback_error or right is None or joints is None:
                gate.anchor = None
                status = "waiting for right-arm feedback"
            else:
                ready, reason = gate.check(
                    right,
                    {"metric_pose_eligible": corners["right_wrist"] >= 8},
                    pose,
                    known,
                    drift,
                    time.monotonic(),
                )
                status = {
                    "等待右臂失能且反馈正常": "AUTO waiting for healthy right-arm feedback",
                    "等待右臂反馈正常": "AUTO waiting for healthy right-arm feedback",
                    "正在移动，等待停稳": "AUTO moving",
                    "等待右腕看清标定板（至少 8 个角点且质量合格）": "right wrist needs a clear board",
                    "检测到新姿态，等待持续停稳": "AUTO new pose, hold still",
                    "等待停稳至少 0.5 秒": "AUTO holding for 0.5 second",
                    "此姿态已采集，请换一个位置或角度": "AUTO pose already buffered",
                    "此姿态采样未通过，请稍作调整后停稳": "AUTO adjust pose, then hold",
                    "正在检查上一组采样结果": "AUTO checking",
                    "姿态稳定且图像合格，自动保存中": "AUTO buffering",
                }.get(reason, "AUTO waiting")
                if ready and pair is None:
                    ready = False
                    status = "matching image and joint timestamps"

            if ready and pair is not None:
                buffer.append(
                    {
                        "images": {
                            role: latest[role]["image"].copy()
                            for role in CAMERAS
                        },
                        "stamps": stamps.copy(),
                        "states": pair["states"],
                        "epoch": time.time(),
                        "sync": {
                            **{key: value for key, value in pair.items() if key != "states"},
                            "camera_stamp_span_s": camera_delta,
                        },
                    }
                )
                known.append(
                    link_pose(kinematics, pair["states"]["right"]["joint_deg"])
                )
                notice = "buffered %d; move to a new pose; q saves" % len(buffer)

            left_panel = panel(rendered["front"])
            right_panel = panel(rendered["right_wrist"])
            canvas = np.hstack([left_panel, right_panel])
            put_lines(
                canvas,
                [
                    "FRONT  corners %d/24" % corners["front"],
                    "serial " + CAMERAS["front"]["serial"],
                ],
                12,
            )
            put_lines(
                canvas,
                [
                    "RIGHT WRIST  corners %d/24" % corners["right_wrist"],
                    "serial " + CAMERAS["right_wrist"]["serial"],
                ],
                652,
            )
            bottom = np.zeros((118, canvas.shape[1], 3), dtype=np.uint8)
            put_lines(
                bottom,
                [
                    "BUFFERED %d   disk %d   camera dt %.1f ms"
                    % (len(buffer), len(known) - len(buffer), camera_delta * 1000.0),
                    status,
                    notice,
                    ("drive ENABLED   " if right and any(right["enabled_drive_bits"]) else "drive DISABLED  ")
                    + "joints "
                    + (
                        " ".join("%.2f" % value for value in joints)
                        if joints is not None
                        else "none"
                    ),
                ],
                12,
                (0, 255, 255),
            )
            cv2.imshow(WINDOW, np.vstack([canvas, bottom]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                save_on_exit = True
                break
            if key == 27:
                break
    finally:
        feedback.close()

    results = []
    if save_on_exit and buffer:
        blank = np.zeros((220, 900, 3), dtype=np.uint8)
        for number, item in enumerate(buffer, 1):
            progress = blank.copy()
            cv2.putText(
                progress,
                "detecting and saving %d / %d" % (number, len(buffer)),
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW, progress)
            cv2.waitKey(1)
            results.append(
                write_group(root, kinematics, profiles, identity, infos, item)
            )
    cv2.destroyAllWindows()
    print(
        json.dumps(
            clean(
                {
                    "buffered": len(buffer),
                    "saved": len(results),
                    "cancelled": not save_on_exit,
                    "written": results,
                    "session": str(root),
                    "robot_motion_commands_sent": 0,
                }
            )
        )
    )


if __name__ == "__main__":
    main()
