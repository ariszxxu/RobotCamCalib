"""Local OpenCV window for left-wrist D435 extrinsic capture.

Stable poses are copied into memory immediately. Press q to detect the
ChArUco board and write every buffered group. No actuator command is sent.
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# The GUI environment provides desktop OpenCV; the deployed RealSense wheel is
# shared read-only from the Python 3.11 control environment for ray geometry.
PYREALSENSE_SITE = Path("/home/agilex/miniconda3/envs/evorl-ljy/lib/python3.11/site-packages")
if PYREALSENSE_SITE.is_dir():
    sys.path.append(str(PYREALSENSE_SITE))

from manual_calibration_gui import AutoGate, next_index
from piper_charuco import clean, detect, sha
from left_wrist_feedback import FeedbackStream, frame_feedback

H = Path(__file__).resolve().parent
CONFIG = Path("/home/agilex/yichen/rsi_sim2real/releases/agx-stack-cube-20261018/existing-hardware-binding.json")
FACTORY = CONFIG.parent / "calibration-v2/factory-current-color-intrinsics.json"
INFO = Path("/dev/shm/left-d435-color-info.json")
SHM = Path("/dev/shm/left-d435-color.bin")
WINDOW = "Left D435 ChArUco"
MAGIC = b"LSD435\0\0"
LEFT_SERIAL = "346522074444"


def read_frame(after_seq=None):
    # The server atomically replaces the file. One open handle keeps header and
    # pixels together, while unchanged frames require only a 36-byte header read.
    with SHM.open("rb") as handle:
        header = handle.read(struct.calcsize("<8sQdIII"))
        if len(header) < struct.calcsize("<8sQdIII"):
            raise ValueError("Partial left-wrist frame header")
        magic, seq, stamp, height, width, channels = struct.unpack("<8sQdIII", header)
        if magic != MAGIC or channels != 3:
            raise ValueError("Unexpected left-wrist frame header")
        if seq == after_seq:
            return int(seq), float(stamp), None
        payload = handle.read()
    image = np.frombuffer(payload, dtype=np.uint8)
    if image.size != height * width * channels:
        raise ValueError("Partial left-wrist frame")
    return int(seq), float(stamp), image.reshape(height, width, channels).copy()


def disk_poses(root):
    poses = []
    for path in sorted(root.glob("left-*/sample.json")):
        sample = json.loads(path.read_text())
        wrist = sample.get("cameras", {}).get("left_wrist", {})
        if sample.get("active_arm") == "left" and wrist.get("metric_pose_eligible"):
            poses.append(np.asarray(sample["T_base_link6_m"]["left"], dtype=float))
    return poses


def link_pose(kinematics, joints):
    pose = kinematics.transform(np.asarray(joints, dtype=float))
    pose[:3, 3] /= 1000
    return pose


def write_group(root, kinematics, profiles, identity, info, item):
    index = next_index(root, "left")
    slot = root / ("left-%04d" % index)
    raw = slot / "raw" / "left_wrist"
    raw.mkdir(parents=True)
    if not cv2.imwrite(str(raw / "rgb.jpg"), item["image"]):
        raise RuntimeError("Could not write buffered RGB")
    stamp = item["stamp"]
    metadata = {
        "serial": LEFT_SERIAL,
        "role": "left_wrist",
        "ros_namespace": "camera_l",
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
        "capture": "memory_buffer_flushed_on_q",
        "robot_camera_extrinsic": None,
        "old_extrinsics_reused": False,
    }
    (raw / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (slot / "joints.json").write_text(json.dumps(clean(item["states"]), indent=2) + "\n")
    view = detect(raw, profiles)
    poses = {}
    for arm in ("left",):
        state = item["states"].get(arm)
        if state and state.get("joint_deg"):
            poses[arm] = link_pose(kinematics, state["joint_deg"]).tolist()
    split = "holdout" if index % 4 == 3 else "train"
    files = {
        "raw/left_wrist/rgb.jpg": sha(raw / "rgb.jpg"),
        "raw/left_wrist/metadata.json": sha(raw / "metadata.json"),
        "joints.json": sha(slot / "joints.json"),
    }
    sample = {
        "index": index,
        "split": split,
        "active_arm": "left",
        "raw": str(slot / "raw"),
        "source": "buffered_left_wrist_flushed_on_q",
        "identity": identity,
        "T_base_link6_m": poses,
        "cameras": {"left_wrist": view},
        "joint_deg_left": item["states"]["left"]["joint_deg"],
        "image_stamp": stamp,
        "buffer_epoch": item["epoch"],
        "capture_sync": item.get("sync"),
        "source_files_sha256": files,
        "robot_motion_commands_sent_by_collector": 0,
        "old_observations_used": False,
        "simulator_information_used": False,
        "physical_board_scale_source": "user confirmed 100 percent print",
        "board_fixed_during_collection": True,
    }
    (slot / "sample.json").write_text(json.dumps(clean(sample), indent=2) + "\n")
    return {
        "index": index,
        "split": split,
        "corners": view.get("corner_count", 0),
        "eligible": bool(view.get("metric_pose_eligible")),
    }


def draw_lines(vis, lines, corner_ok):
    color = (40, 220, 40) if corner_ok else (40, 40, 220)
    for row, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (12, 42 + 36 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15 if row == 0 else 0.7,
            (0, 255, 255) if row == 0 else color if row == 1 else (0, 255, 255),
            3 if row == 0 else 2,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="manual-20261019-left-v1")
    parser.add_argument("--preview-snapshot-seconds", type=float, default=0.0,
                        help="Optional debug JPEG interval; zero disables preview disk writes")
    args = parser.parse_args()
    root = H / "sessions" / args.session
    if not root.is_dir():
        raise ValueError("Existing session is missing: " + args.session)
    cv2.setNumThreads(4)
    board = cv2.aruco.CharucoBoard(
        (7, 5), 0.04, 0.03, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    )
    board.setLegacyPattern(False)
    detector = cv2.aruco.CharucoDetector(board)
    from agilex_control.kinematics import Kinematics

    config = json.loads(CONFIG.read_text())
    kinematics = Kinematics(config)
    profiles = json.loads(FACTORY.read_text())["profiles"]
    identity = json.loads((root / "identity.json").read_text())
    info = json.loads(INFO.read_text())
    gate = AutoGate()
    known = disk_poses(root)
    buffer = []
    disk_total = len(list(root.glob("left-*/sample.json")))
    feedback = FeedbackStream(config)
    feedback_since = 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WINDOW, 1280, 960)
    last_seq = -1
    last_vis = 0.0
    status = "waiting for left wrist image"
    notice = "hold still to buffer    q detects and saves"
    feedback.start()
    try:
        while True:
            try:
                seq, stamp, frame = read_frame(last_seq)
            except (FileNotFoundError, ValueError):
                gate.anchor = None
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            now = time.time()
            if not -0.05 <= now - stamp <= 0.25:
                gate.anchor = None
                stale = np.zeros((240, 900, 3), np.uint8)
                draw_lines(stale, ["BUFFERED %d" % len(buffer), "Camera frame stale - waiting for live images", "q detects and saves buffered groups"], False)
                cv2.imshow(WINDOW, stale)
            elif seq != last_seq:
                last_seq = seq
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
                scale = 2
                vis = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                if marker_corners is not None and marker_ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis, [item * scale for item in marker_corners], marker_ids)
                corner_count = 0 if ids is None else int(len(ids))
                if corners is not None and ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(vis, corners * scale, ids)
                history, state_error = feedback.snapshot()
                latest, drift, pair = frame_feedback(history, stamp, time.time(), feedback_since)
                states = None if latest is None else latest["states"]
                if latest is not None:
                    feedback_since = latest["epoch"]
                left = None if states is None else states.get("left")
                joints = None
                pose = None
                if left is not None and left.get("joint_deg"):
                    joints = np.asarray(left["joint_deg"], dtype=float)
                    pose = link_pose(kinematics, joints)
                view = {"metric_pose_eligible": corner_count >= 8}
                if state_error or left is None or joints is None:
                    gate.anchor = None
                    status = "waiting for joint feedback"
                    ready = False
                else:
                    ready, reason = gate.check(left, view, pose if pose is not None else np.eye(4), known, drift, time.monotonic())
                    status = {
                        "等待右臂失能且反馈正常": "AUTO wait disabled arm",
                        "正在移动，等待停稳": "AUTO moving",
                        "等待右腕看清标定板（至少 8 个角点且质量合格）": "AUTO need a clear board",
                        "检测到新姿态，等待持续停稳": "AUTO new pose, hold still",
                        "等待停稳至少 1 秒": "AUTO holding",
                        "此姿态已采集，请换一个位置或角度": "AUTO pose already buffered",
                        "此姿态采样未通过，请稍作调整后停稳": "AUTO adjust, then hold",
                        "正在检查上一组采样结果": "AUTO checking",
                        "姿态稳定且图像合格，自动保存中": "AUTO buffering",
                    }.get(reason, "AUTO waiting")
                    if reason == "等待停稳至少 1 秒":
                        status += " %.1fs" % max(0.0, 1.0 - (time.monotonic() - gate.since))
                    if ready and pair is None:
                        ready = False
                        status = "AUTO matching frame and joint timestamps"
                lines = [
                    "BUFFERED %d" % len(buffer),
                    "ChArUco corners %d/24  markers %d" % (corner_count, 0 if marker_ids is None else len(marker_ids)),
                    "disk %d   %s" % (disk_total, status),
                    "joints " + (" ".join("%.2f" % value for value in joints) if joints is not None else "none"),
                    notice,
                ]
                draw_lines(vis, lines, corner_count >= 8)
                cv2.imshow(WINDOW, vis)
                now = time.time()
                if args.preview_snapshot_seconds > 0 and now - last_vis > args.preview_snapshot_seconds:
                    cv2.imwrite("/tmp/left-d435-cv2-vis.jpg", vis)
                    last_vis = now
                if ready and pair is not None:
                    buffer.append(
                        {
                            "image": frame.copy(),
                            "stamp": stamp,
                            "states": pair["states"],
                            "epoch": time.time(),
                            "sync": {key: value for key, value in pair.items() if key != "states"},
                        }
                    )
                    known.append(link_pose(kinematics, pair["states"]["left"]["joint_deg"]))
                    notice = "buffered %d    q detects and saves" % len(buffer)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        feedback.close()
    results = []
    if buffer:
        blank = np.zeros((240, 900, 3), np.uint8)
        for number, item in enumerate(buffer, 1):
            canvas = blank.copy()
            cv2.putText(canvas, "detecting %d / %d" % (number, len(buffer)), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)
            cv2.waitKey(1)
            results.append(write_group(root, kinematics, profiles, identity, info, item))
    cv2.destroyAllWindows()
    print(clean({"buffered": len(buffer), "written": results, "robot_motion_commands_sent": 0}))


if __name__ == "__main__":
    main()
