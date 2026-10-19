import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from manual_calibration_gui import AutoGate
from right_wrist_feedback import FeedbackStream, frame_feedback


def row(epoch, q=0):
    state = {"complete": True, "stale": [], "error_code": 0,
             "arm_status": 0, "enabled_drive_bits": [False] * 6,
             "joint_deg": [q] * 6}
    return {"epoch": epoch, "states": {"right": state},
            "joint_epochs": [epoch - 0.005] * 3}


class FeedbackTests(unittest.TestCase):
    def test_healthy_pair_retains_measured_joints_and_timestamps(self):
        history = [row(100), row(100.05, .01), row(100.10, .01)]
        latest, drift, pair = frame_feedback(history, 100.025, 100.11, 100.05)
        self.assertIs(latest, history[-1])
        self.assertAlmostEqual(drift, .01)
        self.assertEqual(pair["states"]["right"]["joint_deg"], [.01] * 6)
        self.assertEqual(pair["feedback_before_epoch"], 100)
        self.assertEqual(pair["feedback_after_epoch"], 100.05)

    def test_rejects_expired_or_future_camera_and_feedback(self):
        for stamp, now in ((99, 100.1), (101, 100.1), (100.025, 100.4)):
            self.assertIsNone(frame_feedback([row(100), row(100.05)], stamp, now)[0])
        self.assertIsNone(frame_feedback([], 100, 100.1)[0])

    def test_missing_or_old_bracket_cannot_be_captured(self):
        for history in ([row(100)], [row(99.8), row(100.05)]):
            self.assertIsNone(frame_feedback(history, 100.025, 100.1)[2])
        history = [row(100), row(100.05)]
        history[0]["joint_epochs"][0] = 99.8
        self.assertIsNone(frame_feedback(history, 100.025, 100.1)[2])

    def test_fault_enabled_stale_and_incomplete_pair_rejected(self):
        for key, value in (("complete", False), ("stale", ["0x2a5"]),
                           ("error_code", 1), ("arm_status", 1),
                           ("enabled_drive_bits", [True] * 6),
                           ("joint_deg", [float("nan")] * 6)):
            history = [row(100), row(100.05)]
            history[0]["states"]["right"][key] = value
            _, drift, pair = frame_feedback(history, 100.025, 100.1)
            self.assertTrue(np.isinf(drift), key)
            self.assertIsNone(pair, key)

    def test_motion_between_gui_checks_resets_stability(self):
        history = [row(100), row(100.05, .1), row(100.10), row(100.15)]
        _, drift, _ = frame_feedback(history, 100.125, 100.16, since=100)
        self.assertAlmostEqual(drift, .1)
        gate = AutoGate()
        state = history[-1]["states"]["right"]
        args = (state, {"metric_pose_eligible": True}, np.eye(4), [])
        gate.check(*args, 0, 10)
        self.assertFalse(gate.check(*args, drift, 11.1)[0])
        self.assertFalse(gate.check(*args, 0, 11.2)[0])
        self.assertTrue(gate.check(*args, 0, 12.21)[0])

    def test_pair_with_motion_is_rejected(self):
        self.assertIsNone(frame_feedback([row(100), row(100.05, .03)], 100.025, 100.1)[2])

    def test_feedback_gap_requires_stability_again(self):
        _, drift, _ = frame_feedback([row(99), row(100), row(100.05)], 100.025, 100.1, since=99)
        self.assertTrue(np.isinf(drift))

    def test_same_frame_skips_pixels_and_bad_frames_are_rejected(self):
        import struct
        import right_wrist_cv2_gui as gui

        with tempfile.TemporaryDirectory() as folder:
            frame = Path(folder) / "frame.bin"
            with patch.object(gui, "SHM", frame):
                frame.write_bytes(struct.pack("<8sQdIII", gui.MAGIC, 1, 100., 2, 3, 3) + bytes(18))
                self.assertEqual(gui.read_frame()[2].shape, (2, 3, 3))
                self.assertIsNone(gui.read_frame(after_seq=1)[2])
                for payload in (b"partial", struct.pack("<8sQdIII", gui.MAGIC, 2, 101., 2, 3, 3)):
                    frame.write_bytes(payload)
                    with self.assertRaises(ValueError):
                        gui.read_frame(after_seq=1)

    def test_stream_is_persistent_and_only_receives(self):
        opened = []

        class Bus:
            def __init__(self, arms):
                opened.append(arms)
                self.stamps = {"right": {}}
                self.events = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def pump(self, period):
                time.sleep(period)
                self.stamps["right"] = {cid: time.monotonic() for cid in (0x2A5, 0x2A6, 0x2A7)}
                self.events.append({})

            def state(self):
                return row(time.time())["states"]

            def send(self, *args):
                raise AssertionError("capture must not send CAN frames")

            open_writer = send

        stream = FeedbackStream({"arms": {"right": "test"}}, period=.005, bus_factory=Bus)
        stream.start()
        try:
            deadline = time.monotonic() + 1
            while len(stream.snapshot()[0]) < 4 and time.monotonic() < deadline:
                time.sleep(.005)
            history, error = stream.snapshot()
            self.assertGreaterEqual(len(history), 4)
            self.assertEqual(error, "")
            self.assertEqual(len(opened), 1)
        finally:
            stream.close()
        self.assertFalse(stream.thread.is_alive())

    def test_hostname_mismatch_rejected_before_opening_bus(self):
        with self.assertRaises(ValueError):
            FeedbackStream({"expected_hostname": "wrong-host", "arms": {}})

    def test_writing_preserves_split_and_sync_without_touching_existing_samples(self):
        import right_wrist_cv2_gui as gui
        from agilex_control.kinematics import Kinematics

        item = {"image": np.zeros((32, 32, 3), np.uint8), "stamp": 100.025,
                "epoch": 100.1, "states": row(100.05)["states"],
                "sync": {"feedback_before_epoch": 100, "feedback_after_epoch": 100.05}}
        info = {"width": 32, "height": 32, "frame_id": "test",
                "distortion_model": "none", "D": [], "K": [1] * 9}
        kinematics = Kinematics(json.loads(gui.CONFIG.read_text()))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "right-0002").mkdir()
            old = root / "right-0002" / "sample.json"
            old.write_text("existing evidence")
            with patch.object(gui, "detect", return_value={"metric_pose_eligible": True, "corner_count": 24}):
                result = gui.write_group(root, kinematics, [], {}, info, item)
            self.assertEqual(result["split"], "holdout")
            sample = json.loads((root / "right-0003" / "sample.json").read_text())
            self.assertEqual(sample["capture_sync"], item["sync"])
            self.assertEqual(old.read_text(), "existing evidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)
