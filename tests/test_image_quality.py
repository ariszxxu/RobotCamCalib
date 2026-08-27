from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

import calibrate_xarm7_g305_eye_in_hand as eye_in_hand
from calibrate_xarm7_g305_eye_in_hand import (
    CaptureFrameCandidate,
    collect_capture_burst,
    select_best_capture_candidate,
)
from robot_cam_calib.image_quality import (
    PlanarTargetQualityConfig,
    focus_metric,
    robust_low_sharpness_indices,
)


def test_focus_metric_prefers_sharp_edges() -> None:
    sharp = np.zeros((240, 320), dtype=np.uint8)
    for row in range(0, 240, 20):
        for column in range(0, 320, 20):
            if (row // 20 + column // 20) % 2:
                sharp[row : row + 20, column : column + 20] = 255
    blurred = cv2.GaussianBlur(sharp, (21, 21), 6.0)
    assert focus_metric(sharp) > 10.0 * focus_metric(blurred)


def test_robust_low_sharpness_rejects_only_extreme_tail() -> None:
    scores = [(index, 1000.0 + index) for index in range(19)] + [(19, 5.0)]
    rejected, report = robust_low_sharpness_indices(scores)
    assert rejected == {19}
    assert report["rejected_indices"] == [19]


def test_capture_candidate_prefers_sharpness_then_reprojection() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    qpos = np.zeros(7)
    lower_reprojection = CaptureFrameCandidate(
        frame,
        SimpleNamespace(reproj_error=0.1),
        None,
        None,
        100.0,
        qpos,
        qpos,
        qpos,
        10,
        20,
        11,
        19,
        0.5,
        0.0,
    )
    sharper = CaptureFrameCandidate(
        frame,
        SimpleNamespace(reproj_error=0.3),
        None,
        None,
        120.0,
        qpos,
        qpos,
        qpos,
        30,
        40,
        31,
        39,
        0.5,
        0.0,
    )
    assert select_best_capture_candidate([lower_reprojection, sharper]) is sharper


def test_burst_pairs_each_new_frame_with_encoder_reads(monkeypatch) -> None:
    class FakeCamera:
        def __init__(self) -> None:
            self.value = 0

        def read_bgr(self):
            self.value += 1
            frame = np.full((16, 16, 3), 10 * self.value, dtype=np.uint8)
            return frame, float(self.value), 1010 if self.value == 1 else 2010

    class FakeRobot:
        def __init__(self) -> None:
            self.values = iter([0.001, 0.002, 0.004, 0.006])

        def read_qpos_once(self) -> np.ndarray:
            qpos = np.zeros(7)
            qpos[0] = np.deg2rad(next(self.values))
            return qpos

    monkeypatch.setattr(
        eye_in_hand,
        "detect_charuco_pose",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True, T=np.eye(4), reproj_error=0.1
        ),
    )
    monkeypatch.setattr(
        eye_in_hand,
        "planar_target_sharpness",
        lambda image, *_args, **_kwargs: float(np.mean(image)),
    )
    system_times = iter([1_000_000, 1_020_000, 2_000_000, 2_020_000])
    monkeypatch.setattr(eye_in_hand.time, "time_ns", lambda: next(system_times))
    intrinsics = SimpleNamespace(
        K=np.eye(3), dist=np.zeros(5), camera_model="pinhole"
    )
    selected, valid_count, burst_range = collect_capture_burst(
        FakeCamera(),
        FakeRobot(),
        np.zeros(7),
        object(),
        object(),
        intrinsics,
        PlanarTargetQualityConfig(0.28, 0.20),
        frame_count=2,
        max_reprojection_error_px=1.0,
        max_joint_range_deg=0.02,
    )
    assert valid_count == 2
    assert np.isclose(np.degrees(selected.qpos_rad[0]), 0.005)
    assert np.isclose(selected.sharpness, 20.0)
    assert np.isclose(burst_range, 0.006)
