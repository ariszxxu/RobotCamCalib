from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_xarm7_g305_eye_in_hand import StabilityGate
from robot_cam_calib.geometry import make_T, transform_delta
from robot_cam_calib.hand_eye import HandEyeObservation, solve_hand_eye_robust


REPO_ROOT = Path(__file__).resolve().parents[1]


def synthetic_observations() -> tuple[np.ndarray, list[HandEyeObservation]]:
    rng = np.random.default_rng(17)
    expected = make_T(
        Rotation.from_rotvec([0.23, -0.14, 0.08]).as_matrix(),
        [0.038, -0.026, 0.079],
    )
    base_target = make_T(
        Rotation.from_rotvec([-0.11, 0.04, 0.19]).as_matrix(),
        [0.52, 0.03, 0.14],
    )
    observations: list[HandEyeObservation] = []
    for index in range(24):
        base_link7 = make_T(
            Rotation.from_rotvec(rng.uniform(-0.9, 0.9, 3)).as_matrix(),
            rng.uniform([-0.2, -0.2, 0.25], [0.55, 0.35, 0.75]),
        )
        camera_target = (
            np.linalg.inv(expected) @ np.linalg.inv(base_link7) @ base_target
        )
        observations.append(
            HandEyeObservation(index, base_link7, camera_target)
        )
    return expected, observations


class HandEyeSolverTests(unittest.TestCase):
    def test_automatic_capture_stability_and_rearm_gate(self) -> None:
        gate = StabilityGate(
            stable_seconds=0.5,
            stable_joint_range_deg=0.12,
            rearm_joint_delta_deg=2.0,
        )
        zero = np.zeros(7)
        self.assertFalse(gate.update(zero, 10.0).ready)
        self.assertFalse(gate.update(zero, 10.49).ready)
        self.assertTrue(gate.update(zero, 10.50).ready)

        gate.mark_captured(zero)
        one_degree = zero.copy()
        one_degree[2] = np.deg2rad(1.0)
        self.assertFalse(gate.update(one_degree, 11.0).armed)

        moved = zero.copy()
        moved[2] = np.deg2rad(2.1)
        rearmed = gate.update(moved, 11.1)
        self.assertTrue(rearmed.armed)
        self.assertTrue(rearmed.rearmed)
        self.assertFalse(rearmed.ready)
        self.assertFalse(gate.update(moved, 11.59).ready)
        self.assertTrue(gate.update(moved, 11.60).ready)

    def test_stability_timer_resets_on_motion(self) -> None:
        gate = StabilityGate(
            stable_seconds=0.5,
            stable_joint_range_deg=0.12,
            rearm_joint_delta_deg=2.0,
        )
        zero = np.zeros(7)
        gate.update(zero, 1.0)
        moved = zero.copy()
        moved[0] = np.deg2rad(0.2)
        reset = gate.update(moved, 1.4)
        self.assertEqual(reset.stable_for_s, 0.0)
        self.assertFalse(gate.update(moved, 1.89).ready)
        self.assertTrue(gate.update(moved, 1.90).ready)

    def test_recovers_link7_camera_transform(self) -> None:
        expected, observations = synthetic_observations()
        result = solve_hand_eye_robust(observations, min_samples=12)
        rotation, translation = transform_delta(
            expected, result["T_gripper_camera"]
        )
        self.assertLess(rotation, 1e-6)
        self.assertLess(translation, 1e-9)
        self.assertEqual(result["excitation"]["relative_rotation_rank"], 3)
        self.assertTrue(result["cross_validation"]["available"])

    def test_robot_adapter_contains_no_set_api_calls(self) -> None:
        source_path = REPO_ROOT / "calibrate_xarm7_g305_eye_in_hand.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        adapter = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ReadOnlyXArm7"
        )
        called_attributes = {
            node.func.attr
            for node in ast.walk(adapter)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(
            {name for name in called_attributes if name.startswith("set_")}
        )


if __name__ == "__main__":
    unittest.main()
