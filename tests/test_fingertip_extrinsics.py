from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_cam_calib.fingertip_extrinsics import (
    FingertipObservation,
    MultiChainObservation,
    PalmTipKinematics,
    solve_fingertip_extrinsics,
    solve_multichain_fingertip_extrinsics,
)
from robot_cam_calib.geometry import inv_T, make_T, transform_delta


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_URDF = Path(
    "/home/CNF2025915223/桌面/FingerEyeV2/assets/thirdparty/"
    "xarm7_wuji_left_description/wuji_left_w_fingereye.urdf"
)


class PalmTipKinematicsTests(unittest.TestCase):
    @unittest.skipUnless(EXTERNAL_URDF.is_file(), "FingerEyeV2 URDF is unavailable")
    def test_index_finger2_chain_uses_hardware_joints_four_through_seven(self) -> None:
        model = PalmTipKinematics(EXTERNAL_URDF, "left_finger2_link4")
        self.assertEqual(model.chain_joint_indices, (4, 5, 6, 7))

    @unittest.skipUnless(EXTERNAL_URDF.is_file(), "FingerEyeV2 URDF is unavailable")
    def test_requested_finger1_chain_uses_first_four_hardware_joints(self) -> None:
        model = PalmTipKinematics(EXTERNAL_URDF, "left_finger1_link4")
        self.assertEqual(model.chain_joint_indices, (0, 1, 2, 3))
        first = model.forward(np.zeros(20))
        moved = np.zeros(20)
        moved[0] = 0.2
        second = model.forward(moved)
        rotation, translation = transform_delta(first, second)
        self.assertGreater(rotation, 1.0)
        self.assertGreaterEqual(translation, 0.0)

    def test_joint_solver_recovers_both_constants(self) -> None:
        rng = np.random.default_rng(24)
        expected_palm_camera = make_T(
            Rotation.from_rotvec([0.35, -0.21, 0.12]).as_matrix(),
            [0.055, -0.018, 0.038],
        )
        expected_tip_cube = make_T(
            Rotation.from_rotvec([-0.14, 0.27, 0.31]).as_matrix(),
            [0.003, -0.006, 0.021],
        )
        observations: list[FingertipObservation] = []
        for index in range(24):
            palm_tip = make_T(
                Rotation.from_rotvec(rng.uniform(-1.0, 1.0, 3)).as_matrix(),
                rng.uniform([-0.04, -0.05, 0.04], [0.06, 0.05, 0.14]),
            )
            camera_cube = (
                inv_T(expected_palm_camera) @ palm_tip @ expected_tip_cube
            )
            observations.append(FingertipObservation(index, palm_tip, camera_cube))
        result = solve_fingertip_extrinsics(
            observations, min_samples=12, starts=8
        )
        camera_rotation, camera_translation = transform_delta(
            expected_palm_camera, result["T_palm_camera"]
        )
        cube_rotation, cube_translation = transform_delta(
            expected_tip_cube, result["T_tip_cube"]
        )
        self.assertLess(camera_rotation, 1.0e-5)
        self.assertLess(camera_translation, 1.0e-8)
        self.assertLess(cube_rotation, 1.0e-5)
        self.assertLess(cube_translation, 1.0e-8)
        self.assertEqual(result["jacobian_rank"], 12)

    def test_multichain_solver_recovers_all_free_constants(self) -> None:
        rng = np.random.default_rng(2408)
        expected_camera = make_T(
            Rotation.from_rotvec([0.31, -0.18, 0.09]).as_matrix(),
            [0.012, -0.052, 0.031],
        )
        expected_targets = {
            "index": make_T(
                Rotation.from_rotvec([-0.2, 0.11, 0.4]).as_matrix(),
                [-0.004, -0.018, 0.033],
            ),
            "thumb": make_T(
                Rotation.from_rotvec([0.15, -0.3, -0.25]).as_matrix(),
                [-0.036, -0.002, 0.034],
            ),
        }
        observations: list[MultiChainObservation] = []
        for chain, target in expected_targets.items():
            for index in range(16):
                palm_tip = make_T(
                    Rotation.from_rotvec(rng.uniform(-1.2, 1.2, 3)).as_matrix(),
                    rng.uniform([-0.06, -0.05, 0.03], [0.06, 0.05, 0.15]),
                )
                camera_target = inv_T(expected_camera) @ palm_tip @ target
                observations.append(
                    MultiChainObservation(
                        chain, index, palm_tip, camera_target
                    )
                )
        result = solve_multichain_fingertip_extrinsics(
            observations, min_samples_per_chain=12, starts=6
        )
        camera_delta = transform_delta(expected_camera, result["T_palm_camera"])
        self.assertLess(camera_delta[0], 1.0e-5)
        self.assertLess(camera_delta[1], 1.0e-8)
        for chain, expected in expected_targets.items():
            delta = transform_delta(
                expected, result["chains"][chain]["T_tip_target"]
            )
            self.assertLess(delta[0], 1.0e-5)
            self.assertLess(delta[1], 1.0e-8)
        self.assertEqual(result["jacobian_rank"], 18)


if __name__ == "__main__":
    unittest.main()
