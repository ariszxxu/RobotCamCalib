from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_cam_calib.capture import TimedFrame, select_synchronized_pair
from robot_cam_calib.config import CalibrationCatalog
from robot_cam_calib.geometry import (
    inv_T,
    make_T,
    solve_x_given_y,
    solve_y_given_x,
)
from robot_cam_calib.targets import (
    project_points_for_intrinsics,
    reproj_error_for_intrinsics,
    solve_pnp_for_intrinsics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_transform_inverse(self) -> None:
        transform = make_T(
            Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix(),
            [0.4, -0.2, 0.7],
        )
        np.testing.assert_allclose(transform @ inv_T(transform), np.eye(4), atol=1e-12)

    def test_closed_form_helpers_return_rigid_transforms(self) -> None:
        rng = np.random.default_rng(42)
        x_true = make_T(
            Rotation.from_rotvec([0.15, -0.2, 0.08]).as_matrix(),
            [0.04, -0.02, 0.08],
        )
        y_true = make_T(
            Rotation.from_rotvec([-0.12, 0.07, 0.18]).as_matrix(),
            [-0.03, 0.06, 0.09],
        )
        left: list[np.ndarray] = []
        right: list[np.ndarray] = []
        for _ in range(20):
            left_i = make_T(
                Rotation.from_rotvec(rng.normal(0.0, 0.5, 3)).as_matrix(),
                rng.uniform(-0.3, 0.3, 3),
            )
            right_i = inv_T(y_true) @ left_i @ x_true
            left.append(left_i)
            right.append(right_i)

        y_est = solve_y_given_x(left, right, x_true)
        x_est = solve_x_given_y(left, right, y_true)
        for transform in (x_est, y_est):
            np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
            np.testing.assert_allclose(
                transform[:3, :3].T @ transform[:3, :3],
                np.eye(3),
                atol=1e-12,
            )
            self.assertAlmostEqual(
                float(np.linalg.det(transform[:3, :3])),
                1.0,
            )


class CaptureTests(unittest.TestCase):
    def test_selects_newest_valid_pair(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        first = [
            TimedFrame(0, 1.00, image),
            TimedFrame(1, 2.00, image),
        ]
        second = [
            TimedFrame(0, 1.01, image),
            TimedFrame(1, 2.02, image),
        ]
        selected = select_synchronized_pair(first, second, None, 0.03)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual((selected[0].index, selected[1].index), (1, 1))


class CatalogTests(unittest.TestCase):
    def test_all_tasks_resolve_to_existing_runners(self) -> None:
        catalog = CalibrationCatalog.load(REPO_ROOT / "configs")
        warnings = catalog.validate(check_files=True)
        # Historical calibrations under ignored outputs/ may be absent in a
        # clean checkout. Tracked target presets and task runners must not be.
        self.assertFalse(
            [
                warning
                for warning in warnings
                if "missing task runner" in warning
                or "charuco_a4_40mm" in warning
                or "compact_apriltag_grid_4x4_tag48mm" in warning
            ]
        )
        for task_name in catalog.tasks:
            command = catalog.build_command(task_name)
            self.assertTrue(Path(command[1]).is_file())
            self.assertNotIn("${", " ".join(command))


class TargetPoseTests(unittest.TestCase):
    def test_pinhole_and_fisheye_pnp_round_trip(self) -> None:
        object_points = np.asarray(
            [[x, y, 0.0] for y in (0.0, 0.03, 0.06) for x in (0.0, 0.03, 0.06)],
            dtype=np.float64,
        )
        matrix = np.asarray(
            [[820.0, 0.0, 640.0], [0.0, 815.0, 400.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        expected_rvec = np.asarray([0.24, -0.18, 0.08], dtype=np.float64)
        expected_tvec = np.asarray([0.01, -0.02, 0.55], dtype=np.float64)
        models = {
            "pinhole": np.asarray([-0.08, 0.02, 0.001, -0.001, 0.0]),
            "fisheye": np.asarray([-0.04, 0.003, 0.0, 0.0]),
        }

        for camera_model, distortion in models.items():
            with self.subTest(camera_model=camera_model):
                image_points = project_points_for_intrinsics(
                    object_points,
                    expected_rvec,
                    expected_tvec,
                    matrix,
                    distortion,
                    camera_model,
                )
                ok, rvec, tvec = solve_pnp_for_intrinsics(
                    object_points,
                    image_points,
                    matrix,
                    distortion,
                    camera_model,
                )
                self.assertTrue(ok)
                self.assertIsNotNone(rvec)
                self.assertIsNotNone(tvec)
                assert rvec is not None and tvec is not None
                self.assertGreater(float(tvec.reshape(3)[2]), 0.0)
                error = reproj_error_for_intrinsics(
                    object_points,
                    image_points,
                    rvec,
                    tvec,
                    matrix,
                    distortion,
                    camera_model,
                )
                self.assertLess(error, 1e-5)


if __name__ == "__main__":
    unittest.main()
