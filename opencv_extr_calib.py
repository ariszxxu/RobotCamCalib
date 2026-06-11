#!/usr/bin/env python3
"""OpenCV hand-eye alternatives for the third-view xArm camera calibration.

The main calibration equation used by this repo is:

    X_CammountTagmount_i @ X_TagmountTag = X_CammountCam @ X_CamTag_i

For the third-view case, cammount is the robot base and tagmount is the
end-effector. This file compares OpenCV's hand-eye solvers on the same
detected AprilTag observations produced by the PyBullet simulator.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parent
SIM_OUTPUTS_DIR = REPO_ROOT / "sim_tests" / "outputs"


@dataclass
class OpenCVCalibrationResult:
    method_name: str
    X_CammountCam: np.ndarray
    X_TagmountTag: np.ndarray
    residual_rot_deg_mean: float
    residual_trans_m_mean: float


@dataclass
class OpenCVErrorSample:
    sample_count: int
    method_name: str
    cam_rot_deg: float
    cam_trans_m: float
    tag_rot_deg: float
    tag_trans_m: float
    residual_rot_deg_mean: float
    residual_trans_m_mean: float


HAND_EYE_METHODS: Dict[str, int] = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

ROBOT_WORLD_HAND_EYE_METHODS: Dict[str, int] = {
    "ROBOT_WORLD_SHAH": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    "ROBOT_WORLD_LI": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
}


def inv_T(X: np.ndarray) -> np.ndarray:
    Y = np.eye(4)
    Y[:3, :3] = X[:3, :3].T
    Y[:3, 3] = -X[:3, :3].T @ X[:3, 3]
    return Y


def make_T(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    X = np.eye(4)
    X[:3, :3] = np.asarray(rotation, dtype=float)
    X[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return X


def split_rt(T_list: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    R_list = [np.asarray(T[:3, :3], dtype=np.float64) for T in T_list]
    t_list = [np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1) for T in T_list]
    return R_list, t_list


def pose_err_deg_m(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    delta = inv_T(T_gt) @ T_est
    rotvec = R.from_matrix(delta[:3, :3]).as_rotvec()
    return float(np.degrees(np.linalg.norm(rotvec))), float(np.linalg.norm(delta[:3, 3]))


def average_transforms(T_list: Sequence[np.ndarray]) -> np.ndarray:
    if len(T_list) == 0:
        raise ValueError("Cannot average an empty transform list.")
    rotations = R.from_matrix(np.stack([T[:3, :3] for T in T_list], axis=0))
    X = np.eye(4)
    X[:3, :3] = rotations.mean().as_matrix()
    X[:3, 3] = np.mean(np.stack([T[:3, 3] for T in T_list], axis=0), axis=0)
    return X


def collect_data(
    X_CamTag_list: Sequence[np.ndarray],
    X_WorldCammount_list: Sequence[np.ndarray],
    X_WorldTagmount_list: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return OpenCV-ready third-view observations.

    Returns:
        X_CammountTagmount_list: cammount-to-tagmount poses.
        X_CamTag_list: AprilTag detector target-to-camera poses.
    """
    if not (
        len(X_CamTag_list)
        == len(X_WorldCammount_list)
        == len(X_WorldTagmount_list)
    ):
        raise ValueError("Observation lists must have equal length.")

    X_CammountTagmount_list = [
        inv_T(X_WorldCammount) @ X_WorldTagmount
        for X_WorldCammount, X_WorldTagmount in zip(
            X_WorldCammount_list,
            X_WorldTagmount_list,
        )
    ]
    return X_CammountTagmount_list, list(X_CamTag_list)


def residual_stats(
    X_CammountTagmount_list: Sequence[np.ndarray],
    X_CamTag_list: Sequence[np.ndarray],
    X_CammountCam: np.ndarray,
    X_TagmountTag: np.ndarray,
) -> Tuple[float, float]:
    rot_errs = []
    trans_errs = []
    for X_CammountTagmount, X_CamTag in zip(X_CammountTagmount_list, X_CamTag_list):
        lhs = X_CammountTagmount @ X_TagmountTag
        rhs = X_CammountCam @ X_CamTag
        rot_err, trans_err = pose_err_deg_m(rhs, lhs)
        rot_errs.append(rot_err)
        trans_errs.append(trans_err)
    return float(np.mean(rot_errs)), float(np.mean(trans_errs))


def estimate_tagmount_tag_from_camera(
    X_CammountTagmount_list: Sequence[np.ndarray],
    X_CamTag_list: Sequence[np.ndarray],
    X_CammountCam: np.ndarray,
) -> np.ndarray:
    per_sample = [
        inv_T(X_CammountTagmount) @ X_CammountCam @ X_CamTag
        for X_CammountTagmount, X_CamTag in zip(X_CammountTagmount_list, X_CamTag_list)
    ]
    return average_transforms(per_sample)


def calibrate_hand_eye_thirdview(
    X_CamTag_list: Sequence[np.ndarray],
    X_WorldCammount_list: Sequence[np.ndarray],
    X_WorldTagmount_list: Sequence[np.ndarray],
    method_name: str,
    method_code: int,
) -> OpenCVCalibrationResult:
    """Use cv2.calibrateHandEye in OpenCV's eye-to-hand convention.

    For third-view, pass tagmount-to-cammount as the robot motion input. OpenCV
    then returns X_CammountCam directly. X_TagmountTag is recovered by averaging
    inv(X_CammountTagmount_i) @ X_CammountCam @ X_CamTag_i.
    """
    X_CammountTagmount_list, X_CamTag_list = collect_data(
        X_CamTag_list,
        X_WorldCammount_list,
        X_WorldTagmount_list,
    )
    X_TagmountCammount_list = [inv_T(X) for X in X_CammountTagmount_list]
    R_robot, t_robot = split_rt(X_TagmountCammount_list)
    R_target2cam, t_target2cam = split_rt(X_CamTag_list)

    R_cammount_cam, t_cammount_cam = cv2.calibrateHandEye(
        R_gripper2base=R_robot,
        t_gripper2base=t_robot,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method_code,
    )

    X_CammountCam = make_T(R_cammount_cam, t_cammount_cam)
    X_TagmountTag = estimate_tagmount_tag_from_camera(
        X_CammountTagmount_list,
        X_CamTag_list,
        X_CammountCam,
    )
    residual_rot, residual_trans = residual_stats(
        X_CammountTagmount_list,
        X_CamTag_list,
        X_CammountCam,
        X_TagmountTag,
    )
    return OpenCVCalibrationResult(
        method_name=method_name,
        X_CammountCam=X_CammountCam,
        X_TagmountTag=X_TagmountTag,
        residual_rot_deg_mean=residual_rot,
        residual_trans_m_mean=residual_trans,
    )


def calibrate_robot_world_thirdview(
    X_CamTag_list: Sequence[np.ndarray],
    X_WorldCammount_list: Sequence[np.ndarray],
    X_WorldTagmount_list: Sequence[np.ndarray],
    method_name: str,
    method_code: int,
) -> OpenCVCalibrationResult:
    """Use cv2.calibrateRobotWorldHandEye for AX=ZB.

    Mapping for third-view:
      A_i = X_CamTag_i
      B_i = X_CammountTagmount_i
      output base2world = inv(X_TagmountTag)
      output gripper2cam = inv(X_CammountCam)
    """
    X_CammountTagmount_list, X_CamTag_list = collect_data(
        X_CamTag_list,
        X_WorldCammount_list,
        X_WorldTagmount_list,
    )
    R_world2cam, t_world2cam = split_rt(X_CamTag_list)
    R_base2gripper, t_base2gripper = split_rt(X_CammountTagmount_list)

    R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
        R_world2cam=R_world2cam,
        t_world2cam=t_world2cam,
        R_base2gripper=R_base2gripper,
        t_base2gripper=t_base2gripper,
        method=method_code,
    )

    X_TagTagmount = make_T(R_base2world, t_base2world)
    X_CamCammount = make_T(R_gripper2cam, t_gripper2cam)
    X_TagmountTag = inv_T(X_TagTagmount)
    X_CammountCam = inv_T(X_CamCammount)
    residual_rot, residual_trans = residual_stats(
        X_CammountTagmount_list,
        X_CamTag_list,
        X_CammountCam,
        X_TagmountTag,
    )
    return OpenCVCalibrationResult(
        method_name=method_name,
        X_CammountCam=X_CammountCam,
        X_TagmountTag=X_TagmountTag,
        residual_rot_deg_mean=residual_rot,
        residual_trans_m_mean=residual_trans,
    )


def calibrate_thirdview(
    X_CamTag_list: Sequence[np.ndarray],
    X_WorldCammount_list: Sequence[np.ndarray],
    X_WorldTagmount_list: Sequence[np.ndarray],
) -> Dict[str, OpenCVCalibrationResult]:
    if len(X_CamTag_list) < 3:
        raise ValueError("At least 3 valid observations are required.")

    results: Dict[str, OpenCVCalibrationResult] = {}
    for method_name, method_code in HAND_EYE_METHODS.items():
        try:
            results[method_name] = calibrate_hand_eye_thirdview(
                X_CamTag_list,
                X_WorldCammount_list,
                X_WorldTagmount_list,
                method_name,
                method_code,
            )
        except Exception as exc:
            print(f"[opencv] {method_name} failed: {exc}")

    for method_name, method_code in ROBOT_WORLD_HAND_EYE_METHODS.items():
        try:
            results[method_name] = calibrate_robot_world_thirdview(
                X_CamTag_list,
                X_WorldCammount_list,
                X_WorldTagmount_list,
                method_name,
                method_code,
            )
        except Exception as exc:
            print(f"[opencv] {method_name} failed: {exc}")

    if not results:
        raise RuntimeError("All OpenCV calibration methods failed.")
    return results


def collect_sim_thirdview_observations(args: argparse.Namespace):
    if importlib.util.find_spec("pupil_apriltags") is None:
        raise RuntimeError("pupil-apriltags is required for simulated AprilTag observations.")

    from pupil_apriltags import Detector
    from sim_tests.pybullet_extr_calib_sim import (
        DEFAULT_INTRINSICS_PATH,
        DEFAULT_ROBOT_URDF,
        PyBulletXarmSim,
        build_ground_truth,
        detect_grid_board_cam_target,
        detect_yaml_board_cam_target,
        load_intrinsics,
        sample_candidate_qpos,
    )

    K, width, height = load_intrinsics(DEFAULT_INTRINSICS_PATH, args.width, args.height)
    gt = build_ground_truth("thirdview")
    rng = np.random.default_rng(args.seed)
    detector = Detector(families="tag36h11", quad_decimate=1.0)

    X_CamTag_list: List[np.ndarray] = []
    X_WorldCammount_list: List[np.ndarray] = []
    X_WorldTagmount_list: List[np.ndarray] = []

    import tempfile

    with tempfile.TemporaryDirectory(prefix="robotcamcalib_opencv_sim_") as tmp:
        sim = PyBulletXarmSim(
            robot_urdf_path=DEFAULT_ROBOT_URDF,
            K=K,
            width=width,
            height=height,
            tag_size=args.tag_size,
            tag_id=args.tag_id,
            tag_marker_fraction=args.tag_marker_fraction,
            tag_rows=args.tag_rows,
            tag_cols=args.tag_cols,
            tag_gap_ratio=args.tag_gap_ratio,
            tag_board_yaml=args.tag_board_yaml,
            tag_board_image=args.tag_board_image,
            tag_board_image_width_m=args.tag_board_image_width_m,
            tag_board_image_height_m=args.tag_board_image_height_m,
            near=args.near,
            far=args.far,
            use_gui=False,
            renderer_name=args.renderer,
            tmpdir=Path(tmp),
        )
        rest_qpos = np.zeros(len(sim.joint_indices))
        attempts = 0
        max_attempts = max(args.samples * args.max_attempts_per_sample, args.samples)
        try:
            while len(X_CamTag_list) < args.samples and attempts < max_attempts:
                attempts += 1
                qpos = sample_candidate_qpos(sim, gt, rng, rest_qpos)
                frame = sim.render(gt, qpos)
                if args.use_gt_tag_pose:
                    detected_X_CamTag = inv_T(frame.X_WorldCam_gt) @ frame.X_WorldTag_gt
                else:
                    if sim.board_layout is not None:
                        detected_X_CamTag = detect_yaml_board_cam_target(
                            detector,
                            frame.rgb,
                            K,
                            sim.board_layout,
                            args.board_min_tags,
                        )
                    else:
                        if sim.grid_board is None:
                            raise RuntimeError("Grid board was not initialized by the simulator.")
                        detected_X_CamTag = detect_grid_board_cam_target(
                            detector,
                            frame.rgb,
                            K,
                            args.tag_size,
                            sim.grid_board,
                            args.board_min_tags,
                        )
                    if detected_X_CamTag is None:
                        continue

                X_CamTag_list.append(detected_X_CamTag)
                X_WorldCammount_list.append(sim.frame_pose(gt.cammount_link_name))
                X_WorldTagmount_list.append(sim.frame_pose(gt.tagmount_link_name))
                rest_qpos = qpos
        finally:
            sim.disconnect()

    if len(X_CamTag_list) < args.samples:
        raise RuntimeError(f"Only collected {len(X_CamTag_list)}/{args.samples} valid detections.")

    return X_CamTag_list, X_WorldCammount_list, X_WorldTagmount_list, gt


def write_error_csv(path: Path, history: Sequence[OpenCVErrorSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_count",
                "method_name",
                "cam_rot_deg",
                "cam_trans_m",
                "tag_rot_deg",
                "tag_trans_m",
                "residual_rot_deg_mean",
                "residual_trans_m_mean",
            ]
        )
        for row in history:
            writer.writerow(
                [
                    row.sample_count,
                    row.method_name,
                    row.cam_rot_deg,
                    row.cam_trans_m,
                    row.tag_rot_deg,
                    row.tag_trans_m,
                    row.residual_rot_deg_mean,
                    row.residual_trans_m_mean,
                ]
            )


def save_camera_error_plot(path: Path, history: Sequence[OpenCVErrorSample]) -> None:
    if not history:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({row.method_name for row in history})

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for method in methods:
        rows = [row for row in history if row.method_name == method]
        rows.sort(key=lambda row: row.sample_count)
        x = np.array([row.sample_count for row in rows], dtype=float)
        axes[0].plot(x, [row.cam_rot_deg for row in rows], marker="o", label=method)
        axes[1].plot(x, [row.cam_trans_m for row in rows], marker="o", label=method)

    axes[0].set_title("OpenCV thirdview camera calibration error vs number of samples")
    axes[0].set_ylabel("Camera rotation error (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Number of accepted samples")
    axes[1].set_ylabel("Camera translation error (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_sim_thirdview(args: argparse.Namespace) -> List[OpenCVErrorSample]:
    X_CamTag_list, X_WorldCammount_list, X_WorldTagmount_list, gt = collect_sim_thirdview_observations(args)
    history: List[OpenCVErrorSample] = []

    for sample_count in range(args.min_samples, len(X_CamTag_list) + 1):
        results = calibrate_thirdview(
            X_CamTag_list[:sample_count],
            X_WorldCammount_list[:sample_count],
            X_WorldTagmount_list[:sample_count],
        )
        for method_name, result in sorted(results.items()):
            cam_rot, cam_trans = pose_err_deg_m(result.X_CammountCam, gt.X_CammountCam)
            tag_rot, tag_trans = pose_err_deg_m(result.X_TagmountTag, gt.X_TagmountTag)
            history.append(
                OpenCVErrorSample(
                    sample_count=sample_count,
                    method_name=method_name,
                    cam_rot_deg=cam_rot,
                    cam_trans_m=cam_trans,
                    tag_rot_deg=tag_rot,
                    tag_trans_m=tag_trans,
                    residual_rot_deg_mean=result.residual_rot_deg_mean,
                    residual_trans_m_mean=result.residual_trans_m_mean,
                )
            )

    final_rows = [row for row in history if row.sample_count == args.samples]
    final_rows.sort(key=lambda row: (row.cam_rot_deg, row.cam_trans_m))
    print("===== OpenCV thirdview camera calibration =====")
    for row in final_rows:
        print(
            f"{row.method_name:18s} "
            f"cam err: {row.cam_rot_deg:8.3f} deg, {row.cam_trans_m:9.5f} m | "
            f"tag err: {row.tag_rot_deg:8.3f} deg, {row.tag_trans_m:9.5f} m | "
            f"residual: {row.residual_rot_deg_mean:8.3f} deg, {row.residual_trans_m_mean:9.5f} m"
        )

    if args.error_csv is not None:
        write_error_csv(args.error_csv, history)
        print(f"Saved OpenCV error CSV to {args.error_csv}")
    if args.error_plot is not None:
        save_camera_error_plot(args.error_plot, history)
        print(f"Saved OpenCV camera error plot to {args.error_plot}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-thirdview", action="store_true", help="Collect PyBullet third-view sim data and calibrate it.")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size", type=float, default=0.048)
    parser.add_argument("--tag-marker-fraction", type=float, default=0.92)
    parser.add_argument("--tag-rows", type=int, default=4)
    parser.add_argument("--tag-cols", type=int, default=4)
    parser.add_argument("--tag-gap-ratio", type=float, default=0.0)
    parser.add_argument("--board-min-tags", type=int, default=4)
    parser.add_argument("--tag-board-yaml", type=Path, default=REPO_ROOT / "assets" / "apriltag_grid" / "compact_apriltag_grid_4x4_tag48mm.yaml")
    parser.add_argument("--tag-board-image", type=Path, default=REPO_ROOT / "assets" / "apriltag_grid" / "compact_apriltag_grid_4x4_tag48mm_board_only.png")
    parser.add_argument("--tag-board-image-width-m", type=float, default=None)
    parser.add_argument("--tag-board-image-height-m", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--near", type=float, default=0.02)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument("--renderer", choices=["tiny", "opengl"], default="tiny")
    parser.add_argument("--max-attempts-per-sample", type=int, default=80)
    parser.add_argument(
        "--use-gt-tag-pose",
        action="store_true",
        help="Diagnostic only: use the simulator GT X_CamTag instead of AprilTag detection.",
    )
    parser.add_argument("--error-plot", type=Path, default=SIM_OUTPUTS_DIR / "opencv_error_plot_thirdview.png")
    parser.add_argument("--error-csv", type=Path, default=SIM_OUTPUTS_DIR / "opencv_error_history_thirdview.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sim_thirdview:
        raise SystemExit("Currently use --sim-thirdview to run the OpenCV comparison on simulated data.")
    if args.samples < 3:
        raise SystemExit("--samples must be at least 3.")
    if args.tag_size <= 0:
        raise SystemExit("--tag-size must be positive.")
    if args.tag_marker_fraction <= 0 or args.tag_marker_fraction > 1:
        raise SystemExit("--tag-marker-fraction must be in (0, 1].")
    if args.tag_rows <= 0 or args.tag_cols <= 0:
        raise SystemExit("--tag-rows and --tag-cols must be positive.")
    if args.tag_gap_ratio < 0:
        raise SystemExit("--tag-gap-ratio must be non-negative.")
    if args.board_min_tags <= 0:
        raise SystemExit("--board-min-tags must be positive.")
    if args.board_min_tags > args.tag_rows * args.tag_cols:
        raise SystemExit("--board-min-tags cannot exceed the number of grid tags.")
    if (args.tag_board_yaml is None) != (args.tag_board_image is None):
        raise SystemExit("--tag-board-yaml and --tag-board-image must be provided together.")
    if args.tag_board_yaml is not None:
        if not args.tag_board_yaml.exists():
            raise SystemExit(f"--tag-board-yaml does not exist: {args.tag_board_yaml}")
        if not args.tag_board_image.exists():
            raise SystemExit(f"--tag-board-image does not exist: {args.tag_board_image}")
    if args.tag_board_image_width_m is not None and args.tag_board_image_width_m <= 0:
        raise SystemExit("--tag-board-image-width-m must be positive.")
    if args.tag_board_image_height_m is not None and args.tag_board_image_height_m <= 0:
        raise SystemExit("--tag-board-image-height-m must be positive.")
    if args.min_samples < 3:
        raise SystemExit("--min-samples must be at least 3.")
    if args.min_samples > args.samples:
        raise SystemExit("--min-samples cannot exceed --samples.")
    run_sim_thirdview(args)


if __name__ == "__main__":
    main()
