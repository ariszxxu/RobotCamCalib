#!/usr/bin/env python3
"""Synthetic intrinsics calibration check using rendered checkerboard frames.

This script does not use a robot. It renders a checkerboard from random camera
poses, runs ``intr_calib.SimpleIntrinsicsCalibrator`` on the rendered RGB
frames, and compares the estimated intrinsics with the known simulation ground
truth.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_TESTS_DIR = Path(__file__).resolve().parent
SIM_OUTPUTS_DIR = SIM_TESTS_DIR / "outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intr_calib import CHECKERBOARD, MIN_SAMPLES, SQUARE_SIZE, SimpleIntrinsicsCalibrator


@dataclass
class RenderedSample:
    rgb: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray


def make_camera_matrix(
    width: int,
    height: int,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
) -> np.ndarray:
    fx_value = float(fx if fx is not None else 0.95 * width)
    fy_value = float(fy if fy is not None else 1.02 * fx_value)
    cx_value = float(cx if cx is not None else 0.51 * width)
    cy_value = float(cy if cy is not None else 0.49 * height)
    return np.array(
        [
            [fx_value, 0.0, cx_value],
            [0.0, fy_value, cy_value],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def checkerboard_square_count() -> Tuple[int, int]:
    inner_cols, inner_rows = CHECKERBOARD
    return inner_cols + 1, inner_rows + 1


def checkerboard_size() -> Tuple[float, float]:
    square_cols, square_rows = checkerboard_square_count()
    return square_cols * SQUARE_SIZE, square_rows * SQUARE_SIZE


def board_outer_corners() -> np.ndarray:
    board_w, board_h = checkerboard_size()
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [board_w, 0.0, 0.0],
            [board_w, board_h, 0.0],
            [0.0, board_h, 0.0],
        ],
        dtype=np.float32,
    )


def scaled_intrinsics(K: np.ndarray, scale: int) -> np.ndarray:
    K_scaled = K.copy()
    K_scaled[0, :] *= scale
    K_scaled[1, :] *= scale
    return K_scaled


def project_points(
    points_board: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    pts, _ = cv2.projectPoints(points_board.astype(np.float32), rvec, tvec, K, dist)
    return pts.reshape(-1, 2)


def transform_points(points_board: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R_board_cam, _ = cv2.Rodrigues(rvec)
    return (R_board_cam @ points_board.T).T + tvec.reshape(1, 3)


def sample_pose(
    rng: np.random.Generator,
    *,
    tilt_deg: float,
    roll_deg: float,
    distance_min: float,
    distance_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rx = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))
    ry = np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))
    rz = np.deg2rad(rng.uniform(-roll_deg, roll_deg))
    R_board_cam = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    rvec, _ = cv2.Rodrigues(R_board_cam)

    board_w, board_h = checkerboard_size()
    board_center = np.array([0.5 * board_w, 0.5 * board_h, 0.0], dtype=np.float64)
    z = rng.uniform(distance_min, distance_max)
    center_cam = np.array(
        [
            rng.uniform(-0.16, 0.16) * z,
            rng.uniform(-0.12, 0.12) * z,
            z,
        ],
        dtype=np.float64,
    )
    tvec = center_cam - R_board_cam @ board_center
    return rvec.reshape(3, 1), tvec.reshape(3, 1)


def pose_is_usable(
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    width: int,
    height: int,
    margin_px: float,
    min_board_px: float,
) -> bool:
    outer = board_outer_corners()
    points_cam = transform_points(outer.astype(np.float64), rvec, tvec)
    if np.any(points_cam[:, 2] <= 1e-6):
        return False

    img_pts = project_points(outer, rvec, tvec, K, dist)
    if np.any(img_pts[:, 0] < margin_px) or np.any(img_pts[:, 0] > width - margin_px):
        return False
    if np.any(img_pts[:, 1] < margin_px) or np.any(img_pts[:, 1] > height - margin_px):
        return False

    bbox = img_pts.max(axis=0) - img_pts.min(axis=0)
    return bool(min(bbox[0], bbox[1]) >= min_board_px)


def render_checkerboard(
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    width: int,
    height: int,
    supersample: int,
    blur_ksize: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    render_w = width * supersample
    render_h = height * supersample
    K_render = scaled_intrinsics(K, supersample)
    image = np.full((render_h, render_w, 3), 255, dtype=np.uint8)
    square_cols, square_rows = checkerboard_square_count()

    for row in range(square_rows):
        for col in range(square_cols):
            if (row + col) % 2 != 0:
                continue

            x0 = col * SQUARE_SIZE
            y0 = row * SQUARE_SIZE
            x1 = x0 + SQUARE_SIZE
            y1 = y0 + SQUARE_SIZE
            square = np.array(
                [
                    [x0, y0, 0.0],
                    [x1, y0, 0.0],
                    [x1, y1, 0.0],
                    [x0, y1, 0.0],
                ],
                dtype=np.float32,
            )
            pts = project_points(square, rvec, tvec, K_render, dist)
            cv2.fillConvexPoly(
                image,
                np.round(pts).astype(np.int32),
                color=(0, 0, 0),
                lineType=cv2.LINE_AA,
            )

    if supersample > 1:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    if noise_std > 0:
        noisy = image.astype(np.float32) + rng.normal(0.0, noise_std, size=image.shape)
        image = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
    return image


def detect_checkerboard(rgb: np.ndarray) -> bool:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ok, _ = SimpleIntrinsicsCalibrator._detect_chessboard(gray)
    return bool(ok)


def collect_samples(args: argparse.Namespace, K_gt: np.ndarray, dist_gt: np.ndarray) -> List[RenderedSample]:
    rng = np.random.default_rng(args.seed)
    samples: List[RenderedSample] = []
    attempts = 0
    while len(samples) < args.samples and attempts < args.max_attempts:
        attempts += 1
        rvec, tvec = sample_pose(
            rng,
            tilt_deg=args.tilt_deg,
            roll_deg=args.roll_deg,
            distance_min=args.distance_min,
            distance_max=args.distance_max,
        )
        if not pose_is_usable(
            rvec,
            tvec,
            K_gt,
            dist_gt,
            width=args.width,
            height=args.height,
            margin_px=args.margin_px,
            min_board_px=args.min_board_px,
        ):
            continue

        rgb = render_checkerboard(
            rvec,
            tvec,
            K_gt,
            dist_gt,
            width=args.width,
            height=args.height,
            supersample=args.supersample,
            blur_ksize=args.blur_ksize,
            noise_std=args.noise_std,
            rng=rng,
        )
        if detect_checkerboard(rgb):
            samples.append(RenderedSample(rgb=rgb, rvec=rvec, tvec=tvec))

    if len(samples) < args.samples:
        raise RuntimeError(
            f"Only rendered {len(samples)} detectable checkerboards after {attempts} attempts; "
            "relax pose bounds or increase --max-attempts."
        )
    print(f"Rendered {len(samples)} detectable checkerboard samples after {attempts} attempts.")
    return samples


def write_preview(path: Path, frames_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(len(frames_rgb), 12)
    thumbs = []
    for i in range(count):
        frame = frames_rgb[i].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ok, corners = SimpleIntrinsicsCalibrator._detect_chessboard(gray)
        if ok and corners is not None:
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ok)
        cv2.putText(
            frame,
            f"{i:02d}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (40, 180, 40),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(cv2.resize(frame, (240, 180), interpolation=cv2.INTER_AREA))

    cols = 4
    rows = int(np.ceil(count / cols))
    canvas = np.full((rows * 180, cols * 240, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        row = i // cols
        col = i % cols
        canvas[row * 180 : (row + 1) * 180, col * 240 : (col + 1) * 240] = thumb
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_frames(frames_dir: Path, frames_rgb: np.ndarray) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames_rgb):
        cv2.imwrite(str(frames_dir / f"frame_{i:03d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def calibration_errors(K_est: np.ndarray, dist_est: np.ndarray, K_gt: np.ndarray, dist_gt: np.ndarray) -> dict:
    dist_est_flat = dist_est.reshape(-1)
    dist_gt_flat = dist_gt.reshape(-1)
    n = min(dist_est_flat.size, dist_gt_flat.size)
    dist_delta = dist_est_flat[:n] - dist_gt_flat[:n]
    return {
        "fx_err_px": float(K_est[0, 0] - K_gt[0, 0]),
        "fy_err_px": float(K_est[1, 1] - K_gt[1, 1]),
        "cx_err_px": float(K_est[0, 2] - K_gt[0, 2]),
        "cy_err_px": float(K_est[1, 2] - K_gt[1, 2]),
        "K_fro_err": float(np.linalg.norm(K_est - K_gt)),
        "mean_focal_rel_err_pct": float(
            50.0
            * (
                abs(K_est[0, 0] - K_gt[0, 0]) / K_gt[0, 0]
                + abs(K_est[1, 1] - K_gt[1, 1]) / K_gt[1, 1]
            )
        ),
        "dist_l2_err": float(np.linalg.norm(dist_delta)),
    }


def save_results(
    args: argparse.Namespace,
    K_gt: np.ndarray,
    dist_gt: np.ndarray,
    results: dict,
    errors: dict,
) -> None:
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "simulation": {
            "checkerboard_inner_corners": list(CHECKERBOARD),
            "square_size": float(SQUARE_SIZE),
            "samples_requested": int(args.samples),
            "image_size": [int(args.width), int(args.height)],
            "seed": int(args.seed),
            "noise_std": float(args.noise_std),
            "blur_ksize": int(args.blur_ksize),
        },
        "ground_truth": {
            "K": K_gt.tolist(),
            "dist": dist_gt.reshape(-1).tolist(),
        },
        "estimated": {
            "K": results["K"].tolist(),
            "dist": results["dist"].reshape(-1).tolist(),
            "rms": float(results["rms"]),
            "mean_reproj_error": float(results["mean_err"]),
            "used_indices": results["used_indices"].astype(int).tolist(),
        },
        "errors": errors,
    }
    with open(args.output_yaml, "w") as f:
        yaml.safe_dump(output, f, sort_keys=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "samples": int(args.samples),
        "used_samples": int(len(results["used_indices"])),
        "width": int(args.width),
        "height": int(args.height),
        "fx_gt": float(K_gt[0, 0]),
        "fx_est": float(results["K"][0, 0]),
        "fy_gt": float(K_gt[1, 1]),
        "fy_est": float(results["K"][1, 1]),
        "cx_gt": float(K_gt[0, 2]),
        "cx_est": float(results["K"][0, 2]),
        "cy_gt": float(K_gt[1, 2]),
        "cy_est": float(results["K"][1, 2]),
        "rms": float(results["rms"]),
        "mean_reproj_error": float(results["mean_err"]),
        **errors,
    }
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--dist", type=float, nargs=5, default=[0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--tilt-deg", type=float, default=32.0)
    parser.add_argument("--roll-deg", type=float, default=35.0)
    parser.add_argument("--distance-min", type=float, default=520.0)
    parser.add_argument("--distance-max", type=float, default=880.0)
    parser.add_argument("--margin-px", type=float, default=18.0)
    parser.add_argument("--min-board-px", type=float, default=135.0)
    parser.add_argument("--supersample", type=int, default=5)
    parser.add_argument("--blur-ksize", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frames-dir", type=Path, default=SIM_OUTPUTS_DIR / "intr_calib_sim_frames")
    parser.add_argument("--preview", type=Path, default=SIM_OUTPUTS_DIR / "intr_calib_sim_preview.png")
    parser.add_argument("--output-yaml", type=Path, default=SIM_OUTPUTS_DIR / "intr_calib_sim_result.yaml")
    parser.add_argument("--output-csv", type=Path, default=SIM_OUTPUTS_DIR / "intr_calib_sim_result.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < MIN_SAMPLES:
        raise SystemExit(f"--samples must be at least {MIN_SAMPLES} to match intr_calib.MIN_SAMPLES.")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive.")
    if args.supersample <= 0:
        raise SystemExit("--supersample must be positive.")
    if args.distance_min <= 0 or args.distance_max <= args.distance_min:
        raise SystemExit("--distance bounds must be positive and increasing.")
    if args.noise_std < 0:
        raise SystemExit("--noise-std must be non-negative.")

    K_gt = make_camera_matrix(args.width, args.height, args.fx, args.fy, args.cx, args.cy)
    dist_gt = np.asarray(args.dist, dtype=np.float64).reshape(1, -1)
    samples = collect_samples(args, K_gt, dist_gt)
    frames_rgb = np.stack([sample.rgb for sample in samples], axis=0)

    calibrator = SimpleIntrinsicsCalibrator()
    results = calibrator.calibrate(frames_rgb)
    errors = calibration_errors(results["K"], results["dist"], K_gt, dist_gt)
    save_results(args, K_gt, dist_gt, results, errors)
    write_preview(args.preview, frames_rgb)
    if args.save_frames:
        save_frames(args.frames_dir, frames_rgb)

    print("Ground-truth K:")
    print(K_gt)
    print("Estimated K:")
    print(results["K"])
    print(f"Used samples: {len(results['used_indices'])}/{args.samples}")
    print(f"RMS reprojection error: {float(results['rms']):.6f} px")
    print(f"Mean reprojection error: {float(results['mean_err']):.6f} px")
    print(
        "Intrinsics error: "
        f"fx={errors['fx_err_px']:.4f}px, fy={errors['fy_err_px']:.4f}px, "
        f"cx={errors['cx_err_px']:.4f}px, cy={errors['cy_err_px']:.4f}px, "
        f"K_fro={errors['K_fro_err']:.4f}, dist_l2={errors['dist_l2_err']:.6f}"
    )
    print(f"Saved result YAML to {args.output_yaml}")
    print(f"Saved result CSV to {args.output_csv}")
    print(f"Saved preview image to {args.preview}")


if __name__ == "__main__":
    main()
