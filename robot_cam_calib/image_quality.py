"""Reusable target-ROI sharpness metrics for calibration images."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PlanarTargetQualityConfig:
    """Geometry and thresholds for comparing planar target sharpness."""

    width_m: float
    height_m: float
    canonical_width_px: int = 700
    canonical_height_px: int = 500
    robust_z_limit: float = -2.5
    max_reject_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("Planar target dimensions must be positive")
        if self.canonical_width_px < 32 or self.canonical_height_px < 32:
            raise ValueError("Canonical target image must be at least 32x32")
        if not 0.0 <= self.max_reject_fraction < 1.0:
            raise ValueError("max_reject_fraction must be in [0, 1)")


def focus_metric(image_gray: np.ndarray) -> float:
    """Return a Tenengrad-dominant focus score; larger means sharper."""

    image = np.asarray(image_gray)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("focus_metric expects a non-empty grayscale image")
    laplacian = cv2.Laplacian(image, cv2.CV_32F)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient_sq = grad_x * grad_x + grad_y * grad_y
    return float(np.mean(gradient_sq) + 0.05 * np.var(laplacian))


def _project_points(
    object_points: np.ndarray,
    T_camera_target: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    camera_model: str,
) -> np.ndarray:
    transform = np.asarray(T_camera_target, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    tvec = transform[:3, 3].reshape(3, 1)
    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    intrinsics = np.asarray(K, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(dist, dtype=np.float64).reshape(-1)
    if str(camera_model).lower() == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            points, rvec, tvec, intrinsics, distortion.reshape(-1, 1)
        )
    else:
        projected, _ = cv2.projectPoints(
            points, rvec, tvec, intrinsics, distortion
        )
    return projected.reshape(-1, 2)


def canonical_planar_target_patch(
    image: np.ndarray,
    T_camera_target: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    config: PlanarTargetQualityConfig,
    *,
    camera_model: str = "pinhole",
) -> np.ndarray:
    """Perspective-normalize a planar rectangular target to a fixed image."""

    frame = np.asarray(image)
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError("Calibration image must be grayscale or BGR")
    outer = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [config.width_m, 0.0, 0.0],
            [config.width_m, config.height_m, 0.0],
            [0.0, config.height_m, 0.0],
        ],
        dtype=np.float64,
    )
    projected = _project_points(
        outer, T_camera_target, K, dist, camera_model
    ).astype(np.float32)
    if not np.all(np.isfinite(projected)):
        raise ValueError("Projected target ROI contains non-finite coordinates")
    area = abs(float(cv2.contourArea(projected.reshape(-1, 1, 2))))
    if area < 100.0:
        raise ValueError(f"Projected target ROI is too small: {area:.1f}px^2")
    width = config.canonical_width_px
    height = config.canonical_height_px
    destination = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(projected, destination)
    return cv2.warpPerspective(
        gray, homography, (width, height), flags=cv2.INTER_LINEAR
    )


def planar_target_sharpness(
    image: np.ndarray,
    T_camera_target: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    config: PlanarTargetQualityConfig,
    *,
    camera_model: str = "pinhole",
) -> float:
    patch = canonical_planar_target_patch(
        image,
        T_camera_target,
        K,
        dist,
        config,
        camera_model=camera_model,
    )
    return focus_metric(patch)


def robust_low_sharpness_indices(
    indexed_scores: list[tuple[Any, float]],
    *,
    robust_z_limit: float = -2.5,
    max_reject_fraction: float = 0.10,
    minimum_scores: int = 8,
) -> tuple[set[Any], dict[str, Any]]:
    """Identify only extreme low-focus outliers within a comparable session."""

    finite = [
        (index, float(score))
        for index, score in indexed_scores
        if np.isfinite(score) and score > 0.0
    ]
    report: dict[str, Any] = {
        "finite_count": len(finite),
        "robust_z_limit": float(robust_z_limit),
        "max_reject_fraction": float(max_reject_fraction),
    }
    if len(finite) < minimum_scores or max_reject_fraction <= 0.0:
        report.update({"rejected_indices": [], "reason": "too_few_scores_or_disabled"})
        return set(), report
    values = np.log(np.asarray([score for _index, score in finite]))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-9)
    robust_z = (values - median) / scale
    candidates = [
        (finite[position][0], float(robust_z[position]))
        for position in range(len(finite))
        if robust_z[position] < robust_z_limit
    ]
    maximum = int(math.floor(len(finite) * max_reject_fraction))
    rejected = {
        index
        for index, _z in sorted(candidates, key=lambda item: item[1])[:maximum]
    }
    report.update(
        {
            "log_median": median,
            "log_mad": mad,
            "rejected_indices": sorted(rejected, key=str),
        }
    )
    return rejected, report
