#!/usr/bin/env python3
"""Offline, blur-aware ChArUco calibration for a pinhole camera.

The script re-detects ChArUco corners from one or more saved capture
directories, rejects low-sharpness frames, removes near-duplicate poses, and
iteratively rejects high-reprojection-error views.  It writes a calibration
YAML plus CSV diagnostics and contact sheets so the selection is auditable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pickle
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import yaml

from intr_calib_charuco import (
    CharucoDetectorAdapter,
    create_charuco_board,
    get_charuco_board_corners,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SAMPLE_RE = re.compile(r"sample_(\d+)_frame_(\d+)", re.IGNORECASE)
_WORKER_LOCAL = threading.local()


@dataclass
class View:
    image_path: str
    source_dir: str
    source_name: str
    sample_index: int
    frame_index: int
    image_width: int
    image_height: int
    quality_ok: bool
    quality_reason: str
    corner_count: int
    marker_count: int
    charuco_ids: Optional[np.ndarray] = None
    image_points: Optional[np.ndarray] = None
    object_points: Optional[np.ndarray] = None
    raw_laplacian_var: float = math.nan
    raw_tenengrad_mean: float = math.nan
    rectified_laplacian_var: float = math.nan
    rectified_tenengrad_mean: float = math.nan
    pixels_per_square: float = math.nan
    board_area_fraction: float = math.nan
    board_center_x: float = math.nan
    board_center_y: float = math.nan
    pose_features: Optional[np.ndarray] = None
    sharpness_score: float = math.nan
    sharpness_percentile: float = math.nan
    status: str = "unclassified"
    final_reproj_error: float = math.nan
    notes: list[str] = field(default_factory=list)

    @property
    def sample_id(self) -> str:
        return f"{self.source_name}:sample_{self.sample_index:04d}:frame_{self.frame_index:06d}"


def relative_or_absolute(path: Path, base: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return str(resolved)


def parse_sample_numbers(path: Path, fallback: int) -> tuple[int, int]:
    match = SAMPLE_RE.search(path.stem)
    if match is None:
        return fallback, fallback
    return int(match.group(1)), int(match.group(2))


def list_images(image_dirs: list[Path]) -> list[tuple[Path, Path]]:
    result: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for directory in image_dirs:
        resolved = directory.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {resolved}")
        paths = sorted(
            path
            for path in resolved.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise RuntimeError(f"No supported images found in: {resolved}")
        for path in paths:
            path = path.resolve()
            if path not in seen:
                result.append((resolved, path))
                seen.add(path)
    return result


def cache_fingerprint(
    image_entries: list[tuple[Path, Path]],
    charuco: dict[str, Any],
    min_corners: int,
    use_opencl: bool,
) -> str:
    payload = {
        "cache_version": 3,
        "charuco": charuco,
        "min_corners": min_corners,
        "gradient_backend": "opencl_umat" if use_opencl else "cpu",
        "images": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for _directory, path in image_entries
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def charuco_quality(
    ids_raw: Optional[np.ndarray],
    squares_x: int,
    squares_y: int,
    min_corners: int,
) -> tuple[bool, str]:
    if ids_raw is None:
        return False, f"corners 0 < {min_corners}"
    ids = np.asarray(ids_raw, dtype=np.int32).reshape(-1)
    if ids.size < min_corners:
        return False, f"corners {ids.size} < {min_corners}"

    inner_cols = squares_x - 1
    inner_rows = squares_y - 1
    valid = (ids >= 0) & (ids < inner_cols * inner_rows)
    ids = ids[valid]
    if ids.size < min_corners:
        return False, f"valid corners {ids.size} < {min_corners}"
    rows = ids // inner_cols
    cols = ids % inner_cols
    row_count = int(np.unique(rows).size)
    col_count = int(np.unique(cols).size)
    if row_count < 2:
        return False, f"grid rows {row_count} < 2"
    if col_count < 4:
        return False, f"grid cols {col_count} < 4"
    bbox_rows = int(rows.max() - rows.min() + 1)
    bbox_cols = int(cols.max() - cols.min() + 1)
    bbox_fraction = bbox_rows * bbox_cols / float(inner_rows * inner_cols)
    if bbox_fraction < 0.35:
        return False, f"board bbox {bbox_fraction:.2f} < 0.35"
    return True, (
        f"corners={ids.size} rows={row_count} cols={col_count} "
        f"board_bbox={bbox_fraction:.2f}"
    )


def gradient_metrics(
    gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    use_opencl: bool = False,
) -> tuple[float, float]:
    if use_opencl:
        gray_umat = cv2.UMat(gray)
        lap = cv2.Laplacian(gray_umat, cv2.CV_32F)
        grad_x = cv2.Sobel(gray_umat, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_umat, cv2.CV_32F, 0, 1, ksize=3)
        grad_sq = cv2.add(
            cv2.multiply(grad_x, grad_x),
            cv2.multiply(grad_y, grad_y),
        )
        # OpenCV 4.5.x can fail when meanStdDev receives a masked UMat after a
        # multi-threaded CPU stage. Keep the expensive filters on OpenCL, then
        # download their outputs and do the small masked reductions on CPU.
        lap_array = lap.get()
        grad_sq_array = grad_sq.get()
        if mask is None:
            laplacian_var = float(np.var(lap_array))
            tenengrad_mean = float(np.mean(grad_sq_array))
        else:
            valid = mask > 0
            if int(np.count_nonzero(valid)) < 100:
                return math.nan, math.nan
            laplacian_var = float(np.var(lap_array[valid]))
            tenengrad_mean = float(np.mean(grad_sq_array[valid]))
        return laplacian_var, tenengrad_mean

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_sq = grad_x * grad_x + grad_y * grad_y
    if mask is None:
        return float(np.var(lap)), float(np.mean(grad_sq))
    valid = mask > 0
    if int(np.count_nonzero(valid)) < 100:
        return math.nan, math.nan
    return float(np.var(lap[valid])), float(np.mean(grad_sq[valid]))


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(np.sum(ordered, axis=1)))
    return np.roll(ordered, -start, axis=0)


def compute_view_metrics(
    gray: np.ndarray,
    image_points: np.ndarray,
    ids: np.ndarray,
    squares_x: int,
    squares_y: int,
    square_length: float,
    use_opencl: bool,
    compute_gradients: bool = True,
) -> dict[str, Any]:
    points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
    height, width = gray.shape[:2]

    x, y, bbox_width, bbox_height = cv2.boundingRect(np.round(points).astype(np.int32))
    padding = max(8, int(round(0.08 * max(bbox_width, bbox_height))))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + bbox_width + padding)
    y1 = min(height, y + bbox_height + padding)
    if compute_gradients:
        raw_lap, raw_ten = gradient_metrics(
            gray[y0:y1, x0:x1],
            use_opencl=use_opencl,
        )
    else:
        raw_lap, raw_ten = math.nan, math.nan

    if compute_gradients:
        canonical_pixels_per_square = 100.0
        canonical_points = np.column_stack(
            (
                (ids_flat % (squares_x - 1) + 1) * canonical_pixels_per_square,
                (ids_flat // (squares_x - 1) + 1) * canonical_pixels_per_square,
            )
        ).astype(np.float32)
        homography, _mask = cv2.findHomography(points, canonical_points, method=0)
        if homography is None:
            rect_lap = math.nan
            rect_ten = math.nan
        else:
            rectified = cv2.warpPerspective(
                gray,
                homography,
                (
                    int(round(squares_x * canonical_pixels_per_square)),
                    int(round(squares_y * canonical_pixels_per_square)),
                ),
                flags=cv2.INTER_LINEAR,
            )
            rectified_mask = np.zeros_like(rectified, dtype=np.uint8)
            hull = cv2.convexHull(np.round(canonical_points).astype(np.int32))
            cv2.fillConvexPoly(rectified_mask, hull, 255)
            rectified_mask = cv2.erode(
                rectified_mask,
                np.ones((7, 7), dtype=np.uint8),
            )
            rect_lap, rect_ten = gradient_metrics(
                rectified,
                rectified_mask,
                use_opencl=use_opencl,
            )
    else:
        rect_lap, rect_ten = math.nan, math.nan

    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    center = np.mean(points, axis=0)
    area_fraction = hull_area / max(float(width * height), 1.0)

    object_xy = np.column_stack(
        (
            (ids_flat % (squares_x - 1) + 1) * square_length,
            (ids_flat // (squares_x - 1) + 1) * square_length,
        )
    ).astype(np.float32)
    board_to_image, _mask = cv2.findHomography(object_xy, points, method=0)
    if board_to_image is None:
        quad = np.asarray(
            [[x, y], [x + bbox_width, y], [x + bbox_width, y + bbox_height], [x, y + bbox_height]],
            dtype=np.float64,
        )
    else:
        board_quad = np.asarray(
            [
                [0.0, 0.0],
                [squares_x * square_length, 0.0],
                [squares_x * square_length, squares_y * square_length],
                [0.0, squares_y * square_length],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(board_quad, board_to_image).reshape(-1, 2)
    quad = order_quad(quad)
    edges = np.asarray(
        [
            np.linalg.norm(quad[1] - quad[0]),
            np.linalg.norm(quad[2] - quad[1]),
            np.linalg.norm(quad[3] - quad[2]),
            np.linalg.norm(quad[0] - quad[3]),
        ],
        dtype=np.float64,
    )
    pixels_per_square = float(
        np.median(
            [
                edges[0] / max(squares_x, 1),
                edges[2] / max(squares_x, 1),
                edges[1] / max(squares_y, 1),
                edges[3] / max(squares_y, 1),
            ]
        )
    )
    direction = quad[1] - quad[0]
    angle = math.atan2(float(direction[1]), float(direction[0]))
    horizontal_tilt = math.log(max(edges[0], 1e-6) / max(edges[2], 1e-6))
    vertical_tilt = math.log(max(edges[1], 1e-6) / max(edges[3], 1e-6))
    features = np.asarray(
        [
            3.0 * center[0] / width,
            3.0 * center[1] / height,
            1.3 * math.log(max(area_fraction, 1e-8)),
            0.5 * math.cos(angle),
            0.5 * math.sin(angle),
            0.8 * horizontal_tilt,
            0.8 * vertical_tilt,
        ],
        dtype=np.float64,
    )
    return {
        "raw_laplacian_var": raw_lap,
        "raw_tenengrad_mean": raw_ten,
        "rectified_laplacian_var": rect_lap,
        "rectified_tenengrad_mean": rect_ten,
        "pixels_per_square": pixels_per_square,
        "board_area_fraction": area_fraction,
        "board_center_x": float(center[0] / width),
        "board_center_y": float(center[1] / height),
        "pose_features": features,
    }


def worker_detector(charuco: dict[str, Any]) -> tuple[Any, CharucoDetectorAdapter, np.ndarray]:
    cached = getattr(_WORKER_LOCAL, "charuco_detector", None)
    if cached is not None:
        return cached
    board, dictionary = create_charuco_board(
        int(charuco["squares_x"]),
        int(charuco["squares_y"]),
        float(charuco["square_length"]),
        float(charuco["marker_length"]),
        str(charuco["dictionary"]),
        bool(charuco.get("legacy_pattern", False)),
    )
    cached = (
        board,
        CharucoDetectorAdapter(board, dictionary),
        get_charuco_board_corners(board),
    )
    _WORKER_LOCAL.charuco_detector = cached
    return cached


def detect_one_view(
    position: int,
    source_dir: Path,
    image_path: Path,
    charuco: dict[str, Any],
    min_corners: int,
    use_opencl: bool,
    compute_gradients: bool,
) -> View:
    if use_opencl and not getattr(_WORKER_LOCAL, "opencl_enabled", False):
        cv2.ocl.setUseOpenCL(True)
        _WORKER_LOCAL.opencl_enabled = bool(cv2.ocl.useOpenCL())
    _board, detector, board_corners = worker_detector(charuco)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    height, width = gray.shape[:2]
    sample_index, frame_index = parse_sample_numbers(image_path, position)
    corners, ids, _marker_corners, marker_ids = detector.detect(gray)
    ok, reason = charuco_quality(
        ids,
        int(charuco["squares_x"]),
        int(charuco["squares_y"]),
        min_corners,
    )
    corner_count = 0 if ids is None else int(np.asarray(ids).size)
    marker_count = 0 if marker_ids is None else int(np.asarray(marker_ids).size)
    view = View(
        image_path=str(image_path),
        source_dir=str(source_dir),
        source_name=source_dir.name,
        sample_index=sample_index,
        frame_index=frame_index,
        image_width=width,
        image_height=height,
        quality_ok=ok,
        quality_reason=reason,
        corner_count=corner_count,
        marker_count=marker_count,
    )
    if ok and corners is not None and ids is not None:
        ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
        valid = (ids_flat >= 0) & (ids_flat < len(board_corners))
        ids_flat = ids_flat[valid]
        image_points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)[valid]
        view.charuco_ids = ids_flat.reshape(-1, 1)
        view.image_points = image_points.reshape(-1, 1, 2)
        view.object_points = board_corners[ids_flat].reshape(-1, 3)
        metrics = compute_view_metrics(
            gray,
            image_points,
            ids_flat,
            int(charuco["squares_x"]),
            int(charuco["squares_y"]),
            float(charuco["square_length"]),
            use_opencl,
            compute_gradients,
        )
        for key, value in metrics.items():
            setattr(view, key, value)
    else:
        view.status = "rejected_detection"
    return view


def detect_views(
    image_entries: list[tuple[Path, Path]],
    charuco: dict[str, Any],
    min_corners: int,
    workers: int,
    use_opencl: bool,
) -> list[View]:
    total = len(image_entries)
    views: list[Optional[View]] = [None] * total
    staged_opencl = bool(use_opencl and workers > 1)
    worker_use_opencl = bool(use_opencl and not staged_opencl)
    worker_compute_gradients = not staged_opencl
    if workers <= 1:
        for position, (source_dir, image_path) in enumerate(image_entries):
            views[position] = detect_one_view(
                position,
                source_dir,
                image_path,
                charuco,
                min_corners,
                worker_use_opencl,
                worker_compute_gradients,
            )
            if (position + 1) % 20 == 0 or position + 1 == total:
                print(f"[INFO] Detected {position + 1}/{total} images", flush=True)
    else:
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_positions = {
                executor.submit(
                    detect_one_view,
                    position,
                    source_dir,
                    image_path,
                    charuco,
                    min_corners,
                    worker_use_opencl,
                    worker_compute_gradients,
                ): position
                for position, (source_dir, image_path) in enumerate(image_entries)
            }
            for future in concurrent.futures.as_completed(future_positions):
                position = future_positions[future]
                views[position] = future.result()
                completed += 1
                if completed % 20 == 0 or completed == total:
                    print(f"[INFO] Detected {completed}/{total} images", flush=True)

    result = [view for view in views if view is not None]
    if staged_opencl:
        cv2.ocl.setUseOpenCL(True)
        valid = [view for view in result if view.quality_ok]
        print(
            f"[INFO] GPU sharpness stage: {len(valid)} valid board images",
            flush=True,
        )
        for position, view in enumerate(valid):
            gray = cv2.imread(view.image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"Failed to re-read image: {view.image_path}")
            metrics = compute_view_metrics(
                gray,
                np.asarray(view.image_points).reshape(-1, 2),
                np.asarray(view.charuco_ids).reshape(-1),
                int(charuco["squares_x"]),
                int(charuco["squares_y"]),
                float(charuco["square_length"]),
                use_opencl=True,
                compute_gradients=True,
            )
            for key in (
                "raw_laplacian_var",
                "raw_tenengrad_mean",
                "rectified_laplacian_var",
                "rectified_tenengrad_mean",
            ):
                setattr(view, key, metrics[key])
            if (position + 1) % 20 == 0 or position + 1 == len(valid):
                print(
                    f"[INFO] GPU sharpness {position + 1}/{len(valid)}",
                    flush=True,
                )
    return result


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, float(np.std(finite)) * 0.1, 1e-6)
    return median, scale


def assign_sharpness_scores(views: list[View], blur_fraction: float) -> float:
    valid = [view for view in views if view.quality_ok]
    if not valid:
        raise RuntimeError("No valid detected views.")

    # Normalize sharpness within pixels-per-square bins.  This separates actual
    # blur from the expected loss of high frequencies when the board is small.
    pps = np.asarray([view.pixels_per_square for view in valid], dtype=np.float64)
    quantiles = np.unique(np.quantile(pps, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if quantiles.size < 3:
        quantiles = np.asarray([float(np.min(pps)) - 1.0, float(np.max(pps)) + 1.0])
    scores = np.zeros(len(valid), dtype=np.float64)
    assigned = np.zeros(len(valid), dtype=bool)
    for bin_index in range(len(quantiles) - 1):
        lower = quantiles[bin_index]
        upper = quantiles[bin_index + 1]
        if bin_index == len(quantiles) - 2:
            mask = (pps >= lower) & (pps <= upper)
        else:
            mask = (pps >= lower) & (pps < upper)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        log_lap = np.log1p(
            np.asarray([valid[index].rectified_laplacian_var for index in indices])
        )
        log_ten = np.log1p(
            np.asarray([valid[index].rectified_tenengrad_mean for index in indices])
        )
        lap_median, lap_scale = robust_location_scale(log_lap)
        ten_median, ten_scale = robust_location_scale(log_ten)
        scores[indices] = 0.5 * (
            (log_lap - lap_median) / lap_scale
            + (log_ten - ten_median) / ten_scale
        )
        assigned[indices] = True
    if not np.all(assigned):
        scores[~assigned] = 0.0

    threshold = float(np.quantile(scores, min(max(blur_fraction, 0.0), 0.49)))
    rank_order = np.argsort(scores)
    percentiles = np.empty_like(scores)
    if scores.size == 1:
        percentiles[:] = 1.0
    else:
        percentiles[rank_order] = np.arange(scores.size) / float(scores.size - 1)
    for index, view in enumerate(valid):
        view.sharpness_score = float(scores[index])
        view.sharpness_percentile = float(percentiles[index])
        if scores[index] <= threshold:
            view.status = "rejected_blur"
            view.notes.append("low rectified board sharpness at comparable board scale")
        else:
            view.status = "candidate"
    return threshold


def standardized_pose_features(views: list[View]) -> np.ndarray:
    matrix = np.vstack([np.asarray(view.pose_features, dtype=np.float64) for view in views])
    medians = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - medians), axis=0)
    scale = np.maximum(1.4826 * mad, np.std(matrix, axis=0) * 0.25)
    scale = np.maximum(scale, 1e-6)
    return (matrix - medians) / scale


def farthest_pose_selection(views: list[View], count: int) -> list[View]:
    if count >= len(views):
        return list(views)
    features = standardized_pose_features(views)
    sharpness = np.asarray([view.sharpness_score for view in views], dtype=np.float64)
    sharp_median, sharp_scale = robust_location_scale(sharpness)
    sharp_z = np.clip((sharpness - sharp_median) / sharp_scale, -3.0, 3.0)

    selected = [int(np.argmax(sharp_z))]
    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)
    min_dist[selected[0]] = -np.inf
    while len(selected) < count:
        score = min_dist + 0.12 * sharp_z
        next_index = int(np.argmax(score))
        selected.append(next_index)
        distance = np.linalg.norm(features - features[next_index], axis=1)
        min_dist = np.minimum(min_dist, distance)
        min_dist[selected] = -np.inf
    return [views[index] for index in selected]


def select_balanced_diverse_views(views: list[View], max_views: int) -> list[View]:
    candidates = [view for view in views if view.status == "candidate"]
    if len(candidates) <= max_views:
        return candidates
    grouped: dict[str, list[View]] = {}
    for view in candidates:
        grouped.setdefault(view.source_dir, []).append(view)
    total = len(candidates)
    allocations: dict[str, int] = {}
    remaining = max_views
    source_items = sorted(grouped.items())
    for position, (source, group) in enumerate(source_items):
        if position == len(source_items) - 1:
            allocation = remaining
        else:
            allocation = int(round(max_views * len(group) / total))
            allocation = max(min(allocation, len(group)), min(12, len(group)))
            allocation = min(allocation, remaining)
        allocations[source] = allocation
        remaining -= allocation
    while remaining > 0:
        expandable = [
            source
            for source, group in source_items
            if allocations[source] < len(grouped[source])
        ]
        if not expandable:
            break
        for source in expandable:
            allocations[source] += 1
            remaining -= 1
            if remaining == 0:
                break

    selected: list[View] = []
    for source, group in source_items:
        selected.extend(farthest_pose_selection(group, allocations[source]))
    selected_ids = {id(view) for view in selected}
    for view in candidates:
        if id(view) not in selected_ids:
            view.status = "rejected_pose_redundancy"
    return selected


def calibration_inputs(views: Iterable[View]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    views_list = list(views)
    return (
        [np.asarray(view.object_points, dtype=np.float32).reshape(-1, 3) for view in views_list],
        [np.asarray(view.image_points, dtype=np.float32).reshape(-1, 1, 2) for view in views_list],
    )


def calibrate(
    views: list[View],
    image_size: tuple[int, int],
    K_seed: Optional[np.ndarray] = None,
    dist_seed: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    object_points, image_points = calibration_inputs(views)
    flags = 0
    K_input = None
    dist_input = None
    if K_seed is not None:
        K_input = np.asarray(K_seed, dtype=np.float64).reshape(3, 3).copy()
        if dist_seed is None:
            dist_input = np.zeros((5, 1), dtype=np.float64)
        else:
            dist_input = np.asarray(dist_seed, dtype=np.float64).reshape(-1, 1).copy()
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-8,
    )
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        K_input,
        dist_input,
        flags=flags,
        criteria=criteria,
    )
    errors: list[float] = []
    total_sq = 0.0
    total_points = 0
    for object_pts, image_pts, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        projected, _jacobian = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
        delta = image_pts.reshape(-1, 2) - projected.reshape(-1, 2)
        squared = np.sum(delta * delta, axis=1)
        errors.append(float(np.sqrt(np.mean(squared))))
        total_sq += float(np.sum(squared))
        total_points += int(len(squared))
    return {
        "rms": float(rms),
        "mean_reproj_error": float(math.sqrt(total_sq / max(total_points, 1))),
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "per_view_errors": errors,
    }


def robust_calibrate(
    initial_views: list[View],
    image_size: tuple[int, int],
    K_seed: np.ndarray,
    dist_seed: np.ndarray,
    min_views: int,
    max_view_error: float,
    max_rounds: int,
) -> tuple[list[View], dict[str, Any], list[dict[str, Any]]]:
    active = list(initial_views)
    history: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    for round_index in range(max_rounds + 1):
        print(
            f"[INFO] Calibration round {round_index + 1}: {len(active)} views",
            flush=True,
        )
        result = calibrate(active, image_size, K_seed, dist_seed)
        errors = np.asarray(result["per_view_errors"], dtype=np.float64)
        median, robust_scale = robust_location_scale(errors)
        threshold = min(max_view_error, median + 3.0 * robust_scale)
        threshold = max(threshold, median + 0.12)
        bad = np.flatnonzero(errors > threshold)
        max_remove = max(1, int(math.ceil(0.10 * len(active))))
        if bad.size > max_remove:
            bad = bad[np.argsort(errors[bad])[::-1][:max_remove]]
        if len(active) - int(bad.size) < min_views:
            allowable = max(0, len(active) - min_views)
            bad = bad[np.argsort(errors[bad])[::-1][:allowable]]
        history.append(
            {
                "round": round_index + 1,
                "num_views": len(active),
                "rms": float(result["rms"]),
                "median_view_error": median,
                "threshold": float(threshold),
                "num_rejected": int(bad.size),
            }
        )
        print(
            f"[INFO] Round {round_index + 1}: rms={result['rms']:.6f} px, "
            f"median={median:.6f} px, threshold={threshold:.6f} px, "
            f"reject={bad.size}",
            flush=True,
        )
        if bad.size == 0 or round_index == max_rounds:
            break
        bad_set = set(int(index) for index in bad)
        for index in bad_set:
            active[index].status = "rejected_reprojection"
            active[index].final_reproj_error = float(errors[index])
            active[index].notes.append(
                f"joint calibration view error above {threshold:.4f} px"
            )
        active = [view for index, view in enumerate(active) if index not in bad_set]
        K_seed = result["K"]
        dist_seed = result["dist"]

    for view, error in zip(active, result["per_view_errors"]):
        view.status = "selected"
        view.final_reproj_error = float(error)
    return active, result, history


def independent_pnp_error(view: View, K: np.ndarray, dist: np.ndarray) -> float:
    object_points = np.asarray(view.object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.asarray(view.image_points, dtype=np.float32).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return math.nan
    projected, _jacobian = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    delta = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def cross_validate(
    selected: list[View],
    image_size: tuple[int, int],
    final_result: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(selected, key=lambda view: (view.source_name, view.frame_index))
    folds = [ordered[::2], ordered[1::2]]
    fold_results: list[dict[str, Any]] = []
    for fold_index in range(2):
        train = folds[fold_index]
        holdout = folds[1 - fold_index]
        if len(train) < 10 or len(holdout) < 10:
            continue
        print(
            f"[INFO] Cross-validation fold {fold_index + 1}: "
            f"train={len(train)}, holdout={len(holdout)}",
            flush=True,
        )
        trained = calibrate(
            train,
            image_size,
            final_result["K"],
            final_result["dist"],
        )
        holdout_errors = [
            independent_pnp_error(view, trained["K"], trained["dist"])
            for view in holdout
        ]
        finite_errors = np.asarray(
            [value for value in holdout_errors if np.isfinite(value)],
            dtype=np.float64,
        )
        fold_results.append(
            {
                "fold": fold_index + 1,
                "train_views": len(train),
                "holdout_views": len(holdout),
                "train_rms": float(trained["rms"]),
                "holdout_pnp_rmse": float(
                    np.sqrt(np.mean(finite_errors * finite_errors))
                ),
                "holdout_pnp_median": float(np.median(finite_errors)),
                "K": trained["K"].tolist(),
                "dist": trained["dist"].reshape(-1).tolist(),
            }
        )
    if not fold_results:
        return {"folds": []}
    fx_values = np.asarray([fold["K"][0][0] for fold in fold_results])
    fy_values = np.asarray([fold["K"][1][1] for fold in fold_results])
    cx_values = np.asarray([fold["K"][0][2] for fold in fold_results])
    cy_values = np.asarray([fold["K"][1][2] for fold in fold_results])
    return {
        "folds": fold_results,
        "fx_range_px": float(np.ptp(fx_values)),
        "fy_range_px": float(np.ptp(fy_values)),
        "cx_range_px": float(np.ptp(cx_values)),
        "cy_range_px": float(np.ptp(cy_values)),
        "mean_holdout_pnp_rmse": float(
            np.mean([fold["holdout_pnp_rmse"] for fold in fold_results])
        ),
    }


def view_to_csv_row(view: View, base: Path) -> dict[str, Any]:
    return {
        "sample_id": view.sample_id,
        "source_name": view.source_name,
        "sample_index": view.sample_index,
        "frame_index": view.frame_index,
        "image_path": relative_or_absolute(Path(view.image_path), base),
        "status": view.status,
        "quality_ok": view.quality_ok,
        "quality_reason": view.quality_reason,
        "corner_count": view.corner_count,
        "marker_count": view.marker_count,
        "raw_laplacian_var": view.raw_laplacian_var,
        "raw_tenengrad_mean": view.raw_tenengrad_mean,
        "rectified_laplacian_var": view.rectified_laplacian_var,
        "rectified_tenengrad_mean": view.rectified_tenengrad_mean,
        "pixels_per_square": view.pixels_per_square,
        "board_area_fraction": view.board_area_fraction,
        "board_center_x": view.board_center_x,
        "board_center_y": view.board_center_y,
        "sharpness_score": view.sharpness_score,
        "sharpness_percentile": view.sharpness_percentile,
        "final_reproj_error": view.final_reproj_error,
        "notes": "; ".join(view.notes),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(
    path: Path,
    views: list[View],
    title: str,
    sort_key: Any,
    max_images: int = 24,
) -> None:
    chosen = sorted(views, key=sort_key)[:max_images]
    if not chosen:
        return
    tile_width, tile_height = 360, 290
    columns = 4
    rows = int(math.ceil(len(chosen) / columns))
    canvas = np.full(
        (60 + rows * tile_height, columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        title,
        (12, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    for index, view in enumerate(chosen):
        image = cv2.imread(view.image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = min((tile_width - 12) / image.shape[1], 220 / image.shape[0])
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        row = index // columns
        column = index % columns
        x0 = column * tile_width + (tile_width - resized.shape[1]) // 2
        y0 = 60 + row * tile_height + 4
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        text_y = 60 + row * tile_height + 235
        cv2.putText(
            canvas,
            view.sample_id,
            (column * tile_width + 8, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"sharp={view.sharpness_score:.2f} err={view.final_reproj_error:.3f}",
            (column * tile_width + 8, text_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            view.status,
            (column * tile_width + 8, text_y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to write contact sheet: {path}")


def build_output_yaml(
    source_yaml: Path,
    image_dirs: list[Path],
    image_size: tuple[int, int],
    charuco: dict[str, Any],
    selected: list[View],
    all_views: list[View],
    result: dict[str, Any],
    source_data: dict[str, Any],
    blur_fraction: float,
    blur_threshold: float,
    history: list[dict[str, Any]],
    validation: dict[str, Any],
    acceleration: dict[str, Any],
    base: Path,
) -> dict[str, Any]:
    K = np.asarray(result["K"], dtype=np.float64)
    dist = np.asarray(result["dist"], dtype=np.float64).reshape(-1)
    source_counts = Counter(view.source_name for view in selected)
    status_counts = Counter(view.status for view in all_views)
    samples = []
    for view in selected:
        samples.append(
            {
                "sample_id": view.sample_id,
                "source_name": view.source_name,
                "sample_index": int(view.sample_index),
                "frame_index": int(view.frame_index),
                "corner_count": int(view.corner_count),
                "marker_count": int(view.marker_count),
                "image_path": relative_or_absolute(Path(view.image_path), base),
                "capture_mode": "offline_blur_filtered",
                "sharpness_score": float(view.sharpness_score),
                "sharpness_percentile": float(view.sharpness_percentile),
                "board_area_fraction": float(view.board_area_fraction),
                "reproj_error": float(view.final_reproj_error),
            }
        )
    return {
        "camera_model": "pinhole",
        "calibration_target": "charuco",
        "calibration_method": "opencv_calibrateCamera_blur_pose_reprojection_filtered",
        "candidate_status": "recommended_filtered",
        "source_yaml": relative_or_absolute(source_yaml, base),
        "source_yaml_rms": float(source_data.get("rms", math.nan)),
        "image_dirs": [
            relative_or_absolute(directory, base)
            for directory in image_dirs
        ],
        "image_size": [int(image_size[0]), int(image_size[1])],
        "K": K.tolist(),
        "dist": dist.tolist(),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "rms": float(result["rms"]),
        "mean_reproj_error": float(result["mean_reproj_error"]),
        "num_samples": len(selected),
        "used_indices": [int(view.frame_index) for view in selected],
        "used_sample_ids": [view.sample_id for view in selected],
        "corner_counts": [int(view.corner_count) for view in selected],
        "per_view_errors": [float(view.final_reproj_error) for view in selected],
        "samples_per_source": dict(sorted(source_counts.items())),
        "selection": {
            "input_images": len(all_views),
            "status_counts": dict(sorted(status_counts.items())),
            "blur_fraction": float(blur_fraction),
            "sharpness_score_threshold": float(blur_threshold),
            "calibration_rounds": history,
        },
        "acceleration": acceleration,
        "samples": samples,
        "charuco": {
            "squares_x": int(charuco["squares_x"]),
            "squares_y": int(charuco["squares_y"]),
            "square_length": float(charuco["square_length"]),
            "marker_length": float(charuco["marker_length"]),
            "dictionary": str(charuco["dictionary"]),
            "legacy_pattern": bool(charuco.get("legacy_pattern", False)),
        },
        "validation": validation,
    }


def default_output_dir(image_dirs: list[Path]) -> Path:
    suffix = "_".join(directory.name for directory in image_dirs)
    return Path("outputs") / f"intrinsics_charuco_offline_eval_{suffix}"


def default_output_name(source_yaml: Path, image_dirs: list[Path]) -> str:
    source_stem = source_yaml.stem
    suffix = "_".join(directory.name for directory in image_dirs)
    return f"{source_stem}_{suffix}_offline_filtered.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline pinhole ChArUco calibration with board-region blur filtering, "
            "pose diversity selection, robust reprojection rejection, and diagnostics."
        )
    )
    parser.add_argument("--source-yaml", required=True, type=Path)
    parser.add_argument(
        "--image-dir",
        action="append",
        dest="image_dirs",
        type=Path,
        default=[],
        help="Saved calibration image directory. Repeat for multiple captures.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-corners", type=int, default=None)
    parser.add_argument("--blur-fraction", type=float, default=0.15)
    parser.add_argument("--max-views", type=int, default=80)
    parser.add_argument("--min-views", type=int, default=40)
    parser.add_argument(
        "--max-view-error",
        type=float,
        default=0.8,
        help="Maximum accepted joint-calibration per-view RMSE in pixels.",
    )
    parser.add_argument("--max-rejection-rounds", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel image detection workers (default: up to 8).",
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "opencl"),
        default="auto",
        help=(
            "Gradient metric backend. auto uses OpenCL/UMat when available; "
            "ChArUco detection and calibration remain CPU operations."
        ),
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-cross-validation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = Path.cwd().resolve()
    source_yaml = args.source_yaml.expanduser().resolve()
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Source YAML does not exist: {source_yaml}")
    source_data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if source_data.get("camera_model") != "pinhole":
        raise ValueError("This offline workflow only supports camera_model=pinhole.")
    if source_data.get("calibration_target") != "charuco":
        raise ValueError("Source YAML must use calibration_target=charuco.")
    charuco = dict(source_data["charuco"])

    image_dirs = [directory.expanduser().resolve() for directory in args.image_dirs]
    yaml_capture_dir = source_data.get("capture", {}).get("sample_image_dir")
    if not image_dirs and yaml_capture_dir:
        image_dirs.append(Path(yaml_capture_dir).expanduser().resolve())
    if not image_dirs:
        raise ValueError("Pass at least one --image-dir.")
    image_entries = list_images(image_dirs)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(image_dirs).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = (
        args.output.expanduser().resolve()
        if args.output is not None
        else output_dir / default_output_name(source_yaml, image_dirs)
    )
    min_corners = (
        int(args.min_corners)
        if args.min_corners is not None
        else int(source_data.get("capture", {}).get("min_corners_per_sample", 12))
    )
    if args.min_views < 20:
        raise ValueError("--min-views must be at least 20.")
    if args.max_views < args.min_views:
        raise ValueError("--max-views must be >= --min-views.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    opencl_available = bool(cv2.ocl.haveOpenCL())
    opencl_device: dict[str, Any] = {}
    if opencl_available and hasattr(cv2.ocl, "Device_getDefault"):
        try:
            device = cv2.ocl.Device_getDefault()
            opencl_available = opencl_available and bool(device.available())
            opencl_device = {
                "name": str(device.name()),
                "vendor": str(device.vendorName()),
                "version": str(device.version()),
            }
        except cv2.error:
            opencl_available = False
    if args.accelerator == "opencl" and not opencl_available:
        raise RuntimeError("--accelerator=opencl requested, but OpenCL is unavailable.")
    use_opencl = args.accelerator != "cpu" and opencl_available
    cv2.ocl.setUseOpenCL(use_opencl)
    cpu_count = os.cpu_count() or 1
    opencv_threads = max(1, cpu_count // args.workers)
    cv2.setNumThreads(opencv_threads)
    acceleration = {
        "requested": str(args.accelerator),
        "gradient_backend": "opencl_umat" if use_opencl else "cpu",
        "opencl_device": opencl_device if use_opencl else {},
        "detection_workers": int(args.workers),
        "opencv_threads_per_worker": int(opencv_threads),
        "calibration_backend": "cpu_opencv",
    }
    print(
        f"[INFO] Parallel detection workers={args.workers}, "
        f"OpenCV threads/worker={opencv_threads}"
    )
    if use_opencl:
        print(
            f"[INFO] Gradient backend=OpenCL/UMat on "
            f"{opencl_device.get('name', 'OpenCL device')}"
        )
    else:
        print("[INFO] Gradient backend=CPU")

    expected_size = tuple(int(value) for value in source_data["image_size"])
    fingerprint = cache_fingerprint(
        image_entries,
        charuco,
        min_corners,
        use_opencl,
    )
    cache_path = output_dir / "detections_cache.pkl"
    views: list[View]
    if cache_path.is_file() and not args.no_cache:
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
        if cache.get("fingerprint") == fingerprint:
            views = cache["views"]
            print(f"[INFO] Loaded {len(views)} cached detections from {cache_path}")
        else:
            views = detect_views(
                image_entries,
                charuco,
                min_corners,
                args.workers,
                use_opencl,
            )
    else:
        views = detect_views(
            image_entries,
            charuco,
            min_corners,
            args.workers,
            use_opencl,
        )
    with cache_path.open("wb") as handle:
        pickle.dump(
            {"fingerprint": fingerprint, "views": views},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    sizes = Counter((view.image_width, view.image_height) for view in views)
    if len(sizes) != 1:
        raise RuntimeError(f"All images must have the same resolution; found {dict(sizes)}")
    image_size = next(iter(sizes))
    if image_size != expected_size:
        raise RuntimeError(
            f"Image resolution {image_size} differs from source YAML {expected_size}."
        )

    valid_count = sum(view.quality_ok for view in views)
    print(f"[INFO] Valid ChArUco detections: {valid_count}/{len(views)}")
    if valid_count < args.min_views:
        raise RuntimeError(
            f"Only {valid_count} valid detections; need at least {args.min_views}."
        )
    blur_threshold = assign_sharpness_scores(views, args.blur_fraction)
    blur_count = sum(view.status == "rejected_blur" for view in views)
    print(
        f"[INFO] Low-sharpness/blur rejection: {blur_count} views "
        f"(score <= {blur_threshold:.4f})"
    )

    initial_views = select_balanced_diverse_views(views, args.max_views)
    print(
        f"[INFO] Pose-diverse calibration views: {len(initial_views)}; "
        f"per source={dict(Counter(view.source_name for view in initial_views))}"
    )
    if len(initial_views) < args.min_views:
        raise RuntimeError(
            f"Only {len(initial_views)} sharp, diverse views remain; "
            f"need at least {args.min_views}."
        )

    K_seed = np.asarray(source_data["K"], dtype=np.float64).reshape(3, 3)
    dist_seed = np.asarray(source_data["dist"], dtype=np.float64).reshape(-1, 1)
    selected, result, history = robust_calibrate(
        initial_views,
        image_size,
        K_seed,
        dist_seed,
        args.min_views,
        args.max_view_error,
        args.max_rejection_rounds,
    )

    for view in views:
        if view.quality_ok and not np.isfinite(view.final_reproj_error):
            view.final_reproj_error = independent_pnp_error(
                view,
                result["K"],
                result["dist"],
            )
    validation = (
        {"folds": [], "skipped": True}
        if args.skip_cross_validation
        else cross_validate(selected, image_size, result)
    )

    output_data = build_output_yaml(
        source_yaml,
        image_dirs,
        image_size,
        charuco,
        selected,
        views,
        result,
        source_data,
        args.blur_fraction,
        blur_threshold,
        history,
        validation,
        acceleration,
        base,
    )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(
        yaml.safe_dump(output_data, sort_keys=False),
        encoding="utf-8",
    )

    csv_rows = [view_to_csv_row(view, base) for view in views]
    write_csv(output_dir / "selection_report.csv", csv_rows)
    make_contact_sheet(
        output_dir / "rejected_blur_contact_sheet.jpg",
        [view for view in views if view.status == "rejected_blur"],
        "Rejected low-sharpness / motion-blur candidates",
        sort_key=lambda view: view.sharpness_score,
    )
    make_contact_sheet(
        output_dir / "rejected_reprojection_contact_sheet.jpg",
        [view for view in views if view.status == "rejected_reprojection"],
        "Rejected reprojection outliers",
        sort_key=lambda view: -view.final_reproj_error,
    )
    make_contact_sheet(
        output_dir / "selected_contact_sheet.jpg",
        selected,
        "Selected sharp, pose-diverse calibration views",
        sort_key=lambda view: (view.source_name, view.frame_index),
    )

    print("[RESULT] Offline pinhole calibration complete")
    print(f"[RESULT] Source RMS: {float(source_data['rms']):.6f} px")
    print(f"[RESULT] Final RMS:  {float(result['rms']):.6f} px")
    print(f"[RESULT] Samples:    {len(selected)}")
    print(f"[RESULT] K:\n{result['K']}")
    print(f"[RESULT] dist: {result['dist'].reshape(-1)}")
    print(f"[RESULT] YAML: {output_yaml}")
    print(f"[RESULT] Diagnostics: {output_dir}")


if __name__ == "__main__":
    main()
