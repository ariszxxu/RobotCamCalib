"""Small, convention-explicit SE(3) helpers shared by calibration solvers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a 3x3 rotation and translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return transform


def inv_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general-purpose matrix inverse."""
    transform = np.asarray(T, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return make_T(rotation.T, -rotation.T @ translation)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Return the SO(3) logarithm as a rotation vector in radians."""
    rotation = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return Rotation.from_matrix(rotation).as_rotvec()


def transform_delta(
    T_ref: np.ndarray,
    T: np.ndarray,
) -> tuple[float, float]:
    """Return rotation (degrees) and translation (metres) between transforms."""
    delta = inv_T(T_ref) @ np.asarray(T, dtype=np.float64).reshape(4, 4)
    rot_deg = float(np.degrees(np.linalg.norm(so3_log(delta[:3, :3]))))
    trans_m = float(np.linalg.norm(delta[:3, 3]))
    return rot_deg, trans_m


def transform_to_params(T: np.ndarray) -> np.ndarray:
    """Pack a transform as ``[rotation-vector, translation]``."""
    transform = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.hstack(
        [
            Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
            transform[:3, 3],
        ]
    )


def params_to_transform(params: np.ndarray) -> np.ndarray:
    """Unpack ``[rotation-vector, translation]`` into a transform."""
    values = np.asarray(params, dtype=np.float64).reshape(6)
    return make_T(
        Rotation.from_rotvec(values[:3]).as_matrix(),
        values[3:],
    )


def wahba(
    rotations_src: list[np.ndarray],
    rotations_tgt: list[np.ndarray],
) -> np.ndarray:
    """Solve the unweighted Wahba rotation alignment problem."""
    if len(rotations_src) != len(rotations_tgt) or not rotations_src:
        raise ValueError("Wahba inputs must be non-empty and equally sized")
    covariance = np.zeros((3, 3), dtype=np.float64)
    for source, target in zip(rotations_src, rotations_tgt):
        covariance += target @ source.T
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return rotation


def solve_y_given_x(
    left_list: list[np.ndarray],
    right_list: list[np.ndarray],
    X: np.ndarray,
) -> np.ndarray:
    """Solve ``left_i @ X ~= Y @ right_i`` for the fixed transform Y."""
    if len(left_list) != len(right_list) or not left_list:
        raise ValueError("Transform lists must be non-empty and equally sized")
    rotation_targets = [left[:3, :3] @ X[:3, :3] for left in left_list]
    rotation_sources = [right[:3, :3] for right in right_list]
    rotation_y = wahba(rotation_sources, rotation_targets)
    translations = [
        left[:3, :3] @ X[:3, 3]
        + left[:3, 3]
        - rotation_y @ right[:3, 3]
        for left, right in zip(left_list, right_list)
    ]
    return make_T(rotation_y, np.mean(translations, axis=0))


def solve_x_given_y(
    left_list: list[np.ndarray],
    right_list: list[np.ndarray],
    Y: np.ndarray,
) -> np.ndarray:
    """Solve ``left_i @ X ~= Y @ right_i`` for the fixed transform X."""
    if len(left_list) != len(right_list) or not left_list:
        raise ValueError("Transform lists must be non-empty and equally sized")
    rotation_sources = [left[:3, :3] for left in left_list]
    rotation_targets = [Y[:3, :3] @ right[:3, :3] for right in right_list]
    rotation_x = wahba(rotation_sources, rotation_targets)
    matrix = np.vstack([left[:3, :3] for left in left_list])
    vector = np.hstack(
        [
            Y[:3, :3] @ right[:3, 3]
            + Y[:3, 3]
            - left[:3, 3]
            for left, right in zip(left_list, right_list)
        ]
    )
    translation_x, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    return make_T(rotation_x, translation_x)


def robust_limit(
    values: list[float],
    minimum: float,
    maximum: float,
    multiplier: float = 3.0,
) -> float:
    """Return a median/MAD threshold clipped to configured hard bounds."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot calculate a robust limit for no values")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    data_limit = median + multiplier * 1.4826 * mad
    return float(np.clip(data_limit, minimum, maximum))


def residual_stats(values: list[float]) -> dict[str, float]:
    """Summarize one scalar residual series."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize no residuals")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }
