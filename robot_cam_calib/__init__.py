"""Reusable building blocks for camera calibration workflows.

Root-level scripts remain the supported command-line entry points.  This
package contains only device-independent code shared by those scripts.
"""

from .geometry import (
    inv_T,
    make_T,
    params_to_transform,
    residual_stats,
    robust_limit,
    so3_log,
    solve_x_given_y,
    solve_y_given_x,
    transform_delta,
    transform_to_params,
    wahba,
)

__all__ = [
    "inv_T",
    "make_T",
    "params_to_transform",
    "residual_stats",
    "robust_limit",
    "so3_log",
    "solve_x_given_y",
    "solve_y_given_x",
    "transform_delta",
    "transform_to_params",
    "wahba",
]
