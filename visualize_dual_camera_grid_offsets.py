#!/usr/bin/env python3
"""Generate human-readable adapter drawings from dual rigid-grid calibration.

The drawings intentionally show the calibrated grid plane, grid frame, camera
optical frame, and a schematic optical frustum.  They do not invent a camera
body envelope, screw-hole pattern, or printable adapter solid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "outputs/dual_camera_rigid_grid_offsets"

AXIS_COLORS = ("#e31a1c", "#19bf32", "#1455ff")
AXIS_NAMES = ("X", "Y", "Z")
GRID_AXIS_LENGTH_MM = 14.0
CAM_AXIS_LENGTH_MM = 14.0


def latest_result_yaml(root: Path) -> Path:
    candidates = list(root.glob("*/dual_camera_rigid_grid_offsets.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No dual_camera_rigid_grid_offsets.yaml under {root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def get_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"OpenCV does not provide {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def generate_marker(dictionary, tag_id: int, side_px: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(
            dictionary, int(tag_id), int(side_px)
        )
    marker = np.zeros((side_px, side_px), dtype=np.uint8)
    cv2.aruco.drawMarker(
        dictionary, int(tag_id), int(side_px), marker, 1
    )
    return marker


def render_grid_texture(
    board: dict[str, Any],
    px_per_mm: int = 24,
) -> tuple[np.ndarray, float, float]:
    width_mm = float(board["board_width_m"]) * 1000.0
    height_mm = float(board["board_height_m"]) * 1000.0
    width_px = int(round(width_mm * px_per_mm))
    height_px = int(round(height_mm * px_per_mm))
    texture = np.full((height_px, width_px), 255, dtype=np.uint8)

    geometry = board.get("geometry", {})
    frame_width_mm = float(geometry.get("black_frame_width_mm", 0.4))
    frame_px = max(1, int(round(frame_width_mm * px_per_mm)))
    texture[:frame_px, :] = 0
    texture[-frame_px:, :] = 0
    texture[:, :frame_px] = 0
    texture[:, -frame_px:] = 0

    dictionary = get_dictionary(str(board["tag_family"]))
    tag_size_mm = float(board["tag_size_m"]) * 1000.0
    tag_px = int(round(tag_size_mm * px_per_mm))
    tag_points = board["tag_object_points"]
    for tag_id_raw, points_raw in tag_points.items():
        tag_id = int(tag_id_raw)
        points = np.asarray(points_raw, dtype=np.float64).reshape(4, 3)
        left_mm = float(np.min(points[:, 0]) * 1000.0)
        top_mm = float(np.min(points[:, 1]) * 1000.0)
        x_px = int(round((left_mm + width_mm / 2.0) * px_per_mm))
        y_px = int(round((top_mm + height_mm / 2.0) * px_per_mm))
        marker = generate_marker(dictionary, tag_id, tag_px)
        texture[y_px : y_px + tag_px, x_px : x_px + tag_px] = marker

    return texture, width_mm, height_mm


def validate_transform(name: str, T: np.ndarray) -> None:
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has invalid homogeneous last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7):
        raise ValueError(f"{name} rotation is not orthogonal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
        raise ValueError(f"{name} rotation determinant is not +1")


def axis_angle_text(R: np.ndarray) -> str:
    rotvec = Rotation.from_matrix(R).as_rotvec()
    angle_deg = float(np.degrees(np.linalg.norm(rotvec)))
    if angle_deg < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = rotvec / np.linalg.norm(rotvec)
    return (
        f"rotation: {angle_deg:.3f} deg about "
        f"[{axis[0]:+.4f}, {axis[1]:+.4f}, {axis[2]:+.4f}] in grid"
    )


def configure_2d_axis(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    equal: bool = True,
) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d7d7d7", linewidth=0.7)
    ax.axhline(0.0, color="#aaaaaa", linewidth=0.7)
    ax.axvline(0.0, color="#aaaaaa", linewidth=0.7)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def draw_projected_frame(
    ax,
    origin_2d: np.ndarray,
    R: np.ndarray,
    coordinate_rows: tuple[int, int],
    axis_length_mm: float,
    prefix: str,
    linewidth: float = 2.2,
) -> None:
    for axis_index, (axis_name, color) in enumerate(
        zip(AXIS_NAMES, AXIS_COLORS)
    ):
        delta = (
            R[list(coordinate_rows), axis_index] * axis_length_mm
        )
        ax.annotate(
            "",
            xy=origin_2d + delta,
            xytext=origin_2d,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": linewidth,
                "mutation_scale": 12,
            },
            zorder=8,
        )
        ax.text(
            *(origin_2d + delta * 1.08),
            f"{prefix}{axis_name}",
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=9,
        )


def draw_dimension(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    color: str = "#555555",
    text_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "lw": 1.2,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )
    midpoint = (
        0.5 * (start[0] + end[0]) + text_offset[0],
        0.5 * (start[1] + end[1]) + text_offset[1],
    )
    ax.text(
        *midpoint,
        label,
        fontsize=8,
        color=color,
        ha="center",
        va="center",
        bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.5},
        zorder=7,
    )


def draw_front_view(
    ax,
    texture: np.ndarray,
    width_mm: float,
    height_mm: float,
    T: np.ndarray,
) -> None:
    t = T[:3, 3]
    R = T[:3, :3]
    half_w = width_mm / 2.0
    half_h = height_mm / 2.0
    ax.imshow(
        texture,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        extent=(-half_w, half_w, half_h, -half_h),
        zorder=1,
    )
    ax.plot(
        [-half_w, half_w, half_w, -half_w, -half_w],
        [-half_h, -half_h, half_h, half_h, -half_h],
        color="#555555",
        linewidth=1.2,
        zorder=4,
    )
    ax.scatter([0], [0], s=28, color="#333333", zorder=9)
    ax.text(1.5, -1.5, "G", fontsize=9, fontweight="bold", zorder=10)
    ax.scatter([t[0]], [t[1]], s=42, color="#ff8c00", zorder=9)
    ax.text(
        t[0] + 1.5,
        t[1] - 1.5,
        "C optical origin",
        fontsize=8,
        color="#8a4600",
        zorder=10,
    )
    ax.plot([0, t[0]], [0, t[1]], "--", color="#ff8c00", lw=1.2)
    draw_projected_frame(
        ax,
        np.array([0.0, 0.0]),
        np.eye(3),
        (0, 1),
        GRID_AXIS_LENGTH_MM,
        "G",
    )
    draw_projected_frame(
        ax,
        t[[0, 1]],
        R,
        (0, 1),
        CAM_AXIS_LENGTH_MM,
        "C",
    )

    dim_y_x = -half_w - 7.0
    ax.plot([0, dim_y_x], [0, 0], ":", color="#777777", lw=0.8)
    ax.plot(
        [t[0], dim_y_x], [t[1], t[1]], ":", color="#777777", lw=0.8
    )
    draw_dimension(
        ax,
        (dim_y_x, 0.0),
        (dim_y_x, t[1]),
        f"ΔY = {t[1]:+.3f} mm",
        text_offset=(-5.0, 0.0),
    )
    dim_x_y = min(t[1], -half_h) - 7.0
    ax.plot([0, 0], [0, dim_x_y], ":", color="#777777", lw=0.8)
    ax.plot(
        [t[0], t[0]], [t[1], dim_x_y], ":", color="#777777", lw=0.8
    )
    draw_dimension(
        ax,
        (0.0, dim_x_y),
        (t[0], dim_x_y),
        f"ΔX = {t[0]:+.3f} mm",
        text_offset=(0.0, -3.5),
    )

    x_min = min(-half_w - 18.0, t[0] - 18.0)
    x_max = max(half_w + 10.0, t[0] + 18.0)
    y_min = min(-half_h - 14.0, t[1] - 22.0)
    y_max = max(half_h + 10.0, t[1] + 18.0)
    ax.set_xlim(x_min, x_max)
    # Positive grid Y is printed-image down, so use an inverted display axis.
    ax.set_ylim(y_max, y_min)
    configure_2d_axis(
        ax,
        "Front view — grid X/Y",
        "grid X [mm]  (+ right)",
        "grid Y [mm]  (+ print down)",
    )


def draw_edge_view(
    ax,
    T: np.ndarray,
    plane: str,
    width_mm: float,
    height_mm: float,
) -> None:
    t = T[:3, 3]
    R = T[:3, :3]
    if plane == "YZ":
        rows = (1, 2)
        origin = t[[1, 2]]
        board_extent = height_mm / 2.0
        ax.plot(
            [-board_extent, board_extent],
            [0.0, 0.0],
            color="#111111",
            linewidth=4,
            solid_capstyle="butt",
        )
        xlabel = "grid Y [mm]"
        title = "Side view — grid Y/Z"
        dx_label = f"ΔY = {t[1]:+.3f} mm"
        dz_label = f"ΔZ = {t[2]:+.3f} mm"
    elif plane == "XZ":
        rows = (0, 2)
        origin = t[[0, 2]]
        board_extent = width_mm / 2.0
        ax.plot(
            [-board_extent, board_extent],
            [0.0, 0.0],
            color="#111111",
            linewidth=4,
            solid_capstyle="butt",
        )
        xlabel = "grid X [mm]"
        title = "Top view — grid X/Z"
        dx_label = f"ΔX = {t[0]:+.3f} mm"
        dz_label = f"ΔZ = {t[2]:+.3f} mm"
    else:
        raise ValueError(plane)

    ax.scatter([0], [0], s=28, color="#333333", zorder=9)
    ax.text(1.0, 1.0, "G", fontsize=9, fontweight="bold")
    ax.scatter([origin[0]], [origin[1]], s=42, color="#ff8c00", zorder=9)
    ax.text(
        origin[0] + 1.0,
        origin[1] + 1.0,
        "C",
        fontsize=9,
        fontweight="bold",
        color="#8a4600",
    )
    ax.plot(
        [0, origin[0]],
        [0, origin[1]],
        "--",
        color="#ff8c00",
        lw=1.2,
    )
    draw_projected_frame(
        ax,
        np.array([0.0, 0.0]),
        np.eye(3),
        rows,
        GRID_AXIS_LENGTH_MM,
        "G",
    )
    draw_projected_frame(
        ax,
        origin,
        R,
        rows,
        CAM_AXIS_LENGTH_MM,
        "C",
    )
    dim_y = min(-8.0, origin[1] - 8.0)
    ax.plot([0, 0], [0, dim_y], ":", color="#777777", lw=0.8)
    ax.plot(
        [origin[0], origin[0]],
        [origin[1], dim_y],
        ":",
        color="#777777",
        lw=0.8,
    )
    draw_dimension(
        ax,
        (0.0, dim_y),
        (origin[0], dim_y),
        dx_label,
        text_offset=(0.0, -3.0),
    )
    dim_x = min(-board_extent - 8.0, origin[0] - 8.0)
    ax.plot([0, dim_x], [0, 0], ":", color="#777777", lw=0.8)
    ax.plot(
        [origin[0], dim_x],
        [origin[1], origin[1]],
        ":",
        color="#777777",
        lw=0.8,
    )
    draw_dimension(
        ax,
        (dim_x, 0.0),
        (dim_x, origin[1]),
        dz_label,
        text_offset=(-4.5, 0.0),
    )
    x_min = min(-board_extent - 18.0, origin[0] - 20.0)
    x_max = max(board_extent + 12.0, origin[0] + 20.0)
    z_min = min(-25.0, origin[1] - 18.0)
    z_max = max(20.0, origin[1] + 18.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    configure_2d_axis(ax, title, xlabel, "grid Z [mm]")


def draw_3d_frame(
    ax,
    origin: np.ndarray,
    R: np.ndarray,
    length_mm: float,
    prefix: str,
) -> None:
    for index, (axis_name, color) in enumerate(
        zip(AXIS_NAMES, AXIS_COLORS)
    ):
        delta = R[:, index] * length_mm
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            delta[0],
            delta[1],
            delta[2],
            color=color,
            linewidth=2.5,
            arrow_length_ratio=0.15,
        )
        endpoint = origin + delta * 1.08
        ax.text(
            endpoint[0],
            endpoint[1],
            endpoint[2],
            f"{prefix}{axis_name}",
            color=color,
            fontsize=8,
            fontweight="bold",
        )


def draw_camera_frustum(ax, T: np.ndarray) -> None:
    t = T[:3, 3]
    R = T[:3, :3]
    local_corners = np.array(
        [
            [-6.0, -4.0, 12.0],
            [6.0, -4.0, 12.0],
            [6.0, 4.0, 12.0],
            [-6.0, 4.0, 12.0],
        ],
        dtype=np.float64,
    )
    corners = (R @ local_corners.T).T + t
    for corner in corners:
        ax.plot(
            [t[0], corner[0]],
            [t[1], corner[1]],
            [t[2], corner[2]],
            color="#ff8c00",
            linewidth=1.0,
        )
    loop = np.vstack([corners, corners[0]])
    ax.plot(
        loop[:, 0],
        loop[:, 1],
        loop[:, 2],
        color="#ff8c00",
        linewidth=1.3,
    )


def draw_isometric(
    ax,
    texture: np.ndarray,
    width_mm: float,
    height_mm: float,
    T: np.ndarray,
) -> None:
    half_w = width_mm / 2.0
    half_h = height_mm / 2.0
    sample_count = 180
    tex_small = cv2.resize(
        texture,
        (sample_count, sample_count),
        interpolation=cv2.INTER_AREA,
    )
    facecolors = (
        cv2.cvtColor(tex_small, cv2.COLOR_GRAY2RGBA).astype(np.float64)
        / 255.0
    )
    x = np.linspace(-half_w, half_w, sample_count)
    y = np.linspace(-half_h, half_h, sample_count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        shade=False,
        antialiased=False,
    )
    border_x = [-half_w, half_w, half_w, -half_w, -half_w]
    border_y = [-half_h, -half_h, half_h, half_h, -half_h]
    ax.plot(border_x, border_y, [0.0] * 5, color="#444444", lw=1.3)

    origin = np.zeros(3, dtype=np.float64)
    t = T[:3, 3]
    ax.plot(
        [0.0, t[0]],
        [0.0, t[1]],
        [0.0, t[2]],
        "--",
        color="#ff8c00",
        lw=1.5,
    )
    ax.scatter([0], [0], [0], s=25, color="#222222")
    ax.scatter([t[0]], [t[1]], [t[2]], s=38, color="#ff8c00")
    draw_3d_frame(
        ax,
        origin,
        np.eye(3),
        GRID_AXIS_LENGTH_MM,
        "G",
    )
    draw_3d_frame(
        ax,
        t,
        T[:3, :3],
        CAM_AXIS_LENGTH_MM,
        "C",
    )
    draw_camera_frustum(ax, T)
    ax.text(
        t[0],
        t[1],
        t[2] - 3.0,
        "camera optical origin",
        color="#8a4600",
        fontsize=8,
    )
    ax.text(
        1.0,
        1.0,
        1.0,
        "grid origin",
        color="#222222",
        fontsize=8,
    )

    all_points = np.vstack(
        [
            np.array(
                [
                    [-half_w, -half_h, 0.0],
                    [half_w, -half_h, 0.0],
                    [half_w, half_h, 0.0],
                    [-half_w, half_h, 0.0],
                ]
            ),
            t,
        ]
    )
    mins = all_points.min(axis=0) - np.array([16.0, 16.0, 16.0])
    maxs = all_points.max(axis=0) + np.array([16.0, 16.0, 16.0])
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    spans = np.maximum(maxs - mins, 1.0)
    ax.set_box_aspect(tuple(spans))
    ax.view_init(elev=23.0, azim=-54.0)
    ax.set_xlabel("grid X [mm]", fontsize=8)
    ax.set_ylabel("grid Y [mm]", fontsize=8)
    ax.set_zlabel("grid Z [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        "Axonometric reference\n"
        "(orange frustum is schematic, not body size)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )


def draw_text_panel(
    ax,
    camera_title: str,
    T: np.ndarray,
    width_mm: float,
    height_mm: float,
) -> None:
    ax.axis("off")
    t = T[:3, 3]
    R = T[:3, :3]
    matrix_rows = [
        "[" + "  ".join(f"{value:+.6f}" for value in row) + "]"
        for row in T
    ]
    lines = [
        "CALIBRATED MOUNT GEOMETRY [R | t_mm]",
        f"T_grid_{camera_title}:",
        *matrix_rows,
        "",
        "Optical origin in grid:",
        f"  [ΔX, ΔY, ΔZ] = [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] mm",
        f"  |G→C| = {np.linalg.norm(t):.3f} mm",
        axis_angle_text(R),
        "",
        f"Grid: {width_mm:.3f} x {height_mm:.3f} mm; origin at boundary center",
        "G: +X print-right, +Y print-down, +Z into target",
        "C: +X image-right, +Y image-down, +Z optical-forward",
        "Frustum is optical-frame only; exact values: dimensions.yaml",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.7,
        family="DejaVu Sans Mono",
        linespacing=1.12,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#f8f8f8",
            "edgecolor": "#bbbbbb",
        },
    )


def create_camera_drawing(
    output_dir: Path,
    camera_label: str,
    camera_title: str,
    T_grid_camera_m: np.ndarray,
    texture: np.ndarray,
    width_mm: float,
    height_mm: float,
    source_result: Path,
) -> tuple[Path, Path]:
    camera_dir = output_dir / camera_label
    camera_dir.mkdir(parents=True, exist_ok=True)
    T = T_grid_camera_m.copy()
    T[:3, 3] *= 1000.0

    fig = plt.figure(figsize=(14, 10), dpi=160)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.05, 1.35],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.30,
    )
    ax_front = fig.add_subplot(grid[0, 0])
    ax_side = fig.add_subplot(grid[0, 1])
    ax_iso = fig.add_subplot(grid[:, 2], projection="3d")
    ax_top = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[1, 1])

    draw_front_view(ax_front, texture, width_mm, height_mm, T)
    draw_edge_view(ax_side, T, "YZ", width_mm, height_mm)
    draw_edge_view(ax_top, T, "XZ", width_mm, height_mm)
    draw_isometric(ax_iso, texture, width_mm, height_mm, T)
    draw_text_panel(ax_text, camera_title, T, width_mm, height_mm)

    fig.suptitle(
        f"{camera_label}: rigid AprilTag grid ↔ camera optical frame",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.01,
        0.012,
        f"Source: {source_result}  |  Units: mm  |  "
        "Red=X, Green=Y, Blue=Z, Orange=camera optical origin/frustum",
        fontsize=8,
        color="#444444",
    )
    png_path = camera_dir / "thumbnail.png"
    svg_path = camera_dir / "dimensions.svg"
    fig.savefig(png_path, dpi=160, facecolor="#f1f1f1")
    fig.savefig(svg_path, facecolor="#f1f1f1")
    plt.close(fig)
    return png_path, svg_path


def create_combined_thumbnail(
    thumb_path: Path,
    third_path: Path,
    output_path: Path,
) -> None:
    images = [Image.open(path).convert("RGB") for path in (thumb_path, third_path)]
    target_width = 1400
    resized = []
    resampling = getattr(Image, "Resampling", Image)
    for image in images:
        height = int(round(image.height * target_width / image.width))
        resized.append(
            image.resize((target_width, height), resampling.LANCZOS)
        )
    gap = 24
    label_height = 52
    canvas = Image.new(
        "RGB",
        (
            target_width,
            sum(image.height for image in resized)
            + gap
            + 2 * label_height,
        ),
        "#ededed",
    )
    draw = ImageDraw.Draw(canvas)
    y = 0
    labels = (
        "THUMB WEB CAM - GRID / OPTICAL FRAME",
        "THIRD VIEW CAM - GRID / OPTICAL FRAME",
    )
    for image, label in zip(resized, labels):
        draw.text((18, y + 14), label, fill="#222222")
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height + gap
    canvas.save(output_path)


def write_dimensions_yaml(
    output_path: Path,
    source_result: Path,
    board_path: Path,
    width_mm: float,
    height_mm: float,
    transforms: dict[str, np.ndarray],
) -> None:
    cameras = {}
    for camera_name, T_m in transforms.items():
        T_mm = T_m.copy()
        T_mm[:3, 3] *= 1000.0
        cameras[camera_name] = {
            "T_grid_camera_mm": T_mm.tolist(),
            "camera_origin_in_grid_mm": T_mm[:3, 3].tolist(),
            "origin_distance_mm": float(np.linalg.norm(T_mm[:3, 3])),
            "camera_x_axis_in_grid": T_mm[:3, 0].tolist(),
            "camera_y_axis_in_grid": T_mm[:3, 1].tolist(),
            "camera_z_axis_in_grid": T_mm[:3, 2].tolist(),
            "rotation_angle_deg": float(
                np.degrees(
                    Rotation.from_matrix(T_mm[:3, :3]).magnitude()
                )
            ),
        }
    data = {
        "schema": "robot_cam_calib.adapter_reference_dimensions.v1",
        "units": "mm",
        "source_result_yaml": str(source_result),
        "source_grid_yaml": str(board_path),
        "frame_convention": (
            "T_grid_camera maps camera optical coordinates into the "
            "corresponding rigid grid frame"
        ),
        "grid": {
            "outer_size_mm": [width_mm, height_mm],
            "origin": "center of complete outer boundary",
            "x_axis": "print right",
            "y_axis": "print down",
            "z_axis": "into target",
        },
        "cameras": cameras,
        "drawing_note": (
            "Camera frusta in thumbnails indicate optical frames only. "
            "They are not camera body envelopes or mounting-hole drawings."
        ),
    }
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def write_readme(
    output_path: Path,
    source_result: Path,
) -> None:
    text = f"""# Grid-to-camera adapter reference drawings

Source calibration: `{source_result}`

## Files

- `tag_grid_front.png`: exact 3x3 AprilTag grid illustration.
- `thumb_web_cam/thumbnail.png`: thumb camera four-view drawing.
- `thumb_web_cam/dimensions.svg`: vector version.
- `third_view_cam/thumbnail.png`: third-view camera four-view drawing.
- `third_view_cam/dimensions.svg`: vector version.
- `combined_thumbnail.png`: both camera drawings in one image.
- `dimensions.yaml`: exact millimetre transforms and axis vectors for CAD.

The orange camera frustum is schematic. It shows the calibrated optical origin
and optical direction only; it does not encode the camera enclosure, mounting
holes, fasteners, wall thickness, or manufacturing clearance.
"""
    output_path.write_text(text, encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    result_path = (
        args.result_yaml.expanduser().resolve()
        if args.result_yaml is not None
        else latest_result_yaml(args.results_root).resolve()
    )
    result = load_yaml(result_path)
    if (
        result.get("schema")
        != "robot_cam_calib.dual_rigid_apriltag_grid_offsets.v1"
    ):
        raise ValueError(f"Unexpected result schema in {result_path}")

    board_path = Path(result["inputs"]["apriltag_grid_yaml"]).resolve()
    board = load_yaml(board_path)
    texture, width_mm, height_mm = render_grid_texture(board)

    requested = result["solution"]["requested_transforms"]
    transforms = {
        "thumb_web_cam": np.asarray(
            requested["T_grid_thumb_web_cam"], dtype=np.float64
        ),
        "third_view_cam": np.asarray(
            requested["T_grid_third_view_cam"], dtype=np.float64
        ),
    }
    for name, T in transforms.items():
        validate_transform(name, T)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_path.parent / "adapter_reference_drawings"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_front_path = output_dir / "tag_grid_front.png"
    cv2.imwrite(str(grid_front_path), texture)

    thumb_png, _thumb_svg = create_camera_drawing(
        output_dir,
        "thumb_web_cam",
        "thumb_web_cam",
        transforms["thumb_web_cam"],
        texture,
        width_mm,
        height_mm,
        result_path,
    )
    third_png, _third_svg = create_camera_drawing(
        output_dir,
        "third_view_cam",
        "third_view_cam",
        transforms["third_view_cam"],
        texture,
        width_mm,
        height_mm,
        result_path,
    )
    create_combined_thumbnail(
        thumb_png,
        third_png,
        output_dir / "combined_thumbnail.png",
    )
    write_dimensions_yaml(
        output_dir / "dimensions.yaml",
        result_path,
        board_path,
        width_mm,
        height_mm,
        transforms,
    )
    write_readme(output_dir / "README.md", result_path)

    print(f"Generated adapter reference drawings: {output_dir}")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(output_dir)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw calibrated AprilTag grid and camera optical frames for "
            "human/CAD adapter reference."
        )
    )
    parser.add_argument(
        "--result-yaml",
        type=Path,
        default=None,
        help="Calibration result YAML; defaults to the latest run.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults beside the selected result YAML.",
    )
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
