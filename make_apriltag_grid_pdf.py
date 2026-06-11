#!/usr/bin/env python3
"""Generate the print-ready AprilTag grid and matching board-layout YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parent

MM_PER_INCH = 25.4


def april_tag_dictionary():
    aruco = cv2.aruco
    dict_id = getattr(aruco, "DICT_APRILTAG_36h11", None)
    if dict_id is None:
        dict_id = getattr(aruco, "DICT_APRILTAG_36H11")
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def make_marker(tag_id: int, marker_px: int):
    dictionary = april_tag_dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, marker_px)

    marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, tag_id, marker_px, marker, 1)
    return marker


def draw_board(
    ax,
    *,
    page_w_mm: float,
    page_h_mm: float,
    board_left_mm: float,
    board_bottom_mm: float,
    rows: int,
    cols: int,
    tag_id_start: int,
    tag_size_mm: float,
    tile_size_mm: float,
    gap_mm: float,
    marker_px: int,
    add_crop_marks: bool,
) -> None:
    pitch_mm = tile_size_mm + gap_mm
    marker_margin_mm = (tile_size_mm - tag_size_mm) / 2.0
    board_w_mm = (cols - 1) * pitch_mm + tile_size_mm
    board_h_mm = (rows - 1) * pitch_mm + tile_size_mm

    ax.set_xlim(0.0, page_w_mm)
    ax.set_ylim(0.0, page_h_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    markers = {
        tag_id_start + index: make_marker(tag_id_start + index, marker_px)
        for index in range(rows * cols)
    }

    for row in range(rows):
        for col in range(cols):
            tag_id = tag_id_start + row * cols + col
            tile_left = board_left_mm + col * pitch_mm
            tile_bottom = board_bottom_mm + (rows - 1 - row) * pitch_mm
            marker_left = tile_left + marker_margin_mm
            marker_right = marker_left + tag_size_mm
            marker_bottom = tile_bottom + marker_margin_mm
            marker_top = marker_bottom + tag_size_mm

            ax.imshow(
                markers[tag_id],
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="nearest",
                origin="upper",
                extent=[marker_left, marker_right, marker_bottom, marker_top],
            )

    if not add_crop_marks:
        return

    ax.add_patch(
        Rectangle(
            (board_left_mm, board_bottom_mm),
            board_w_mm,
            board_h_mm,
            fill=False,
            edgecolor="0.70",
            linewidth=0.35,
        )
    )
    crop_len_mm = 7.0
    for x in (board_left_mm, board_left_mm + board_w_mm):
        direction = -1.0 if x == board_left_mm else 1.0
        ax.plot([x, x + direction * crop_len_mm], [board_bottom_mm, board_bottom_mm], color="0.65", linewidth=0.35)
        ax.plot(
            [x, x + direction * crop_len_mm],
            [board_bottom_mm + board_h_mm, board_bottom_mm + board_h_mm],
            color="0.65",
            linewidth=0.35,
        )
    for y in (board_bottom_mm, board_bottom_mm + board_h_mm):
        direction = -1.0 if y == board_bottom_mm else 1.0
        ax.plot([board_left_mm, board_left_mm], [y, y + direction * crop_len_mm], color="0.65", linewidth=0.35)
        ax.plot(
            [board_left_mm + board_w_mm, board_left_mm + board_w_mm],
            [y, y + direction * crop_len_mm],
            color="0.65",
            linewidth=0.35,
        )


def save_page(
    path_pdf: Optional[Path],
    path_png: Optional[Path],
    *,
    page_w_mm: float,
    page_h_mm: float,
    board_left_mm: float,
    board_bottom_mm: float,
    rows: int,
    cols: int,
    tag_id_start: int,
    tag_size_mm: float,
    tile_size_mm: float,
    gap_mm: float,
    board_w_mm: float,
    board_h_mm: float,
    marker_px: int,
    add_title: bool,
) -> None:
    fig = plt.figure(figsize=(page_w_mm / MM_PER_INCH, page_h_mm / MM_PER_INCH), dpi=300)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_board(
        ax,
        page_w_mm=page_w_mm,
        page_h_mm=page_h_mm,
        board_left_mm=board_left_mm,
        board_bottom_mm=board_bottom_mm,
        rows=rows,
        cols=cols,
        tag_id_start=tag_id_start,
        tag_size_mm=tag_size_mm,
        tile_size_mm=tile_size_mm,
        gap_mm=gap_mm,
        marker_px=marker_px,
        add_crop_marks=True,
    )

    if add_title:
        ax.text(
            page_w_mm / 2.0,
            page_h_mm - 22.0,
            f"AprilTag 36h11 {rows}x{cols}, IDs {tag_id_start}-{tag_id_start + rows * cols - 1}, marker edge = {tag_size_mm:.2f} mm",
            fontsize=8,
            color="0.2",
            ha="center",
            va="center",
        )
        ax.text(
            page_w_mm / 2.0,
            page_h_mm - 31.0,
            f"Print at 100% / actual size. Board outer size = {board_w_mm:.2f} mm x {board_h_mm:.2f} mm.",
            fontsize=7,
            color="0.35",
            ha="center",
            va="center",
        )
        ax.plot([20.0, 120.0], [20.0, 20.0], color="black", linewidth=0.8)
        ax.plot([20.0, 20.0], [18.0, 22.0], color="black", linewidth=0.8)
        ax.plot([120.0, 120.0], [18.0, 22.0], color="black", linewidth=0.8)
        ax.text(70.0, 24.0, "100 mm check", fontsize=7, ha="center", va="bottom", color="0.2")

    if path_pdf is not None:
        path_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_pdf, format="pdf", facecolor="white")
    if path_png is not None:
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, format="png", dpi=300, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "assets" / "apriltag_grid")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--tag-id-start", type=int, default=0)
    parser.add_argument("--tag-size-mm", type=float, default=48.0)
    parser.add_argument("--marker-fraction", type=float, default=0.92)
    parser.add_argument("--gap-ratio", type=float, default=0.0)
    parser.add_argument("--marker-px", type=int, default=1024)
    parser.add_argument("--stem", default=None)
    return parser.parse_args()


def tag_corners_pupil_order_mm(tag_size_mm: float) -> list[list[float]]:
    half = tag_size_mm / 2.0
    return [
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ]


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("--rows and --cols must be positive.")
    if args.tag_size_mm <= 0:
        raise SystemExit("--tag-size-mm must be positive.")
    if args.gap_ratio < 0:
        raise SystemExit("--gap-ratio must be non-negative.")
    if args.marker_fraction <= 0 or args.marker_fraction > 1:
        raise SystemExit("--marker-fraction must be in (0, 1].")

    tile_size_mm = args.tag_size_mm / args.marker_fraction
    gap_mm = args.tag_size_mm * args.gap_ratio
    pitch_mm = tile_size_mm + gap_mm
    board_w_mm = (args.cols - 1) * pitch_mm + tile_size_mm
    board_h_mm = (args.rows - 1) * pitch_mm + tile_size_mm

    a3_w_mm = 297.0
    a3_h_mm = 420.0
    if board_w_mm > a3_w_mm or board_h_mm > a3_h_mm - 70.0:
        raise SystemExit("The generated board does not fit on the current A3 layout.")

    stem = args.stem
    if stem is None:
        stem = f"compact_apriltag_grid_{args.rows}x{args.cols}_tag{int(round(args.tag_size_mm))}mm"
    board_left_mm = (a3_w_mm - board_w_mm) / 2.0
    board_bottom_mm = max((a3_h_mm - board_h_mm) / 2.0 - 5.0, 20.0)
    save_page(
        args.out_dir / f"{stem}_a3.pdf",
        None,
        page_w_mm=a3_w_mm,
        page_h_mm=a3_h_mm,
        board_left_mm=board_left_mm,
        board_bottom_mm=board_bottom_mm,
        rows=args.rows,
        cols=args.cols,
        tag_id_start=args.tag_id_start,
        tag_size_mm=args.tag_size_mm,
        tile_size_mm=tile_size_mm,
        gap_mm=gap_mm,
        board_w_mm=board_w_mm,
        board_h_mm=board_h_mm,
        marker_px=args.marker_px,
        add_title=True,
    )

    quiet_margin_mm = 10.0
    save_page(
        None,
        args.out_dir / f"{stem}_board_only.png",
        page_w_mm=board_w_mm + 2.0 * quiet_margin_mm,
        page_h_mm=board_h_mm + 2.0 * quiet_margin_mm,
        board_left_mm=quiet_margin_mm,
        board_bottom_mm=quiet_margin_mm,
        rows=args.rows,
        cols=args.cols,
        tag_id_start=args.tag_id_start,
        tag_size_mm=args.tag_size_mm,
        tile_size_mm=tile_size_mm,
        gap_mm=gap_mm,
        board_w_mm=board_w_mm,
        board_h_mm=board_h_mm,
        marker_px=args.marker_px,
        add_title=False,
    )

    tag_corners_tag_mm = tag_corners_pupil_order_mm(args.tag_size_mm)
    tags_yaml = []
    for row in range(args.rows):
        for col in range(args.cols):
            tag_id = args.tag_id_start + row * args.cols + col
            x_mm = ((args.cols - 1) / 2.0 - col) * pitch_mm
            y_mm = ((args.rows - 1) / 2.0 - row) * pitch_mm
            corners_board_mm = [
                [corner[0] + x_mm, corner[1] + y_mm, corner[2]]
                for corner in tag_corners_tag_mm
            ]
            tags_yaml.append(
                {
                    "id": tag_id,
                    "row": row,
                    "col": col,
                    "center_mm": [x_mm, y_mm, 0.0],
                    "T_board_tag": [
                        [1.0, 0.0, 0.0, x_mm],
                        [0.0, 1.0, 0.0, y_mm],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "corners_board_mm": corners_board_mm,
                }
            )

    board_yaml = {
        "schema": "robot_cam_calib.apriltag_board.v1",
        "name": stem,
        "family": "tag36h11",
        "units": "mm",
        "layout": {
            "rows": args.rows,
            "cols": args.cols,
            "tag_id_start": args.tag_id_start,
            "tag_id_end": args.tag_id_start + args.rows * args.cols - 1,
        },
        "geometry": {
            "tag_size_mm": args.tag_size_mm,
            "marker_fraction": args.marker_fraction,
            "tile_size_mm": tile_size_mm,
            "explicit_gap_mm": gap_mm,
            "pitch_mm": pitch_mm,
            "black_marker_edge_gap_mm": pitch_mm - args.tag_size_mm,
            "board_size_mm": [board_w_mm, board_h_mm],
        },
        "target_frame": {
            "name": "board",
            "origin": "center of complete 4x4 printed board",
            "x_axis": "toward lower printed column index",
            "y_axis": "toward lower printed row index",
            "z_axis": "normal to printed tag plane",
        },
        "detection_corner_order": {
            "source": "pupil_apriltags.Detection.corners",
            "tag_frame_corners_mm": tag_corners_tag_mm,
            "description": "Use each detection's four image corners in this exact order with the matching corners_board_mm entry for bundle PnP.",
        },
        "tags": tags_yaml,
    }
    with open(args.out_dir / f"{stem}.yaml", "w") as f:
        yaml.safe_dump(board_yaml, f, sort_keys=False)

    print(f"marker_edge_mm={args.tag_size_mm:.6f}")
    print(f"marker_fraction={args.marker_fraction:.6f}")
    print(f"tile_size_mm={tile_size_mm:.6f}")
    print(f"gap_mm={gap_mm:.6f}")
    print(f"pitch_mm={pitch_mm:.6f}")
    print(f"black_marker_edge_gap_mm={pitch_mm - args.tag_size_mm:.6f}")
    print(f"board_size_mm={board_w_mm:.6f}x{board_h_mm:.6f}")


if __name__ == "__main__":
    main()
