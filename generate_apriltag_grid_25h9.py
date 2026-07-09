from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


# ---------------------------- User macros ---------------------------- #
TAG_FAMILY = "DICT_APRILTAG_36h10"

# ID layout as printed on the board, row by row from top to bottom.
# This near-field A4 target keeps tags small enough for ~5 cm working distance.
ID_GRID = [
    list(range(row_start, row_start - 20, -1))
    for row_start in range(579, -1, -20)
]

# Physical dimensions. These numbers define the board geometry used later
# for solvePnP, so measure the printed board and keep them consistent.
TAG_SIZE_M = 0.008      # printed outer square edge length of one AprilTag
TAG_GAP_M = 0.002       # white gap between adjacent tag outer squares
OUTER_MARGIN_M = 0.003  # white margin around the whole grid
MIN_CORNERS_PER_SAMPLE = 120

# Print/export settings.
DPI = 600
OUTPUT_DIR = Path("outputs/apriltag_grid_36h10_a4_near_8mm")
OUTPUT_STEM = "apriltag_36h10_grid_20x29_ids_579_to_0_tag8mm_gap2mm_margin3mm_a4_near"
ADD_HUMAN_READABLE_IDS = False
ADD_CUT_BORDER = False
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
A4_MARGIN_MM = 12.0


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def m_to_px(meters: float) -> int:
    return mm_to_px(meters * 1000.0)


def get_dictionary():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, TAG_FAMILY):
        raise ValueError(f"OpenCV does not provide {TAG_FAMILY}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_FAMILY))


def generate_marker(dictionary, tag_id: int, marker_px: int) -> np.ndarray:
    max_id = len(dictionary.bytesList)
    if tag_id < 0 or tag_id >= max_id:
        raise ValueError(f"tag id {tag_id} out of range for {TAG_FAMILY}; max id is {max_id - 1}")
    marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), int(marker_px))
    return marker.astype(np.uint8)


def board_geometry() -> dict:
    rows = len(ID_GRID)
    cols = len(ID_GRID[0])
    if any(len(row) != cols for row in ID_GRID):
        raise ValueError("ID_GRID must be rectangular.")

    board_width_m = 2.0 * OUTER_MARGIN_M + cols * TAG_SIZE_M + (cols - 1) * TAG_GAP_M
    board_height_m = 2.0 * OUTER_MARGIN_M + rows * TAG_SIZE_M + (rows - 1) * TAG_GAP_M
    return {
        "rows": rows,
        "cols": cols,
        "board_width_m": board_width_m,
        "board_height_m": board_height_m,
    }


def tag_object_points() -> dict[int, list[list[float]]]:
    """Return tag corner coordinates in board frame.

    Board frame:
    - origin: center of the whole printed grid rectangle
    - +X: image right / increasing column
    - +Y: image down / increasing row
    - +Z: away from printed page is not encoded here; all points have z=0

    Corner order follows OpenCV/pupil_apriltags convention:
    top-left, top-right, bottom-right, bottom-left in printed image.
    """
    geom = board_geometry()
    width = geom["board_width_m"]
    height = geom["board_height_m"]
    x0 = -width / 2.0 + OUTER_MARGIN_M
    y0 = -height / 2.0 + OUTER_MARGIN_M

    out: dict[int, list[list[float]]] = {}
    pitch = TAG_SIZE_M + TAG_GAP_M
    for r, row in enumerate(ID_GRID):
        for c, tag_id in enumerate(row):
            left = x0 + c * pitch
            top = y0 + r * pitch
            right = left + TAG_SIZE_M
            bottom = top + TAG_SIZE_M
            out[int(tag_id)] = [
                [left, top, 0.0],
                [right, top, 0.0],
                [right, bottom, 0.0],
                [left, bottom, 0.0],
            ]
    return out


def render_board() -> Image.Image:
    dictionary = get_dictionary()
    geom = board_geometry()

    tag_px = m_to_px(TAG_SIZE_M)
    gap_px = m_to_px(TAG_GAP_M)
    margin_px = m_to_px(OUTER_MARGIN_M)
    board_w_px = 2 * margin_px + geom["cols"] * tag_px + (geom["cols"] - 1) * gap_px
    board_h_px = 2 * margin_px + geom["rows"] * tag_px + (geom["rows"] - 1) * gap_px

    canvas = Image.new("L", (board_w_px, board_h_px), 255)
    draw = ImageDraw.Draw(canvas)

    for r, row in enumerate(ID_GRID):
        for c, tag_id in enumerate(row):
            x = margin_px + c * (tag_px + gap_px)
            y = margin_px + r * (tag_px + gap_px)
            marker = generate_marker(dictionary, int(tag_id), tag_px)
            marker_img = Image.fromarray(marker, mode="L")
            canvas.paste(marker_img, (x, y))
            if ADD_HUMAN_READABLE_IDS:
                draw.text((x + 4, y + 4), str(tag_id), fill=128)

    if ADD_CUT_BORDER:
        draw.rectangle(
            (0, 0, board_w_px - 1, board_h_px - 1),
            outline=0,
            width=max(1, mm_to_px(0.2)),
        )

    return canvas


def load_font(size_px: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size_px)
    return ImageFont.load_default()


def render_a4_page(board: Image.Image) -> Image.Image:
    page_w = mm_to_px(A4_WIDTH_MM)
    page_h = mm_to_px(A4_HEIGHT_MM)
    page = Image.new("RGB", (page_w, page_h), "white")

    board_rgb = board.convert("RGB")
    board_x = (page_w - board_rgb.width) // 2
    board_y = (page_h - board_rgb.height) // 2
    page.paste(board_rgb, (board_x, board_y))

    return page


def save_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = render_board()
    a4_page = render_a4_page(image)
    geom = board_geometry()
    tag_points = tag_object_points()

    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}_{DPI}dpi.pdf"
    yaml_path = OUTPUT_DIR / f"{OUTPUT_STEM}.yaml"

    a4_page.save(pdf_path, "PDF", resolution=float(DPI))

    config = {
        "target_type": "apriltag_grid",
        "tag_family": TAG_FAMILY,
        "id_grid": ID_GRID,
        "rows": geom["rows"],
        "cols": geom["cols"],
        "tag_size_m": float(TAG_SIZE_M),
        "tag_gap_m": float(TAG_GAP_M),
        "min_corners_per_sample": int(MIN_CORNERS_PER_SAMPLE),
        "outer_margin_m": float(OUTER_MARGIN_M),
        "board_width_m": float(geom["board_width_m"]),
        "board_height_m": float(geom["board_height_m"]),
        "board_frame": {
            "origin": "center of full printed grid rectangle",
            "x_axis": "right in printed image",
            "y_axis": "down in printed image",
            "z_axis": "board normal; object points have z=0",
            "corner_order": "top-left, top-right, bottom-right, bottom-left",
        },
        "tag_object_points": {
            int(tag_id): points
            for tag_id, points in sorted(tag_points.items())
        },
        "outputs": {
            "pdf": str(pdf_path),
            "yaml": str(yaml_path),
            "dpi": int(DPI),
            "pdf_page": "A4 portrait",
        },
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("[INFO] Generated AprilTag grid")
    print(f"  family: {TAG_FAMILY}")
    print(f"  grid: {geom['cols']} cols x {geom['rows']} rows")
    print(f"  tag_size: {TAG_SIZE_M * 1000.0:.2f} mm")
    print(f"  tag_gap: {TAG_GAP_M * 1000.0:.2f} mm")
    print(f"  board: {geom['board_width_m'] * 1000.0:.2f} x {geom['board_height_m'] * 1000.0:.2f} mm")
    print(f"  pdf: {pdf_path}")
    print(f"  yaml: {yaml_path}")


if __name__ == "__main__":
    save_outputs()
