#!/usr/bin/env python3
"""Generate the 50 mm AprilTag target for optical/physical-origin calibration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


NAME = "tiny_physical_optics_frame_offset"
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs" / NAME
PDF_PATH = OUTPUT_DIR / f"{NAME}.pdf"
YAML_PATH = OUTPUT_DIR / f"{NAME}.yaml"

TAG_FAMILY_OPENCV = "DICT_APRILTAG_36h11"
TAG_FAMILY_PUPIL = "tag36h11"
ROWS = 3
COLS = 3
ID_GRID = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
]

# The outer edge of the black boundary is exactly 50 mm square. The tag size is
# an exact multiple of the 8 modules in a 36h11 marker (6 data + 2 border).
BOARD_SIZE_MM = 50.0
TAG_SIZE_MM = 13.6
TAG_GAP_MM = 2.0
TAG_PITCH_MM = TAG_SIZE_MM + TAG_GAP_MM
TAG_ENVELOPE_MM = COLS * TAG_SIZE_MM + (COLS - 1) * TAG_GAP_MM
OUTER_MARGIN_MM = (BOARD_SIZE_MM - TAG_ENVELOPE_MM) / 2.0
FRAME_WIDTH_MM = 0.4
QUIET_MARGIN_MM = OUTER_MARGIN_MM - FRAME_WIDTH_MM
TAG_MODULE_COUNT = 8

PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 297.0
VALIDATION_PX_PER_MM = 100


def rounded(value: float) -> float:
    return float(f"{value:.12f}")


def get_dictionary():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing; install opencv-contrib-python.")
    if not hasattr(cv2.aruco, TAG_FAMILY_OPENCV):
        raise RuntimeError(f"OpenCV does not provide {TAG_FAMILY_OPENCV}.")
    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, TAG_FAMILY_OPENCV)
    )


def generate_marker(dictionary, tag_id: int, side_px: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, side_px)
    marker = np.zeros((side_px, side_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, tag_id, side_px, marker, 1)
    return marker


def marker_modules(dictionary, tag_id: int) -> np.ndarray:
    """Return the exact 8x8 black/white module raster for one 36h11 marker."""
    pixels_per_module = 40
    rendered = generate_marker(
        dictionary,
        tag_id,
        TAG_MODULE_COUNT * pixels_per_module,
    )
    modules = np.empty((TAG_MODULE_COUNT, TAG_MODULE_COUNT), dtype=np.uint8)
    for row in range(TAG_MODULE_COUNT):
        for col in range(TAG_MODULE_COUNT):
            block = rendered[
                row * pixels_per_module : (row + 1) * pixels_per_module,
                col * pixels_per_module : (col + 1) * pixels_per_module,
            ]
            if not np.all(block == block[0, 0]):
                raise RuntimeError(
                    f"Marker {tag_id} module ({row}, {col}) is not uniform."
                )
            modules[row, col] = block[0, 0]
    return modules


def tag_left_top_mm(row: int, col: int) -> tuple[float, float]:
    return (
        OUTER_MARGIN_MM + col * TAG_PITCH_MM,
        OUTER_MARGIN_MM + row * TAG_PITCH_MM,
    )


def render_validation_image(dictionary) -> np.ndarray:
    """Render a high-resolution board raster for clean-image detection QA."""
    px_per_mm = VALIDATION_PX_PER_MM
    board_px = int(round(BOARD_SIZE_MM * px_per_mm))
    frame_px = int(round(FRAME_WIDTH_MM * px_per_mm))
    tag_px = int(round(TAG_SIZE_MM * px_per_mm))
    image = np.full((board_px, board_px), 255, dtype=np.uint8)

    image[:frame_px, :] = 0
    image[-frame_px:, :] = 0
    image[:, :frame_px] = 0
    image[:, -frame_px:] = 0

    for row, ids in enumerate(ID_GRID):
        for col, tag_id in enumerate(ids):
            left_mm, top_mm = tag_left_top_mm(row, col)
            left_px = int(round(left_mm * px_per_mm))
            top_px = int(round(top_mm * px_per_mm))
            marker = generate_marker(dictionary, tag_id, tag_px)
            image[top_px : top_px + tag_px, left_px : left_px + tag_px] = marker
    return image


def validate_clean_detection(
    image: np.ndarray,
    dictionary,
) -> tuple[int, int, list[int]]:
    padding_px = 200
    padded = cv2.copyMakeBorder(
        image,
        padding_px,
        padding_px,
        padding_px,
        padding_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _rejected = detector.detectMarkers(padded)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            padded,
            dictionary,
            parameters=parameters,
        )

    detected_ids = [] if ids is None else sorted(int(v) for v in ids.reshape(-1))
    expected_ids = sorted(tag_id for row in ID_GRID for tag_id in row)
    if detected_ids != expected_ids:
        raise RuntimeError(
            "Generated board failed clean-image detection: "
            f"detected IDs {detected_ids}, expected {expected_ids}."
        )
    return len(detected_ids), len(detected_ids) * 4, detected_ids


def draw_black_frame(
    pdf: canvas.Canvas,
    left: float,
    bottom: float,
    size: float,
) -> None:
    """Draw the cutting frame entirely inside the exact 50 mm boundary."""
    width = FRAME_WIDTH_MM * mm
    pdf.setFillColorRGB(0.0, 0.0, 0.0)
    pdf.rect(left, bottom, size, width, stroke=0, fill=1)
    pdf.rect(left, bottom + size - width, size, width, stroke=0, fill=1)
    pdf.rect(left, bottom + width, width, size - 2.0 * width, stroke=0, fill=1)
    pdf.rect(
        left + size - width,
        bottom + width,
        width,
        size - 2.0 * width,
        stroke=0,
        fill=1,
    )


def draw_vector_marker(
    pdf: canvas.Canvas,
    modules: np.ndarray,
    left: float,
    bottom: float,
) -> None:
    tag_size = TAG_SIZE_MM * mm
    module_size = tag_size / TAG_MODULE_COUNT
    pdf.setFillColorRGB(0.0, 0.0, 0.0)
    for row in range(TAG_MODULE_COUNT):
        for col in range(TAG_MODULE_COUNT):
            if modules[row, col] != 0:
                continue
            x0 = col * module_size
            x1 = (col + 1) * module_size
            y0 = (TAG_MODULE_COUNT - row - 1) * module_size
            y1 = (TAG_MODULE_COUNT - row) * module_size
            # A tiny overlap prevents hairline seams between adjacent black cells.
            # Clamp it at the marker edge so the printed tag geometry stays exact.
            overlap = 0.002 * mm
            pdf.rect(
                left + max(0.0, x0 - overlap),
                bottom + max(0.0, y0 - overlap),
                min(tag_size, x1 + overlap) - max(0.0, x0 - overlap),
                min(tag_size, y1 + overlap) - max(0.0, y0 - overlap),
                stroke=0,
                fill=1,
            )


def draw_dimension_check(
    pdf: canvas.Canvas,
    page_width: float,
    y: float,
) -> None:
    length = BOARD_SIZE_MM * mm
    left = (page_width - length) / 2.0
    pdf.setStrokeColorRGB(0.0, 0.0, 0.0)
    pdf.setLineWidth(0.2 * mm)
    pdf.line(left, y, left + length, y)
    pdf.line(left, y - 1.5 * mm, left, y + 1.5 * mm)
    pdf.line(left + length, y - 1.5 * mm, left + length, y + 1.5 * mm)
    pdf.setFillColorRGB(0.1, 0.1, 0.1)
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(page_width / 2.0, y + 2.5 * mm, "50.00 mm print check")


def write_pdf(dictionary) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page_size = (PAGE_WIDTH_MM * mm, PAGE_HEIGHT_MM * mm)
    pdf = canvas.Canvas(str(PDF_PATH), pagesize=page_size, pageCompression=1)
    pdf.setTitle(NAME)
    pdf.setAuthor("RobotCamCalib1")
    pdf.setSubject(
        "Print at 100% / Actual Size. The outer edge of the black frame is "
        "exactly 50.00 mm square."
    )

    board_size = BOARD_SIZE_MM * mm
    board_left = (page_size[0] - board_size) / 2.0
    board_bottom = (page_size[1] - board_size) / 2.0

    pdf.setFillColorRGB(1.0, 1.0, 1.0)
    pdf.rect(board_left, board_bottom, board_size, board_size, stroke=0, fill=1)
    draw_black_frame(pdf, board_left, board_bottom, board_size)

    for row, ids in enumerate(ID_GRID):
        for col, tag_id in enumerate(ids):
            left_mm, top_mm = tag_left_top_mm(row, col)
            tag_left = board_left + left_mm * mm
            tag_bottom = (
                board_bottom
                + (BOARD_SIZE_MM - top_mm - TAG_SIZE_MM) * mm
            )
            draw_vector_marker(
                pdf,
                marker_modules(dictionary, tag_id),
                tag_left,
                tag_bottom,
            )

    pdf.setFillColorRGB(0.1, 0.1, 0.1)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawCentredString(
        page_size[0] / 2.0,
        board_bottom + board_size + 22.0 * mm,
        NAME,
    )
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(
        page_size[0] / 2.0,
        board_bottom + board_size + 16.5 * mm,
        "AprilTag 36h11 | 3 x 3 | IDs 0-8 | up to 36 detected corners",
    )
    pdf.drawCentredString(
        page_size[0] / 2.0,
        board_bottom - 16.0 * mm,
        "Print at 100% / Actual Size. Disable Fit, Shrink, and Oversize.",
    )
    pdf.drawCentredString(
        page_size[0] / 2.0,
        board_bottom - 21.0 * mm,
        "Cut along the OUTER edge of the black frame: 50.00 x 50.00 mm.",
    )
    draw_dimension_check(pdf, page_size[0], board_bottom - 31.0 * mm)
    pdf.showPage()
    pdf.save()


def physical_tag_corners_m(row: int, col: int) -> list[list[float]]:
    """OpenCV order in the physical frame: TL, TR, BR, BL; +x right, +y down."""
    left_mm, top_mm = tag_left_top_mm(row, col)
    left_mm -= BOARD_SIZE_MM / 2.0
    top_mm -= BOARD_SIZE_MM / 2.0
    right_mm = left_mm + TAG_SIZE_MM
    bottom_mm = top_mm + TAG_SIZE_MM
    return [
        [rounded(left_mm / 1000.0), rounded(top_mm / 1000.0), 0.0],
        [rounded(right_mm / 1000.0), rounded(top_mm / 1000.0), 0.0],
        [rounded(right_mm / 1000.0), rounded(bottom_mm / 1000.0), 0.0],
        [rounded(left_mm / 1000.0), rounded(bottom_mm / 1000.0), 0.0],
    ]


def pupil_tag_corners_mm() -> list[list[float]]:
    """Corner order used by the generated target geometry."""
    half = TAG_SIZE_MM / 2.0
    return [
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ]


def make_legacy_tags() -> list[dict]:
    local_corners = pupil_tag_corners_mm()
    records = []
    for row, ids in enumerate(ID_GRID):
        for col, tag_id in enumerate(ids):
            # In this convention +x is printed
            # left and +y is printed up.
            center_x = ((COLS - 1) / 2.0 - col) * TAG_PITCH_MM
            center_y = ((ROWS - 1) / 2.0 - row) * TAG_PITCH_MM
            corners = [
                [
                    rounded(corner[0] + center_x),
                    rounded(corner[1] + center_y),
                    0.0,
                ]
                for corner in local_corners
            ]
            records.append(
                {
                    "id": tag_id,
                    "row": row,
                    "col": col,
                    "center_mm": [
                        rounded(center_x),
                        rounded(center_y),
                        0.0,
                    ],
                    "T_board_tag": [
                        [1.0, 0.0, 0.0, rounded(center_x)],
                        [0.0, 1.0, 0.0, rounded(center_y)],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "corners_board_mm": corners,
                }
            )
    return records


def write_yaml(
    detected_tag_count: int,
    detected_corner_count: int,
    detected_ids: list[int],
) -> None:
    tag_object_points = {}
    for row, ids in enumerate(ID_GRID):
        for col, tag_id in enumerate(ids):
            tag_object_points[tag_id] = physical_tag_corners_m(row, col)

    config = {
        # The v1 fields keep the target geometry self-describing.
        "schema": "robot_cam_calib.apriltag_board.v1",
        "target_type": "apriltag_grid",
        "name": NAME,
        "family": TAG_FAMILY_PUPIL,
        "units": "mm",
        "layout": {
            "rows": ROWS,
            "cols": COLS,
            "tag_id_start": 0,
            "tag_id_end": 8,
            "id_grid": ID_GRID,
        },
        "geometry": {
            "tag_size_mm": TAG_SIZE_MM,
            "marker_fraction": rounded(TAG_SIZE_MM / TAG_PITCH_MM),
            "tile_size_mm": TAG_SIZE_MM,
            "explicit_gap_mm": TAG_GAP_MM,
            "pitch_mm": TAG_PITCH_MM,
            "black_marker_edge_gap_mm": TAG_GAP_MM,
            "tag_envelope_size_mm": [TAG_ENVELOPE_MM, TAG_ENVELOPE_MM],
            "outer_margin_mm": rounded(OUTER_MARGIN_MM),
            "black_frame_width_mm": FRAME_WIDTH_MM,
            "quiet_margin_to_frame_inner_edge_mm": rounded(QUIET_MARGIN_MM),
            "board_size_mm": [BOARD_SIZE_MM, BOARD_SIZE_MM],
            "outer_size_includes_black_frame": True,
            "physical_origin_tag_id": 4,
        },
        "target_frame": {
            "name": "board",
            "origin": "center of the complete 50 mm outer boundary",
            "x_axis": "left in the printed image",
            "y_axis": "up in the printed image",
            "z_axis": "x cross y; into the target when viewed from the printed front",
        },
        "detection_corner_order": {
            "source": "pupil_apriltags.Detection.corners",
            "tag_frame_corners_mm": pupil_tag_corners_mm(),
            "description": (
                "Use each pupil_apriltags detection's corners in its returned order "
                "with the matching corners_board_mm entry."
            ),
        },
        "tags": make_legacy_tags(),
        # The following fields are consumed by this project's OpenCV grid loaders.
        "tag_family": TAG_FAMILY_OPENCV,
        "id_grid": ID_GRID,
        "rows": ROWS,
        "cols": COLS,
        "tag_size_m": rounded(TAG_SIZE_MM / 1000.0),
        "tag_gap_m": rounded(TAG_GAP_MM / 1000.0),
        "board_width_m": rounded(BOARD_SIZE_MM / 1000.0),
        "board_height_m": rounded(BOARD_SIZE_MM / 1000.0),
        "min_corners_per_sample": 16,
        "tag_object_points": tag_object_points,
        "frames": {
            "physical_dimensions": {
                "origin": (
                    "center of the complete 50 mm outer boundary; also the center "
                    "of tag ID 4"
                ),
                "x_axis": "right in the printed image",
                "y_axis": "down in the printed image",
                "z_axis": "x cross y; into the target when viewed from the printed front",
                "object_points": "top-level tag_object_points, in metres",
            },
            "board": {
                "origin": "same physical point as physical_dimensions",
                "x_axis": "left in the printed image",
                "y_axis": "up in the printed image",
                "z_axis": "same as physical_dimensions",
                "object_points": "tags[].corners_board_mm, in millimetres",
            },
            "camera_optical": {
                "origin": "camera optical center",
                "x_axis": "right in the camera image",
                "y_axis": "down in the camera image",
                "z_axis": "forward along the optical axis",
            },
        },
        "frame_transforms": {
            "notation": "T_A_B maps coordinates expressed in frame B into frame A",
            "T_physical_dimensions_board": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "T_board_physical_dimensions": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "pose_usage": {
            "opencv_detector_result": "T_camera_optical_physical_dimensions",
            "pupil_detector_result": "T_camera_optical_board",
            "pupil_to_physical_formula": (
                "T_camera_optical_physical_dimensions = "
                "T_camera_optical_board @ T_board_physical_dimensions"
            ),
            "optical_to_physical_origin_offset": (
                "translation column of T_camera_optical_physical_dimensions"
            ),
            "physical_to_optical_origin_offset": (
                "translation column of inverse(T_camera_optical_physical_dimensions)"
            ),
            "note": (
                "The board and physical_dimensions origins coincide, so both "
                "detector conventions produce the same origin translation."
            ),
        },
        "detection_recommendations": {
            "corner_refinement": "APRILTAG or SUBPIX",
            "maximum_corner_count": ROWS * COLS * 4,
            "recommended_minimum_tags": 4,
            "recommended_minimum_corners": 16,
            "use_all_visible_tags_in_one_bundle_pnp_solve": True,
        },
        "print": {
            "pdf": PDF_PATH.relative_to(REPO_ROOT).as_posix(),
            "paper": "A4 portrait",
            "page_size_mm": [PAGE_WIDTH_MM, PAGE_HEIGHT_MM],
            "print_scale_percent": 100,
            "scaling_instruction": (
                "Print at 100% / Actual Size; disable Fit, Shrink, and Oversize."
            ),
            "cut_instruction": (
                "Cut along the outer edge of the black frame; the result is "
                "exactly 50.00 x 50.00 mm including the frame."
            ),
            "verification": (
                "Measure both the target outer edge and the separate 50.00 mm "
                "check line after printing."
            ),
        },
        "generation_validation": {
            "clean_image_detected_tags": detected_tag_count,
            "clean_image_detected_corners": detected_corner_count,
            "clean_image_detected_ids": detected_ids,
        },
    }
    class NoAliasSafeDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    with YAML_PATH.open("w", encoding="utf-8") as stream:
        yaml.dump(
            config,
            stream,
            Dumper=NoAliasSafeDumper,
            sort_keys=False,
            allow_unicode=True,
        )


def validate_geometry() -> None:
    if len(ID_GRID) != ROWS or any(len(row) != COLS for row in ID_GRID):
        raise RuntimeError("ID_GRID does not match ROWS/COLS.")
    if not np.isclose(TAG_ENVELOPE_MM + 2.0 * OUTER_MARGIN_MM, BOARD_SIZE_MM):
        raise RuntimeError("Tag envelope and margins do not sum to board size.")
    if QUIET_MARGIN_MM <= TAG_SIZE_MM / TAG_MODULE_COUNT:
        raise RuntimeError("The outer tag quiet margin is less than one tag module.")
    if int(round(TAG_SIZE_MM * VALIDATION_PX_PER_MM)) % TAG_MODULE_COUNT != 0:
        raise RuntimeError("Validation raster does not sample tag modules evenly.")


def main() -> None:
    validate_geometry()
    dictionary = get_dictionary()
    validation_image = render_validation_image(dictionary)
    tag_count, corner_count, ids = validate_clean_detection(
        validation_image,
        dictionary,
    )
    write_pdf(dictionary)
    write_yaml(tag_count, corner_count, ids)

    print(f"Generated: {PDF_PATH}")
    print(f"Generated: {YAML_PATH}")
    print(
        f"Target: {BOARD_SIZE_MM:.2f} x {BOARD_SIZE_MM:.2f} mm including frame, "
        f"{tag_count} tags, {corner_count} available tag corners"
    )
    print(
        f"Tag edge: {TAG_SIZE_MM:.2f} mm, gap: {TAG_GAP_MM:.2f} mm, "
        f"black frame: {FRAME_WIDTH_MM:.2f} mm"
    )


if __name__ == "__main__":
    main()
