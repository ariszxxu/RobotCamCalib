from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------------------------- User macros ---------------------------- #
DEFAULT_SAMPLE_PKL = Path(
    "outputs/extrinsics_fingertip_apriltag_grid_usable_samples/"
    "extrinsics_fingertip_Q_root_tip_apriltag_grid_0705_214543/"
    "usable_samples.pkl"
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8081
DEFAULT_MAX_IMAGE_WIDTH = 900
DEFAULT_JPEG_QUALITY = 85
AUTO_PLAY_PERIOD_S = 0.5


def load_pickle(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"PKL root must be a dict: {resolved}")
    samples = payload.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"PKL has no samples: {resolved}")
    payload["_resolved_path"] = str(resolved)
    return payload


def decode_png(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode PNG image from pkl.")
    return image


def resize_for_display(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr
    scale = float(max_width) / float(w)
    return cv2.resize(
        image_bgr,
        (max_width, max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def bgr_to_rgb_for_viser(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    image_bgr = resize_for_display(image_bgr, max_width)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def sample_image(sample: dict[str, Any], key: str) -> np.ndarray:
    images_png = sample.get("images_png", {})
    if key in images_png:
        return decode_png(images_png[key])
    if key == "third_view_Q_and_B_reprojection" and "third_view_aprilcube_reprojection" in images_png:
        fallback = decode_png(images_png["third_view_aprilcube_reprojection"])
        return draw_missing_overlay_notice(fallback, "missing third_view_Q_and_B_reprojection in pkl")
    if key == "third_view_aprilcube_reprojection" and "third_view" in images_png:
        fallback = decode_png(images_png["third_view"])
        return draw_missing_overlay_notice(fallback, "missing third_view_aprilcube_reprojection in pkl")
    raise KeyError(f"Sample {sample.get('index')} has no image key: {key}")


def draw_missing_overlay_notice(image_bgr: np.ndarray, text: str) -> np.ndarray:
    out = image_bgr.copy()
    cv2.putText(
        out,
        text,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    return out


def residual_text(sample: dict[str, Any]) -> str:
    residual = sample.get("solution_residual") or {}
    errors = sample.get("errors") or {}
    lines = [
        f"`sample_index`: `{sample.get('index')}`",
        f"`capture_mode`: `{sample.get('capture_mode')}`",
        f"`root_rot_deg`: `{float(residual.get('root_rot_deg', float('nan'))):.3f}`",
        f"`tip_rot_deg`: `{float(residual.get('tip_rot_deg', float('nan'))):.3f}`",
        f"`root_trans_mm`: `{float(residual.get('root_trans_m', float('nan'))) * 1000.0:.2f}`",
        f"`tip_trans_mm`: `{float(residual.get('tip_trans_m', float('nan'))) * 1000.0:.2f}`",
        f"`E_aprilcube_reproj_px`: `{float(errors.get('E_aprilcube_reproj_px', float('nan'))):.3f}`",
        f"`E_apriltag_grid_reproj_px`: `{float(errors.get('E_apriltag_grid_reproj_px', float('nan'))):.3f}`",
        f"`root_grid_reproj_px`: `{float(errors.get('root_apriltag_grid_reproj_px', float('nan'))):.3f}`",
        f"`tip_grid_reproj_px`: `{float(errors.get('tip_apriltag_grid_reproj_px', float('nan'))):.3f}`",
    ]
    return "\n".join(lines)


def transform_text(sample: dict[str, Any]) -> str:
    def fmt_t(key: str) -> str:
        T = np.asarray(sample.get(key), dtype=np.float64).reshape(4, 4)
        t = T[:3, 3] * 1000.0
        return f"`{key}` t_mm = `[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]`"

    return "\n".join(
        [
            fmt_t("T_E_Q"),
            fmt_t("T_E_B"),
            fmt_t("T_Q_root"),
            fmt_t("T_Q_tip"),
        ]
    )


def pkl_summary(payload: dict[str, Any]) -> str:
    diagnostics = payload.get("diagnostics", {})
    lines = [
        f"`pkl`: `{payload.get('_resolved_path')}`",
        f"`source_yaml`: `{payload.get('source_extrinsics_yaml')}`",
        f"`num_usable_samples`: `{payload.get('num_usable_samples')}`",
    ]
    for key in (
        "root_residual_rot_deg",
        "root_residual_trans_m",
        "tip_residual_rot_deg",
        "tip_residual_trans_m",
    ):
        if key in diagnostics:
            lines.append(f"`{key}`: `{diagnostics[key]}`")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize fingertip extrinsic calibration usable_samples.pkl in viser."
    )
    parser.add_argument("--pkl", type=Path, default=DEFAULT_SAMPLE_PKL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_IMAGE_WIDTH)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = load_pickle(args.pkl)
    samples = payload["samples"]

    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("viser is required. Run this script in the pyroki environment.") from exc

    first_sample = samples[0]
    max_width = int(args.max_width)
    jpeg_quality = int(args.jpeg_quality)

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.world_axes.visible = False

    with server.gui.add_folder("Sample Controls"):
        frame_slider = server.gui.add_slider(
            "Sample",
            min=0,
            max=len(samples) - 1,
            step=1,
            initial_value=0,
        )
        auto_play = server.gui.add_checkbox("Auto play", initial_value=False)
        status_text = server.gui.add_text(
            "Status",
            initial_value=f"sample 1/{len(samples)} index={first_sample.get('index')}",
            disabled=True,
        )

    with server.gui.add_folder("Third View"):
        third_Q_B_handle = server.gui.add_image(
            bgr_to_rgb_for_viser(sample_image(first_sample, "third_view_Q_and_B_reprojection"), max_width),
            label="AprilCube Q + AprilTag grid B reprojection",
            format="jpeg",
            jpeg_quality=jpeg_quality,
        )
        third_reproj_handle = server.gui.add_image(
            bgr_to_rgb_for_viser(sample_image(first_sample, "third_view_aprilcube_reprojection"), max_width),
            label="AprilCube reprojection",
            format="jpeg",
            jpeg_quality=jpeg_quality,
        )
        third_raw_handle = server.gui.add_image(
            bgr_to_rgb_for_viser(sample_image(first_sample, "third_view"), max_width),
            label="Raw third-view",
            format="jpeg",
            jpeg_quality=jpeg_quality,
        )

    with server.gui.add_folder("Fingertip Images"):
        root_handle = server.gui.add_image(
            bgr_to_rgb_for_viser(sample_image(first_sample, "root"), max_width),
            label="root",
            format="jpeg",
            jpeg_quality=jpeg_quality,
        )
        tip_handle = server.gui.add_image(
            bgr_to_rgb_for_viser(sample_image(first_sample, "tip"), max_width),
            label="tip",
            format="jpeg",
            jpeg_quality=jpeg_quality,
        )

    with server.gui.add_folder("Per-sample Metrics"):
        residual_markdown = server.gui.add_markdown(residual_text(first_sample))
        transform_markdown = server.gui.add_markdown(transform_text(first_sample))

    with server.gui.add_folder("PKL Summary"):
        server.gui.add_markdown(pkl_summary(payload))

    print(f"[INFO] loaded pkl: {payload['_resolved_path']}")
    print(f"[INFO] samples: {len(samples)}")
    print(f"[INFO] viser: http://localhost:{int(args.port)}")

    current_idx = -1
    last_auto_step = time.monotonic()
    while True:
        if bool(auto_play.value):
            now = time.monotonic()
            if now - last_auto_step >= AUTO_PLAY_PERIOD_S:
                frame_slider.value = (int(frame_slider.value) + 1) % len(samples)
                last_auto_step = now
        else:
            last_auto_step = time.monotonic()

        idx = int(frame_slider.value)
        if idx != current_idx:
            sample = samples[idx]
            third_Q_B_handle.image = bgr_to_rgb_for_viser(
                sample_image(sample, "third_view_Q_and_B_reprojection"),
                max_width,
            )
            third_reproj_handle.image = bgr_to_rgb_for_viser(
                sample_image(sample, "third_view_aprilcube_reprojection"),
                max_width,
            )
            third_raw_handle.image = bgr_to_rgb_for_viser(sample_image(sample, "third_view"), max_width)
            root_handle.image = bgr_to_rgb_for_viser(sample_image(sample, "root"), max_width)
            tip_handle.image = bgr_to_rgb_for_viser(sample_image(sample, "tip"), max_width)
            status_text.value = f"sample {idx + 1}/{len(samples)} index={sample.get('index')}"
            residual_markdown.content = residual_text(sample)
            transform_markdown.content = transform_text(sample)
            current_idx = idx
        time.sleep(0.03)


if __name__ == "__main__":
    main()
