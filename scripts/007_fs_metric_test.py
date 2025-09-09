#!/usr/bin/env python3
import os
import time
import cv2
import yaml
import numpy as np
import imageio.v2 as imageio

from contaccams.fs_wrapper import FStereoConfig, FoundationStereoWrapper
from contaccams.stereo_rectify import StereoRectifier  # must provide .load_or_build_maps(), .rectify_pair(), .alpha

# ------------------- Defaults (edit if needed) ------------------- #
camera_to_port = {
    "tip_cam":  "3-10:1.0",   # LEFT
    "root_cam": "3-9:1.0",    # RIGHT
}
left_name, right_name = "tip_cam", "root_cam"

out_dir = "outputs"
stereo_yaml     = f"{out_dir}/{left_name}_{right_name}_stereo.yaml"
maps_npz        = f"{out_dir}/{left_name}_{right_name}_rectify_maps.npz"  # or None to force rebuild
ckpt_path = "/home/ps/projects/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"

# Rectification alpha (must match how you intend to run)
rectify_alpha   = -1.0  # try 0.0, 0.5, 1.0, or -1 (OpenCV chooses)

# Display window size
WIN_W, WIN_H = 1400, 800
# FS speed/quality
VALID_ITERS = 24      # lower → faster, higher → better
SCALE       = 1.0     # set <1.0 to speed up (wrapper scales K)
Z_FAR       = 10.0
# --------------------------------------------------------------- #


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _recompute_rectified_Ps(stereo_data: dict, alpha: float):
    W, H = int(stereo_data["image_size"][0]), int(stereo_data["image_size"][1])
    K1 = np.asarray(stereo_data["K1"], np.float64)
    D1 = np.asarray(stereo_data["D1"], np.float64).reshape(-1, 1)
    K2 = np.asarray(stereo_data["K2"], np.float64)
    D2 = np.asarray(stereo_data["D2"], np.float64).reshape(-1, 1)
    R  = np.asarray(stereo_data["R"],  np.float64)
    T  = np.asarray(stereo_data["T"],  np.float64).reshape(3,)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (W, H), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
    )
    return R1, R2, P1, P2, Q


def _rectified_K_and_baseline(stereo_yaml_path: str, alpha: float):
    data = _load_yaml(stereo_yaml_path)
    R1, R2, P1, P2, Q = _recompute_rectified_Ps(data, alpha)
    fx = float(P1[0, 0]); fy = float(P1[1, 1])
    cx = float(P1[0, 2]); cy = float(P1[1, 2])
    K_rect = np.array([[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    baseline = float(-P2[0, 3] / fx)
    return K_rect, baseline


def _open_caps_sync() -> tuple[cv2.VideoCapture, cv2.VideoCapture]:
    """Open both cameras by USB port strings; returns two V4L2 captures."""
    import pyudev
    ctx = pyudev.Context()

    def dev_for_port(port_str: str) -> str:
        for dev in ctx.list_devices(subsystem='video4linux'):
            if dev.parent and dev.parent.subsystem == 'usb':
                usb_port = dev.parent.get('DEVPATH', '').split('/')[-1]
                if port_str in usb_port:
                    return dev.device_node
        raise RuntimeError(f"USB port not found: {port_str}")

    left_dev  = dev_for_port(camera_to_port[left_name])
    right_dev = dev_for_port(camera_to_port[right_name])
    capL = cv2.VideoCapture(left_dev,  cv2.CAP_V4L2)
    capR = cv2.VideoCapture(right_dev, cv2.CAP_V4L2)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Failed to open one or both cameras.")
    return capL, capR


def _grab_pair(capL: cv2.VideoCapture, capR: cv2.VideoCapture) -> tuple[np.ndarray, np.ndarray]:
    """Two-phase (grab→retrieve) to better sync the pair; returns BGR frames."""
    capL.grab(); capR.grab()
    okL, fL = capL.retrieve()
    okR, fR = capR.retrieve()
    if not okL or not okR:
        raise RuntimeError("Failed to read from cameras.")
    return fL, fR


def _vis_disparity(disp: np.ndarray) -> np.ndarray:
    """Return a BGR colormap of disparity."""
    d = disp.copy()
    bad = ~np.isfinite(d) | (d <= 0)
    if np.all(bad):
        return np.zeros((*d.shape, 3), np.uint8)
    d[bad] = 0
    vmax = float(max(np.percentile(d[~bad], 99.0), 1e-6))
    d = np.clip(d / vmax, 0, 1)
    d8 = (d * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_PLASMA)


def main():
    os.makedirs(out_dir, exist_ok=True)

    # 1) Rectifier + rectified intrinsics/baseline
    rectifier = StereoRectifier(
        camera_to_port=camera_to_port,
        left_name=left_name,
        right_name=right_name,
        stereo_yaml=stereo_yaml,
        maps_npz=maps_npz,
        alpha=rectify_alpha,
        use_npz_if_available=True,
    )
    rectifier.load_or_build_maps()
    K_rect, baseline = _rectified_K_and_baseline(stereo_yaml, rectifier.alpha)

    # 2) FS wrapper (stays in memory)
    cfg = FStereoConfig(
        ckpt_path=ckpt_path,
        scale=SCALE,                # if <1.0, wrapper scales K accordingly
        hiera=False,
        valid_iters=VALID_ITERS,
        remove_invisible=True,
        z_far=Z_FAR,
        device="cuda",
        amp=True,
    )
    fs = FoundationStereoWrapper(cfg)

    # 3) Open cameras at calibrated size
    capL, capR = _open_caps_sync()
    capL.set(cv2.CAP_PROP_FRAME_WIDTH,  rectifier.W); capL.set(cv2.CAP_PROP_FRAME_HEIGHT, rectifier.H)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH,  rectifier.W); capR.set(cv2.CAP_PROP_FRAME_HEIGHT, rectifier.H)

    # 4) Interactive window & mouse callback
    win = "Interactive FS (click left panel for depth)  [q: quit, p: pause]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIN_W, WIN_H)

    # state for clicks
    click_xy = None   # image coords in rectified space (x,y)
    last_depth = np.nan
    paused = False
    fps = 0.0
    last_t = time.time()

    # mapping info for mouse callback (updated each frame)
    panel_w = rectifier.W
    panel_h = rectifier.H
    disp_scale = 1.0
    off_x_left = 0
    off_y = 0

    def on_mouse(event, x, y, flags, userdata):
        nonlocal click_xy, last_depth
        # Map (x,y) in window to rectified-left image coords
        if event == cv2.EVENT_LBUTTONDOWN:
            # check if inside left panel region on the canvas
            # canvas layout before resizing: [left | disp] side-by-side
            # after resizing, both panels scaled by disp_scale and centered with offsets
            # We compute coordinates assuming the same layout.
            # Left panel area in window:
            x0 = off_x_left
            y0 = off_y
            x1 = x0 + int(panel_w * disp_scale)
            y1 = y0 + int(panel_h * disp_scale)
            if x0 <= x < x1 and y0 <= y < y1:
                # map to image coords
                ix = int((x - x0) / disp_scale)
                iy = int((y - y0) / disp_scale)
                click_xy = (ix, iy)

    cv2.setMouseCallback(win, on_mouse)

    depth = None
    disp = None
    rectL_rgb = None

    while True:
        if not paused or rectL_rgb is None:
            # acquire
            rawL_bgr, rawR_bgr = _grab_pair(capL, capR)
            if (rawL_bgr.shape[1], rawL_bgr.shape[0]) != (rectifier.W, rectifier.H):
                rawL_bgr = cv2.resize(rawL_bgr, (rectifier.W, rectifier.H), interpolation=cv2.INTER_AREA)
            if (rawR_bgr.shape[1], rawR_bgr.shape[0]) != (rectifier.W, rectifier.H):
                rawR_bgr = cv2.resize(rawR_bgr, (rectifier.W, rectifier.H), interpolation=cv2.INTER_AREA)

            # rectify → RGB for FS
            rectL_rgb, rectR_rgb = rectifier.rectify_pair(
                rawL_bgr, rawR_bgr,
                input_color="bgr",
                output_color="rgb",
                resize_to_calib=False
            )

            # FS inference
            out = fs.infer(
                left_rgb=rectL_rgb,
                right_rgb=rectR_rgb,
                K=K_rect,
                baseline=baseline / 1000.0,
                return_xyz=False,
            )
            disp  = out["disp"]
            depth = out["depth"]
            valid_mask = out["valid_mask"]

            # fps update
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 1.0 / dt

        # Build display canvas (rectL + disparity colormap)
        left_bgr = cv2.cvtColor(rectL_rgb, cv2.COLOR_RGB2BGR)
        disp_vis = _vis_disparity(disp)
        canvas = np.concatenate([left_bgr, disp_vis], axis=1)  # (H, 2W, 3)

        # Place a marker & depth text if clicked
        if click_xy is not None and depth is not None:
            ix, iy = click_xy
            if 0 <= ix < rectifier.W and 0 <= iy < rectifier.H:
                z = float(depth[iy, ix])
                if not np.isfinite(z) or z <= 0:
                    txt = f"({ix},{iy})  Z = invalid"
                else:
                    txt = f"({ix},{iy})  Z = {z:.3f} m"
                # draw on left panel (image-space coords)
                cv2.circle(canvas, (ix, iy), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, txt, (ix + 10, max(24, iy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Add HUD (fps)
        cv2.putText(canvas, f"{fps:4.1f} FPS", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 30), 2, cv2.LINE_AA)

        # Resize canvas to window, track scale & offsets for click mapping
        H, W2 = canvas.shape[:2]            # H x (2W)
        scale = min(WIN_W / W2, WIN_H / H)  # uniform
        disp_scale = scale
        view = cv2.resize(canvas, (int(W2 * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

        # center in window → compute offsets for left panel
        pad_x = max(0, (WIN_W - view.shape[1]) // 2)
        pad_y = max(0, (WIN_H - view.shape[0]) // 2)
        # Build a fixed-size window image with letterboxing
        framed = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        framed[pad_y:pad_y + view.shape[0], pad_x:pad_x + view.shape[1]] = view

        # For click mapping:
        off_x_left = pad_x  # left panel starts at left edge within view
        off_y      = pad_y
        panel_w    = rectifier.W
        panel_h    = rectifier.H

        cv2.imshow(win, framed)

        # keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    capL.release(); capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
