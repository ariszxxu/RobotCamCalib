#!/usr/bin/env python3
import os
import json
import cv2
import yaml
import numpy as np
import imageio.v2 as imageio

from contaccams.fs_wrapper import FStereoConfig, FoundationStereoWrapper, maybe_save_pointcloud
from contaccams.stereo_rectify import StereoRectifier  # must expose .load_or_build_maps(), .rectify_pair(), .alpha

# ------------------- Defaults (edit if needed) ------------------- #
camera_to_port = {
    "tip_cam":  "3-10:1.0",   # LEFT
    "root_cam": "3-9:1.0",    # RIGHT
}
left_name, right_name = "tip_cam", "root_cam"

out_dir = "outputs"
intr_left_yaml  = f"{out_dir}/{left_name}_intrinsics.yaml"
intr_right_yaml = f"{out_dir}/{right_name}_intrinsics.yaml"
stereo_yaml     = f"{out_dir}/{left_name}_{right_name}_stereo.yaml"
maps_npz        = f"{out_dir}/{left_name}_{right_name}_rectify_maps.npz"  # or None to force rebuild

# FoundationStereo checkpoint (edit to your actual ckpt)
ckpt_path = "/home/ps/projects/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"

# Choose rectification alpha (MUST match what you want to use at runtime)
rectify_alpha = -1.0   # try 0.0, 0.5, 1.0, or -1 (OpenCV chooses)
# --------------------------------------------------------------- #


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _recompute_rectified_Ps(stereo_data: dict, alpha: float):
    """Recompute (R1,R2,P1,P2,Q) with the same alpha used to build maps."""
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
    """
    Return (K_rect, baseline_m) consistent with the rectification alpha.
    K_rect is derived from P1; baseline_m = -P2[0,3]/fx.
    Units follow your calibration (meters if your board square size was meters).
    """
    data = _load_yaml(stereo_yaml_path)
    R1, R2, P1, P2, Q = _recompute_rectified_Ps(data, alpha)
    fx = float(P1[0, 0])
    cx = float(P1[0, 2])
    fy = float(P1[1, 1])
    cy = float(P1[1, 2])
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

    # 1) Build rectifier (loads or rebuilds maps, depending on maps_npz)
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

    # 2) Derive rectified intrinsics and baseline consistent with alpha
    K_rect, baseline = _rectified_K_and_baseline(stereo_yaml, rectifier.alpha)

    # 3) Open cameras, set resolution to calibrated size, grab one synced pair
    capL, capR = _open_caps_sync()
    capL.set(cv2.CAP_PROP_FRAME_WIDTH,  rectifier.W); capL.set(cv2.CAP_PROP_FRAME_HEIGHT, rectifier.H)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH,  rectifier.W); capR.set(cv2.CAP_PROP_FRAME_HEIGHT, rectifier.H)

    # (You can loop here; we take a single pair for a quick test)
    rawL_bgr, rawR_bgr = _grab_pair(capL, capR)

    # 4) RECTIFY → RGB output for the model
    rectL_rgb, rectR_rgb = rectifier.rectify_pair(
        rawL_bgr, rawR_bgr,
        input_color="bgr",
        output_color="rgb",
        resize_to_calib=True
    )

    capL.release(); capR.release()

    # 5) Run FoundationStereo on the rectified pair
    cfg = FStereoConfig(
        ckpt_path=ckpt_path,
        scale=1.0,              # you can downscale here; wrapper will scale K accordingly
        hiera=False,
        valid_iters=32,
        remove_invisible=True,
        z_far=10.0,
        device="cuda",
        amp=True,
    )
    fs = FoundationStereoWrapper(cfg)
    out = fs.infer(
        left_rgb=rectL_rgb,
        right_rgb=rectR_rgb,
        K=K_rect,               # rectified intrinsics (for this size & alpha)
        baseline=baseline / 1000.0,      # meters (or whatever unit your calibration used)
        return_xyz=True,
    )

    # 6) Save a small viz (rectified left + disparity colormap)
    vis_disp = _vis_disparity(out["disp"])  # BGR
    vis_side = np.concatenate([cv2.cvtColor(rectL_rgb, cv2.COLOR_RGB2BGR), vis_disp], axis=1)
    cv2.imwrite(os.path.join(out_dir, "rectified_left_and_disp.png"), vis_side)
    np.save(os.path.join(out_dir, "depth_meter.npy"), out["depth"])
    print("[ok] Saved outputs in", out_dir)

    if "xyz" in out and out["xyz"] is not None:
        maybe_save_pointcloud(
            out_dir,
            xyz=out["xyz"],
            rgb=rectL_rgb,           # use rectified left RGB as colors
            mask=out["valid_mask"],  # mask from FS output
            z_far=cfg.z_far,
            denoise=True,
            nb_points=30,
            radius=0.03
        )


    


if __name__ == "__main__":
    main()
