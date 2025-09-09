#!/usr/bin/env python3
import os
import json
import cv2
import yaml
import numpy as np
from typing import Tuple, Dict, Optional

# ---- Your pair (edit if needed) ----
camera_to_port = {
    "tip_cam":  "3-10:1.0",   # LEFT
    "root_cam": "3-9:1.0",    # RIGHT
}
left_name, right_name = "tip_cam", "root_cam"

# ---- Paths to calibration outputs ----
out_dir = "outputs"
intr_left_yaml  = f"{out_dir}/{left_name}_intrinsics.yaml"
intr_right_yaml = f"{out_dir}/{right_name}_intrinsics.yaml"
stereo_yaml     = f"{out_dir}/{left_name}_{right_name}_stereo.yaml"
maps_npz        = f"{out_dir}/{left_name}_{right_name}_rectify_maps.npz"

# ======================================================================

def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_intrinsics(intr_yaml_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Return (K, D, (W,H))"""
    data = load_yaml(intr_yaml_path)
    K = np.asarray(data["K"], dtype=np.float64)
    D = np.asarray(data["dist"], dtype=np.float64).reshape(-1, 1)
    W, H = int(data["image_size"][0]), int(data["image_size"][1])
    return K, D, (W, H)

def build_rectify_maps_from_yaml(stereo_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int,int]]:
    W, H = int(stereo_data["image_size"][0]), int(stereo_data["image_size"][1])
    imageSize = (W, H)
    K1 = np.asarray(stereo_data["K1"], np.float64)
    D1 = np.asarray(stereo_data["D1"], np.float64).reshape(-1, 1)
    K2 = np.asarray(stereo_data["K2"], np.float64)
    D2 = np.asarray(stereo_data["D2"], np.float64).reshape(-1, 1)
    R1 = np.asarray(stereo_data["R1"], np.float64)
    P1 = np.asarray(stereo_data["P1"], np.float64)
    R2 = np.asarray(stereo_data["R2"], np.float64)
    P2 = np.asarray(stereo_data["P2"], np.float64)

    R  = np.asarray(stereo_data["R"],  np.float64)
    T  = np.asarray(stereo_data["T"],  np.float64).reshape(3,)

    # re-rectify with desired alpha
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, imageSize

def load_or_build_maps() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int,int]]:
    # if os.path.isfile(maps_npz):
    #     z = np.load(maps_npz, allow_pickle=False)

    #     map1x, map1y = z["map1x"], z["map1y"]
    #     map2x, map2y = z["map2x"], z["map2y"]

    #     # ---- robust meta loader ----
    #     def _load_meta_from_npz(npz) -> Optional[dict]:
    #         if "meta" not in npz.files:
    #             return None
    #         raw = npz["meta"]
    #         # meta was saved as a JSON string; NPZ stores it as a 0-D ndarray
    #         try:
    #             # common case: 0-D <U... array
    #             meta_str = raw.item() if hasattr(raw, "item") else str(raw)
    #             return json.loads(meta_str)
    #         except Exception:
    #             # rare fallback: try decoding bytes
    #             try:
    #                 meta_str = raw.tobytes().decode("utf-8", "ignore")
    #                 return json.loads(meta_str)
    #             except Exception:
    #                 return None

    #     meta = _load_meta_from_npz(z)
    #     if meta is not None and "image_size" in meta:
    #         W, H = int(meta["image_size"][0]), int(meta["image_size"][1])
    #     else:
    #         # fallback to stereo yaml if meta missing/bad
    #         stereo_data = load_yaml(stereo_yaml)
    #         W, H = int(stereo_data["image_size"][0]), int(stereo_data["image_size"][1])

    #     return map1x, map1y, map2x, map2y, (W, H)

    # else: build maps from stereo yaml
    stereo_data = load_yaml(stereo_yaml)
    return build_rectify_maps_from_yaml(stereo_data)


def open_caps() -> Tuple[cv2.VideoCapture, cv2.VideoCapture]:
    # Resolve /dev/video* via your USB port strings using udev (quick inline)
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

def grab_retrieve_pair(capL: cv2.VideoCapture, capR: cv2.VideoCapture) -> Tuple[np.ndarray, np.ndarray]:
    # Two-phase to roughly sync
    capL.grab(); capR.grab()
    okL, frameL = capL.retrieve()
    okR, frameR = capR.retrieve()
    if not okL or not okR:
        raise RuntimeError("Failed to read from cameras.")
    return frameL, frameR  # BGR

def tile_2x2(a, b, c, d, pad=6, bg=(24,24,24)):
    """Stack 4 images (BGR) into a 2x2 canvas with same sizes."""
    h, w = a.shape[:2]
    H = h*2 + pad*3
    W = w*2 + pad*3
    canvas = np.full((H, W, 3), bg, np.uint8)
    # positions
    canvas[pad:pad+h, pad:pad+w] = a
    canvas[pad:pad+h, 2*pad+w:2*pad+2*w] = b
    canvas[2*pad+h:2*pad+2*h, pad:pad+w] = c
    canvas[2*pad+h:2*pad+2*h, 2*pad+w:2*pad+2*w] = d
    # small labels
    cv2.putText(canvas, "RAW L",   (pad+10, pad+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RAW R",   (2*pad+w+10, pad+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RECT L",  (pad+10, 2*pad+h+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,220,30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RECT R",  (2*pad+w+10, 2*pad+h+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,220,30), 2, cv2.LINE_AA)
    return canvas

def overlay_epilines(img_bgr: np.ndarray, step: int = 40, color=(0, 255, 255), thickness: int = 1) -> np.ndarray:
    """
    Draw evenly spaced horizontal lines across the image to visually check epipolar alignment.
    For rectified pairs, corresponding content should align to the same y (scanline) on L/R.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    y = step
    while y < h:
        cv2.line(out, (0, y), (w, y), color, thickness, lineType=cv2.LINE_AA)
        y += step
    # optional: draw a thicker center line
    cv2.line(out, (0, h // 2), (w, h // 2), (0, 180, 255), max(2, thickness), lineType=cv2.LINE_AA)
    return out

def main():
    # Load calibration (maps preferred)
    map1x, map1y, map2x, map2y, (W, H) = load_or_build_maps()
    print(f"[info] Rectify maps ready for image size: {(W, H)}")

    # Optional: enforce camera resolution to calibration size (if supported by driver)
    capL, capR = open_caps()
    # Set only if your drivers accept it; otherwise comment out:
    capL.set(cv2.CAP_PROP_FRAME_WIDTH,  W); capL.set(cv2.CAP_PROP_FRAME_HEIGHT,  H)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH,  W); capR.set(cv2.CAP_PROP_FRAME_HEIGHT,  H)


    win = "Stereo Before/After (q: quit, p: pause, e: epilines)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 900)

    paused = False
    show_epi = True  # start with epipolar guides ON
    lastL = lastR = lastLr = lastRr = None

    while True:
        if not paused or lastL is None:
            rawL_bgr, rawR_bgr = grab_retrieve_pair(capL, capR)

            # Resize to calibration size if mismatch
            if (rawL_bgr.shape[1], rawL_bgr.shape[0]) != (W, H):
                rawL_bgr = cv2.resize(rawL_bgr, (W, H), interpolation=cv2.INTER_AREA)
            if (rawR_bgr.shape[1], rawR_bgr.shape[0]) != (W, H):
                rawR_bgr = cv2.resize(rawR_bgr, (W, H), interpolation=cv2.INTER_AREA)

            # Rectify
            rectL_bgr = cv2.remap(rawL_bgr, map1x, map1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            rectR_bgr = cv2.remap(rawR_bgr, map2x, map2y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            lastL, lastR, lastLr, lastRr = rawL_bgr, rawR_bgr, rectL_bgr, rectR_bgr

        # --- draw epipolar guides (same y-positions on all four) ---
        dispL  = overlay_epilines(lastL)  if show_epi else lastL
        dispR  = overlay_epilines(lastR)  if show_epi else lastR
        dispLr = overlay_epilines(lastLr) if show_epi else lastLr
        dispRr = overlay_epilines(lastRr) if show_epi else lastRr

        # tile and show
        canvas = tile_2x2(dispL, dispR, dispLr, dispRr)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('e'):
            show_epi = not show_epi  # toggle epipolar lines

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    capL.release(); capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
