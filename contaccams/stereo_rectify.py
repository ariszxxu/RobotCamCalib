#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import cv2
import yaml
import numpy as np
from typing import Tuple, Dict, Optional

def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _overlay_epilines(img_bgr: np.ndarray, step: int = 40,
                      color=(0, 255, 255), thickness: int = 1) -> np.ndarray:
    """Draw evenly spaced horizontal lines to visually check epipolar alignment."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for y in range(step, h, step):
        cv2.line(out, (0, y), (w, y), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(out, (0, h // 2), (w, h // 2), (0, 180, 255), max(2, thickness), cv2.LINE_AA)
    return out


def _tile_2x2(a, b, c, d, pad=6, bg=(24, 24, 24)):
    """Stack 4 BGR images of the same size into a 2x2 canvas."""
    h, w = a.shape[:2]
    H = h * 2 + pad * 3
    W = w * 2 + pad * 3
    canvas = np.full((H, W, 3), bg, np.uint8)

    canvas[pad:pad + h, pad:pad + w] = a
    canvas[pad:pad + h, 2 * pad + w:2 * pad + 2 * w] = b
    canvas[2 * pad + h:2 * pad + 2 * h, pad:pad + w] = c
    canvas[2 * pad + h:2 * pad + 2 * h, 2 * pad + w:2 * pad + 2 * w] = d

    cv2.putText(canvas, "RAW L",  (pad + 10, pad + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RAW R",  (2 * pad + w + 10, pad + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RECT L", (pad + 10, 2 * pad + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RECT R", (2 * pad + w + 10, 2 * pad + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2, cv2.LINE_AA)
    return canvas


class StereoRectifier:
    """
    Live stereo rectification viewer.

    - Loads intrinsics/stereo results from YAML.
    - Optionally loads precomputed rectify maps from NPZ.
    - Or rebuilds maps using cv2.stereoRectify with chosen alpha (default -1).
    - Opens two cameras by USB port strings (via pyudev).
    - Displays RAW (top) and RECTIFIED (bottom) with epipolar guides.
    """

    def __init__(
        self,
        camera_to_port: Dict[str, str],
        left_name: str,
        right_name: str,
        stereo_yaml: str,
        maps_npz: Optional[str] = None,
        alpha: float = -1.0,  # -1 lets OpenCV choose; try 0.0 (tighter) .. 1.0 (wider)
        use_npz_if_available: bool = True,
        window_title: str = "Stereo Before/After (q: quit, p: pause, e: epilines)",
        window_size: Tuple[int, int] = (1600, 900),
    ):
        self.camera_to_port = camera_to_port
        self.left_name = left_name
        self.right_name = right_name
        self.stereo_yaml = stereo_yaml
        self.maps_npz = maps_npz
        self.alpha = alpha
        self.use_npz_if_available = use_npz_if_available

        self.window_title = window_title
        self.window_size = window_size

        # Will be filled by load_or_build_maps()
        self.map1x = self.map1y = self.map2x = self.map2y = None
        self.W = self.H = None

        # caps
        self.capL = None
        self.capR = None

    # ------------------- Maps / calibration ------------------- #
    @staticmethod
    def _build_rectify_maps_from_yaml(stereo_data: Dict, alpha: float):
        W, H = int(stereo_data["image_size"][0]), int(stereo_data["image_size"][1])
        imageSize = (W, H)

        K1 = np.asarray(stereo_data["K1"], np.float64)
        D1 = np.asarray(stereo_data["D1"], np.float64).reshape(-1, 1)
        K2 = np.asarray(stereo_data["K2"], np.float64)
        D2 = np.asarray(stereo_data["D2"], np.float64).reshape(-1, 1)

        R = np.asarray(stereo_data["R"], np.float64)
        T = np.asarray(stereo_data["T"], np.float64).reshape(3,)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
        )

        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv2.CV_32FC1)
        return map1x, map1y, map2x, map2y, (W, H)

    @staticmethod
    def _npz_meta(npz) -> Optional[dict]:
        if "meta" not in npz.files:
            return None
        raw = npz["meta"]
        try:
            s = raw.item() if hasattr(raw, "item") else str(raw)
            return json.loads(s)
        except Exception:
            try:
                return json.loads(raw.tobytes().decode("utf-8", "ignore"))
            except Exception:
                return None

    def load_or_build_maps(self):
        """Load rectify maps from NPZ (if allowed and present), else build from YAML with alpha."""
        if self.use_npz_if_available and self.maps_npz and os.path.isfile(self.maps_npz):
            z = np.load(self.maps_npz, allow_pickle=False)
            self.map1x, self.map1y = z["map1x"], z["map1y"]
            self.map2x, self.map2y = z["map2x"], z["map2y"]

            meta = self._npz_meta(z)
            if meta and "image_size" in meta:
                self.W, self.H = int(meta["image_size"][0]), int(meta["image_size"][1])
            else:
                stereo_data = _load_yaml(self.stereo_yaml)
                self.W, self.H = int(stereo_data["image_size"][0]), int(stereo_data["image_size"][1])
            print(f"[info] Loaded rectify maps from NPZ for image size {(self.W, self.H)}")
            return

        # else: build from YAML (and chosen alpha)
        stereo_data = _load_yaml(self.stereo_yaml)
        m1x, m1y, m2x, m2y, (W, H) = self._build_rectify_maps_from_yaml(stereo_data, self.alpha)
        self.map1x, self.map1y, self.map2x, self.map2y = m1x, m1y, m2x, m2y
        self.W, self.H = W, H
        print(f"[info] Built rectify maps from YAML with alpha={self.alpha} for image size {(W, H)}")

    # ------------------- Camera IO ------------------- #
    def _open_caps(self) -> Tuple[cv2.VideoCapture, cv2.VideoCapture]:
        import pyudev
        ctx = pyudev.Context()

        def dev_for_port(port_str: str) -> str:
            for dev in ctx.list_devices(subsystem='video4linux'):
                if dev.parent and dev.parent.subsystem == 'usb':
                    usb_port = dev.parent.get('DEVPATH', '').split('/')[-1]
                    if port_str in usb_port:
                        return dev.device_node
            raise RuntimeError(f"USB port not found: {port_str}")

        left_dev = dev_for_port(self.camera_to_port[self.left_name])
        right_dev = dev_for_port(self.camera_to_port[self.right_name])

        capL = cv2.VideoCapture(left_dev, cv2.CAP_V4L2)
        capR = cv2.VideoCapture(right_dev, cv2.CAP_V4L2)
        if not capL.isOpened() or not capR.isOpened():
            raise RuntimeError("Failed to open one or both cameras.")
        return capL, capR

    @staticmethod
    def _grab_retrieve_pair(capL: cv2.VideoCapture, capR: cv2.VideoCapture) -> Tuple[np.ndarray, np.ndarray]:
        """Two-phase read to roughly sync both cameras."""
        capL.grab(); capR.grab()
        okL, frameL = capL.retrieve()
        okR, frameR = capR.retrieve()
        if not okL or not okR:
            raise RuntimeError("Failed to read from cameras.")
        return frameL, frameR  # BGR

    # ------------------- Run loop ------------------- #
    def run(self):
        if self.map1x is None:
            self.load_or_build_maps()

        self.capL, self.capR = self._open_caps()
        # Try to enforce calibrated resolution
        self.capL.set(cv2.CAP_PROP_FRAME_WIDTH,  self.W)
        self.capL.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.capR.set(cv2.CAP_PROP_FRAME_WIDTH,  self.W)
        self.capR.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)

        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, *self.window_size)

        paused = False
        show_epi = True
        lastL = lastR = lastLr = lastRr = None

        while True:
            if not paused or lastL is None:
                rawL_bgr, rawR_bgr = self._grab_retrieve_pair(self.capL, self.capR)

                # Resize to calibration size if mismatch
                if (rawL_bgr.shape[1], rawL_bgr.shape[0]) != (self.W, self.H):
                    rawL_bgr = cv2.resize(rawL_bgr, (self.W, self.H), interpolation=cv2.INTER_AREA)
                if (rawR_bgr.shape[1], rawR_bgr.shape[0]) != (self.W, self.H):
                    rawR_bgr = cv2.resize(rawR_bgr, (self.W, self.H), interpolation=cv2.INTER_AREA)

                # Rectify
                rectL_bgr = cv2.remap(rawL_bgr, self.map1x, self.map1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                rectR_bgr = cv2.remap(rawR_bgr, self.map2x, self.map2y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                lastL, lastR, lastLr, lastRr = rawL_bgr, rawR_bgr, rectL_bgr, rectR_bgr

            # Optional epipolar guides
            dispL  = _overlay_epilines(lastL)  if show_epi else lastL
            dispR  = _overlay_epilines(lastR)  if show_epi else lastR
            dispLr = _overlay_epilines(lastLr) if show_epi else lastLr
            dispRr = _overlay_epilines(lastRr) if show_epi else lastRr

            canvas = _tile_2x2(dispL, dispR, dispLr, dispRr)
            cv2.imshow(self.window_title, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('e'):
                show_epi = not show_epi

            if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.capL.release(); self.capR.release()
        cv2.destroyAllWindows()

    def rectify_pair(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        *,
        input_color: str = "bgr",     # "bgr" or "rgb"
        output_color: str = "rgb",    # "bgr" or "rgb" (use "rgb" for your model)
        resize_to_calib: bool = True,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify one image pair using preloaded maps.

        Args:
            left_img, right_img: (H,W,3) images
            input_color: color space of inputs ("bgr" or "rgb")
            output_color: color space to return ("bgr" or "rgb")
            resize_to_calib: if True, resize inputs to calibrated (W,H) when mismatched
            interpolation, border_mode: passed to cv2.remap

        Returns:
            rect_left, rect_right: rectified images in output_color
        """
        if self.map1x is None or self.map1y is None:
            # Build or load maps if not ready
            self.load_or_build_maps()

        assert left_img.ndim == 3 and right_img.ndim == 3 and left_img.shape[2] == 3 and right_img.shape[2] == 3, \
            "Inputs must be (H,W,3)"

        # Convert input color to BGR for OpenCV remap
        if input_color.lower() == "rgb":
            L_bgr = cv2.cvtColor(left_img,  cv2.COLOR_RGB2BGR)
            R_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)
        else:
            L_bgr, R_bgr = left_img, right_img

        # Ensure size matches calibration
        if (L_bgr.shape[1], L_bgr.shape[0]) != (self.W, self.H) or (R_bgr.shape[1], R_bgr.shape[0]) != (self.W, self.H):
            if not resize_to_calib:
                raise ValueError(
                    f"Input sizes {(L_bgr.shape[1], L_bgr.shape[0])} and {(R_bgr.shape[1], R_bgr.shape[0])} "
                    f"do not match calibrated size {(self.W, self.H)} and resize_to_calib=False."
                )
            L_bgr = cv2.resize(L_bgr, (self.W, self.H), interpolation=cv2.INTER_AREA)
            R_bgr = cv2.resize(R_bgr, (self.W, self.H), interpolation=cv2.INTER_AREA)

        # Rectify
        rectL_bgr = cv2.remap(L_bgr, self.map1x, self.map1y, interpolation, borderMode=border_mode)
        rectR_bgr = cv2.remap(R_bgr, self.map2x, self.map2y, interpolation, borderMode=border_mode)

        # Convert to requested output color
        if output_color.lower() == "rgb":
            rectL = cv2.cvtColor(rectL_bgr, cv2.COLOR_BGR2RGB)
            rectR = cv2.cvtColor(rectR_bgr, cv2.COLOR_BGR2RGB)
        else:
            rectL, rectR = rectL_bgr, rectR_bgr

        return rectL, rectR
    


if __name__ == "__main__":

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
    maps_npz        = f"{out_dir}/{left_name}_{right_name}_rectify_maps.npz"
    # --------------------------------------------------------------- #

    # ------------------- CLI entry ------------------- #
    def main():
        viewer = StereoRectifier(
            camera_to_port=camera_to_port,
            left_name=left_name,
            right_name=right_name,
            stereo_yaml=stereo_yaml,
            maps_npz=maps_npz,          # set to None to always rebuild from YAML
            alpha=-1.0,                 # try 0.0 (crop/less zoom), 0.5 (middle), 1.0 (full FOV), or -1 (OpenCV picks)
            use_npz_if_available=True,
            window_title="Stereo Before/After (q: quit, p: pause, e: epilines)",
            window_size=(1600, 900),
        )
        viewer.run()

    main()
