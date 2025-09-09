import cv2
import os 
import time
import numpy as np
from typing import Tuple, Dict, Optional
from contaccams.cameras import CameraManager, InteractiveCameraManager
from contaccams.intr_calib import SimpleIntrinsicsCalibrator
from contaccams.stereo_calib import SimpleStereoCalibrator  # path as you place it
# ---------- subclass ----------
class InteractiveCameraManagerForCalibration(InteractiveCameraManager):
    def __init__(
        self,
        manager,
        window_name: str = "Calibration Capture",
        window_wh: Tuple[int, int] = (1600, 900),
        checkerboard: Tuple[int, int] = (8, 6),   # (cols, rows) of inner corners
        show_fps: bool = True,
        display_scale: Optional[float] = None,
        undistort_fn=None,
        buffer_max: Optional[int] = None,
        rgb_input: bool = False,
        require_all_detected_on_store: bool = False,
    ):
        super().__init__(
            manager,
            window_name=window_name,
            window_wh=window_wh,
            display_scale=display_scale,
            show_fps=show_fps,
            undistort_fn=undistort_fn,
            buffer_max=buffer_max,
            rgb_input=rgb_input,
        )
        self.cb_cols, self.cb_rows = checkerboard
        self.require_all_detected_on_store = require_all_detected_on_store

    @staticmethod
    def _detect_corners(gray: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
        cols, rows = pattern_size
        if hasattr(cv2, "findChessboardCornersSB"):
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
            if ok:
                corners = corners.reshape(-1, 1, 2).astype(np.float32)
            return ok, corners
        else:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
            if ok:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return ok, corners

    def _overlay_detections(self, rgb: np.ndarray, ok: bool, corners: Optional[np.ndarray], name: str) -> np.ndarray:
        """Draw chessboard corners (green if ok) and a status label."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ok and corners is not None:
            cv2.drawChessboardCorners(bgr, (self.cb_cols, self.cb_rows), corners, ok)
            label = f"{name}: DETECTED"
            color = (40, 220, 40)
        else:
            label = f"{name}: not detected"
            color = (40, 40, 220)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.6, 2
        (tw, th_pix), _ = cv2.getTextSize(label, font, fs, th)

        margin = 10
        x2 = bgr.shape[1] - margin        # right edge
        y1 = margin                       # top edge
        x1 = x2 - (tw + 12)               # box width = text + padding
        y2 = y1 + th_pix + 14

        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(bgr, label, (x1 + 6, y2 - 6), font, fs, color, th, cv2.LINE_AA)

        return bgr

    def run(self):
        """
        Start the UI loop with chessboard overlays.
        Returns dict(name -> stacked buffer ndarray) on exit.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_wh[0], self.window_wh[1])

        while True:
            # read or hold
            if not self.paused:
                frames = self._read_all()  # RGB frames
                for k, f in frames.items():
                    if f is not None:
                        self.last_frames[k] = f
                self._update_fps()

            # build display frames w/ detection overlays
            disp_frames: Dict[str, np.ndarray] = {}
            detect_ok: Dict[str, bool] = {}

            for name in self.names:
                rgb = self.last_frames.get(name)
                if rgb is None:
                    continue
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                ok, corners = self._detect_corners(gray, (self.cb_cols, self.cb_rows))
                bgr_with_overlay = self._overlay_detections(rgb, ok, corners, name)
                # convert back to RGB for tiler (which expects RGB → it converts to BGR internally)
                disp_frames[name] = cv2.cvtColor(bgr_with_overlay, cv2.COLOR_BGR2RGB)
                detect_ok[name] = ok

            # composite
            canvas = self._tile_frames(disp_frames, self.window_wh)

            # optional global scaling (display only)
            if self.display_scale and self.display_scale > 0:
                W, H = self.window_wh
                sw, sh = int(W * self.display_scale), int(H * self.display_scale)
                canvas = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)

            # bottom HUD hint
            if self.show_help:
                hint = "[s] store ALL  [p] pause  [h] help  [c] clear all  [q] quit"
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs, th = 0.6, 2
                (tw, th_pix), _ = cv2.getTextSize(hint, font, fs, th)
                Hc, Wc = canvas.shape[:2]
                x1, y1 = 10, Hc - 10 - th_pix - 10
                x2, y2 = x1 + tw + 12, y1 + th_pix + 14
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), -1)
                cv2.putText(canvas, hint, (x1 + 6, y2 - 6), font, fs, (240, 240, 240), th, cv2.LINE_AA)

            cv2.imshow(self.window_name, canvas)

            # keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('p'):
                self.paused = not self.paused
            elif key == ord('c'):
                self.clear_all_buffers()
            elif key == ord('s'):
                # store ALL cameras; optionally require all detected
                if self.require_all_detected_on_store and not all(detect_ok.get(n, False) for n in self.names):
                    print("[store] skipped: not all cameras have valid detections")
                else:
                    self._store_all()

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyWindow(self.window_name)
        return self.get_all_buffers()


class AutoCameraManagerForCalibration(InteractiveCameraManagerForCalibration):
    def __init__(
        self,
        manager,
        window_name: str = "Auto Calibration Capture",
        window_wh: Tuple[int, int] = (1600, 900),
        checkerboard: Tuple[int, int] = (8, 6),
        show_fps: bool = True,
        display_scale: Optional[float] = None,
        undistort_fn=None,
        buffer_max: Optional[int] = None,
        rgb_input: bool = False,
        sleep_after_store: float = 0.2,
        max_captures: Optional[int] = None,   # stop after N auto-captures; None = unlimited
    ):
        super().__init__(
            manager,
            window_name=window_name,
            window_wh=window_wh,
            checkerboard=checkerboard,
            show_fps=show_fps,
            display_scale=display_scale,
            undistort_fn=undistort_fn,
            buffer_max=buffer_max,
            rgb_input=rgb_input,
            require_all_detected_on_store=False,  # we will store on ANY detected
        )
        self.sleep_after_store = float(sleep_after_store)
        self.max_captures = max_captures
        self.capture_count = 0

    def run(self):
        """
        Auto-capture loop:
          - overlays chessboard detections
          - if ANY camera detects the board, store ALL cameras' frames into buffers
          - sleep self.sleep_after_store after each store
        Returns dict(name -> stacked buffer ndarray) on exit.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_wh[0], self.window_wh[1])

        while True:
            # read or hold
            if not self.paused:
                frames = self._read_all()  # RGB frames
                for k, f in frames.items():
                    if f is not None:
                        self.last_frames[k] = f
                self._update_fps()

            # build display frames w/ detection overlays
            disp_frames: Dict[str, np.ndarray] = {}
            detect_ok: Dict[str, bool] = {}

            for name in self.names:
                rgb = self.last_frames.get(name)
                if rgb is None:
                    continue
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                ok, corners = self._detect_corners(gray, (self.cb_cols, self.cb_rows))
                bgr_with_overlay = self._overlay_detections(rgb, ok, corners, name)
                # convert back to RGB for tiler (tiler expects RGB and converts internally)
                disp_frames[name] = cv2.cvtColor(bgr_with_overlay, cv2.COLOR_BGR2RGB)
                detect_ok[name] = ok

            # Auto-capture if ANY camera has a detection
            if not self.paused and any(detect_ok.values()):
                self._store_all()
                self.capture_count += 1
                # brief visual toast on the canvas later; sleep to avoid duplicates
                time.sleep(self.sleep_after_store)

            # composite
            canvas = self._tile_frames(disp_frames, self.window_wh)

            # optional scale for display
            if self.display_scale and self.display_scale > 0:
                W, H = self.window_wh
                sw, sh = int(W * self.display_scale), int(H * self.display_scale)
                canvas = cv2.resize(canvas, (sw, sh), interpolation=cv2.INTER_AREA)

            # HUD: help + capture count
            hint = "[auto] capturing when ANY detected  |  [p] pause  [h] help  [c] clear  [q] quit"
            cnt  = f"captures: {self.capture_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, th = 0.6, 2
            Hc, Wc = canvas.shape[:2]
            # bottom-left: hint
            (tw, th_pix), _ = cv2.getTextSize(hint, font, fs, th)
            x1, y2 = 10, Hc - 10
            x2, y1 = x1 + tw + 12, y2 - (th_pix + 14)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.putText(canvas, hint, (x1 + 6, y2 - 6), font, fs, (240, 240, 240), th, cv2.LINE_AA)
            # top-right: counter
            (tw2, th2), _ = cv2.getTextSize(cnt, font, fs, th)
            margin = 10
            xr2 = Wc - margin
            yr1 = margin
            xr1 = xr2 - (tw2 + 12)
            yr2 = yr1 + th2 + 14
            cv2.rectangle(canvas, (xr1, yr1), (xr2, yr2), (0, 0, 0), -1)
            cv2.putText(canvas, cnt, (xr1 + 6, yr2 - 6), font, fs, (40, 220, 40), th, cv2.LINE_AA)

            # show (canvas is BGR from tiler)
            cv2.imshow(self.window_name, canvas)

            # stop automatically if target reached
            if self.max_captures is not None and self.capture_count >= self.max_captures:
                print(f"[auto] Reached max_captures={self.max_captures}, stopping.")
                break

            # keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('p'):
                self.paused = not self.paused
            elif key == ord('c'):
                self.clear_all_buffers()
                self.capture_count = 0

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyWindow(self.window_name)
        return self.get_all_buffers()


if __name__ == "__main__":
    camera_to_port = {
        "tip_cam": "3-10:1.0",
        "root_cam": "3-9:1.0",
    }
    left_name, right_name = "tip_cam", "root_cam"  # choose your pair

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    manager = CameraManager(camera_to_port)
    icm = InteractiveCameraManagerForCalibration(manager, window_wh=(1400, 800), show_fps=True)
    # icm = AutoCameraManagerForCalibration(
    #     manager,
    #     window_wh=(1400, 800),
    #     checkerboard=(8, 6),
    #     show_fps=True,
    #     sleep_after_store=0.2,   # as requested
    #     max_captures=None          # optional: stop after 40 snapshots
    # )
    opened_count = manager.open_all_cameras()
    print(f"Successfully opened {opened_count} cameras")

    # 1) Capture
    buffers = icm.run()  # dict: name -> (N,H,W,3) or None
    print("Captured buffers:", {k: (None if v is None else v.shape) for k, v in buffers.items()})

    # 2) Mono intrinsics (per camera)
    intrinsics: dict = {}
    intr_calib = SimpleIntrinsicsCalibrator()
    for name, rgb_frames in buffers.items():
        if rgb_frames is None or len(rgb_frames) == 0:
            print(f"[warn] No frames in buffer for {name}; skipping.")
            continue
        res = intr_calib.calibrate(rgb_frames)
        intrinsics[name] = res
        intr_calib.save_yaml(f"{out_dir}/{name}_intrinsics.yaml")
        print(f"[mono] {name}: rms={res['rms']:.4f}, mean_err={res['mean_err']:.4f}")

    # 3) Stereo calibration using the same buffers (paired by index)
    if left_name in buffers and right_name in buffers:
        L = buffers[left_name]
        R = buffers[right_name]
        if L is None or R is None or len(L) == 0 or len(R) == 0:
            print("[stereo] Missing frames for one side; skip.")
        else:
            stereo = SimpleStereoCalibrator()
            stereo_res = stereo.calibrate(
                left_frames_rgb=L,
                right_frames_rgb=R,
                intr_left=intrinsics[left_name],
                intr_right=intrinsics[right_name],
                alpha=0.0,  # tighter crop for stereo
                flags_stereo=cv2.CALIB_FIX_INTRINSIC
            )
            SimpleStereoCalibrator.save_yaml(f"{out_dir}/{left_name}_{right_name}_stereo.yaml", stereo_res)
            print(f"[stereo] Saved to {out_dir}/{left_name}_{right_name}_stereo.yaml")
            stereo.build_and_attach_maps()
            stereo.save_rectify_maps_npz(f"{out_dir}/{left_name}_{right_name}_rectify_maps.npz")

    manager.release_all()

