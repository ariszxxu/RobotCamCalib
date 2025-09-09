import os
import cv2
import yaml
import numpy as np
from typing import Dict, Tuple, List, Optional
import hashlib
import json
# Reuse your checkerboard config
CHECKERBOARD: Tuple[int, int] = (8, 6)  # (cols, rows) inner corners
SQUARE_SIZE: float = 25.0               # same unit as intrinsics

def _detect_chessboard(gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    cols, rows = CHECKERBOARD
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

def _make_object_points(n_views: int) -> List[np.ndarray]:
    cols, rows = CHECKERBOARD
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= float(SQUARE_SIZE)
    return [objp.copy() for _ in range(n_views)]

def _avg_row_col_dirs(corners: np.ndarray, cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return average row-direction and col-direction vectors (2,) from row-major corners (N,1,2)."""
    pts = corners.reshape(rows, cols, 2)
    # average over rows: vector from first to last in each row
    vrow = (pts[:, -1, :] - pts[:, 0, :]).mean(axis=0)
    # average over cols: vector from first to last in each col
    vcol = (pts[-1, :, :] - pts[0, :, :]).mean(axis=0)
    return vrow, vcol

def _flip_horiz(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Reverse each row (mirror horizontally)."""
    pts = corners.reshape(rows, cols, 2)[:, ::-1, :]
    return pts.reshape(-1, 1, 2)

def _flip_vert(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Reverse row order (mirror vertically)."""
    pts = corners.reshape(rows, cols, 2)[::-1, :, :]
    return pts.reshape(-1, 1, 2)

def _align_orientation(cL: np.ndarray, cR: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """
    Make right corners orientation consistent with left.
    Returns adjusted right corners (or original if already consistent).
    """
    vrowL, vcolL = _avg_row_col_dirs(cL, cols, rows)
    vrowR, vcolR = _avg_row_col_dirs(cR, cols, rows)

    # If row direction is opposite, flip horizontally on right
    if float(vrowL @ vrowR) < 0:
        cR = _flip_horiz(cR, cols, rows)
        vrowR, vcolR = _avg_row_col_dirs(cR, cols, rows)

    # If column direction is opposite, flip vertically on right
    if float(vcolL @ vcolR) < 0:
        cR = _flip_vert(cR, cols, rows)
        # vrowR, vcolR = _avg_row_col_dirs(cR, cols, rows)  # no need to recompute again

    return cR

class SimpleStereoCalibrator:
    """
    Stereo calibrator using pre-calibrated intrinsics.
    Use CALIB_FIX_INTRINSIC to estimate R,T,E,F only, then stereoRectify.

    calibrate(left_frames_rgb, right_frames_rgb, intr_left, intr_right) -> dict
    save_yaml(path, data)
    """
    def __init__(self):
        self.results: Optional[Dict] = None

    @staticmethod
    def _ensure_u8(frames_rgb: np.ndarray) -> np.ndarray:
        if frames_rgb.dtype == np.uint8:
            return frames_rgb
        arr = frames_rgb.astype(np.float32)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        return arr.astype(np.uint8)

    def calibrate(
        self,
        left_frames_rgb: np.ndarray,
        right_frames_rgb: np.ndarray,
        intr_left: Dict,
        intr_right: Dict,
        alpha: float = 0.0,   # 0 for tighter crop (good for stereo matching)
        flags_stereo: int = cv2.CALIB_FIX_INTRINSIC
    ) -> Dict:
        """
        Args:
            left_frames_rgb / right_frames_rgb: (N,H,W,3) RGB, paired by index
            intr_left/intr_right: dicts from SimpleIntrinsicsCalibrator.calibrate()
        Returns:
            dict with K1,D1,K2,D2,R,T,E,F,R1,R2,P1,P2,Q,roi1,roi2,imageSize,rms,mean_err
        """
        assert left_frames_rgb.shape == right_frames_rgb.shape, "Left/right buffers must have same shape"
        n, h, w, _ = left_frames_rgb.shape
        imageSize = (w, h)

        # intr
        K1 = np.array(intr_left["K"], dtype=np.float64)
        D1 = np.array(intr_left["dist"], dtype=np.float64).reshape(-1, 1)
        K2 = np.array(intr_right["K"], dtype=np.float64)
        D2 = np.array(intr_right["dist"], dtype=np.float64).reshape(-1, 1)

        # to uint8
        Lg = self._ensure_u8(left_frames_rgb)
        Rg = self._ensure_u8(right_frames_rgb)

        # detect per-pair, keep only pairs where BOTH succeed
        objpoints: List[np.ndarray] = []
        imgpoints1: List[np.ndarray] = []
        imgpoints2: List[np.ndarray] = []
        used_idx: List[int] = []

        for i in range(n):
            gl = cv2.cvtColor(Lg[i], cv2.COLOR_RGB2GRAY)
            gr = cv2.cvtColor(Rg[i], cv2.COLOR_RGB2GRAY)  # <--- fixed typo: Rg -> R
            ok1, c1 = _detect_chessboard(gl)
            ok2, c2 = _detect_chessboard(gr)
            if not (ok1 and ok2 and c1 is not None and c2 is not None and c1.shape == c2.shape):
                continue

            # Align right orientation to left (handles mirrored ordering)
            c2 = _align_orientation(c1, c2, cols=CHECKERBOARD[0], rows=CHECKERBOARD[1])

            # RANSAC fundamental matrix: use it as a PAIR-QUALITY check only
            pts1 = c1.reshape(-1, 2).astype(np.float32)
            pts2 = c2.reshape(-1, 2).astype(np.float32)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
            if F is None or mask is None or float(mask.mean()) < 0.85:
                # Too many outliers / inconsistent geometry -> drop this pair
                continue

            # Append FULL corner sets (do NOT subset by inliers here)
            imgpoints1.append(c1.astype(np.float32))
            imgpoints2.append(c2.astype(np.float32))
            used_idx.append(i)

        if len(imgpoints1) < 8:
            raise RuntimeError(f"Not enough valid stereo detections: {len(imgpoints1)}; need >= 8")

        objpoints = _make_object_points(len(imgpoints1))

        # stereoCalibrate with fixed intrinsics
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        rms, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            K1, D1, K2, D2, imageSize,
            criteria=criteria, flags=flags_stereo
        )

        # stereoRectify
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, imageSize, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
        )

        self.results = {
            "image_size": [int(imageSize[0]), int(imageSize[1])],
            "used_indices": [int(i) for i in used_idx],
            "K1": K1.tolist(), "D1": D1.reshape(-1).tolist(),
            "K2": K2.tolist(), "D2": D2.reshape(-1).tolist(),
            "R": R.tolist(), "T": T.reshape(-1).tolist(),
            "E": E.tolist(), "F": F.tolist(),
            "R1": R1.tolist(), "R2": R2.tolist(),
            "P1": P1.tolist(), "P2": P2.tolist(),
            "Q": Q.tolist(),
            "roi1": [int(roi1[0]), int(roi1[1]), int(roi1[2]), int(roi1[3])],
            "roi2": [int(roi2[0]), int(roi2[1]), int(roi2[2]), int(roi2[3])],
            "rms": float(rms),
        }
        # quick human-readable summary
        baseline = float(np.linalg.norm(T.reshape(3)))
        print(f"[stereo] RMS={rms:.4f}  | baseline=~{baseline:.3f} (same units as SQUARE_SIZE)")
        return self.results

    @staticmethod
    def save_yaml(path: str, data: Dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def _make_rectify_maps(self, K1, D1, R1, P1, K2, D2, R2, P2, imageSize):
        map1x, map1y = cv2.initUndistortRectifyMap(
            np.asarray(K1, np.float64), np.asarray(D1, np.float64).reshape(-1, 1),
            np.asarray(R1, np.float64), np.asarray(P1, np.float64),
            tuple(imageSize), cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            np.asarray(K2, np.float64), np.asarray(D2, np.float64).reshape(-1, 1),
            np.asarray(R2, np.float64), np.asarray(P2, np.float64),
            tuple(imageSize), cv2.CV_32FC1
        )
        return map1x, map1y, map2x, map2y

    def build_and_attach_maps(self):
        """Build rectify maps and attach them to self.results (must be called after calibrate)."""
        if self.results is None:
            raise RuntimeError("Run calibrate() first.")
        K1, D1 = self.results["K1"], self.results["D1"]
        K2, D2 = self.results["K2"], self.results["D2"]
        R1, P1 = self.results["R1"], self.results["P1"]
        R2, P2 = self.results["R2"], self.results["P2"]
        imageSize = tuple(self.results["image_size"])
        m1x, m1y, m2x, m2y = self._make_rectify_maps(K1, D1, R1, P1, K2, D2, R2, P2, imageSize)
        self.results["map1x"] = m1x
        self.results["map1y"] = m1y
        self.results["map2x"] = m2x
        self.results["map2y"] = m2y
        return m1x, m1y, m2x, m2y

    @staticmethod
    def _fingerprint(meta: dict) -> str:
        """Create a short fingerprint for sanity checks."""
        dump = json.dumps(meta, sort_keys=True).encode("utf-8")
        return hashlib.sha1(dump).hexdigest()[:10]

    def save_rectify_maps_npz(self, path: str):
        """
        Save only the rectify maps + minimal metadata to an NPZ.
        Use np.load(path) at runtime and feed directly to cv2.remap.
        """
        if self.results is None:
            raise RuntimeError("Nothing to save; run calibrate() first.")
        if not all(k in self.results for k in ("map1x", "map1y", "map2x", "map2y")):
            # build maps if not present
            self.build_and_attach_maps()

        meta = {
            "image_size": self.results["image_size"],  # [W, H]
            "K1": self.results["K1"],
            "D1": self.results["D1"],
            "K2": self.results["K2"],
            "D2": self.results["D2"],
            "R1": self.results["R1"],
            "P1": self.results["P1"],
            "R2": self.results["R2"],
            "P2": self.results["P2"],
        }
        fp = self._fingerprint(meta)
        np.savez_compressed(
            path,
            map1x=self.results["map1x"],
            map1y=self.results["map1y"],
            map2x=self.results["map2x"],
            map2y=self.results["map2y"],
            meta=json.dumps(meta),   # store as JSON string
            fingerprint=fp
        )
        print(f"[stereo] Rectify maps saved to {path}  (fp={fp})")
