import numpy as np
import cv2
import yaml
from typing import Dict, Tuple, List, Optional

# Fixed checkerboard you print from PDF (edit if needed)
CHECKERBOARD: Tuple[int, int] = (8, 11)  # (cols, rows) of inner corners
SQUARE_SIZE: float = 12.0                 # square length in your chosen unit
MIN_SAMPLES: int = 12                     # minimum valid detections recommended

class SimpleIntrinsicsCalibrator:
    """
    Minimal, camera-agnostic intrinsics calibrator:
      - calibrate(frames_rgb): input (n, h, w, 3) RGB array; returns K, dist, etc.
      - undistort(image_rgb): undistort a single RGB image using the result
      - save_yaml(path): save intrinsics to YAML (essential fields only)
    """
    def __init__(self):
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.rms: Optional[float] = None
        self.mean_reproj_error: Optional[float] = None
        self.image_size: Optional[Tuple[int, int]] = None  # (w, h)

    @staticmethod
    def _detect_chessboard(gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        cols, rows = CHECKERBOARD
        # Prefer the SB detector when available (OpenCV >= 4.5); fallback otherwise
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

    @staticmethod
    def _make_object_points(n_views: int) -> List[np.ndarray]:
        cols, rows = CHECKERBOARD
        objp = np.zeros((rows * cols, 3), np.float32)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp[:, :2] = grid
        objp *= float(SQUARE_SIZE)
        return [objp.copy() for _ in range(n_views)]

    def _compute_mean_reproj_error(self, K, dist, rvecs, tvecs, objpoints, imgpoints) -> float:
        total_err = 0.0
        total_pts = 0
        for i in range(len(objpoints)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
            total_err += err * err
            total_pts += len(objpoints[i])
        return float(np.sqrt(total_err / max(total_pts, 1)))

    def calibrate(self, frames_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args:
            frames_rgb: (n, h, w, 3) RGB array (uint8, or float in [0,1]/[0,255])
        Returns:
            dict with keys: K, dist, rvecs, tvecs, rms, mean_err, used_indices, image_size, fx, fy, cx, cy
        """
        assert frames_rgb.ndim == 4 and frames_rgb.shape[-1] == 3, \
            "frames_rgb must be (n, h, w, 3) RGB array"
        n, h, w, _ = frames_rgb.shape
        self.image_size = (w, h)

        # Normalize to uint8 for OpenCV
        if frames_rgb.dtype != np.uint8:
            arr = frames_rgb.astype(np.float32)
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            frames_u8 = arr.astype(np.uint8)
        else:
            frames_u8 = frames_rgb

        imgpoints: List[np.ndarray] = []
        used_idx: List[int] = []

        # Detect chessboard corners per frame
        for i in range(n):
            gray = cv2.cvtColor(frames_u8[i], cv2.COLOR_RGB2GRAY)
            ok, corners = self._detect_chessboard(gray)
            if ok:
                imgpoints.append(corners)
                used_idx.append(i)

        if len(imgpoints) < MIN_SAMPLES:
            raise RuntimeError(f"Not enough valid chessboard samples: {len(imgpoints)}; need >= {MIN_SAMPLES}")

        objpoints = self._make_object_points(len(imgpoints))

        # Pinhole + standard distortion calibration
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.image_size, None, None
        )
        if not ret:
            raise RuntimeError("cv2.calibrateCamera failed; check samples/visibility.")

        mean_err = self._compute_mean_reproj_error(K, dist, rvecs, tvecs, objpoints, imgpoints)

        # Store results
        self.K, self.dist = K, dist
        self.rvecs, self.tvecs = rvecs, tvecs
        self.rms = float(ret)
        self.mean_reproj_error = float(mean_err)

        return {
            "K": K, "dist": dist, "rvecs": rvecs, "tvecs": tvecs,
            "rms": float(ret), "mean_err": float(mean_err),
            "used_indices": np.array(used_idx, dtype=int),
            "image_size": (w, h),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        }

    def undistort(self, image_rgb: np.ndarray) -> np.ndarray:
        """Undistort a single RGB image using the current intrinsics."""
        if self.K is None or self.dist is None or self.image_size is None:
            raise RuntimeError("Run calibrate() first.")
        newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, self.image_size, 0)
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        und_bgr = cv2.undistort(bgr, self.K, self.dist, None, newK)
        return cv2.cvtColor(und_bgr, cv2.COLOR_BGR2RGB)

    def save_yaml(self, path: str = "intrinsics.yaml") -> None:
        """Save essential intrinsics to YAML (no extra fields)."""
        if self.K is None or self.dist is None or self.image_size is None:
            raise RuntimeError("Nothing to save; run calibrate() first.")
        data = {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "K": self.K.tolist(),
            "dist": self.dist.reshape(-1).tolist(),
            "fx": float(self.K[0, 0]),
            "fy": float(self.K[1, 1]),
            "cx": float(self.K[0, 2]),
            "cy": float(self.K[1, 2]),
            "rms": float(self.rms) if self.rms is not None else None,
            "mean_reproj_error": float(self.mean_reproj_error) if self.mean_reproj_error is not None else None,
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
