# contaccams/foundation_stereo_wrapper.py
import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import os,sys
sys.path.append("/home/ps/projects/FoundationStereo")
import numpy as np
import cv2
import torch
from omegaconf import OmegaConf

#!/usr/bin/env python3
import os, argparse, logging
import numpy as np
import cv2
import imageio.v2 as imageio
# Project deps (same as your demo)
from core.utils.utils import InputPadder
from core.foundation_stereo import FoundationStereo


@dataclass
class FStereoConfig:
    ckpt_path: str                                  # path to model_best_*.pth
    scale: float = 1.0                              # <= 1.0
    hiera: bool = False                             # hierarchical inference
    valid_iters: int = 32                           # network iterations
    remove_invisible: bool = True                   # mask non-overlap
    z_far: Optional[float] = None                   # clip Z in meters (None = no z clip)
    device: str = "cuda"                            # "cuda" or "cpu"
    amp: bool = True                                # autocast mixed precision


class FoundationStereoWrapper:
    """
    Thin wrapper around NVIDIA FoundationStereo demo that operates on in-memory arrays.
    Returns arrays (disp/depth/xyz/masks) without touching disk.
    """

    def __init__(self, cfg: FStereoConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")

        # Load OmegaConf cfg.yaml next to the checkpoint (to match model hyperparams)
        ckpt_dir = os.path.dirname(cfg.ckpt_path)
        yaml_path = os.path.join(ckpt_dir, "cfg.yaml")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"cfg.yaml not found next to checkpoint: {yaml_path}")

        base = OmegaConf.load(yaml_path)
        # Carry over expected keys from CLI demo
        if "vit_size" not in base:
            base["vit_size"] = "vitl"
        base["valid_iters"] = cfg.valid_iters
        base["hiera"] = int(cfg.hiera)
        base["scale"] = float(cfg.scale)

        # Build OmegaConf args for FoundationStereo
        self.args = OmegaConf.create(base)
        logging.info(f"[FStereoWrapper] Using checkpoint: {cfg.ckpt_path}")

        # Model
        self.model = FoundationStereo(self.args)
        ckpt = torch.load(cfg.ckpt_path, map_location=self.device)
        logging.info(f"[FStereoWrapper] ckpt step={ckpt.get('global_step')}, epoch={ckpt.get('epoch')}")
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        # Small helper to keep autocast off on CPU
        self._amp_enabled = cfg.amp and self.device.type == "cuda"

    @staticmethod
    def _ensure_rgb_u8(img: np.ndarray) -> np.ndarray:
        """(H,W,3) RGB uint8."""
        assert img.ndim == 3 and img.shape[2] == 3, f"Expected (H,W,3), got {img.shape}"
        if img.dtype == np.uint8:
            return img
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        return arr.astype(np.uint8)

    @staticmethod
    def _resize_if_needed(img: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return img
        assert 0 < scale <= 1.0, "scale must be in (0,1]"
        return cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _build_depth_from_disp(disp: np.ndarray, K_scaled: np.ndarray, baseline: float) -> np.ndarray:
        """Depth Z = fx * B / disp. Returns float32 with NaN where disp <= 0."""
        fx = float(K_scaled[0, 0])
        depth = np.empty_like(disp, dtype=np.float32)
        depth[:] = np.nan
        valid = disp > 0
        depth[valid] = fx * float(baseline) / disp[valid]
        return depth

    @staticmethod
    def _depth_to_xyz(depth: np.ndarray, K_scaled: np.ndarray) -> np.ndarray:
        """Back-project to XYZ (H,W,3) from depth and intrinsics."""
        h, w = depth.shape
        fx, fy = float(K_scaled[0, 0]), float(K_scaled[1, 1])
        cx, cy = float(K_scaled[0, 2]), float(K_scaled[1, 2])
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        Z = depth
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy
        xyz = np.dstack([X, Y, Z]).astype(np.float32)
        return xyz

    @torch.inference_mode()
    def infer(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        K: np.ndarray,
        baseline: float,
        *,
        remove_invisible: Optional[bool] = None,
        z_far: Optional[float] = None,
        return_xyz: bool = True,
    ) -> Dict[str, Any]:
        """
        Args:
            left_rgb, right_rgb: (H,W,3) RGB arrays (uint8 or float)
            K: (3,3) float64/float32 camera intrinsics for the rectified inputs' size
               (same as raw if you don't resize; we will scale K if cfg.scale != 1)
            baseline: float (meters or your desired unit)
            remove_invisible: override cfg.remove_invisible (mask non-overlap)
            z_far: override cfg.z_far; keep points with 0<Z<=z_far
            return_xyz: also compute dense xyz map

        Returns:
            {
              "disp": (H,W) float32 pixels,
              "depth": (H,W) float32 (same unit as baseline),
              "xyz": (H,W,3) float32 (optional; present if return_xyz=True),
              "valid_mask": (H,W) bool  (disp>0 and finite depth and z clip applied),
              "scale": float,
              "K_scaled": (3,3) float64 intrinsics after scale,
            }
        """
        cfg = self.cfg
        if remove_invisible is None:
            remove_invisible = cfg.remove_invisible
        if z_far is None:
            z_far = cfg.z_far

        # 1) Prep images (RGB uint8) and optional downscale
        L = self._ensure_rgb_u8(left_rgb)
        R = self._ensure_rgb_u8(right_rgb)
        L = self._resize_if_needed(L, cfg.scale)
        R = self._resize_if_needed(R, cfg.scale)
        assert L.shape == R.shape, "Left/Right sizes differ after scaling"

        H, W = L.shape[:2]
        L_t = torch.as_tensor(L, device=self.device).float()[None].permute(0, 3, 1, 2)  # 1x3xHxW
        R_t = torch.as_tensor(R, device=self.device).float()[None].permute(0, 3, 1, 2)

        # 2) Pad to network-friendly size
        padder = InputPadder(L_t.shape, divis_by=32, force_square=False)
        L_t, R_t = padder.pad(L_t, R_t)

        # 3) Forward
        autocast_ctx = torch.cuda.amp.autocast(enabled=self._amp_enabled)
        with autocast_ctx:
            if not cfg.hiera:
                disp_t = self.model.forward(L_t, R_t, iters=cfg.valid_iters, test_mode=True)
            else:
                disp_t = self.model.run_hierachical(L_t, R_t, iters=cfg.valid_iters, test_mode=True, small_ratio=0.5)

        disp_t = padder.unpad(disp_t.float())             # 1x1xH x W (assumed)
        disp = disp_t.detach().cpu().numpy().reshape(H, W).astype(np.float32)

        # 4) Optional non-overlap removal (same as your demo)
        if remove_invisible:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            us_right = xx.astype(np.float32) - disp
            invalid = us_right < 0
            disp[invalid] = np.inf  # mark invalid before depth

        # 5) Build scaled intrinsics (if scale < 1, scale fx, fy, cx, cy)
        K = np.asarray(K, dtype=np.float64).copy()
        K_scaled = K.copy()
        K_scaled[0, :2] *= cfg.scale  # (fx, cx)
        K_scaled[1, :2] *= cfg.scale  # (fy, cy)

        # 6) Depth & XYZ
        depth = self._build_depth_from_disp(disp, K_scaled, baseline)   # NaN where invalid
        valid_mask = np.isfinite(depth) & (depth > 0)
        if z_far is not None:
            valid_mask &= (depth <= float(z_far))

        result: Dict[str, Any] = {
            "disp": disp,
            "depth": depth,
            "valid_mask": valid_mask,
            "scale": float(cfg.scale),
            "K_scaled": K_scaled,
        }

        if return_xyz:
            xyz = self._depth_to_xyz(depth, K_scaled)
            # mask out invalid XYZ for sanity (keep NaN in depth)
            xyz[~valid_mask] = np.nan
            result["xyz"] = xyz

        return result


def set_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def vis_disparity(disp: np.ndarray) -> np.ndarray:
    """Simple disparity visualization (uint8 BGR)."""
    d = disp.copy()
    # mask inf/NaN and negatives
    bad = ~np.isfinite(d) | (d <= 0)
    if np.all(bad):
        return np.zeros((*d.shape, 3), np.uint8)
    d[bad] = 0
    vmax = np.percentile(d[~bad], 99.0)
    vmax = float(max(vmax, 1e-6))
    d = np.clip(d / vmax, 0, 1)
    d8 = (d * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_PLASMA)

def maybe_save_pointcloud(out_dir: str, xyz: np.ndarray, rgb: np.ndarray, mask: np.ndarray,
                          z_far: float | None, denoise: bool, nb_points: int, radius: float):
    try:
        import open3d as o3d
    except Exception:
        logging.warning("Open3D not available; skipping PLY export.")
        return
    pts = xyz[mask].reshape(-1, 3)
    cols = (rgb[mask].reshape(-1, 3).astype(np.float32) / 255.0)
    if pts.size == 0:
        logging.warning("No valid 3D points to save.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    ply_path = os.path.join(out_dir, "cloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    logging.info(f"Point cloud saved: {ply_path}")
    if denoise:
        logging.info("[Optional] Denoising point cloud...")
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        inlier = pcd.select_by_index(ind)
        ply2 = os.path.join(out_dir, "cloud_denoise.ply")
        o3d.io.write_point_cloud(ply2, inlier)
        logging.info(f"Denoised point cloud saved: {ply2}")
        pcd = inlier

    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()
    
def main():
    set_logging()
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    # --- keep the same args & defaults as your demo ---
    parser.add_argument('--left_file',   default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument('--right_file',  default=f'{code_dir}/../assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str,
                        help='camera intrinsic matrix and baseline file: first line 3x3 K (row-major), second line baseline')
    parser.add_argument('--ckpt_dir',    default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str,
                        help='pretrained model path (*.pth)')
    parser.add_argument('--out_dir',     default=f'{code_dir}/../output/', type=str)
    parser.add_argument('--scale',       default=1.0, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera',       default=0, type=int, help='hierarchical inference (for >1K images)')
    parser.add_argument('--z_far',       default=10.0, type=float, help='max depth to keep in point cloud (meters)')
    parser.add_argument('--valid_iters', default=32, type=int, help='number of flow-field updates')
    parser.add_argument('--get_pc',      default=1, type=int, help='save point cloud output (requires Open3D)')
    parser.add_argument('--remove_invisible', default=1, type=int,
                        help='remove non-overlapping (left-only) pixels from depth/pc')
    parser.add_argument('--denoise_cloud',    default=1, type=int, help='denoise the point cloud')
    parser.add_argument('--denoise_nb_points', default=30, type=int)
    parser.add_argument('--denoise_radius',    default=0.03, type=float)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load inputs (same as demo) ---
    logging.info(f"Loading left image:  {args.left_file}")
    logging.info(f"Loading right image: {args.right_file}")
    imgL = imageio.imread(args.left_file)   # RGB
    imgR = imageio.imread(args.right_file)  # RGB

    if args.scale != 1.0:
        assert args.scale <= 1.0, "scale must be <= 1"
        imgL = cv2.resize(imgL, dsize=None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        imgR = cv2.resize(imgR, dsize=None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

    H, W = imgL.shape[:2]
    logging.info(f"Input size: {(W, H)}")

    # K + baseline file (first line 9 floats row-major, second line baseline)
    with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].strip().split())), dtype=np.float32).reshape(3, 3)
        baseline = float(lines[1].strip())
    # IMPORTANT: wrapper will scale K internally when scale<1; we pass original K here.

    # --- Build and run wrapper ---
    cfg = FStereoConfig(
        ckpt_path=args.ckpt_dir,
        scale=float(args.scale),
        hiera=bool(args.hiera),
        valid_iters=int(args.valid_iters),
        remove_invisible=bool(args.remove_invisible),
        z_far=float(args.z_far),
        device="cuda",
        amp=True,
    )
    wrapper = FoundationStereoWrapper(cfg)

    out = wrapper.infer(
        left_rgb=imgL,
        right_rgb=imgR,
        K=K,
        baseline=baseline,
        remove_invisible=bool(args.remove_invisible),
        z_far=float(args.z_far),
        return_xyz=True,
    )

    disp  = out["disp"]          # (H,W) float32
    depth = out["depth"]         # (H,W) float32 (meters if baseline in meters)
    mask  = out["valid_mask"]    # (H,W) bool
    K_scaled = out["K_scaled"]   # (3,3) after scale (for reference)

    # --- Save a quick disparity visualization like the demo ---
    vis_d = vis_disparity(disp)
    # concatenate original left image (RGB) with disparity vis (BGR) → convert left to BGR for side-by-side
    vis_side = np.concatenate([cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR), vis_d], axis=1)
    vis_path = os.path.join(args.out_dir, "vis.png")
    cv2.imwrite(vis_path, vis_side)
    logging.info(f"Saved: {vis_path}")

    # Save depth as .npy (meters if baseline in meters)
    depth_path = os.path.join(args.out_dir, "depth_meter.npy")
    np.save(depth_path, depth)
    logging.info(f"Saved: {depth_path}")

    # Optional: save point cloud (requires Open3D)
    if args.get_pc:
        maybe_save_pointcloud(
            args.out_dir,
            xyz=out["xyz"],
            rgb=imgL,                # color from left image (RGB)
            mask=mask,
            z_far=args.z_far,
            denoise=bool(args.denoise_cloud),
            nb_points=args.denoise_nb_points,
            radius=args.denoise_radius
        )



if __name__ == "__main__":
    main()
