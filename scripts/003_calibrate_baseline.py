import os
import cv2
import numpy as np

from contaccams.cameras import CameraManager, InteractiveCameraManager
from contaccams.intr_calib import SimpleIntrinsicsCalibrator
from contaccams.stereo_calib import SimpleStereoCalibrator  # path as you place it

if __name__ == "__main__":
    camera_to_port = {
        "tip_cam": "3-10:1.0",
        "root_cam": "3-9:1.0",
    }
    left_name, right_name = "tip_cam", "root_cam"  # choose your pair

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    manager = CameraManager(camera_to_port)
    icm = InteractiveCameraManager(manager, window_wh=(1400, 800), show_fps=True)

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
