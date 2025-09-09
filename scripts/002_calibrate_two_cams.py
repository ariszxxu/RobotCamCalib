import cv2
import numpy as np
import time
from math import ceil, sqrt
from contaccams.cameras import CameraManager, InteractiveCameraManager
from contaccams.intr_calib import SimpleIntrinsicsCalibrator

if __name__ == "__main__":
    camera_to_port = {
        "tip_cam": "3-10:1.0",
        "root_cam": "3-9:1.0",
    }

    manager = CameraManager(camera_to_port)
    icm = InteractiveCameraManager(manager)

    # Open all cameras
    opened_count = manager.open_all_cameras()
    print(f"Successfully opened {opened_count} cameras")

    # Live visualize in a single window (press 'q' to quit)
    # visualize_all_cameras(manager, window_wh=(1400, 800), convert_rgb=False)
    buffers = icm.run()
    print("Captured buffers from all cameras:", {name: buf.shape for name, buf in buffers.items()})

    intr_calib = SimpleIntrinsicsCalibrator()

    for name, rgb_frames in buffers.items():
        calib_results = intr_calib.calibrate(rgb_frames)
        print("Calibration results:", calib_results)
        intr_calib.save_yaml(f"outputs/{name}_intrinsics.yaml")
        print(f"Saved intrinsics to outputs/{name}_intrinsics.yaml")

    # Release all cameras when done
    manager.release_all()

