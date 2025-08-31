# Minimal HandEye Calibration

Minimal python implementaiton of camera calibration with full visualization for robotic manipulation. 

## Requirements & Installation 

`pip install -r requirements.txt`

## Intrinsics Calibration 

1. Print `assets/intr_calib_checkerboard.pdf` on an A4 paper with 100% scale.
2. We provide reference code for calibrating intrinsics for a realsense camera in `realsense_intrinsics_calibration_example` in `intr_calib.py`. Make adaptation to your camera settings in `cameras.py`. 
3. When running `cam_ui.run()`, put the chessboard A4 in front of your camera and press `s` on your keyboard to store the current image to buffer. After saving tens of pictures, press `q` to get result. 

## Extrinsics Calibration 

1. Print `assets/extr_calib_apriltag.pdf` on an A4 paper with 100% scale. Cut out the apriltag and paste it on a flat surface. In eye-on-base calibration, the AprialTag is assumed to be rigidly attached to the robot base frame, e.g. pasted on the table. In eye-on-hand calibration, the apriltag is assumed to be be rigidly attached to a robot end-effector frame.
2. We provide reference code for calibrating extrinsics for a realsense camera and xarm6 system in `extr_calib.py`. Make adaptation to `thirdview_realsense_xarm6_example` and `wrist_realsense_xarm6_example` based on your camera and robot settings. 
3. After running the script, open `http://localhost:8080/` in your browser. Drag your robot around in a manual mode, make sure the camera can see the AprialTag. If the AprialTag can be clearly seen in the image, click `click_and_append` button to store the current robot and tag poses to buffer. After saving 8 groups of data, the visualizer will automatically update the solution visualization, i.e. `frames/X_WorldCam` and `frames/X_WorldTag`. The more diverse the data, the better the result. Click `click_and_save` button to save the results.

