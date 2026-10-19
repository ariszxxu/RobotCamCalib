# AgileX Piper three-camera calibration

This directory contains the source used on 2026-10-19 to calibrate the two
wrist D435 cameras and the fixed front D435 camera on the AgileX dual-arm
system.

The calibration conventions are explicit:

- `T_A_B` maps points expressed in frame B into frame A.
- Matrix translations are in metres.
- Wrist outputs are `T_link6_camera`.
- The fixed-camera output is `T_base_front_color_optical`.

## Capture programs

- `right_wrist_frame_server.py` and `right_wrist_cv2_gui.py`: right wrist
  capture.
- `left_wrist_frame_server.py` and `left_wrist_cv2_gui.py`: left wrist
  capture.
- `front_right_frame_server.py` and `front_from_right_cv2_gui.py`: synchronized
  front/right capture with a fixed board and a manually moved right arm.
- `right_wrist_feedback.py` and `left_wrist_feedback.py`: passive joint and
  drive-state feedback. The capture programs do not send actuator commands.

The GUI buffers accepted samples in memory. Press `q` to flush the buffer to a
session; pressing Escape discards the unsaved buffer.

## Solvers

- `solve_right_combined_filtered.py`: right-wrist hand-eye solution with image
  sharpness and geometric outlier filtering.
- `solve_left_best_cv.py`: left-wrist training-only model selection and
  holdout evaluation.
- `solve_three_camera_extrinsics.py`: joint multi-session right-wrist solution
  and fixed-front-camera solution, followed by a local three-camera manifest.

The solvers preserve predetermined holdouts and write local candidates. They
do not update a robot control configuration. Generated `sessions/` and
`exports/` contain robot-specific observations and results and are deliberately
excluded from Git.

The workstation paths, camera serials, ROS namespaces, and CAN bindings in
these scripts describe the deployed AgileX machine. Review them before using
the programs on another system.
