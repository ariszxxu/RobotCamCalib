# RobotCamCalib1

面向多相机、多标定板的配置化内参与外参标定工具。相机、采集 profile、标定板和任务分别维护，避免复制脚本后修改常量。

## 环境

本仓库使用 Conda 环境 `fingereye_policy`：

```bash
conda activate fingereye_policy
pip install -r requirements.txt
python calibctl.py validate --check-files
```

G305 采集依赖环境中的 Orbbec SDK / `pyorbbecsdk`；AprilCube 检测还依赖 `configs/targets.yaml` 中配置的外部 AprilCube 工程。`reportlab` 仅用于重新生成 50 mm AprilTag Grid PDF。

## 配置方式

- `configs/cameras.yaml`：物理相机、连接方式及采集 profile。分辨率、裁剪、ISP/矫正模式、镜头焦距或相机模型变化时，应新增 profile 并重新标定内参。
- `configs/targets.yaml`：ChArUco、AprilTag Grid、AprilCube 的几何和配置文件。不同实体打印件应使用不同 ID。
- `configs/tasks.yaml`：把相机 profile、标定板和标定脚本组合成可重复运行的任务。
- `robot_cam_calib/`：公共配置、采集、坐标变换、文件写入和目标位姿检测代码。
- `outputs/`：每次采集、诊断和候选标定结果。

常用命令：

```bash
python calibctl.py list all
python calibctl.py show tasks middle_finger_hand_back_extrinsics
python calibctl.py command middle_finger_intrinsics_charuco
python calibctl.py run middle_finger_intrinsics_charuco
python calibctl.py run dual_grid_mount_offsets -- --max-samples 60
```

## 当前标定任务

| 任务 | 功能 |
|---|---|
| `third_view_intrinsics_charuco` | 第三视角针孔相机的 ChArUco 内参标定 |
| `thumb_web_intrinsics_charuco` | 虎口鱼眼相机的 ChArUco 内参标定 |
| `middle_finger_intrinsics_charuco` | 中指针孔相机的 ChArUco 内参标定 |
| `thumb_web_cube_charuco_extrinsics` | 用第三视角 AprilCube 与虎口 ChArUco 成对观测求外参 |
| `dual_grid_mount_offsets` | 两台相机各带一块 AprilTag Grid，利用相互观测求两个安装偏移 |
| `g305_hand_back_extrinsics` | 求手背 AprilCube 坐标系到 G305 原始左 RGB 光学坐标系的外参 |
| `middle_finger_hand_back_extrinsics` | 求手背 AprilCube 坐标系到中指相机光学坐标系的外参 |

## 保留脚本及职责

| 脚本 | 职责 |
|---|---|
| `calibctl.py` | 检查、查看、解析并运行配置任务；日常统一入口 |
| `intr_calib_charuco.py` | OpenCV 相机的针孔/鱼眼内参采集、离线筛选、交叉验证与结果保存 |
| `extr_calib_d435_cube_cv2_apriltag_grid.py` | AprilCube + ChArUco 双相机成对目标外参求解 |
| `calibrate_dual_camera_rigid_apriltag_grids.py` | 双相机、双刚性 AprilTag Grid 的闭环安装外参求解 |
| `calibrate_g305_left_hand_back_palm.py` | 第三视角相机 + Orbbec G305 的手背外参采集与鲁棒求解 |
| `calibrate_middle_finger_hand_back_cube.py` | 第三视角相机 + 中指相机的手背外参采集与鲁棒求解 |
| `generate_tiny_physical_optics_frame_offset.py` | 生成 50 mm、3x3 AprilTag Grid 的 PDF、纹理和几何 YAML |
| `recompute_extrinsics_filtered.py` | 从已保存样本中排除指定索引，重新计算成对目标外参 |
| `visualize_extr_fingertip.py` | 在 Viser 中查看通用相机/指尖外参坐标关系 |
| `visualize_dual_camera_grid_offsets.py` | 为双 Grid 外参生成坐标系、视锥和安装偏移示意图 |
| `visualize_g305_hand_back_extrinsic.py` | 在 Viser 中查看手背到 G305 的外参 |

详细的新增相机、标定板选择、内外参流程、坐标约定和质量门槛见 [`docs/calibration_workflows.md`](docs/calibration_workflows.md)。

## 验证

不连接相机即可执行：

```bash
python -m unittest discover -s tests -v
python extr_calib_d435_cube_cv2_apriltag_grid.py --self-test
python calibrate_dual_camera_rigid_apriltag_grids.py --self-test
python calibrate_g305_left_hand_back_palm.py --self-test --solver-workers 2
python calibrate_middle_finger_hand_back_cube.py --self-test --solver-workers 2 --no-progress
```

所有外参均使用 `T_A_B` 约定：把 B 坐标系中的点变换到 A 坐标系。不要只记录模糊的 `extrinsic` 字段而不标明方向。
