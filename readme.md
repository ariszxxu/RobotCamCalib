# RobotCamCalib1

面向多相机、多标定板的配置化内参与外参标定工具。相机、采集 profile、标定板和任务分别维护，避免复制脚本后修改常量。

## 环境

本机使用 Conda 环境 `fingereye`：

```bash
conda activate fingereye
pip install -r requirements.txt
python calibctl.py validate --check-files
```

G305 采集依赖该环境中的 Orbbec SDK / `pyorbbecsdk`；xArm7 手眼标定还使用
`xarm-python-sdk`。AprilCube 检测依赖 `configs/targets.yaml` 中配置的外部
AprilCube 工程。`reportlab` 仅用于重新生成 50 mm AprilTag Grid PDF。

## 配置方式

- `configs/robots.yaml`：机器人身份、地址及只读位姿 profile。
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
| `xarm7_g305_eye_in_hand` | 只读 xArm7 qpos，求 `link7_T_wuji_g305_raw_left_optical` |
| `wuji_g305_fingertip_extrinsics` | 用 WujiHand URDF、G305 与指尖 AprilCube 联合求掌部相机和指尖标靶两个安装外参 |

## xArm7 + G305 眼在手上标定

物理布置：G305 刚性固定在 `link7`，ChArUco 板相对 xArm 基座保持不动。
程序不会移动机械臂；操作者在 xArm 网页中移动机械臂。检测到 qpos 连续稳定
0.5 秒、标定板检测合格后，程序自动保存一组；成功采样后需要再次明显移动机械臂
才会重新布防，避免同一姿态重复采样。建议采集 30 组，覆盖多个位置并绕至少两个、
最好三个旋转轴改变姿态。

先做配置和硬件检查：

```bash
/home/CNF2025915223/miniconda3/envs/fingereye/bin/python \
  calibctl.py run xarm7_g305_eye_in_hand -- --check-config

/home/CNF2025915223/miniconda3/envs/fingereye/bin/python \
  calibctl.py run xarm7_g305_eye_in_hand -- --check-hardware
```

正式采集：

```bash
/home/CNF2025915223/miniconda3/envs/fingereye/bin/python \
  calibctl.py run xarm7_g305_eye_in_hand
```

默认自动采样阈值为稳定 1.0 秒、同步 burst 关节稳定范围 0.02°，采样后至少移动 2° 才重新
布防。`s` 可手动补采，`q` 或 `Esc` 提前求解。少于 12 组时只保存 manifest，
不输出候选外参。程序临时切换 G305 到 `Dual Color Streams`，并在退出时恢复原工作
模式。每组样本保存原始左目图、7D qpos、`T_base_link7`、
`T_wuji_g305_raw_left_optical_charuco` 和重投影误差。输出采用 `T_A_B` 约定，
字段名为 `T_link7_wuji_g305_raw_left_optical`。

中断后可从 manifest 重新求解而不连接硬件：

```bash
/home/CNF2025915223/miniconda3/envs/fingereye/bin/python \
  calibctl.py run xarm7_g305_eye_in_hand -- \
  --offline-manifest /absolute/path/to/capture_manifest.yaml
```

## WujiHand + G305 指尖双外参标定

每个样本使用 Wuji 实测 `qpos20` 从指定 URDF 计算
`T_left_palm_link_left_finger2_link4(q)`，并从 G305 原始左目检测 IDs 6–11 的
AprilCube。联合方程为：

```text
T_left_palm_link_left_finger2_link4(q_i)
@ T_left_finger2_link4_index_wuji_w_cube_update
= T_left_palm_link_wuji_g305_raw_left_optical
@ T_wuji_g305_raw_left_optical_index_wuji_w_cube_update_i
```

先做只读检查；检查成功后必须显式加 `--execute-motion` 才会移动 WujiHand：

```bash
/home/CNF2025915223/miniconda3/envs/fingereye/bin/python \
  calibctl.py run wuji_g305_fingertip_extrinsics -- --check-config

/home/CNF2025915223/miniconda3/envs/fingereye/bin/python \
  calibctl.py run wuji_g305_fingertip_extrinsics -- --check-hardware

/home/CNF2025915223/miniconda3/envs/fingereye/bin/python \
  calibctl.py run wuji_g305_fingertip_extrinsics -- --execute-motion
```

所有目标先经过 FingerEyeV2 现有的实时硬件/回放限位、保守自碰撞检查、分段路点、
新鲜反馈和关闭电机清理。每个接受样本保存原始图、标注图、qpos20、URDF FK、PnP
位姿和时间戳；无效检测另存到 `rejected/`。退出时默认回到初始 qpos8。

当前 URDF 的标准语义是 `finger1=thumb`、`finger2=index`；实动探测确认 IDs 6–11
的实体 AprilCube 属于 `left_finger2_link4`。其 frame 与
`fingereye_mesh/index_wuji_w_cube.stl` 的源 frame 重合。

Thumb mesh 标注使用 IDs 12–17 的 18.75 mm AprilCube；其 frame 与
`fingereye_mesh/thumb.obj` 的源 frame 重合。已有 palm-to-G305 外参时可在不移动
WujiHand 的情况下连续采集并鲁棒平均：

```bash
/home/CNF2025915223/miniconda3/envs/fingereye/bin/python \
  calibctl.py run wuji_g305_thumb_fingertip_extrinsics
```

输出字段为
`T_left_finger1_link4_thumb_fingertip_mesh_frame`，并保存每帧图像、qpos20、
单帧候选变换和离群点诊断。

## 保留脚本及职责

| 脚本 | 职责 |
|---|---|
| `calibctl.py` | 检查、查看、解析并运行配置任务；日常统一入口 |
| `intr_calib_charuco.py` | OpenCV 相机的针孔/鱼眼内参采集、离线筛选、交叉验证与结果保存 |
| `extr_calib_d435_cube_cv2_apriltag_grid.py` | AprilCube + ChArUco 双相机成对目标外参求解 |
| `calibrate_dual_camera_rigid_apriltag_grids.py` | 双相机、双刚性 AprilTag Grid 的闭环安装外参求解 |
| `calibrate_g305_left_hand_back_palm.py` | 第三视角相机 + Orbbec G305 的手背外参采集与鲁棒求解 |
| `calibrate_middle_finger_hand_back_cube.py` | 第三视角相机 + 中指相机的手背外参采集与鲁棒求解 |
| `calibrate_xarm7_g305_eye_in_hand.py` | 只读 xArm7 qpos 的 G305 eye-in-hand 采集、求解和诊断 |
| `generate_tiny_physical_optics_frame_offset.py` | 生成 50 mm、3x3 AprilTag Grid 的 PDF、纹理和几何 YAML |
| `recompute_extrinsics_filtered.py` | 从已保存样本中排除指定索引，重新计算成对目标外参 |
| `visualize_extr_fingertip.py` | 在 Viser 中查看通用相机/指尖外参坐标关系 |
| `visualize_dual_camera_grid_offsets.py` | 为双 Grid 外参生成坐标系、视锥和安装偏移示意图 |
| `visualize_g305_hand_back_extrinsic.py` | 在 Viser 中查看手背到 G305 的外参 |
| `visualize_wuji_calibrated_frames.py` | 用 floating-joint Wuji URDF 同时查看 palm-to-G305 与 link4-to-fingertip-mesh 标定 frame |

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
