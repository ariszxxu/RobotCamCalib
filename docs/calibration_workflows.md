# 多相机、多标定板的仓库组织与标定流程

## 设计原则

把容易混在一起的三类信息分开：

1. **相机设备**：序列号、SDK/backend、物理安装名称。
2. **采集 profile**：分辨率、帧率、像素格式、是否原始/矫正、对应内参。
3. **标定任务**：使用哪些相机 profile、哪些实体标定板、运行哪个求解流程。

因此，同一台相机的 `2592x1944@50 MJPG` 和 `1280x720@30 YUYV` 必须是两个
profile，并分别标定内参；同一份 PDF 打印出的两块板也是两个物理实例，在外参闭环
中不能当作同一坐标系。

仓库当前使用：

- `configs/robots.yaml`：机器人身份、连接及位姿读取 profile。
- `configs/cameras.yaml`：相机身份和采集 profile。
- `configs/targets.yaml`：标定板几何、字典和配置文件。
- `assets/targets/`：可复用且受版本控制的实体标定板几何 preset。
- `configs/tasks.yaml`：把相机和标定板组合成可运行任务。
- `robot_cam_calib/`：与设备无关的公共几何、同步采集和文件工具。
- 根目录脚本：保留兼容的命令行入口。
- `outputs/`：一次运行产生的原图、诊断和候选结果，不提交 Git。

建议逐步演进为以下结构；根目录脚本可继续作为兼容 wrapper：

```text
RobotCamCalib1/
├── configs/
│   ├── cameras.yaml
│   ├── targets.yaml
│   └── tasks.yaml
├── robot_cam_calib/
│   ├── cameras/          # OpenCV / RealSense / Orbbec adapters
│   ├── targets/          # ChArUco / AprilGrid / AprilCube detectors
│   ├── solvers/          # intrinsics / paired-target / hand-eye solvers
│   ├── capture.py
│   ├── geometry.py
│   └── io.py
├── apps/                 # 薄 CLI，负责解析参数和显示 UI
├── assets/targets/       # 可打印 PDF 和受版本控制的几何 YAML
├── calibrations/         # 人工验收后“晋升”的 current/baseline 标定结果
│   ├── intrinsics/<camera>/<profile>/current.yaml
│   └── extrinsics/<task>/current.yaml
├── outputs/              # 每次运行的原始结果，Git ignored
│   ├── intrinsics/<task>/<timestamp>/
│   └── extrinsics/<task>/<timestamp>/
├── tests/
└── docs/
```

不要让业务程序直接依赖带时间戳的 `outputs/...yaml`。完成质量审核后，把选中的结果
复制/晋升到 `calibrations/.../current.yaml`，配置文件只引用这个稳定路径；原始运行目录
继续保留，便于追溯。

## 配置与运行

检查配置引用：

```bash
python calibctl.py validate --check-files
```

查看可用设备、板和任务：

```bash
python calibctl.py list all
python calibctl.py show cameras middle_finger_cam
python calibctl.py show tasks middle_finger_hand_back_extrinsics
```

先查看 task 最终解析成的兼容命令：

```bash
python calibctl.py command middle_finger_intrinsics_charuco
```

运行任务；`--` 后可临时覆盖 task 中没有配置的脚本参数：

```bash
python calibctl.py run middle_finger_intrinsics_charuco
python calibctl.py run dual_grid_mount_offsets -- --max-samples 60
```

推荐把经常使用的参数写入 task，把一次性的采样数量、预览开关等放在 `--` 后。

## 新增相机

在 `configs/cameras.yaml` 中只登记一次物理设备，然后按实际使用模式增加 profile：

```yaml
new_camera:
  backend: opencv
  description: End-effector camera, serial ABC123.
  connections:
    lab_bench: "4-2:1.0"
  profiles:
    full_30fps:
      resolution: [1920, 1080]
      fps: 30
      fourcc: MJPG
      camera_model: pinhole
      intrinsics: calibrations/intrinsics/new_camera/full_30fps/current.yaml
```

以下任一项变化都应视为不同的内参 profile：

- 分辨率、binning、ROI/crop；
- 原始流与 ISP/SDK 矫正后的流；
- pinhole/fisheye 模型；
- 可变焦、对焦位置或镜头被重新安装；
- 会改变成像几何的 SDK work mode。

USB 端口变化通常不改变内参，只修改 `connections`；应优先用相机序列号确认设备身份，USB
拓扑只负责定位采集节点。

## 新增标定板

标定板必须有稳定的物理 ID，例如 `charuco_a4_40mm_print_20260817_A`。配置应记录：

- target 类型和字典；
- 行列数、方格/标签真实尺寸；
- 坐标原点、轴方向和角点顺序；
- PDF/几何 YAML 的版本或哈希；
- 打印比例和实测尺寸。

建议选择：

- **ChArUco**：普通/鱼眼内参首选，部分遮挡仍可用，角点精度好。
- **Checkerboard**：环境可控、整板始终可见时最简单。
- **AprilTag Grid**：外参、大视角、远距离或容易遮挡时更稳。
- **AprilCube**：需要从多个方向观察刚体坐标系时使用。

打印后必须按两个方向实测尺寸。只修改 YAML 中的尺寸去“补偿”错误打印，会让板的
非均匀缩放无法被模型表达。

## 内参标定顺序

1. 固定镜头、焦距、对焦和采集 profile。
2. 用 task 启动对应相机和标定板。
3. 覆盖画面中心、四角和边缘；包含不同距离以及绕 X/Y 轴倾斜的姿态。
4. 避免全部样本正对标定板或集中在同一距离。
5. 查看重投影误差、逐视图离群点、交叉验证和焦距稳定性。
6. 验收后晋升为该 profile 的 `current.yaml`。

内参输出必须至少保存：相机 ID、profile、图像尺寸、相机模型、K、畸变参数、目标几何、
样本/离群点和诊断信息。

## 外参任务选择

必须先有参与相机对应 profile 的有效内参，然后根据物理约束选择任务：

| 场景 | 推荐流程 |
|---|---|
| 相机固定在刚体/手指上，另一台固定相机可观察刚体目标 | paired-target extrinsics（AprilCube + ChArUco） |
| 两台相机各自固定一块板并能互相看见 | mutual-grid extrinsics |
| 相机与机器人基座或末端的关系可由机器人位姿提供 | robot hand-eye |
| 只知道两个相机同时看到同一块板 | stereo/relative-camera extrinsics |

外参结果必须明确使用 `T_A_B` 约定：它把 B 坐标表达映射到 A。禁止仅写 `extrinsic`
而不写方向。每个 task 还应记录：参与的内参文件、标定板物理 ID、采集 profile、时间同步
方式、样本索引、离群点、残差和雅可比秩/条件数。

### xArm7 + G305 eye-in-hand

`xarm7_g305_eye_in_hand` 只读取 xArm7 qpos，并调用控制器 FK 得到
`T_base_link7`；它不调用任何机器人 `set_*` 或运动接口。因为控制器 FK 会包含 TCP
offset，任务要求 TCP offset 为零。xArm7 URDF 中 `link7 -> link_eef` 为恒等变换，
所以此时 FK 末端帧就是所需 `link7`。

固定板假设下，每个样本满足：

```text
T_base_link7_i
@ T_link7_wuji_g305_raw_left_optical
@ T_wuji_g305_raw_left_optical_charuco_i
= T_base_charuco
```

程序连续只读 qpos，检测到机械臂稳定 0.5 秒后自动采样。一次采样完成后，至少一个
关节相对上次采样移动 2° 才重新布防；自动触发时还会再次读取一组 qpos 验证相机帧
与机械臂姿态没有错配。采集必须保存原始 qpos，而不能只保存控制器显示的 TCP 位姿。
求解结果需要同时检查板在 base 中的旋转/平移残差、方法间结果、奇偶子集稳定性和
相对旋转激励秩。旋转激励秩低于 3 时，不应把候选 YAML 晋升为正式标定。

## 推荐质量门槛

- 内参样本至少覆盖图像大部分区域，并有明显的深度和倾斜变化。
- 软件同步的双相机外参只在目标短暂停稳后采样。
- 外参运动应覆盖多个旋转轴；只平移或只绕单轴旋转容易退化。
- 先按检测/清晰度质量排除明显坏图，再由鲁棒求解器处理几何离群点。
- 使用奇偶样本或交叉子集重算，检查结果是否稳定。
- 保存原始图片和 capture manifest；重算时只排除索引，不删除原图。
