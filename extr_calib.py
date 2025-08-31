from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
from loguru import logger
import yaml
import time 
import viser
from viser.extras import ViserUrdf
from dataclasses import dataclass
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R


# --------------------- small SE3/SO3 helpers --------------------- #

def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti

def angle_from_R(R: np.ndarray) -> float:
    c = (np.trace(R) - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))

def log_SO3(R: np.ndarray) -> np.ndarray:
    th = angle_from_R(R)
    if th < 1e-12:
        return np.zeros(3)
    W = (R - R.T) * (0.5 / np.sin(th))
    return th * np.array([W[2,1], W[0,2], W[1,0]])

def se3_mean(Ts: List[np.ndarray]) -> np.ndarray:
    R_sum = np.zeros((3, 3))
    t_sum = np.zeros(3)
    for T in Ts:
        R_sum += T[:3, :3]
        t_sum += T[:3, 3]
    U, _, Vt = np.linalg.svd(R_sum)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    Tm = np.eye(4)
    Tm[:3, :3] = Rm
    Tm[:3, 3]  = t_sum / len(Ts)
    return Tm

# --------------------- core pieces for AX = XB ------------------- #

def _relative_pairs(A_list: List[np.ndarray], B_list: List[np.ndarray],
                    min_rot_deg: float = 1.0,
                    max_pairs: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build relative motions C_ij=A_i A_j^{-1}, D_ij=B_i B_j^{-1}, filter by rotation size."""
    n = len(A_list)
    C_rel, D_rel = [], []
    for i in range(n):
        for j in range(i+1, n):
            C = A_list[i] @ inv_T(A_list[j])
            D = B_list[i] @ inv_T(B_list[j])
            # skip nearly identity rotations (ill-conditioned)
            if np.degrees(angle_from_R(C[:3, :3])) < min_rot_deg: 
                continue
            C_rel.append(C); D_rel.append(D)
    if max_pairs is not None and len(C_rel) > max_pairs:
        idx = np.random.permutation(len(C_rel))[:max_pairs]
        C_rel = [C_rel[k] for k in idx]
        D_rel = [D_rel[k] for k in idx]
    return C_rel, D_rel

def _fit_RX_from_pairs(C_rel: List[np.ndarray], D_rel: List[np.ndarray]) -> np.ndarray:
    """Closed-form rotation for AX=XB using screw-axis (log) Procrustes."""
    A = np.stack([log_SO3(C[:3,:3]) for C in C_rel], 0)  # kx3
    B = np.stack([log_SO3(D[:3,:3]) for D in D_rel], 0)  # kx3
    H = B.T @ A
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def _fit_tX_from_pairs(RX: np.ndarray, C_rel: List[np.ndarray], D_rel: List[np.ndarray]) -> np.ndarray:
    """Linear LS for translation: (I - R_D) tX = t_D - R_X t_C."""
    rows, rhs = [], []
    for C, D in zip(C_rel, D_rel):
        RC, tC = C[:3, :3], C[:3, 3]
        RD, tD = D[:3, :3], D[:3, 3]
        rows.append(np.eye(3) - RD)
        rhs.append(tD - RX @ tC)
    M = np.concatenate(rows, 0)  # (3k,3)
    b = np.concatenate(rhs, 0)   # (3k,)
    lam = 1e-9  # tiny Tikhonov
    t = np.linalg.lstsq(M.T @ M + lam*np.eye(3), M.T @ b, rcond=None)[0]
    return t

def _residuals_X(C_rel: List[np.ndarray], D_rel: List[np.ndarray], RX: np.ndarray, tX: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Residuals on D ≈ X C X^{-1}: angle error (rad) and translation norm (same unit as input)."""
    ang = []
    tran = []
    X = np.eye(4); X[:3,:3]=RX; X[:3,3]=tX
    Xi = inv_T(X)
    for C, D in zip(C_rel, D_rel):
        Pred = X @ C @ Xi
        E = inv_T(Pred) @ D  # residual transform
        ang.append(angle_from_R(E[:3,:3]))
        tran.append(np.linalg.norm(E[:3,3]))
    return np.array(ang), np.array(tran)

# --------------------- RANSAC + refine, then Y ------------------- #

def calibrate_cammount_and_tag(
    X_CamTag_list: np.ndarray,        # B_i, (n,4,4)
    X_WorldCammount_list: np.ndarray, # (n,4,4)
    X_WorldTagmount_list: np.ndarray, # (n,4,4)
    ransac_iters: int = 500,
    sample_size: int = 8,             # relative-pair samples per hypothesis
    rot_thresh_deg: float = 1.0,      # inlier threshold on rotation (deg)
    trans_thresh: float = 0.01,       # inlier threshold on translation (meters)
    min_rot_deg_pair: float = 1.0,    # ignore nearly identity pairs
    max_pairs: Optional[int] = None,  # cap total relative pairs
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Robustly solve B_i = X A_i Y.
    Returns (X_CammountCam, X_TagmountTag, info).
    """
    n = X_CamTag_list.shape[0]
    assert X_CamTag_list.shape == (n,4,4)
    assert X_WorldCammount_list.shape == (n,4,4)
    assert X_WorldTagmount_list.shape == (n,4,4)

    # Build A_i, B_i
    A_list = [inv_T(X_WorldCammount_list[i]) @ X_WorldTagmount_list[i] for i in range(n)]
    B_list = [X_CamTag_list[i] for i in range(n)]

    # Relative motions
    C_rel, D_rel = _relative_pairs(A_list, B_list, min_rot_deg=min_rot_deg_pair, max_pairs=max_pairs)
    k = len(C_rel)
    if k < sample_size:
        raise RuntimeError("Not enough informative relative pairs for robust estimation.")

    # Precompute thresholds in radians
    rot_thr = np.radians(rot_thresh_deg)

    # RANSAC loop
    best_inliers = None
    best_RX, best_tX = None, None
    idx_all = np.arange(k)

    for _ in range(ransac_iters):
        idx = np.random.choice(k, size=sample_size, replace=False)
        RX_h = _fit_RX_from_pairs([C_rel[i] for i in idx], [D_rel[i] for i in idx])
        tX_h = _fit_tX_from_pairs(RX_h, [C_rel[i] for i in idx], [D_rel[i] for i in idx])

        ang_res, tr_res = _residuals_X(C_rel, D_rel, RX_h, tX_h)
        inliers = (ang_res <= rot_thr) & (tr_res <= trans_thresh)

        if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
            best_inliers = inliers
            best_RX, best_tX = RX_h, tX_h

    # Refit using all inliers (or fall back to all pairs if RANSAC failed)
    if best_inliers is None or np.count_nonzero(best_inliers) < sample_size:
        in_idx = idx_all
    else:
        in_idx = np.where(best_inliers)[0]

    C_in = [C_rel[i] for i in in_idx]
    D_in = [D_rel[i] for i in in_idx]
    RX = _fit_RX_from_pairs(C_in, D_in)
    tX = _fit_tX_from_pairs(RX, C_in, D_in)

    # Recover Y and average
    X = np.eye(4); X[:3,:3]=RX; X[:3,3]=tX
    Y_list = [inv_T(A) @ inv_T(X) @ B for A, B in zip(A_list, B_list)]
    Y = se3_mean(Y_list)

    # Report residuals on original equations
    rot_err_deg, trans_err = [], []
    for A, B in zip(A_list, B_list):
        E = inv_T(X @ A @ Y) @ B
        rot_err_deg.append(np.degrees(angle_from_R(E[:3,:3])))
        trans_err.append(np.linalg.norm(E[:3,3]))

    info = {
        "num_pairs": k,
        "num_inliers": int(len(in_idx)),
        "rot_err_deg_mean": float(np.mean(rot_err_deg)),
        "rot_err_deg_med": float(np.median(rot_err_deg)),
        "rot_err_deg_max": float(np.max(rot_err_deg)),
        "trans_err_mean": float(np.mean(trans_err)),
        "trans_err_med": float(np.median(trans_err)),
        "trans_err_max": float(np.max(trans_err)),
    }
    logger.info(f"Calibration residuals (mean/med/max): rot {info['rot_err_deg_mean']:.4f}/{info['rot_err_deg_med']:.4f}/{info['rot_err_deg_max']:.4f}, trans {info['trans_err_mean']:.4f}/{info['trans_err_med']:.4f}/{info['trans_err_max']:.4f}")

    # Return in your convention
    X_CammountCam = inv_T(X)
    X_TagmountTag = Y
    return X_CammountCam, X_TagmountTag

class ViserUrdfUser:
    """Owns a Viser server + URDF; exposes joint names and live updates."""

    def __init__(self, urdf_path: Path,):
        self.urdf_path = urdf_path.absolute()
        self.server = viser.ViserServer()
        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=self.urdf_path,
            load_meshes=True,
            load_collision_meshes=False,
            collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        )
        self.ndof = len(self.get_actuated_joint_names())
        self.qpos = np.zeros(self.ndof)
        self.actuated_joint_names = self.get_actuated_joint_names()
        self.actuated_joint_limits = self.viser_urdf.get_actuated_joint_limits()
        self.lower_joint_limits = np.array([self.actuated_joint_limits[k][0] for k in self.actuated_joint_names])
        self.upper_joint_limits = np.array([self.actuated_joint_limits[k][1] for k in self.actuated_joint_names])
        self.default_qpos = 0.5 * (self.lower_joint_limits + self.upper_joint_limits)
        self.viser_urdf.update_cfg(self.default_qpos)

    def get_actuated_joint_names(self) -> List[str]:
        return self.viser_urdf.get_actuated_joint_names()

    def update(self, qpos_user: np.ndarray, input_joint_names: List[str]=None) -> None:
        if input_joint_names is not None:
            reorder_idx = np.array([input_joint_names.index(n) for n in self.actuated_joint_names], dtype=int)
            self.qpos = qpos_user[reorder_idx]
        else:
            self.qpos = qpos_user
        self.viser_urdf.update_cfg(self.qpos)

    def get_frame_pose(self, frame_name: str) -> np.ndarray:
        return np.array(self.viser_urdf._urdf.get_transform(frame_name))
    
@dataclass
class ExtrinsicsCalibConfig:
    # robot parameters
    robot_urdf_path: Path  # this should align with the robot in the real world

    # camera and tag's mounting link frame names in the URDF, can be either the base link or an end effector link 
    cammount_link_name: str
    tagmount_link_name: str

    # camera intrinsics
    K: np.ndarray # (3,3) camera intrinsics

    # output paths 
    output_file_path: Path = Path("outputs/extrinsics.yaml")

    # apriltag parameters
    tag_size: float = 0.048
    tag_family: str = "tag36h11"

    # ransac param 
    ransac_sample_size : int = 8


class CamTagCalibrator:
    """Calibrates the camera and tag poses."""

    def __init__(self, config: ExtrinsicsCalibConfig):
        self.config = config

        self.urdf_viser = ViserUrdfUser(urdf_path=config.robot_urdf_path)
        self.server = self.urdf_viser.server
        self.tag_detector = Detector(
            families=config.tag_family,
            quad_decimate=1.0,
        )
        self.camera_params = (
            config.K[0, 0],  # fx
            config.K[1, 1],  # fy
            config.K[0, 2],  # cx
            config.K[1, 2],  # cy
        )

        # global variables to store calibration data
        self.X_WorldCammount_list = []  # list of (4,4) camera mount poses in world frame
        self.X_WorldTagmount_list = []  # list of (4,4) tag mount poses in world frame
        self.X_CamTag_list = []        # list of (4,4) camera-to-tag poses

        # current observations 
        self.X_CamTag = np.eye(4)  # (4,4) camera-to-tag pose
        self.X_WorldCammount = np.eye(4)  # (4,4) camera mount pose in world frame
        self.X_WorldTagmount = np.eye(4)  # (4,4) tag mount pose in world frame

        # current solution estimates
        self.X_CammountCam = np.eye(4) # (4,4) camera mount to camera pose
        self.X_TagmountTag = np.eye(4) # (4,4) tag mount to tag pose

        # add GUI buttons
        self.append_button = self.server.gui.add_button("click_and_append")
        self.save_button = self.server.gui.add_button("click_and_save")
        self.append_button.on_click(lambda _: self.append_and_solve())
        self.save_button.on_click(lambda _: self.save_results())

    def vis_frame(self, X: np.ndarray, frame_name: str):
        link_wxyz = R.from_matrix(X[:3, :3]).as_quat()[[3, 0, 1, 2]] 
        link_xyz = X[:3, 3]
        self.server.scene.add_frame(
            name=f"frames/{frame_name}",
            axes_length=0.2,
            axes_radius=0.01,
            wxyz=link_wxyz,
            position=link_xyz,
        )

    def vis_step(self, rgb_image: np.ndarray, qpos: np.ndarray, input_joint_names: List[str]=None) -> None:
        """
        Loop this vis_step() function to update the Viser server and get user inputs.

        Input; 
        - rgb_image: (H,W,3) uint8 BGR image from the camera
        - qpos: (ndof,) robot joint values
        - input_joint_names: List[str], we will reorder qpos according to urdf_viser.get_actuated_joint_names()

        ---
        scene part: 
        - robot current configuration 
        - X_WorldCammount
        - X_WorldTagmount
        - X_WorldCam (current estimate)
        - X_WorldTag (current estimate)
        - rgb image in the camera frustum

        --- 
        gui part: 
        - click_and_append_button: append current X_CamTag, X_WorldCammount, X_WorldTagmount to the list
        - click_and_save_button: save current X_CammountCam, X_TagmountTag

        """
        # update viser digital robot model 
        self.urdf_viser.update(qpos, input_joint_names)
        X_WorldCammount = self.urdf_viser.get_frame_pose(self.config.cammount_link_name)
        X_WorldTagmount = self.urdf_viser.get_frame_pose(self.config.tagmount_link_name)
        self.X_WorldCammount = X_WorldCammount
        self.X_WorldTagmount = X_WorldTagmount
        self.vis_frame(X_WorldCammount, "X_WorldCammount")
        self.vis_frame(X_WorldTagmount, "X_WorldTagmount")

        # update current estimates
        X_WorldCam = X_WorldCammount @ self.X_CammountCam
        X_WorldTag = X_WorldTagmount @ self.X_TagmountTag
        self.vis_frame(X_WorldCam, "X_WorldCam")
        self.vis_frame(X_WorldTag, "X_WorldTag")

        # update camera frustum 
        self.server.scene.add_camera_frustum(
            name="camera_frustum",
            fov=120,
            aspect=rgb_image.shape[1] / rgb_image.shape[0],
            image=rgb_image,
            wxyz=R.from_matrix(X_WorldCam[:3, :3]).as_quat()[[3, 0, 1, 2]],
            position=X_WorldCam[:3, 3],
        )

        # update apriltag detection
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        tags = self.tag_detector.detect(gray_image, True, self.camera_params, self.config.tag_size)
        if len(tags) == 1:
            tag = tags[0]
            self.X_CamTag[:3, :3] = tag.pose_R
            self.X_CamTag[:3, 3] = tag.pose_t[:, 0]

    def append_and_solve(self):
        """Append current observations and solve for new estimates."""
        self.X_CamTag_list.append(self.X_CamTag.copy())
        self.X_WorldCammount_list.append(self.X_WorldCammount.copy())
        self.X_WorldTagmount_list.append(self.X_WorldTagmount.copy())
        logger.info(f"Appended {len(self.X_CamTag_list)} observations.")

        if len(self.X_CamTag_list) >= self.config.ransac_sample_size:  
            self.X_CammountCam, self.X_TagmountTag = calibrate_cammount_and_tag(
                np.asarray(self.X_CamTag_list),
                np.asarray(self.X_WorldCammount_list),
                np.asarray(self.X_WorldTagmount_list),
                sample_size=self.config.ransac_sample_size,
            )
            logger.info("Solved for new estimates.")
            logger.info(f"X_CammountCam:\n{self.X_CammountCam}")
            logger.info(f"X_TagmountTag:\n{self.X_TagmountTag}")

    def save_results(self):
        """Save current estimates to a yaml file."""
        results = {
            "X_CammountCam": self.X_CammountCam.tolist(),
            "X_TagmountTag": self.X_TagmountTag.tolist(),
        }
        with open(self.config.output_file_path, "w") as f:
            yaml.dump(results, f)
        logger.info(f"Saved calibration results to {self.config.output_file_path}.")


def thirdview_realsense_xarm6_example():
    # A third-view realsense camera & xarm6 example 
    from cameras import RealsenseCamera
    from xarm.wrapper import XArmAPI

    # camera 
    cam = RealsenseCamera()
    cam.start()
    intr_calib_yaml_path = Path("outputs/intrinsics.yaml")  # replace with your intrinsics yaml path
    with open(intr_calib_yaml_path, "r") as f:
        intr_data = yaml.safe_load(f)
    K = np.array(intr_data["K"], dtype=float).reshape(3, 3)

    # robot model & real robot
    robot_urdf_path = Path("assets/robots/xarm6/xarm6_wo_ee.urdf")  # replace with your robot urdf path
    XARM6_IP = "192.168.1.208"
    xarm = XArmAPI(XARM6_IP, is_radian=True)
    xarm.motion_enable(enable=True)
    xarm.set_mode(0)
    xarm.set_state(state=0)
    xarm.set_gripper_mode(0)
    xarm.set_gripper_enable(True)
    xarm.set_gripper_speed(5000)

    config = ExtrinsicsCalibConfig(
        robot_urdf_path=robot_urdf_path,
        cammount_link_name="link_base",  
        tagmount_link_name="link_eef",
        K=K,
        output_file_path=Path("outputs/extrinsics.yaml"),
        tag_size=0.048,
        tag_family="tag36h11",
    )
    calibrator = CamTagCalibrator(config)

    while True:
        rgb_frame = cam.read() 
        _, joint_values = xarm.get_servo_angle(is_radian=True) 
        calibrator.vis_step(rgb_frame, np.array(joint_values)[:6], input_joint_names=None)   # first 6 joints
        time.sleep(0.1)


def wrist_realsense_xarm6_example():
    # A wrist-view realsense camera & xarm6 example 
    # compared to the third-view example, only exchange the cammount and tagmount link names
    from cameras import RealsenseCamera
    from xarm.wrapper import XArmAPI

    # camera 
    cam = RealsenseCamera()
    cam.start()
    intr_calib_yaml_path = Path("outputs/intrinsics.yaml")  # replace with your intrinsics yaml path
    with open(intr_calib_yaml_path, "r") as f:
        intr_data = yaml.safe_load(f)
    K = np.array(intr_data["K"], dtype=float).reshape(3, 3)

    # robot model & real robot
    robot_urdf_path = Path("assets/robots/xarm6/xarm6_wo_ee.urdf")  # replace with your robot urdf path
    XARM6_IP = "192.168.1.208"
    xarm = XArmAPI(XARM6_IP, is_radian=True)
    xarm.motion_enable(enable=True)
    xarm.set_mode(0)
    xarm.set_state(state=0)
    xarm.set_gripper_mode(0)
    xarm.set_gripper_enable(True)
    xarm.set_gripper_speed(5000)

    config = ExtrinsicsCalibConfig(
        robot_urdf_path=robot_urdf_path,
        cammount_link_name="link_eef",  
        tagmount_link_name="link_base",
        K=K,
        output_file_path=Path("outputs/extrinsics_wrist.yaml"),
        tag_size=0.048,
        tag_family="tag36h11",
    )
    calibrator = CamTagCalibrator(config)

    while True:
        rgb_frame = cam.read() 
        _, joint_values = xarm.get_servo_angle(is_radian=True) 
        calibrator.vis_step(rgb_frame, np.array(joint_values)[:6], input_joint_names=None)   # first 6 joints
        time.sleep(0.1)


if __name__ == "__main__":
    # thirdview_realsense_xarm6_example()
    wrist_realsense_xarm6_example()
