from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
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


# ---------- Lie helpers ----------
def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=float)

def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def so3_log(R):
    # numerically stable log
    cos_th = (np.trace(R)-1.0)/2.0
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-12:
        return np.array([0.,0.,0.])
    w_hat = (R - R.T) / (2*np.sin(th))
    return np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]]) * th

def dexp_inv(w):
    # inverse of differential of exp on SO(3) (Eq. (44) in paper)
    th = np.linalg.norm(w)
    I = np.eye(3)
    if th < 1e-8:
        return I - 0.5*skew(w) + (1/12.0)*(skew(w)@skew(w))
    A = 0.5
    B = (1.0/(th**2))*(1 - th*np.cos(th)/(2*np.sin(th)))
    W = skew(w)
    return I - A*W + B*(W@W)

def se3_exp(dw, dq):
    R = so3_exp(dw)
    # first-order adequate for small steps; Jacobian could be added if needed
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=dq
    return T

def se3_log(T):
    w = so3_log(T[:3,:3])
    q = T[:3,3]
    return w, q

def inv_T(T):
    R = T[:3,:3]; t=T[:3,3]
    Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]= -R.T@t
    return Ti

def pose_err_deg_m(T_est, T_gt):
    d = inv_T(T_gt) @ T_est
    w = so3_log(d[:3,:3])
    return np.degrees(np.linalg.norm(w)), np.linalg.norm(d[:3,3])

# --- Huber on Mahalanobis^2 ---
def huber_weight_mahal(r2, delta2):
    w = np.ones_like(r2)
    mask = r2 > delta2
    w[mask] = np.sqrt(delta2 / r2[mask])
    return w

def compose_left(T, xi):
    # left-multiplicative update: T_new = Exp(xi) * T
    dw = xi[:3]; dq = xi[3:]
    return se3_exp(dw, dq) @ T

def pack_r(w, p):
    return np.hstack([w, p])  # 6,

# --- utilities used above (same as before) ---
def _wahba(Rs_src, Rs_tgt):
    H = np.zeros((3,3))
    for Rsrc,Rtgt in zip(Rs_src, Rs_tgt):
        H += Rtgt @ Rsrc.T
    U,S,Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt
    return R

def _solve_Y_given_X(A_list, B_list, X):
    RX = X[:3,:3]; tX = X[:3,3]
    R_targets = []; R_sources = []
    for A,B in zip(A_list, B_list):
        R_targets.append(A[:3,:3] @ RX)
        R_sources.append(B[:3,:3])
    RY = _wahba(R_sources, R_targets)
    # t: A tX + a ≈ RY b + tY
    M=[]; v=[]
    for A,B in zip(A_list, B_list):
        M.append(np.eye(3))
        v.append((A[:3,:3]@tX + A[:3,3]) - (RY@B[:3,3]))
    M = np.vstack(M); v = np.hstack(v)
    tY, *_ = np.linalg.lstsq(M, v, rcond=None)
    T = np.eye(4); T[:3,:3]=RY; T[:3,3]=tY
    return T

def _solve_X_given_Y(A_list, B_list, Y):
    RY = Y[:3,:3]; tY = Y[:3,3]
    R_sources = []; R_targets = []
    for A,B in zip(A_list, B_list):
        R_sources.append(A[:3,:3])
        R_targets.append(RY @ B[:3,:3])
    RX = _wahba(R_sources, R_targets)
    # A tX + a ≈ RY b + tY
    M=[]; v=[]
    for A,B in zip(A_list, B_list):
        M.append(A[:3,:3])
        v.append((RY@B[:3,3] + tY) - A[:3,3])
    M = np.vstack(M); v = np.hstack(v)
    tX, *_ = np.linalg.lstsq(M, v, rcond=None)
    T = np.eye(4); T[:3,:3]=RX; T[:3,3]=tX
    return T

def block_diag(blocks):
    # simple block-diagonal constructor
    r = sum(b.shape[0] for b in blocks)
    c = sum(b.shape[1] for b in blocks)
    out = np.zeros((r,c))
    i = j = 0
    for B in blocks:
        rr, cc = B.shape
        out[i:i+rr, j:j+cc] = B
        i += rr; j += cc
    return out


# ---------- PROBABILISTIC MLE (Config-3) with GN + numerical Jacobians ----------
def calibrate_cammount_and_tag_prob(
    X_CamTag_list: np.ndarray,        # B_i (n,4,4)
    X_WorldCammount_list: np.ndarray, # (n,4,4)
    X_WorldTagmount_list: np.ndarray, # (n,4,4)
    Sigma_w_list: Optional[List[np.ndarray]] = None,  # rot cov of B_i
    Sigma_p_list: Optional[List[np.ndarray]] = None,  # trans cov of B_i
    max_iters: int = 200,
    huber_delta_rot_deg: float = 3.0,
    huber_delta_trans: float = 0.01,
    eps_dx: float = 1e-8,
    eps_stop_deg: float = 1e-6,
    eps_stop_trans: float = 1e-8,
    damping: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:

    n = X_CamTag_list.shape[0]
    assert X_CamTag_list.shape == (n,4,4)
    assert X_WorldCammount_list.shape == (n,4,4)
    assert X_WorldTagmount_list.shape == (n,4,4)

    # Build A_i (trusted) and B_i (noisy)
    A_list = [inv_T(X_WorldCammount_list[i]) @ X_WorldTagmount_list[i] for i in range(n)]
    B_list = [X_CamTag_list[i] for i in range(n)]

    # Cov defaults
    if Sigma_w_list is None:
        Sigma_w_list = [ (np.radians(3.0)**2) * np.eye(3) for _ in range(n) ]
    if Sigma_p_list is None:
        Sigma_p_list = [ (0.01**2) * np.eye(3) for _ in range(n) ]
    SwI = [np.linalg.inv(Sw) for Sw in Sigma_w_list]
    SpI = [np.linalg.inv(Sp) for Sp in Sigma_p_list]

    # Simple LS init (same as before)
    X = np.eye(4)
    Y = _solve_Y_given_X(A_list, B_list, X)
    X = _solve_X_given_Y(A_list, B_list, Y)
    Y = _solve_Y_given_X(A_list, B_list, X)

    # Robust params
    delta_w2 = (np.radians(huber_delta_rot_deg))**2
    delta_p2 = (huber_delta_trans)**2

    # parameter vector is [xi_X (6), xi_Y (6)] but we update via left-mult on (X,Y) directly
    for it in range(max_iters):
        # assemble residuals and block-diagonal weight inverses
        r_all = []
        W_inv_blocks = []
        for i,(A,B,Sw,Sp) in enumerate(zip(A_list, B_list, SwI, SpI)):
            # residual transform: E_i = X^{-1} A^{-1} Y B  (should be ~ Identity)
            Ei = inv_T(X) @ inv_T(A) @ Y @ B
            w_i, p_i = se3_log(Ei)

            # Mahalanobis^2 per part
            r2w = w_i.T @ Sw @ w_i
            r2p = p_i.T @ Sp @ p_i
            ww = huber_weight_mahal(np.array([r2w]), delta_w2)[0]
            wp = huber_weight_mahal(np.array([r2p]), delta_p2)[0]

            # stack 6x1 residual (apply sqrt weights inside W_inv later)
            r_i = pack_r(w_i, p_i)
            r_all.append(r_i)

            # Build W^{-1} per sample (6x6), using robust weights
            # We put W^{-1} = diag( ww*Sw, wp*Sp )
            Wi_inv = np.block([
                [ww*Sw,            np.zeros((3,3))],
                [np.zeros((3,3)),  wp*Sp         ]
            ])
            W_inv_blocks.append(Wi_inv)

        r_all = np.hstack(r_all)            # (6n,)
        W_inv = block_diag(W_inv_blocks)    # (6n x 6n)

        # Numerically build J (6n x 12): columns for [δx(6), δy(6)]
        J = np.zeros((6*n, 12))
        # basis for se3 perturb
        E6 = np.eye(6)

        # finite-diff step size
        h = eps_dx

        # columns 0..5: effect of δx on residuals
        for k in range(6):
            xi = np.zeros(6); xi[k] = h
            Xp = compose_left(X, xi)  # left-mult perturb
            col = []
            for i,(A,B) in enumerate(zip(A_list, B_list)):
                Ei_p = inv_T(Xp) @ inv_T(A) @ Y @ B
                w_p, p_p = se3_log(Ei_p)
                Ei = inv_T(X) @ inv_T(A) @ Y @ B
                w_0, p_0 = se3_log(Ei)
                dr = pack_r(w_p - w_0, p_p - p_0) / h
                col.append(dr)
            J[:, k] = np.hstack(col)

        # columns 6..11: effect of δy on residuals
        for k in range(6):
            xi = np.zeros(6); xi[k] = h
            Yp = compose_left(Y, xi)
            col = []
            for i,(A,B) in enumerate(zip(A_list, B_list)):
                Ei_p = inv_T(X) @ inv_T(A) @ Yp @ B
                w_p, p_p = se3_log(Ei_p)
                Ei = inv_T(X) @ inv_T(A) @ Y @ B
                w_0, p_0 = se3_log(Ei)
                dr = pack_r(w_p - w_0, p_p - p_0) / h
                col.append(dr)
            J[:, 6+k] = np.hstack(col)

        # Solve Gauss–Newton normal equations with damping (Levenberg)
        # (J^T W^{-1} J + λI) Δ = - J^T W^{-1} r
        JT_Wi = J.T @ W_inv
        H = JT_Wi @ J
        g = JT_Wi @ r_all
        H += damping * np.eye(12)
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(H) @ g

        # split and update
        dx = delta[:6]; dy = delta[6:]
        X = compose_left(X, dx)
        Y = compose_left(Y, dy)

        if verbose:
            rm = np.sqrt(np.mean(r_all**2))
            print(f"[it {it:02d}] |delta_x|={np.linalg.norm(dx):.2e}, |delta_y|={np.linalg.norm(dy):.2e}, rmse={rm:.3e}")

        # stopping
        if np.linalg.norm(dx[:3]) < np.radians(eps_stop_deg) and \
           np.linalg.norm(dx[3:]) < eps_stop_trans and \
           np.linalg.norm(dy[:3]) < np.radians(eps_stop_deg) and \
           np.linalg.norm(dy[3:]) < eps_stop_trans:
            break

    # report residuals on Ai X ≈ Y Bi  (identity if perfect)
    rot_err_deg, trans_err = [], []
    for A,B in zip(A_list, B_list):
        E = inv_T(A @ X) @ (Y @ B)
        wE, qE = se3_log(E)
        rot_err_deg.append(np.degrees(np.linalg.norm(wE)))
        trans_err.append(np.linalg.norm(qE))

    info = {
        "iters": it+1,
        "rot_err_deg_mean": float(np.mean(rot_err_deg)),
        "rot_err_deg_med": float(np.median(rot_err_deg)),
        "rot_err_deg_max": float(np.max(rot_err_deg)),
        "trans_err_mean": float(np.mean(trans_err)),
        "trans_err_med": float(np.median(trans_err)),
        "trans_err_max": float(np.max(trans_err)),
    }

    X_CammountCam = inv_T(X)
    X_TagmountTag = Y
    return X_CammountCam, X_TagmountTag, info


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
            self.X_CammountCam, self.X_TagmountTag, info = calibrate_cammount_and_tag_prob(
                np.asarray(self.X_CamTag_list),
                np.asarray(self.X_WorldCammount_list),
                np.asarray(self.X_WorldTagmount_list),
                max_iters=200,
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


# ---------- random SE(3) sampling ----------
def rand_unit_vec(rng):
    v = rng.normal(size=3)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def sample_so3(rng, max_deg=30.0):
    ang = np.radians(rng.uniform(-max_deg, max_deg))
    axis = rand_unit_vec(rng)
    return so3_exp(axis * ang)

def sample_se3(rng, rot_deg=30.0, trans_range=0.3):
    R = sample_so3(rng, rot_deg)
    t = rng.uniform(-trans_range, trans_range, size=3)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
    return T

# ---------- build synthetic dataset ----------
def make_synth_dataset(
    n=30,
    seed=0,
    Xgt_rot_deg=20.0, Xgt_trans=0.05,
    Ygt_rot_deg=20.0, Ygt_trans=0.05,
    A_rot_deg=50.0,  A_trans=0.5,
    noise_rot_deg=2.0, noise_trans=0.005,
    anisotropic=False
):
    rng = np.random.default_rng(seed)

    # ground-truth X and Y
    X_gt = sample_se3(rng, rot_deg=Xgt_rot_deg, trans_range=Xgt_trans)
    Y_gt = sample_se3(rng, rot_deg=Ygt_rot_deg, trans_range=Ygt_trans)

    # noiseless A_i
    A_list = [sample_se3(rng, rot_deg=A_rot_deg, trans_range=A_trans) for _ in range(n)]

    # true B_i from AX = YB  =>  B_i_true = Y^{-1} A_i X
    B_true_list = [inv_T(Y_gt) @ A @ X_gt for A in A_list]

    # add noise on B only (config 3)
    X_CamTag_list = []
    Sigma_w_list, Sigma_p_list = [], []

    for B_true in B_true_list:
        if anisotropic:
            # example: different variances on axes
            std_w = np.radians(np.array([noise_rot_deg, noise_rot_deg*0.5, noise_rot_deg*2.0]))
            std_p = np.array([noise_trans, noise_trans*0.5, noise_trans*2.0])
            Sw = np.diag(std_w**2)
            Sp = np.diag(std_p**2)
        else:
            Sw = (np.radians(noise_rot_deg)**2) * np.eye(3)
            Sp = (noise_trans**2) * np.eye(3)

        # draw noise and compose on the RIGHT: B_meas = B_true * Exp(ξ)
        w_noise = rng.multivariate_normal(np.zeros(3), Sw)
        p_noise = rng.multivariate_normal(np.zeros(3), Sp)
        Xi = se3_exp(w_noise, p_noise)
        B_meas = B_true @ Xi

        X_CamTag_list.append(B_meas)
        Sigma_w_list.append(Sw)
        Sigma_p_list.append(Sp)

    X_CamTag_list = np.stack(X_CamTag_list, axis=0)

    # To pass A_i through your function, set:
    #   A_i = inv(X_WorldCammount_i) @ X_WorldTagmount_i
    # Choose X_WorldCammount_i = I, X_WorldTagmount_i = A_i
    X_WorldCammount_list = np.stack([np.eye(4) for _ in range(n)], axis=0)
    X_WorldTagmount_list  = np.stack(A_list, axis=0)

    return (X_gt, Y_gt,
            X_CamTag_list, X_WorldCammount_list, X_WorldTagmount_list,
            Sigma_w_list, Sigma_p_list)




# --------- random SE(3) sampling ----------
def rand_unit_vec(rng):
    v = rng.normal(size=3)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def sample_so3(rng, max_deg=30.0):
    ang = np.radians(rng.uniform(-max_deg, max_deg))
    axis = rand_unit_vec(rng)
    return so3_exp(axis * ang)

def sample_se3(rng, rot_deg=30.0, trans_range=0.3):
    R = sample_so3(rng, rot_deg)
    t = rng.uniform(-trans_range, trans_range, size=3)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
    return T

# --------- synthetic dataset (A noiseless, B noisy) ----------
def make_synth_dataset_for_original(
    n=30,
    seed=0,
    Xgt_rot_deg=20.0, Xgt_trans=0.05,
    Ygt_rot_deg=20.0, Ygt_trans=0.05,
    A_rot_deg=50.0,  A_trans=0.5,
    noise_rot_deg=2.0, noise_trans=0.005,
    anisotropic=False
):
    rng = np.random.default_rng(seed)

    # ground-truth X and Y
    X_gt = sample_se3(rng, rot_deg=Xgt_rot_deg, trans_range=Xgt_trans)
    Y_gt = sample_se3(rng, rot_deg=Ygt_rot_deg, trans_range=Ygt_trans)

    # noiseless A_i
    A_list = [sample_se3(rng, rot_deg=A_rot_deg, trans_range=A_trans) for _ in range(n)]

    # true B_i from AX = YB  =>  B_i_true = Y^{-1} A_i X
    B_true_list = [inv_T(Y_gt) @ A @ X_gt for A in A_list]

    # add noise on B only (compose on the right)
    X_CamTag_list = []
    for B_true in B_true_list:
        if anisotropic:
            std_w = np.radians(np.array([noise_rot_deg, noise_rot_deg*0.5, noise_rot_deg*2.0]))
            std_p = np.array([noise_trans, noise_trans*0.5, noise_trans*2.0])
            w_noise = rng.normal(0, std_w, size=3)
            p_noise = rng.normal(0, std_p, size=3)
        else:
            w_noise = rng.normal(0, np.radians(noise_rot_deg), size=3)
            p_noise = rng.normal(0, noise_trans, size=3)
        Xi = se3_exp(w_noise, p_noise)
        B_meas = B_true @ Xi
        X_CamTag_list.append(B_meas)
    X_CamTag_list = np.stack(X_CamTag_list, axis=0)

    # Your function expects:
    #   A_i = inv(X_WorldCammount_i) @ X_WorldTagmount_i
    # We can set X_WorldCammount_i = I, X_WorldTagmount_i = A_i
    X_WorldCammount_list = np.stack([np.eye(4) for _ in range(n)], axis=0)
    X_WorldTagmount_list  = np.stack(A_list, axis=0)

    return X_gt, Y_gt, X_CamTag_list, X_WorldCammount_list, X_WorldTagmount_list


# --------- optional: side-by-side compare with probabilistic solver ----------
def test_solver(n=30, seed=0, anisotropic=False, noise_rot_deg=2.0, noise_trans=0.005):

    # Build the same dataset again to keep identical randomness
    (X_gt, Y_gt,
     X_CamTag_list, X_WorldCammount_list, X_WorldTagmount_list) = \
        make_synth_dataset_for_original(
            n=n, seed=seed, anisotropic=anisotropic,
            noise_rot_deg=noise_rot_deg, noise_trans=noise_trans
        )

    # Isotropic default covariances inside the function are fine for this test
    t0 = time.perf_counter()
    X_CammountCam_prob, Y_prob, info_prob = calibrate_cammount_and_tag_prob(
        X_CamTag_list, X_WorldCammount_list, X_WorldTagmount_list,
        Sigma_w_list=None, Sigma_p_list=None,  # or pass anisotropic covs if you want
        max_iters=3000, 
        huber_delta_rot_deg=3.0, huber_delta_trans=0.01,
        verbose=False,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0

    X_est_prob = inv_T(X_CammountCam_prob)
    rot_err_X_deg, trans_err_X = pose_err_deg_m(X_est_prob, X_gt)
    rot_err_Y_deg, trans_err_Y = pose_err_deg_m(Y_prob, Y_gt)

    print("===== Probabilistic (Config-3, Huber) =====")
    print(f"samples (n):                 {n}")
    print(f"noise (rot deg, trans m):    ({noise_rot_deg:.3f}, {noise_trans:.4f})   anisotropic={anisotropic}")
    print(f"inference time (s):          {elapsed:.4f}")
    print(f"iters:                       {info_prob['iters']}")
    print(f"[X] rot err (deg):           {rot_err_X_deg:.4f}")
    print(f"[X] trans err (m):           {trans_err_X:.6f}")
    print(f"[Y] rot err (deg):           {rot_err_Y_deg:.4f}")
    print(f"[Y] trans err (m):           {trans_err_Y:.6f}")
    print(f"residual mean (rot deg):     {info_prob['rot_err_deg_mean']:.4f}")
    print(f"residual mean (trans m):     {info_prob['trans_err_mean']:.6f}")
    print("=============================================================\n")

    return {
            "prob": {
                "elapsed_s": elapsed,
                "rot_err_X_deg": rot_err_X_deg,
                "trans_err_X_m": trans_err_X,
                "rot_err_Y_deg": rot_err_Y_deg,
                "trans_err_Y_m": trans_err_Y,
                "info": info_prob
            }}

if __name__ == "__main__":
    # A few quick runs
    # _ = test_original_solver(n=30, seed=42, anisotropic=False, noise_rot_deg=2.0, noise_trans=0.005)
    # _ = test_original_solver(n=10,  seed=7,  anisotropic=False, noise_rot_deg=2.0, noise_trans=0.005)
    # _ = test_original_solver(n=30, seed=99, anisotropic=True,  noise_rot_deg=3.0, noise_trans=0.01)

    # Side-by-side (optional, only if you imported the probabilistic version into extr_calib.py)
    _ = test_solver(n=30, seed=3, anisotropic=False, noise_rot_deg=20.0, noise_trans=0.03)


