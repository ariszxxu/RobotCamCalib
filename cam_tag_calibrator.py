from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
from loguru import logger
import yaml
import viser
from viser.extras import ViserUrdf
from dataclasses import dataclass
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# ---------- SE(3) helpers ----------
def inv_T(T):
    """4x4 rigid transform inverse."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti

def se3_mean(T_list, weights=None):
    """
    Simple SE(3) mean: translation weighted-average; rotation via SVD projection.
    T_list: list of (4,4)
    """
    n = len(T_list)
    if n == 0:
        raise ValueError("Empty list for SE(3) mean.")
    if weights is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(weights, dtype=float)
        w = w / np.sum(w)

    # translation mean
    t_mean = sum(w[i] * T_list[i][:3, 3] for i in range(n))

    # rotation mean (Procrustes projection)
    R_sum = np.zeros((3, 3))
    for i in range(n):
        R_sum += w[i] * T_list[i][:3, :3]
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:  # enforce proper rotation
        U[:, -1] *= -1
        R_mean = U @ Vt

    Tm = np.eye(4)
    Tm[:3, :3] = R_mean
    Tm[:3, 3]  = t_mean
    return Tm

# ---------- Core alternating solver ----------
def calibrate_cammount_and_tag(
    X_CamTag_list,        # (n,4,4)  B_i
    X_WorldCammount_list, # (n,4,4)
    X_WorldTagmount_list, # (n,4,4)
    X_CammountCam_init=None,   # (4,4)    initial guess for X_CammountCam
    weights=None,
    iters=15,
    tol_deg=1e-4,
    tol_trans=1e-6,
):
    """
    Solve for:
      - X_CammountCam  = (X_CamCammount)^{-1}
      - X_TagmountTag  = Y

    Model:
      B_i = X * A_i * Y
      where
        B_i = X_CamTag[i]
        A_i = inv(X_WorldCammount[i]) @ X_WorldTagmount[i]
        X   = X_CamCammount (unknown)
        Y   = X_TagmountTag  (unknown)

    Alternating updates:
      X <- mean_i( B_i * inv(Y) * inv(A_i) )
      Y <- mean_i( inv(A_i) * inv(X) * B_i )
    """
    X_CamTag_list        = np.asarray(X_CamTag_list)
    X_WorldCammount_list = np.asarray(X_WorldCammount_list)
    X_WorldTagmount_list = np.asarray(X_WorldTagmount_list)

    n = X_CamTag_list.shape[0]
    assert X_CamTag_list.shape == (n, 4, 4)
    assert X_WorldCammount_list.shape == (n, 4, 4)
    assert X_WorldTagmount_list.shape == (n, 4, 4)

    # Build A_i
    A_list = [inv_T(X_WorldCammount_list[i]) @ X_WorldTagmount_list[i] for i in range(n)]
    B_list = [X_CamTag_list[i] for i in range(n)]

    # Initialize: given X_CammountCam_init => X_init = inv(X_CammountCam_init)
    X = inv_T(X_CammountCam_init)     # X = X_CamCammount
    Y = np.eye(4)                     # start with identity if unknown

    def delta_pose(T_new, T_old):
        """Return small pose deltas for stopping: (deg, meters)."""
        R = T_new[:3, :3] @ T_old[:3, :3].T
        # angle from rotation matrix
        cosang = max(min((np.trace(R) - 1) / 2, 1.0), -1.0)
        ang = np.degrees(np.arccos(cosang))
        dt = np.linalg.norm(T_new[:3, 3] - T_old[:3, 3])
        return ang, dt

    for _ in range(iters):
        X_old, Y_old = X.copy(), Y.copy()

        # Update X with current Y
        Ts_X = [B_list[i] @ inv_T(Y) @ inv_T(A_list[i]) for i in range(n)]
        X = se3_mean(Ts_X, weights)

        # Update Y with current X
        Ts_Y = [inv_T(A_list[i]) @ inv_T(X) @ B_list[i] for i in range(n)]
        Y = se3_mean(Ts_Y, weights)

        # Convergence check
        ang_X, dt_X = delta_pose(X, X_old)
        ang_Y, dt_Y = delta_pose(Y, Y_old)
        if max(ang_X, ang_Y) < tol_deg and max(dt_X, dt_Y) < tol_trans:
            break

    # Return in your requested forms
    X_CammountCam = inv_T(X)   # desired
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
        if input_joint_names is None:
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
    output_file_path: Path = Path("extrinsics_calib_results.yaml")

    # apriltag parameters
    tag_size: float = 0.048
    tag_family: str = "tag36h11"

    # initial guesses 
    X_CammountCam_init: Optional[np.ndarray] = None  # (4,4) initial guess for camera mount to camera pose
    X_TagmountTag_init: Optional[np.ndarray] = None  # (4,4) initial guess for tag mount to tag pose


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
        self.X_CammountCam = np.eye(4) if self.config.X_CammountCam_init is None else self.config.X_CammountCam_init  # (4,4) camera mount to camera pose
        self.X_TagmountTag = np.eye(4) if self.config.X_TagmountTag_init is None else self.config.X_TagmountTag_init  # (4,4) tag mount to tag pose

        # add GUI buttons
        self.append_button = self.server.gui.add_button("click_and_append", "Append current poses")
        self.save_button = self.server.gui.add_button("click_and_save", "Save calibration results")
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

        if len(self.X_CamTag_list) >= 3:  # need at least 3 observations to solve 
            self.X_CammountCam, self.X_TagmountTag = calibrate_cammount_and_tag(
                self.X_CamTag_list,
                self.X_WorldCammount_list,
                self.X_WorldTagmount_list,
                X_CammountCam_init=self.config.X_CammountCam_init,
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

