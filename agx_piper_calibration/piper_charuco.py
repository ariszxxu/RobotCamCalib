"""New-board Piper observations and RobotCamCalib solve, without actuators.

Translations are metres; T_A_B maps frame B into frame A. Native RealSense
inverse Brown coefficients are deprojected by the installed geometry module,
never passed to OpenCV as forward Brown coefficients. No old hand-eye is used.
"""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np

DRIVER=Path('/home/agilex/wcx_gpt6_astra_agilex/src')
HERE=Path(__file__).resolve().parent
CORE_ROOT=HERE.parent if (HERE.parent/'robot_cam_calib').is_dir() else HERE/'RobotCamCalib'
sys.path.insert(0,str(DRIVER))
sys.path.insert(0,str(CORE_ROOT))
from robot_cam_calib.geometry import inv_T,transform_delta
from robot_cam_calib.hand_eye import HandEyeObservation,solve_hand_eye_robust,mean_transform

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v

def detect(folder,profiles):
    from agilex_control.geometry import pixel_rays
    folder=Path(folder);meta=read(folder/'metadata.json')
    matching=[p for p in profiles if p['serial']==meta['serial'] and p['width']==meta['color']['width'] and p['height']==meta['color']['height']]
    if len(matching)!=1:raise ValueError('Current factory profile identity missing')
    profile=matching[0];k=np.asarray(meta['color']['K']).reshape(3,3)
    if max(abs(profile['fx']-k[0,0]),abs(profile['fy']-k[1,1]),abs(profile['cx']-k[0,2]),abs(profile['cy']-k[1,2]))>.5 or not np.allclose(profile['coeffs'],meta['color']['D'],atol=1e-9):
        raise ValueError('Current ROS and factory camera models disagree')
    board=cv2.aruco.CharucoBoard((7,5),.04,.03,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50));board.setLegacyPattern(False)
    im=cv2.imread(str(folder/'rgb.jpg'))
    if im is None:raise ValueError('Missing captured RGB')
    corners,ids,markers,mids=cv2.aruco.CharucoDetector(board).detectBoard(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
    out={'serial':meta['serial'],'corner_count':0 if ids is None else len(ids),'metric_pose_eligible':False,
         'rgb_sha256':sha(folder/'rgb.jpg'),'metadata_sha256':sha(folder/'metadata.json'),
         'color_stamp':meta['color_stamp'],'actual_depth_registered_to_color':meta['depth_registered_to_rgb'],
         'depth_used_for_pnp':False,'old_extrinsics_used':False}
    if ids is None or len(ids)<8:return out
    points=board.getChessboardCorners()[ids.flatten()].astype(np.float64)
    pixels=corners.reshape(-1,2).astype(np.float64)
    if len(set(ids.flatten().tolist()))!=len(ids):raise ValueError('Duplicate corner identities')
    norm=pixel_rays(profile,pixels)[:,:2].astype(np.float64)
    solved=cv2.solvePnPGeneric(points,norm,np.eye(3),np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
    candidates=[]
    for r,t in zip(solved[1],solved[2]):
        rotation=cv2.Rodrigues(r)[0];camera=points@rotation.T+t.reshape(3)
        if np.any(camera[:,2]<=.05):continue
        residual=(camera[:,:2]/camera[:,2:]-norm)*[profile['fx'],profile['fy']]
        error=float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
        T=np.eye(4);T[:3,:3]=rotation;T[:3,3]=t.flatten()
        candidates.append((error,T,r,t))
    if not candidates:return out
    candidates.sort(key=lambda x:x[0]);best=candidates[0]
    if len(candidates)>1:
        angle,_=transform_delta(best[1],candidates[1][1])
        if candidates[1][0]-best[0]<.10 and angle>5:
            out.update(reason='planar_pose_ambiguous',hypothesis_errors_pixel_equivalent=[x[0] for x in candidates]);return out
    r,t=cv2.solvePnPRefineLM(points,norm,np.eye(3),np.zeros(5),best[2],best[3])
    rotation=cv2.Rodrigues(r)[0];camera=points@rotation.T+t.reshape(3)
    if np.any(camera[:,2]<=.05):return out
    residual=(camera[:,:2]/camera[:,2:]-norm)*[profile['fx'],profile['fy']]
    error=float(np.sqrt(np.mean(np.sum(residual**2,axis=1))))
    T=np.eye(4);T[:3,:3]=rotation;T[:3,3]=t.flatten()
    out.update(T_camera_board_m=T.tolist(),normalized_ray_reprojection_rmse_pixel_equivalent=error,
               distortion_model=profile['model'],corner_ids=ids.flatten().tolist(),corners_px=pixels.tolist(),
               metric_pose_eligible=error<=1.,error_is_normalized_ray_pixel_equivalent=True)
    return out

def make_sample(raw,config,profiles,index,split):
    from agilex_control.kinematics import Kinematics
    if split not in ('train','holdout'):raise ValueError('Split must be fixed before capture')
    raw=Path(raw);before=read(raw/'before-state.json')['data'];after=read(raw/'after-state.json')['data']
    if before['control_frames'] or after['control_frames']:raise ValueError('Unexpected writer during snapshot')
    kin=Kinematics(config);base={}
    for arm in ('left','right'):
        a=before['states'][arm];b=after['states'][arm]
        if not a['complete'] or not b['complete'] or a['stale'] or b['stale'] or a['error_code'] or b['error_code']:raise ValueError('Incomplete feedback')
        q1=np.array(a['joint_deg']);q2=np.array(b['joint_deg'])
        if np.max(abs(q1-q2))>.02:raise ValueError('Joint stability worse than .02 degree')
        t=kin.transform((q1+q2)/2);t[:3,3]/=1000;base[arm]=t.tolist()
    observation=read(raw/'observation.json')
    if observation['camera_stamp_span_s']>.05:raise ValueError('Three-camera snapshot span exceeds 50 ms')
    cams={role:detect(raw/role,profiles) for role in ('front','left_wrist','right_wrist')}
    return {'index':index,'split':split,'raw':str(raw),'source':'new_physical_7x5_40mm_charuco_fixed_board',
            'T_base_link6_m':base,'cameras':cams,'robot_motion_commands_sent_by_collector':0,
            'old_observations_used':False,'simulator_information_used':False}

def solve_arm(samples,arm):
    rows=[s for s in samples if s['active_arm']==arm and s['cameras'][arm+'_wrist'].get('metric_pose_eligible')]
    train=[s for s in rows if s['split']=='train'];hold=[s for s in rows if s['split']=='holdout']
    if len(train)<20 or len(hold)<10:raise ValueError('Need at least 20 training and 10 predetermined holdout poses for '+arm)
    obs=[HandEyeObservation(s['index'],np.array(s['T_base_link6_m'][arm]),np.array(s['cameras'][arm+'_wrist']['T_camera_board_m'])) for s in train]
    result=solve_hand_eye_robust(obs,min_samples=14)
    x=result['T_gripper_camera'];board=result['T_base_target_mean']
    residuals=[]
    for s in hold:
        predicted=np.array(s['T_base_link6_m'][arm])@x@np.array(s['cameras'][arm+'_wrist']['T_camera_board_m'])
        rot,trans=transform_delta(board,predicted)
        residuals.append({'index':s['index'],'rotation_deg':rot,'translation_m':trans})
    cv=result['cross_validation'];checks={
        'rotation_rank3':result['excitation']['relative_rotation_rank']==3,
        'holdout_translation_max_le4mm':max(s['translation_m'] for s in residuals)<=.004,
        'holdout_rotation_max_le1_5deg':max(s['rotation_deg'] for s in residuals)<=1.5,
        'odd_even_translation_le5mm':cv.get('available',False) and cv.get('translation_delta_m',1) <=.005,
        'odd_even_rotation_le1_5deg':cv.get('available',False) and cv.get('rotation_delta_deg',999)<=1.5,
        'camera_mount_translation_norm_lt300mm':np.linalg.norm(x[:3,3])<.3}
    return {'arm':arm,'T_link6_camera_m':x,'T_base_board_m':board,'training_count':len(train),
            'holdout_count':len(hold),'holdout_residuals':residuals,'qualified':all(checks.values()),'checks':checks,
            'solver':result,'contact_tcp_qualified':False,'accepted_for_control':False}

def solve_session(samples):
    left=solve_arm(samples,'left');right=solve_arm(samples,'right')
    front=[np.array(s['cameras']['front']['T_camera_board_m']) for s in samples if s['split']=='train' and s['cameras']['front'].get('metric_pose_eligible')]
    if len(front)<20:raise ValueError('Insufficient fixed-board front training observations')
    front_board=mean_transform(front)
    front_residuals=[transform_delta(front_board,np.array(s['cameras']['front']['T_camera_board_m'])) for s in samples if s['split']=='holdout' and s['cameras']['front'].get('metric_pose_eligible')]
    if len(front_residuals)<10:raise ValueError('Insufficient predetermined front holdouts')
    fixed=max(t for r,t in front_residuals)<=.004 and max(r for r,t in front_residuals)<=1.5
    bL=left['T_base_board_m'];bR=right['T_base_board_m']
    return clean({'status':'candidate_requires_independent_physical_validation','qualified':left['qualified'] and right['qualified'] and fixed,
        'left':left,'right':right,'T_leftbase_front_optical_m':bL@inv_T(front_board),
        'T_rightbase_leftbase_m':bR@inv_T(bL),'front_heldout_fixed_board_residuals':front_residuals,
        'front_board_fixed_check':fixed,'privileged_simulator_information_used':False,
        'old_calibration_used':False,'contact_tcp_included':False,'accepted_for_control':False})
