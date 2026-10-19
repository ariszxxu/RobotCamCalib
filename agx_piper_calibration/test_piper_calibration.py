"""Pure calibration convention/holdout/PnP checks; no robot or camera I/O."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from piper_charuco import detect,solve_session
from robot_cam_calib.geometry import make_T,inv_T,transform_delta

def synthetic():
    rng=np.random.default_rng(20261019)
    left_x=make_T(Rotation.from_rotvec([.20,-.13,.1]).as_matrix(),[.04,-.02,.07])
    right_x=make_T(Rotation.from_rotvec([-.15,.21,-.1]).as_matrix(),[.02,.03,.09])
    left_board=make_T(Rotation.from_rotvec([-.1,.04,.2]).as_matrix(),[.40,.1,.02])
    right_left=make_T(Rotation.from_rotvec([0,0,.04]).as_matrix(),[0,.65,.01])
    right_board=right_left@left_board
    front=make_T(Rotation.from_rotvec([.4,-.1,.2]).as_matrix(),[.2,-.1,.65])
    front_board=inv_T(front)@left_board
    rows=[]
    for arm,x,board in [('left',left_x,left_board),('right',right_x,right_board)]:
        for i in range(40):
            g=make_T(Rotation.from_rotvec(rng.uniform(-.9,.9,3)).as_matrix(),rng.uniform([-.2,-.2,.25],[.55,.35,.75]))
            c=inv_T(x)@inv_T(g)@board
            rows.append({'active_arm':arm,'index':i,'split':'holdout' if i%4==3 else 'train',
                         'T_base_link6_m':{arm:g},'cameras':{
                             arm+'_wrist':{'metric_pose_eligible':True,'T_camera_board_m':c},
                             'front':{'metric_pose_eligible':True,'T_camera_board_m':front_board}}})
    return rows,left_x,right_x,front,right_left

class CalibrationTests(unittest.TestCase):
    def test_three_cameras_and_interbase_conventions(self):
        rows,xl,xr,f,rl=synthetic();r=solve_session(rows)
        self.assertTrue(r['qualified']);self.assertFalse(r['accepted_for_control'])
        for expected,actual in [(xl,r['left']['T_link6_camera_m']),(xr,r['right']['T_link6_camera_m']),
                                (f,r['T_leftbase_front_optical_m']),(rl,r['T_rightbase_leftbase_m'])]:
            angle,position=transform_delta(expected,np.array(actual))
            self.assertLess(angle,1e-6);self.assertLess(position,1e-8)

    def test_holdout_is_not_fitted(self):
        rows,*_=synthetic();baseline=solve_session(rows)
        for row in rows:
            if row['split']=='holdout':row['cameras'][row['active_arm']+'_wrist']['T_camera_board_m'][0,3]+=.015
        changed=solve_session(rows)
        self.assertFalse(changed['qualified'])
        np.testing.assert_allclose(baseline['left']['T_link6_camera_m'],changed['left']['T_link6_camera_m'],atol=1e-12)
        np.testing.assert_allclose(baseline['right']['T_link6_camera_m'],changed['right']['T_link6_camera_m'],atol=1e-12)

    def test_board_movement_cannot_pass(self):
        rows,*_=synthetic()
        for row in rows:
            if row['split']=='holdout':row['cameras']['front']['T_camera_board_m']=np.array(row['cameras']['front']['T_camera_board_m']).copy();row['cameras']['front']['T_camera_board_m'][1,3]+=.02
        result=solve_session(rows);self.assertFalse(result['qualified']);self.assertFalse(result['front_board_fixed_check'])

    def test_no_wrist_view_is_not_a_calibration(self):
        rows,*_=synthetic()
        for row in rows:row['cameras'][row['active_arm']+'_wrist']['metric_pose_eligible']=False
        with self.assertRaises(ValueError):solve_session(rows)

    def test_charuco_pnp_recovers_known_rendered_pose(self):
        board=cv2.aruco.CharucoBoard((7,5),.04,.03,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50));board.setLegacyPattern(False)
        pattern=board.generateImage((1400,1000),marginSize=0)
        K=np.array([[530.,0,320],[0,530.,240],[0,0,1.]])
        rot=np.array([.15,-.12,.08]);t=np.array([-.13,-.10,.55])
        corners=np.array([[0,0,0],[.28,0,0],[.28,.20,0],[0,.20,0]])
        image_points=cv2.projectPoints(corners,rot,t,K,np.zeros(5))[0].reshape(-1,2).astype('float32')
        transform=cv2.getPerspectiveTransform(np.array([[0,0],[1399,0],[1399,999],[0,999]],dtype='float32'),image_points)
        im=cv2.warpPerspective(pattern,transform,(640,480),borderValue=255)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cv2.imwrite(str(root/'rgb.jpg'),im)
            profile={'serial':'synthetic','width':640,'height':480,'fx':530.,'fy':530.,'cx':320.,'cy':240.,'model':'none','coeffs':[0.]*5}
            meta={'serial':'synthetic','color':{'width':640,'height':480,'K':K.flatten().tolist(),'D':[0.]*5},'color_stamp':0,'depth_registered_to_rgb':False}
            (root/'metadata.json').write_text(json.dumps(meta))
            result=detect(root,[profile]);self.assertTrue(result['metric_pose_eligible'])
            expected=make_T(cv2.Rodrigues(rot)[0],t)
            angle,position=transform_delta(expected,np.array(result['T_camera_board_m']))
            self.assertLess(angle,1.);self.assertLess(position,.003)

if __name__=='__main__':unittest.main(verbosity=2)
