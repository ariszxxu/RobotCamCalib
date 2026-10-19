"""Loopback Piper GUI using RobotCamCalib. Camera/feedback reads only; no actuator API."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np
from piper_charuco import clean, detect, read, sha, solve_arm, solve_session
from robot_cam_calib.geometry import transform_delta

H = Path(__file__).resolve().parent
CONFIG = Path('/home/agilex/yichen/rsi_sim2real/releases/agx-stack-cube-20261018/existing-hardware-binding.json')
PROFILES = CONFIG.parent/'calibration-v2/factory-current-color-intrinsics.json'
ROLES = ('front','left_wrist','right_wrist')

def verified_rows(root):
    identity_file=root/'identity.json'
    if not identity_file.exists():return []
    identity=read(identity_file); rows=[]; poses={'left':[], 'right':[]}
    for path in sorted(root.glob('*/sample.json')):
        s=read(path);arm=s['active_arm']
        if arm not in poses or s['identity']!=identity:raise ValueError('Sample identity mismatch')
        if s['split']!=('holdout' if s['index']%4==3 else 'train'):raise ValueError('Changed holdout split')
        for rel,digest in s['source_files_sha256'].items():
            f=(path.parent/rel).resolve()
            if not f.is_relative_to(path.parent.resolve()) or sha(f)!=digest:raise ValueError('Raw sample evidence changed')
        # Invalid views must not suppress a later valid sample at the same pose.
        if not s['cameras'][arm+'_wrist'].get('metric_pose_eligible'):continue
        pose=np.asarray(s['T_base_link6_m'][arm])
        if any(r<5 and t<.010 for r,t in (transform_delta(old,pose) for old in poses[arm])):continue
        poses[arm].append(pose);rows.append(s)
    return rows

def next_index(root,arm):
    ids=[int(p.name.split('-')[1]) for p in root.glob(arm+'-*') if re.fullmatch(arm+r'-\d+',p.name)]
    return max(ids,default=-1)+1

class AutoGate:
    """Measured-pose stability, geometric novelty and bounded failed-capture retries."""
    def __init__(self):
        self.anchor=None;self.since=0;self.attempt_q=None;self.attempts=0;self.last_attempt=0
    def check(self,state,view,pose,known,drift,now):
        q=np.asarray(state['joint_deg'],dtype=float)
        if not state['complete'] or state['stale'] or state['error_code'] or state['arm_status'] or any(state['enabled_drive_bits']):
            self.anchor=None;return False,'等待右臂失能且反馈正常'
        if drift>.02:
            self.anchor=None;return False,'正在移动，等待停稳'
        if not view.get('metric_pose_eligible'):
            self.anchor=None;return False,'等待右腕看清标定板（至少 8 个角点且质量合格）'
        if self.anchor is None or np.max(abs(q-self.anchor))>.02:
            self.anchor=q.copy();self.since=now;return False,'检测到新姿态，等待持续停稳'
        if now-self.since<1:return False,'等待停稳至少 1 秒'
        if any(r<5 and t<.010 for r,t in (transform_delta(old,pose) for old in known)):
            return False,'此姿态已采集，请换一个位置或角度'
        if self.attempt_q is not None and np.max(abs(q-self.attempt_q))<1 and self.attempts>=2:
            return False,'此姿态采样未通过，请稍作调整后停稳'
        if now-self.last_attempt<6:return False,'正在检查上一组采样结果'
        return True,'姿态稳定且图像合格，自动保存中'
    def submitted(self,q,now):
        q=np.asarray(q,dtype=float)
        self.attempts=self.attempts+1 if self.attempt_q is not None and np.max(abs(q-self.attempt_q))<1 else 1
        self.attempt_q=q.copy();self.last_attempt=now

class App:
    def __init__(self,session):
        if not re.fullmatch(r'manual-[A-Za-z0-9_-]+',session):raise ValueError('Invalid session')
        self.session=session;self.root=H/'sessions'/session;self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.request_lock=threading.Lock();self.job=None;self.preview={};self.images={};self.result=None
        self.counts={a:{'train':0,'holdout':0} for a in ('left','right')}
        self.message='手动摆姿采样。保持标定板固定，移动后完全停稳再保存。'
        self.auto_enabled=False;self.board_fixed=False;self.auto_reason='待确认标定板固定后开启';self.gate=AutoGate()
        from agilex_control.kinematics import Kinematics
        self.kin=Kinematics(read(CONFIG))
        self.token=secrets.token_urlsafe(24);self.last_preview=0;self.heartbeat=0
        threading.Thread(target=self.preview_loop,daemon=True).start()

    def snapshot(self):
        if self.job!='capture':
            rows=verified_rows(self.root)
            self.counts={arm:{kind:sum(s['active_arm']==arm and s['split']==kind for s in rows) for kind in ('train','holdout')} for arm in ('left','right')}
        return {'session':self.session,'job':self.job,'message':self.message,'preview':self.preview,
                'counts':self.counts,'result':self.result,'robot_commands_sent':0,
                'auto_enabled':self.auto_enabled,'board_fixed':self.board_fixed,'auto_reason':self.auto_reason}

    def set_auto(self,enabled,board_fixed):
        if enabled and not board_fixed:raise ValueError('Confirm fixed board before automatic sampling')
        self.auto_enabled=bool(enabled);self.board_fixed=bool(board_fixed)
        self.gate=AutoGate();self.auto_reason='等待新鲜画面与稳定姿态' if enabled else '自动采样已暂停'
        (self.root/'auto-sampling.json').write_text(json.dumps({'enabled':self.auto_enabled,'board_fixed':self.board_fixed,'epoch':time.time(),'robot_commands_sent':0}))

    def preview_loop(self):
        while True:
            time.sleep(.2)
            if self.job or time.time()-self.heartbeat>15 or time.time()-self.last_preview<3:continue
            if not self.lock.acquire(False):continue
            try:
                self.last_preview=time.time()
                with tempfile.TemporaryDirectory(prefix='.manual-preview-',dir=H/'sessions') as temp:
                    raw=Path(temp)/'raw'
                    env=dict(os.environ,PYTHONPATH='/opt/ros/noetic/lib/python3/dist-packages')
                    p=subprocess.run(['/usr/bin/python3',str(H/'capture_three_ros.py'),'--output',str(raw),'--timeout','10'],env=env,capture_output=True,text=True,timeout=20)
                    if p.returncode:raise RuntimeError(p.stderr[-800:] or 'Camera snapshot unavailable')
                    profiles=read(PROFILES)['profiles'];views={};images={}
                    for role in ROLES:
                        d=detect(raw/role,profiles);views[role]=d
                        im=cv2.imread(str(raw/role/'rgb.jpg'))
                        for x,y in d.get('corners_px',[]):cv2.circle(im,(round(x),round(y)),4,(40,230,40),-1)
                        im=cv2.resize(im,(640,round(im.shape[0]*640/im.shape[1])))
                        ok,jpg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,82])
                        if not ok:raise RuntimeError('Preview encoding failed')
                        images[role]=jpg.tobytes()
                    self.images=images
                    states=read(raw/'after-state.json')['data']['states']
                    self.preview={'epoch':time.time(),'views':views,'states':states,
                                  'camera_stamp_span_s':read(raw/'observation.json')['camera_stamp_span_s']}
                    if self.auto_enabled and self.board_fixed:
                        before=read(raw/'before-state.json')['data']['states']
                        drift=max(abs(a-b) for arm in states for a,b in zip(before[arm]['joint_deg'],states[arm]['joint_deg']))
                        pose=self.kin.transform(states['right']['joint_deg']);pose[:3,3]/=1000
                        known=[np.asarray(s['T_base_link6_m']['right']) for s in verified_rows(self.root) if s['active_arm']=='right']
                        ready,self.auto_reason=self.gate.check(states['right'],views['right_wrist'],pose,known,drift,time.time())
                        if ready and self.start('capture','right'):
                            self.gate.submitted(states['right']['joint_deg'],time.time())
            except Exception as e:
                self.preview={'epoch':time.time(),'error':str(e)}
            finally:self.lock.release()

    def start(self,action,arm):
        if action not in ('capture','solve') or arm not in ('left','right'):raise ValueError('Invalid request')
        with self.request_lock:
            if self.job:return False
            self.job=action
            threading.Thread(target=self.run,args=(action,arm),daemon=True).start();return True

    def run(self,action,arm):
        self.lock.acquire()
        try:
            if action=='capture':
                index=next_index(self.root,arm)
                p=subprocess.run([sys.executable,'-B',str(H/'collect_sample.py'),'--session',self.session,'--arm',arm,'--index',str(index)],capture_output=True,text=True,timeout=40)
                if p.returncode:raise RuntimeError(p.stderr[-1000:] or 'Capture failed; original evidence retained')
                s=read(self.root/(arm+'-'+str(index).zfill(4))/'sample.json')
                eligible=s['cameras'][arm+'_wrist']['metric_pose_eligible']
                self.message=f'{arm} 样本 {index} 已保存；'+('进入质量与姿态去重检查。' if eligible else '腕部标定板检测不合格，不计入有效样本。')
            else:
                rows=verified_rows(self.root)
                result=solve_arm(rows,arm)
                other='left' if arm=='right' else 'right'
                if sum(s['active_arm']==other and s['split']=='train' for s in rows)>=20 and sum(s['active_arm']==other and s['split']=='holdout' for s in rows)>=10:
                    result=solve_session(rows)
                result=clean(result);result['session_identity']=read(self.root/'identity.json')
                result['accepted_for_control']=False
                output=self.root/('candidate-'+arm+'-'+str(time.time_ns())+'.json')
                with output.open('x') as f:json.dump(result,f,indent=2)
                self.result={'qualified':result['qualified'],'path':str(output),'accepted_for_control':False,'checks':result.get('checks')}
                self.message='候选外参已保存，尚未安装到真机控制。'
        except Exception as e:self.message='未完成：'+str(e)
        finally:self.job=None;self.lock.release()

class Handler(BaseHTTPRequestHandler):
    def respond(self,data,status=200,kind='application/json'):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/':self.respond(self.server.html_path.read_text().replace('__TOKEN__',self.server.app.token).encode(),kind='text/html; charset=utf-8')
        elif path=='/status':
            self.server.app.heartbeat=time.time()
            try:self.respond(self.server.app.snapshot())
            except Exception as e:self.respond({'error':str(e)},500)
        elif path.startswith('/image/'):
            role=path.removeprefix('/image/')
            if role not in ROLES or role not in self.server.app.images:self.respond({'error':'No fresh frame'},404)
            else:self.respond(self.server.app.images[role],kind='image/jpeg')
        else:self.respond({'error':'Not found'},404)
    def do_POST(self):
        if self.headers.get('X-Calibration-Token')!=self.server.app.token:self.respond({'error':'Invalid session'},403);return
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=1024:raise ValueError('Invalid request length')
            d=json.loads(self.rfile.read(size))
            if urlsplit(self.path).path=='/auto':
                if not isinstance(d.get('enabled'),bool) or not isinstance(d.get('board_fixed'),bool):raise ValueError('Expected booleans')
                self.server.app.set_auto(d['enabled'],d['board_fixed']);self.respond({'ok':True});return
            if urlsplit(self.path).path!='/action':self.respond({'error':'Not found'},404);return
            if d.get('board_fixed') is not True:raise ValueError('Confirm board remains fixed')
            ok=self.server.app.start(d['action'],d['arm'])
            self.respond({'started':ok},202 if ok else 409)
        except Exception as e:self.respond({'error':str(e)},400)
    def log_message(self,*args):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--port',type=int,default=19877)
    p.add_argument('--html',default='manual_calibration_gui.html');a=p.parse_args()
    html_path=(H/a.html).resolve()
    if html_path.parent!=H or html_path.suffix!='.html' or not html_path.is_file():raise ValueError('Invalid GUI HTML')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);server.app=App(a.session)
    server.html_path=html_path
    (server.app.root/'gui-process.json').write_text(json.dumps({'pid':os.getpid(),'port':a.port,'robot_commands_sent':0,'epoch':time.time()}))
    server.serve_forever()

if __name__=='__main__':main()
