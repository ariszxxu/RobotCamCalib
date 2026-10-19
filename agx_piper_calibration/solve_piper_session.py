"""Offline solution from fresh, source-bound Piper samples; never publishes."""
import argparse
import json
from pathlib import Path
import numpy as np
from piper_charuco import read,sha,solve_session,clean
from robot_cam_calib.geometry import transform_delta

H=Path(__file__).resolve().parent

def load_samples(root):
    identity=read(root/'identity.json');samples=[];poses={'left':[],'right':[]}
    for path in sorted(root.glob('*/sample.json')):
        s=read(path);arm=s['active_arm']
        if arm not in poses or s['identity']!=identity:raise ValueError('Sample ownership or source changed')
        if s['split']!=('holdout' if s['index']%4==3 else 'train'):raise ValueError('Predetermined split changed')
        for rel,expected in s['source_files_sha256'].items():
            source=(path.parent/rel).resolve()
            if not source.is_relative_to(path.parent) or sha(source)!=expected:raise ValueError('Original captured evidence changed')
        pose=np.array(s['T_base_link6_m'][arm])
        if any(r<5 and t<.010 for r,t in (transform_delta(old,pose) for old in poses[arm])):
            continue  # Diversity filter uses measured robot pose only, not fit outcome.
        poses[arm].append(pose);samples.append(s)
    return samples

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);a=p.parse_args()
    if Path(a.session).name!=a.session:raise ValueError('Simple session name required')
    root=(H/'sessions'/a.session).resolve();out=root/'candidate-extrinsics.json'
    if out.exists():raise FileExistsError('Preserve first candidate and review')
    rows=load_samples(root);result=solve_session(rows)
    result['session_identity']=read(root/'identity.json');result['samples']=[str(p) for p in root.glob('*/sample.json')]
    with out.open('x') as f:json.dump(clean(result),f,indent=2)
    print(json.dumps({'candidate':str(out),'qualified':result['qualified'],'accepted_for_control':False,
                     'robot_motion_commands_sent':0,'left_checks':result['left']['checks'],'right_checks':result['right']['checks']}))

if __name__=='__main__':main()
