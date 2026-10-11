"""Magnet pull and stem cone needed by a commercial magnetic ball joint (ball + stem on
platform 2, magnetic cup on platform 1) for several cup orientations.
    python3 magnetic_joint_orientation.py d35_core_R100"""
import numpy as np, json, sys
sys.path.insert(0,'.')
import magnetic_joint as mj
from robot_setup import setup
from trajectories import Showcase, pose_and_rates
from params import lumped_inertia
from design import workspace
DEG=np.pi/180
name=sys.argv[1]
S=setup(name,mode='ideal',payload=(0.05,0.05),motors=[]); r=S['robot']; h=S['h']; r.set_inertia(lumped_inertia(S['P'],r.a,'3mass'))
F=[]; R12=[]
tr=Showcase(h,speed=2.0,yaw=30*DEG)
for t in np.arange(0,tr.duration,0.02):
    p,Q1,Q2,cd,cdd=tr(t); F.append(Q1.T@mj.solve(r,p,Q1,Q2,cd,cdd)[0]); R12.append(Q1.T@Q2)
dirs=np.array([[i,j,k] for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if (i,j,k)!=(0,0,0)],float); dirs/=np.linalg.norm(dirs,axis=1)[:,None]
for v in [np.r_[0,0,0,0,0,0,8*DEG],np.r_[0.02,0,0.01,10*DEG,0,15*DEG,15*DEG],np.r_[-0.02,0.02,-0.01,0,-10*DEG,-15*DEG,0]]:
    p,Q1,Q2,cd,cdd=pose_and_rates(h,v,np.zeros(7),np.zeros(7))
    for k,Q in ((1,Q1),(2,Q2)):
        for d_ in dirs: F.append(Q1.T@mj.solve(r,p,Q1,Q2,cd,cdd,[(k,Q@mj.TIP,10*d_)])[0])
    for Fg in (5.,10.):
        for s in (1,-1):
            ey=np.array([0,1.,0]); F.append(Q1.T@mj.solve(r,p,Q1,Q2,cd,cdd,[(1,Q1@mj.TIP,-s*Fg*(Q1@ey)),(2,Q2@mj.TIP,s*Fg*(Q2@ey))])[0])
for v in workspace():
    p,Q1,Q2,*_=pose_and_rates(h,v,np.zeros(7),np.zeros(7)); R12.append(Q1.T@Q2)
F=np.array(F); R12=np.array(R12)
def cone(n):
    # stem fixed in body 2, aligned with the cup axis n at the middle of the range: search the stem direction minimising the cone
    best=1e9
    g=np.random.default_rng(1).normal(size=(3000,3)); g/=np.linalg.norm(g,axis=1)[:,None]
    for s in g:
        sd=R12@s                     # stem in body-1 frame over all poses
        ang=np.degrees(np.arccos(np.clip(sd@n,-1,1))).max()
        best=min(best,ang)
    return best
def pull(n): return max(0,-(F@n).min())
g=np.random.default_rng(0).normal(size=(4000,3)); g/=np.linalg.norm(g,axis=1)[:,None]
worst=(F@g.T).min(0); nopt=g[np.argmax(worst)]
out={}
for lab,n in (('eje óptimo de carga',nopt),('eje x (eje de la pinza)',np.array([1.,0,0])),('eje -x',np.array([-1.,0,0])),('vertical hacia arriba +z',np.array([0,0,1.])),('vertical hacia abajo -z',np.array([0,0,-1.]))):
    out[lab]=dict(axis=n.round(2).tolist(),pull_N=round(pull(n),1),cone_deg=round(cone(n),1),max_force_N=round(float(np.linalg.norm(F,axis=1).max()),1))
    print(lab,out[lab])
json.dump(out,open(f'results/magnetic_joint_orient_{name}.json','w'),indent=1,ensure_ascii=False)
