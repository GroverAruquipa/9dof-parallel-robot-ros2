"""Isaac Sim (PhysX) run of the reference trajectory, same protocol as mujoco_cctomm.py.

Steps PhysX directly at 2000 Hz (no timeline, no rendering): 0.5 s settling at the reference pose, then
for every 100 Hz sample j of reference_trajectory.npz the nine drive targets are set to q[j], 20 physics
steps run, and the poses of both platforms are read back from USD. Writes result_isaac.npz with the same
fields as result_mujoco.npz (t, p1, R1, p2, R2, q).

Run inside the Isaac Sim container (from /isaac-sim):
  ./python.sh /path/to/isaac_sim/run_validation.py --usd /path/to/ninedof_cctomm_simple.usda [--no-gravity] [--kp 1e6]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = '/World/ninedof'

ap = argparse.ArgumentParser()
ap.add_argument('--usd', default=os.path.join(HERE, 'ninedof_cctomm_simple.usda'))
ap.add_argument('--no-gravity', action='store_true')
ap.add_argument('--kp', type=float, help='stiffer servo (critical damping with the same armature)')
ap.add_argument('-o', '--out', default=os.path.join(HERE, 'results', 'isaac_gravity.npz'))
args = ap.parse_args()

from isaacsim import SimulationApp                     # noqa: E402  (must come before any omni import)
app = SimulationApp({'headless': True})

import carb                                            # noqa: E402
import omni.physx                                      # noqa: E402
import omni.usd                                        # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics           # noqa: E402

carb.settings.get_settings().set('/physics/updateToUsd', True)
ctx = omni.usd.get_context()
ctx.open_stage(os.path.abspath(args.usd))
for _ in range(10):
    app.update()
stage = ctx.get_stage()

DT = 1.0 / 2000.0
scene = UsdPhysics.Scene(stage.GetPrimAtPath('/World/physicsScene'))
if args.no_gravity:
    scene.GetGravityMagnitudeAttr().Set(0.0)
targets = []
for i in range(1, 10):
    j = stage.GetPrimAtPath(f'{ROBOT}/joints/q{i}')
    d = UsdPhysics.DriveAPI(j, 'linear')
    if args.kp:
        arm = j.GetAttribute('physxJoint:armature').Get() or 0.01
        d.GetStiffnessAttr().Set(args.kp); d.GetDampingAttr().Set(2 * (args.kp * arm) ** 0.5)
        d.GetMaxForceAttr().Set(200.0)
    targets.append(d.GetTargetPositionAttr())
plat = [UsdGeom.Xformable(stage.GetPrimAtPath(f'{ROBOT}/plat{k}')) for k in (1, 2)]
car = [UsdGeom.Xformable(stage.GetPrimAtPath(p.GetPath()))
       for p in stage.Traverse() if p.GetName().startswith('carriage_')]
car_names = [p.GetName() for p in stage.Traverse() if p.GetName().startswith('carriage_')]
car_b0z = [c.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2] for c in car]

physx = omni.physx.get_physx_interface()
physx.start_simulation()
t_sim = 0.0


def step(n):
    global t_sim
    for _ in range(n):
        physx.update_simulation(DT, t_sim)
        t_sim += DT


def pose(x):
    M = x.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    R = np.array(M.ExtractRotationMatrix()).T              # Gf row-vector convention -> column convention
    return np.array(M.ExtractTranslation()), R


ref = np.load(os.path.join(HERE, 'reference_trajectory.npz'))
leg_order = [str(s) for s in ref['legs']]                  # q columns follow the CAD instances
idx = [car_names.index(f'carriage_{k}') for k in leg_order]
sub = int(round((ref['t'][1] - ref['t'][0]) / DT))


def set_q(q):
    for i in range(9):
        targets[i].Set(float(q[i]))


set_q(ref['q'][0]); step(int(0.5 / DT))
out = {k: [] for k in ('p1', 'R1', 'p2', 'R2', 'q')}
n = len(ref['t'])
for j in range(n):
    set_q(ref['q'][j]); step(sub)
    p1, R1 = pose(plat[0]); p2, R2 = pose(plat[1])
    out['p1'].append(p1); out['R1'].append(R1); out['p2'].append(p2); out['R2'].append(R2)
    zs = [pose(car[i])[0][2] - car_b0z[i] for i in idx]
    out['q'].append(zs)
    if j % 500 == 0:
        print(f'  sample {j}/{n}  t = {ref["t"][j]:.2f} s', flush=True)
np.savez_compressed(args.out, t=ref['t'], **{k: np.array(v) for k, v in out.items()})
print('wrote', args.out)
app.close()
