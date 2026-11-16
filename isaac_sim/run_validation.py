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
ap.add_argument('--samples', type=int, help='only the first N samples (quick check, e.g. 600)')
ap.add_argument('-o', '--out', default=os.path.join(HERE, 'results', 'isaac_gravity.npz'))
args = ap.parse_args()

from isaacsim import SimulationApp                     # noqa: E402  (must come before any omni import)
app = SimulationApp({'headless': True})

import carb                                            # noqa: E402
import omni.physx                                      # noqa: E402
import omni.usd                                        # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics           # noqa: E402

settings = carb.settings.get_settings()
settings.set('/physics/updateToUsd', True)
settings.set('/physics/fabricEnabled', False)            # first run: poses never reached USD (Fabric)
try:
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.physx.fabric', False)
except Exception as e:                                   # noqa: BLE001
    print('note: could not disable omni.physx.fabric:', e)
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
try:
    physx.force_load_physics_from_usd()
except Exception as e:                                   # noqa: BLE001
    print('note: force_load_physics_from_usd:', e)
physx.start_simulation()
t_sim = 0.0


def step(n):
    global t_sim
    for _ in range(n):
        physx.update_simulation(DT, t_sim)
        t_sim += DT


def mat_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose(x):
    """World pose read from PhysX itself (independent of the USD and Fabric write-back)."""
    path = str(x.GetPath())
    r = physx.get_rigidbody_transformation(path)
    if r and r.get('ret_val', True) and 'position' in r:
        return np.array(r['position'], float), mat_xyzw(np.array(r['rotation'], float))
    M = x.ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # fallback: USD
    return np.array(M.ExtractTranslation()), np.array(M.ExtractRotationMatrix()).T


ref = np.load(os.path.join(HERE, 'reference_trajectory.npz'))
leg_order = [str(s) for s in ref['legs']]                  # q columns follow the CAD instances
idx = [car_names.index(f'carriage_{k}') for k in leg_order]
sub = int(round((ref['t'][1] - ref['t'][0]) / DT))


def set_q(q):
    for i in range(9):
        targets[i].Set(float(q[i]))


# every spherical joint: anchor on body 0 and on body 1, to measure how far the joint opens
anchors = []
for p in stage.Traverse():
    if p.IsA(UsdPhysics.SphericalJoint):
        jt = UsdPhysics.Joint(p)
        b0 = UsdGeom.Xformable(stage.GetPrimAtPath(jt.GetBody0Rel().GetTargets()[0]))
        b1 = UsdGeom.Xformable(stage.GetPrimAtPath(jt.GetBody1Rel().GetTargets()[0]))
        anchors.append((p.GetName(), bool(jt.GetExcludeFromArticulationAttr().Get()), b0, b1,
                        np.array(jt.GetLocalPos0Attr().Get(), float), np.array(jt.GetLocalPos1Attr().Get(), float)))


def joint_gaps():
    """Opening of the loop closures (excluded joints) and of the tree joints, in mm."""
    loop, tree = [], []
    for name, excl, b0, b1, l0, l1 in anchors:
        pa, Ra = pose(b0); pb, Rb = pose(b1)
        g = np.linalg.norm((pa + Ra @ l0) - (pb + Rb @ l1)) * 1000
        (loop if excl else tree).append(g)
    return max(loop), max(tree)


print('DIAG raw PhysX pose of plat1 before stepping:', physx.get_rigidbody_transformation(f'{ROBOT}/plat1'), flush=True)
p_start = pose(plat[0])[0]
set_q(ref['q'][0]); step(int(0.5 / DT))
p_settle = pose(plat[0])[0]
gl, gt = joint_gaps()
print(f'DIAG after 0.5 s settling: plat1 moved {np.linalg.norm(p_settle - p_start) * 1000:.3f} mm, '
      f'loop closures open {gl:.4f} mm, tree joints {gt:.4f} mm', flush=True)

out = {k: [] for k in ('p1', 'R1', 'p2', 'R2', 'q', 'gap_loop', 'gap_tree')}
n = min(len(ref['t']), args.samples) if args.samples else len(ref['t'])
for j in range(n):
    set_q(ref['q'][j]); step(sub)
    p1, R1 = pose(plat[0]); p2, R2 = pose(plat[1])
    out['p1'].append(p1); out['R1'].append(R1); out['p2'].append(p2); out['R2'].append(R2)
    zs = [pose(car[i])[0][2] - car_b0z[i] for i in idx]
    out['q'].append(zs)
    gl, gt = joint_gaps(); out['gap_loop'].append(gl); out['gap_tree'].append(gt)
    if j % 100 == 0:
        e = np.max(np.abs(np.array(zs) - ref['q'][j])) * 1000
        print(f'  sample {j}/{n}  t = {ref["t"][j]:.2f} s  q tracking {e:.3f} mm  '
              f'p - p_ref {np.linalg.norm(p1 - ref["p"][j]) * 1000:.3f} mm  '
              f'loops open {gl:.4f} mm', flush=True)
print(f'DIAG loop closures over the run: max {max(out["gap_loop"]):.4f} mm, '
      f'tree joints max {max(out["gap_tree"]):.4f} mm', flush=True)
# actuator check at the end, so it does not disturb the run: q1 + 2 mm for 0.1 s
k1 = idx[0]; z0 = pose(car[k1])[0][2]
targets[0].Set(float(ref['q'][n - 1][0]) + 0.002); step(200)
print(f'DIAG actuator check: target +2.000 mm, carriage 1 moved {(pose(car[k1])[0][2] - z0) * 1000:.3f} mm', flush=True)
np.savez_compressed(args.out, t=ref['t'][:n], **{k: np.array(v) for k, v in out.items()})
print('wrote', args.out)
app.close()
