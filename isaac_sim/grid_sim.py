"""PhysX run of N copies of the robot at once, each with its own servo stiffness (make_grid.py).

Same stepping and reading as run_validation.py (validated against MuJoCo): PhysX stepped directly (1000 Hz by default),
drive targets set from reference_trajectory.npz at 100 Hz, poses read from PhysX itself. Records, for every copy
and every 100 Hz sample, both platform poses and the nine carriage positions (results/grid_run.npz), then prints the
largest tracking error of each copy against the inverse kinematics.

Run in the container:  ./python.sh /work/isaac_sim/grid_sim.py [--n 16] [--t-end 20.5] [--hz 1000]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=16)
ap.add_argument('--spacing', type=float, default=0.25)
ap.add_argument('--kp-min', type=float, default=2e4)
ap.add_argument('--kp-max', type=float, default=2e6)
ap.add_argument('--t-end', type=float, default=20.5, help='s of the showcase (20.5: hold, translations, rigid rotation x)')
ap.add_argument('--hz', type=int, default=1000, help='PhysX steps per second')
ap.add_argument('--out', default=os.path.join(HERE, 'results', 'grid_run.npz'))
args = ap.parse_args()

from isaacsim import SimulationApp                     # noqa: E402
app = SimulationApp({'headless': True})

import carb                                            # noqa: E402
import omni.physx                                      # noqa: E402
import omni.usd                                        # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics               # noqa: E402

import make_grid                                       # noqa: E402

settings = carb.settings.get_settings()
settings.set('/physics/updateToUsd', False)            # poses are read from PhysX, no write-back needed
settings.set('/physics/fabricEnabled', False)
try:
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.physx.fabric', False)
except Exception as e:                                 # noqa: BLE001
    print('note: could not disable omni.physx.fabric:', e)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
grid_usd = os.path.join(HERE, 'results', 'grid_physics.usda')
_, envs = make_grid.build(grid_usd, os.path.join(HERE, 'ninedof_cctomm_simple.usda'), n=args.n,
                          spacing=args.spacing, kp_min=args.kp_min, kp_max=args.kp_max, hz=args.hz)
ctx = omni.usd.get_context()
ctx.open_stage(grid_usd)
for _ in range(10):
    app.update()
stage = ctx.get_stage()

ref = np.load(os.path.join(HERE, 'reference_trajectory.npz'))
legs = [str(k) for k in ref['legs']]                   # q columns follow the CAD instances
DT = 1.0 / args.hz
sub = int(round((ref['t'][1] - ref['t'][0]) / DT))
n_s = int(np.searchsorted(ref['t'], args.t_end)) + 1

targets, car_paths, car_z0 = [], [], []
for e in envs:
    targets.append([UsdPhysics.DriveAPI(stage.GetPrimAtPath(f'{e["path"]}/joints/q{i}'), 'linear').GetTargetPositionAttr()
                    for i in range(1, 10)])
    paths = [f'{e["path"]}/carriage_{k}' for k in legs]
    car_paths.append(paths)
    car_z0.append([UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                   .ExtractTranslation()[2] for p in paths])
# joint qi drives the carriage of leg i in the robot file; legs follow the same order as the reference columns
physx = omni.physx.get_physx_interface()
try:
    physx.force_load_physics_from_usd()
except Exception as e:                                 # noqa: BLE001
    print('note: force_load_physics_from_usd:', e)
physx.start_simulation()


def mat_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose(path):
    r = physx.get_rigidbody_transformation(path)
    return np.array(r['position'], float), mat_xyzw(np.array(r['rotation'], float))


t_sim = 0.0


def step(n):
    global t_sim
    for _ in range(n):
        physx.update_simulation(DT, t_sim)
        t_sim += DT


def set_q(q):
    for tg in targets:
        for i in range(9):
            tg[i].Set(float(q[i]))


set_q(ref['q'][0]); step(int(round(1.0 / DT)))                 # 1 s settling at the reference pose
N = len(envs)
kp_all = np.array([e['kp'] for e in envs])
out = {'p1': np.zeros((n_s, N, 3)), 'R1': np.zeros((n_s, N, 3, 3)), 'p2': np.zeros((n_s, N, 3)),
       'R2': np.zeros((n_s, N, 3, 3)), 'q': np.zeros((n_s, N, 9))}
t0 = time.time()
for j in range(n_s):
    set_q(ref['q'][j]); step(sub)
    for k, e in enumerate(envs):
        out['p1'][j, k], out['R1'][j, k] = pose(f'{e["path"]}/plat1')
        out['p2'][j, k], out['R2'][j, k] = pose(f'{e["path"]}/plat2')
        out['q'][j, k] = [pose(p)[0][2] - z0 for p, z0 in zip(car_paths[k], car_z0[k])]
    if j % 200 == 0:
        el = time.time() - t0
        e_soft = np.abs(out['q'][j, int(np.argmin(kp_all))] - ref['q'][j]).max() * 1000
        e_stiff = np.abs(out['q'][j, int(np.argmax(kp_all))] - ref['q'][j]).max() * 1000
        print(f'  sample {j}/{n_s}  t = {ref["t"][j]:.2f} s  q tracking softest {e_soft:.3f} mm, stiffest '
              f'{e_stiff:.3f} mm  elapsed {el:.0f} s  remaining ~{el / max(j, 1) * (n_s - j):.0f} s', flush=True)
# actuator check after the run (so it does not disturb it): q1 of the stiffest copy + 2 mm for 0.1 s
ks = int(np.argmax(kp_all)); z_before = pose(car_paths[ks][0])[0][2]
targets[ks][0].Set(float(ref['q'][n_s - 1][0]) + 0.002); step(int(round(0.1 / DT)))
print(f'DIAG actuator check: target +2.000 mm, carriage moved {(pose(car_paths[ks][0])[0][2] - z_before) * 1000:.3f} mm',
      flush=True)
offsets = np.array([e['offset'] for e in envs]); kps = kp_all
np.savez_compressed(args.out, t=ref['t'][:n_s], offsets=offsets, kp=kps, **out)
print('wrote', args.out)
el = time.time() - t0
full = el / (n_s * N) * (int(np.searchsorted(ref['t'], 20.5)) + 1) * 16
print(f'DIAG speed: {el:.0f} s for {N} robots x {n_s} samples; the default run (16 robots, 20.5 s) would take '
      f'about {full / 60:.0f} min if the cost grows with the number of robots', flush=True)

# tracking of each copy against the inverse kinematics (positions relative to its own offset)
print(f'\n  {"kp (N/m)":>10s} {"p error max (mm)":>17s} {"Q1 error max (deg)":>19s} {"q tracking max (mm)":>20s}')
for k in np.argsort(kps):
    ep = np.linalg.norm(out['p1'][:, k] - offsets[k] - ref['p'][:n_s], axis=1).max() * 1000
    c = np.clip((np.einsum('tij,tij->t', out['R1'][:, k], ref['Q1'][:n_s]) - 1) / 2, -1, 1)
    eq = np.abs(out['q'][:, k] - ref['q'][:n_s]).max() * 1000
    print(f'  {kps[k]:10.0f} {ep:17.3f} {np.degrees(np.arccos(c)).max():19.3f} {eq:20.3f}', flush=True)
app.close()
