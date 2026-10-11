"""Translational workspace of a design: positions of O_p (orientation at home, gripper
open 8 deg per platform) where every constraint of design.py holds and 1/kappa >= ICMIN.

    python3 trans_workspace.py d35_core [d35_core_R70 ...]   -> results/tws_<name>.json
"""
import json
import os
import sys
import numpy as np

from design import (make_robot, pose_metrics, evaluate, workspace, NAMES, LIMITS, HERE, MM, DEG)

ICMIN = float(os.environ.get('ICMIN', '0.005'))   # 0.005: kappa <= 200
GX = np.arange(-60, 60.1, 3) * MM
GZ = np.arange(-45, 45.1, 3) * MM
JAW = 8 * DEG


def tws(name):
    D = json.load(open(os.path.join(HERE, 'results', f'design_{name}.json')))
    x = np.array([D['x'][k] for k in NAMES]); d = D['d']
    r = make_robot(x, d)
    axis = evaluate(x, d, workspace(), details=True)['socket_axis']
    ok = np.zeros((len(GX), len(GX), len(GZ)), bool)
    for i, px in enumerate(GX):
        for j, py in enumerate(GX):
            for k, pz in enumerate(GZ):
                m = pose_metrics(r, x, d, np.array([px, py, pz, 0, 0, 0, JAW]))
                if m is None:
                    continue
                cone = np.arccos(np.clip(np.einsum('ij,ij->i', m['lp'], axis), -1, 1)).max()
                ok[i, j, k] = (m['icond'] >= ICMIN and m['trans'] >= LIMITS['trans']
                               and m['travel'] <= LIMITS['travel'] and m['mis'] <= LIMITS['mis']
                               and cone <= LIMITS['cone'] and m['leg_leg'] >= LIMITS['leg_leg']
                               and m['leg_crank'] >= LIMITS['leg_crank'])
    dv = (3 * MM) ** 3
    # connected to home only
    from scipy.ndimage import label
    lab, _ = label(ok)
    c = lab[np.argmin(abs(GX)), np.argmin(abs(GX)), np.argmin(abs(GZ))]
    reach = (lab == c) if c > 0 else np.zeros_like(ok)
    # largest centred cube and cylinder
    X, Y, Z = np.meshgrid(GX, GX, GZ, indexing='ij')
    cube = 0
    for a in np.arange(0, 60.1, 1) * MM:
        if np.all(reach[(abs(X) <= a) & (abs(Y) <= a) & (abs(Z) <= a)]):
            cube = a
        else:
            break
    cyl = (0, 0)
    best = 0
    for rr in np.arange(0, 60.1, 3) * MM:
        for hz in np.arange(0, 45.1, 3) * MM:
            if np.all(reach[(np.hypot(X, Y) <= rr) & (abs(Z) <= hz)]) and np.pi * rr ** 2 * 2 * hz > best:
                best = np.pi * rr ** 2 * 2 * hz; cyl = (rr, hz)
    env = np.max(np.hypot(r.c[:, 0], r.c[:, 1])) + d
    out = dict(name=name, crank_envelope_radius_mm=env / MM, volume_cm3=float(reach.sum() * dv * 1e6),
               centred_cube_half_mm=cube / MM, cylinder_radius_mm=cyl[0] / MM, cylinder_half_height_mm=cyl[1] / MM,
               x_range_mm=[float(GX[reach.any((1, 2))].min() / MM), float(GX[reach.any((1, 2))].max() / MM)] if reach.any() else None,
               z_range_mm=[float(GZ[reach.any((0, 1))].min() / MM), float(GZ[reach.any((0, 1))].max() / MM)] if reach.any() else None,
               icond_core=D['objective'], icond_min_showcase=D['worst']['icond'], fails=D.get('fails'))
    out['icond_threshold'] = ICMIN
    tag = name + ('' if ICMIN == 0.005 else f'_k{1 / ICMIN:.0f}')
    json.dump(out, open(os.path.join(HERE, 'results', f'tws_{tag}.json'), 'w'), indent=1)
    np.save(os.path.join(HERE, 'results', f'tws_{tag}.npy'), reach)
    print(json.dumps(out), flush=True)
    return out


if __name__ == '__main__':
    for n in sys.argv[1:]:
        tws(n)
