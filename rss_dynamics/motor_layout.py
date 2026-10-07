"""Placement of the nine MPX-40C4WA motors and GT2 belts under the cranks.

Each motor axis is parallel to its crank axis v_i (a plain belt between parallel
shafts).  The crank pulley sits on the crank shaft, PULLEY_OFFSET behind the
crank along -v_i.  Per motor we choose the belt centre distance C_i, the belt
angle phi_i from the vertical (in the plane normal to v_i) and the side s_i to
which the motor body extends along its axis.  Motor envelope (measured on the
CAD of the 3_unlimited_rotation robot): body diameter 51 mm, length 140 mm with
the encoder, square flange 67 mm (taken as a 34 mm radius over 20 mm).

Objective: smallest footprint radius plus depth, with every motor 4 mm away from
every other motor, from every belt and below the base plate.

    python3 motor_layout.py d25       # results/motors_d25.json, figures/packaging_d25.png
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from design import seg_dist, make_robot, NAMES, HERE
from params import pulley

R_BODY, L_BODY, R_FLANGE, L_FLANGE = 0.0255, 0.140, 0.034, 0.020
PULLEY_OFFSET = -0.0215          # crank pulley centre along v_i (behind the crank)
SHAFT_OUT = 0.018                # motor pulley to flange face
CLEAR = 0.004


def motor_geometry(r, z_plate, X):
    """Motor capsules and belts for the decision vector X (3 per motor)."""
    n = r.n
    C = X[:n]; phi = X[n:2 * n]; side = np.where(X[2 * n:] >= 0, 1.0, -1.0)
    ez = np.array([0, 0, 1.0])
    out = []
    for i in range(n):
        v = r.v[i]
        pc = r.c[i] + PULLEY_OFFSET * v
        down = -ez + (ez @ v) * v; down /= np.linalg.norm(down)
        lat = np.cross(v, down)
        pm = pc + C[i] * (np.cos(phi[i]) * down + np.sin(phi[i]) * lat)
        a0 = pm + side[i] * SHAFT_OUT * v                     # flange face
        a1 = a0 + side[i] * L_BODY * v                         # motor end
        out.append(dict(pc=pc, pm=pm, a0=a0, a1=a1, axis=v * side[i], C=C[i]))
    return out


def cost(X, r, z_plate, detail=False):
    M = motor_geometry(r, z_plate, X)
    pen = 0.0
    n = len(M)
    A0 = np.array([m['a0'] for m in M]); A1 = np.array([m['a1'] for m in M])
    PC = np.array([m['pc'] for m in M]); PM = np.array([m['pm'] for m in M])
    for i in range(n):
        for j in range(i + 1, n):
            dmin = seg_dist(A0[i], A1[i], A0[j], A1[j])
            pen += max(0, 2 * R_FLANGE + CLEAR - dmin) * 100
        for j in range(n):
            if j != i:   # belt j against motor i
                pen += max(0, R_FLANGE + 0.004 + CLEAR - seg_dist(A0[i], A1[i], PC[j], PM[j])) * 100
        # motor below the base plate
        top = max(A0[i, 2], A1[i, 2]) + R_FLANGE
        pen += max(0, top - (z_plate - CLEAR)) * 100
    pts = np.r_[A0, A1]
    radius = np.max(np.hypot(pts[:, 0], pts[:, 1])) + R_FLANGE
    depth = z_plate - (np.min(pts[:, 2]) - R_FLANGE)
    f = radius + 0.5 * depth + pen
    if detail:
        return dict(radius=radius, depth=depth, pen=pen, motors=M)
    return f


def main(name='d25'):
    D = json.load(open(os.path.join(HERE, 'results', f'design_{name}.json')))
    x = np.array([D['x'][k] for k in NAMES])
    r = make_robot(x, D['d'])
    z_plate = r.c[:, 2].min() - 0.024
    n = r.n
    bounds = [(0.05, 0.30)] * n + [(-1.2, 1.2)] * n + [(-1, 1)] * n
    best = None
    for seed in (0, 1):
        res = differential_evolution(cost, bounds, args=(r, z_plate), seed=seed, maxiter=400,
                                     popsize=10, tol=1e-10, polish=False, init='sobol')
        print(name, seed, res.fun, flush=True)
        if best is None or res.fun < best.fun:
            best = res
    det = cost(best.x, r, z_plate, detail=True)
    N = D.get('N', 3.0)
    pc_, pm_ = pulley(round(16 * N)), pulley(16)
    motors = []
    for i, m in enumerate(det['motors']):
        C = m['C']
        Lb = 2 * C + np.pi * (pc_['r'] + pm_['r']) + (pc_['r'] - pm_['r']) ** 2 / C
        mid = (m['a0'] + m['a1']) / 2
        motors.append(dict(pos=mid.tolist(), axis=m['axis'].tolist(), len=L_BODY, r=R_BODY,
                           pulley_pos=m['pm'].tolist(), belt_offset=PULLEY_OFFSET,
                           C_mm=1e3 * C, belt_length_mm=1e3 * Lb, belt_teeth=int(round(1e3 * Lb / 2))))
    json.dump(motors, open(os.path.join(HERE, 'results', f'motors_{name}.json'), 'w'), indent=1)
    summary = dict(footprint_radius_mm=1e3 * det['radius'], depth_below_plate_mm=1e3 * det['depth'],
                   penalty=det['pen'], z_plate_mm=1e3 * z_plate)
    json.dump(summary, open(os.path.join(HERE, 'results', f'packaging_{name}.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    # figure: top and side views
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    th = np.linspace(0, 2 * np.pi, 200)
    for k, (a, (ix, iy)) in enumerate(zip(ax, ((0, 1), (0, 2)))):
        for i, m in enumerate(det['motors']):
            a.plot([m['a0'][ix] * 1e3, m['a1'][ix] * 1e3], [m['a0'][iy] * 1e3, m['a1'][iy] * 1e3],
                   lw=2 * R_BODY * 1e3 * (0.62 if k == 0 else 0.62), color='0.35', alpha=0.55, solid_capstyle='butt')
            a.plot([m['pc'][ix] * 1e3, m['pm'][ix] * 1e3], [m['pc'][iy] * 1e3, m['pm'][iy] * 1e3], 'k-', lw=1)
            a.text(m['a1'][ix] * 1e3, m['a1'][iy] * 1e3, str(i + 1), fontsize=9)
        tips = r.c + r.crank(r.ik(np.array([0, 0, x[5]]), np.eye(3), np.eye(3)))[0]
        for i in range(n):
            a.plot([r.c[i, ix] * 1e3, tips[i, ix] * 1e3], [r.c[i, iy] * 1e3, tips[i, iy] * 1e3], 'r-', lw=3)
        a.set_aspect('equal'); a.grid(alpha=0.3)
    ax[0].set_title(f'Vista superior: motores (gris), correas (negro), manivelas (rojo)\n'
                    f'radio ocupado {1e3 * det["radius"]:.0f} mm')
    ax[1].set_title(f'Vista frontal (x-z): profundidad bajo la placa {1e3 * det["depth"]:.0f} mm')
    ax[0].set_xlabel('x (mm)'); ax[0].set_ylabel('y (mm)'); ax[1].set_xlabel('x (mm)'); ax[1].set_ylabel('z (mm)')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', f'packaging_{name}.png'), dpi=140)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'd25')
