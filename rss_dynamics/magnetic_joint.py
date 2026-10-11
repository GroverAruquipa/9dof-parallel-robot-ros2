"""Loads on the central spherical joint, to size a magnetic ball joint.

The document eliminates f_O (force of body 1 on body 2 at O_p) by adding the two
Newton equations.  It is recovered from the Newton equation of body 2 (eq. F2):

    f_O = M2 (a_C2 - g) + sum_{i in body 2} F_i/2 - sum_{i in body 2} sigma_i m_i - F_ext,2

(F_i/2: middle-mass loads of the 3-mass link model; F_ext,2: external force on body 2).
A magnetic joint (steel ball on body 2 in a cup with a magnet on body 1) holds while
the contact force stays in the cup: f_O . n + P >= 0, with n the cup axis (fixed in
body 1, pointing from the cup to the ball) and P the magnet pull on the ball.

Cases: showcase x2 with 50 g per platform; human force of 10 N in every direction at
the fingertips of either platform; grip (object squeezed between the fingers).

    python3 magnetic_joint.py d35_core_R100
    PSCALE=2.6667 python3 magnetic_joint.py d70_s26667_core_R150
"""
import json
import os
import sys
import numpy as np

from robot_setup import setup, HERE
from trajectories import Showcase, pose_and_rates
from params import lumped_inertia
from geometry import PSCALE
from design import workspace
from control import log_so3

DEG = np.pi / 180
TIP = np.array([0.0, 0.0, 0.038]) * PSCALE      # fingertip region of each half (CAD: z 36-40 mm)


def solve(r, p, Q1, Q2, cd, cdd, ext=()):
    """sigma and f_O.  ext: list of (body, point rel. O_p (world), force (world))."""
    th = r.ik(p, Q1, Q2)
    kn = r.kin(p, Q1, Q2, th)
    thd = r.joint_rates(kn, cd)
    uu = r.accel_terms(kn, cd, thd)
    thdd = (kn['J'] @ cdd + uu) / kn['K']
    Mc, hc, gc = r.Mc_hc_gc(Q1, Q2, cd)
    rhs = Mc @ cdd + hc - gc
    Fh = np.zeros((r.n, 3))
    if np.any(r.m_mid > 0):
        Fh, wrench = r.mid_loads(kn, cd, cdd, thd, thdd)
        rhs = rhs + wrench
    F2ext = np.zeros(3)
    for k, pt, F in ext:
        w = np.zeros(9); w[:3] = F; w[3 * k:3 * k + 3] = np.cross(pt, F)
        rhs = rhs - w
        if k == 2:
            F2ext += F
    sig = np.linalg.solve(kn['J'].T, rhs)
    (s1, _, _, _), (s2, _, _, _) = r.bodies(Q1, Q2)
    w2, w2d = cd[6:9], cdd[6:9]
    a2 = cdd[:3] + np.cross(w2d, s2) + np.cross(w2, np.cross(w2, s2))
    b2 = r.body == 2
    fO = r.M[1] * (a2 - r.g) + Fh[b2].sum(0) - (sig[b2, None] * kn['m'][b2]).sum(0) - F2ext
    return fO, sig


def main(name):
    S = setup(name, mode='ideal', payload=(0.05, 0.05), motors=[])
    r = S['robot']; h = S['h']
    r.set_inertia(lumped_inertia(S['P'], r.a, '3mass'))
    rows = []                                   # (case, f_O in body-1 frame)
    # 1. showcase x2
    tr = Showcase(h, speed=2.0, yaw=30 * DEG)
    rel = []
    for t in np.arange(0, tr.duration, 0.01):
        p, Q1, Q2, cd, cdd = tr(t)
        fO, _ = solve(r, p, Q1, Q2, cd, cdd)
        rows.append(('showcase', Q1.T @ fO))
        rel.append(np.linalg.norm(log_so3(Q1.T @ Q2)))
    # 2. human force 10 N at the fingertips, 26 directions, at a few poses
    dirs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)], float)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    poses = [np.r_[0, 0, 0, 0, 0, 0, 8 * DEG], np.r_[0.02, 0, 0.01, 10 * DEG, 0, 15 * DEG, 15 * DEG],
             np.r_[-0.02, 0.02, -0.01, 0, -10 * DEG, -15 * DEG, 0]]
    for v in poses:
        p, Q1, Q2, cd, cdd = pose_and_rates(h, v, np.zeros(7), np.zeros(7))
        for k, Q in ((1, Q1), (2, Q2)):
            for d_ in dirs:
                fO, _ = solve(r, p, Q1, Q2, cd, cdd, [(k, Q @ TIP, 10.0 * d_)])
                rows.append(('humano 10 N', Q1.T @ fO))
        # 3. grip: the object pushes the two fingertips apart along the closing direction
        for Fg in (5.0, 10.0):
            for sgn in (1, -1):
                ey = np.array([0, 1.0, 0])
                ext = [(1, Q1 @ TIP, -sgn * Fg * (Q1 @ ey)), (2, Q2 @ TIP, sgn * Fg * (Q2 @ ey))]
                fO, _ = solve(r, p, Q1, Q2, cd, cdd, ext)
                rows.append((f'agarre {Fg:.0f} N', Q1.T @ fO))
    # relative rotation over the design workspace
    for v in workspace():
        p, Q1, Q2, *_ = pose_and_rates(h, v, np.zeros(7), np.zeros(7))
        rel.append(np.linalg.norm(log_so3(Q1.T @ Q2)))
    F = np.array([f for _, f in rows]); cases = np.array([c for c, _ in rows])
    # best cup axis: maximise the worst compression f.n over all cases
    g = np.random.default_rng(0).normal(size=(6000, 3)); g /= np.linalg.norm(g, axis=1)[:, None]
    out = dict(design=name, platform_scale=PSCALE)
    for subset in ('showcase', 'todo'):
        Fs = F if subset == 'todo' else F[cases == 'showcase']
        worst = (Fs @ g.T).min(0)
        n = g[np.argmax(worst)]
        out[subset] = dict(max_force_N=float(np.linalg.norm(Fs, axis=1).max()),
                           cup_axis_body1=n.round(3).tolist(),
                           magnet_pull_needed_N=float(max(0, -worst.max())),
                           axial_compression_max_N=float((Fs @ n).max()),
                           lateral_max_N=float(np.linalg.norm(Fs - np.outer(Fs @ n, n), axis=1).max()))
    out['per_case'] = {c: dict(max_force_N=float(np.linalg.norm(F[cases == c], axis=1).max()),
                               min_along_axis_N=float((F[cases == c] @ np.array(out['todo']['cup_axis_body1'])).min()))
                       for c in sorted(set(cases))}
    out['relative_rotation_max_deg'] = float(np.degrees(max(rel)))
    json.dump(out, open(os.path.join(HERE, 'results', f'magnetic_joint_{name}.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'd35_core_R100')
