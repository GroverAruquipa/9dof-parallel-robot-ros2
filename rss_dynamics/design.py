"""Dimensional synthesis of the 5RSS-S-4RSS robot (cranks of 25 or 35 mm).

For a crank length d, find (Rt, dpsi, gamma, beta, l, h) that maximise the
worst inverse condition number of the homogenised Jacobian
    H = K^-1 J diag(L, L, L, 1, ..., 1),      L = 37.5 mm (platform radius)
over the working poses (the showcase motions of the grasping robot plus
combined poses), subject to
    - inverse kinematics on every pose, crank travel <= 100 deg from home,
    - transmission |K_ii| / (l d) >= 0.5 (angle between leg and tip velocity <= 60 deg),
    - crank-side joint: rod end with its bolt parallel to the crank axis
      (free about the bolt), misalignment |angle(leg, plane normal to v_i)| <= 30 deg,
    - platform-side ball joint: cone <= 40 deg about its socket axis (the mean
      direction of the leg in the platform frame over the poses),
    - leg-leg axis distance >= 9 mm, leg-crank >= 9 mm, crank-crank >= 12 mm.

    python3 design.py            # writes results/design_d25.json, results/design_d35.json
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution

from model import RSS9
from geometry import build_geometry, fix_branch, pose_from_params, home_pose, MM, DEG

HERE = os.path.dirname(os.path.abspath(__file__))
L_CHAR = 0.0375
WORKERS = int(os.environ.get('WORKERS', '-1'))
DUMMY_INERTIA = dict(M=[1, 1], s0=np.zeros((2, 3)), IC0=np.array([np.eye(3)] * 2),
                     Im=np.ones(9), mm=np.ones(9), rc=np.ones(9))
LIMITS = dict(travel=100 * DEG, trans=0.5, mis=30 * DEG, cone=40 * DEG, leg_leg=9 * MM,
              leg_crank=9 * MM, crank_crank=12 * MM, trim=8 * MM)
AMP = np.array([30 * MM, 30 * MM, 15 * MM, 20 * DEG, 20 * DEG, 45 * DEG, 15 * DEG])


def workspace(n_random=40, seed=1):
    """Poses as 7-parameter vectors (x, y, z, roll, pitch, yaw, jaw)."""
    W = [np.zeros(7)]
    for k in range(6):                          # single-axis sweeps of the showcase
        for s in (-1, -0.5, 0.5, 1):
            v = np.zeros(7); v[k] = s * AMP[k]; W.append(v)
    for s in (0.5, 1):                          # gripper alone and with each motion
        v = np.zeros(7); v[6] = s * AMP[6]; W.append(v)
    for k in range(6):
        for s in (-1, 1):
            v = np.zeros(7); v[k] = s * AMP[k]; v[6] = AMP[6]; W.append(v)
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):   # circle and cone
        W.append(np.array([25 * MM * np.cos(a), 25 * MM * np.sin(a), 0, 0, 0, 0, 0]))
        W.append(np.array([0, 0, 0, 20 * DEG * np.cos(a), 20 * DEG * np.sin(a), 0, 0]))
    rng = np.random.default_rng(seed)           # combined poses at half amplitude
    for _ in range(n_random):
        v = rng.uniform(-0.5, 0.5, 7) * AMP; v[6] = rng.uniform(0, 1) * AMP[6]; W.append(v)
    return np.array(W)


def core_workspace(n_random=30, seed=2):
    """Neighbourhood of home where the pHRI demonstrations happen: half of the
    showcase amplitudes on each axis, the gripper up to 15 deg, combined poses at
    a quarter of the amplitudes."""
    W = [np.zeros(7)]
    for k in range(6):
        for s in (-0.5, 0.5):
            v = np.zeros(7); v[k] = s * AMP[k]; W.append(v)
            v = v.copy(); v[6] = AMP[6]; W.append(v)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        v = rng.uniform(-0.25, 0.25, 7) * AMP; v[6] = rng.uniform(0, 1) * AMP[6]; W.append(v)
    return np.array(W)


OBJ = os.environ.get('OBJ', 'worst')       # 'worst': min 1/kappa over the showcase; 'core': over core_workspace
ICOND_FLOOR = 1.5e-3                        # with OBJ = core: required 1/kappa on the whole showcase


def seg_dist(P0, P1, Q0, Q1):
    """Minimum distance between segments P0P1 and Q0Q1 (arrays (..., 3))."""
    d1, d2, r = P1 - P0, Q1 - Q0, P0 - Q0
    a = np.sum(d1 * d1, -1); e = np.sum(d2 * d2, -1); f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1); b = np.sum(d1 * d2, -1)
    den = a * e - b * b
    s = np.clip(np.where(den > 1e-14, (b * f - c * e) / np.where(den > 1e-14, den, 1), 0), 0, 1)
    t = (b * s + f) / e
    t = np.clip(t, 0, 1)
    s = np.clip((b * t - c) / a, 0, 1)
    return np.linalg.norm(P0 + d1 * s[..., None] - Q0 - d2 * t[..., None], axis=-1)


PAIRS = np.array([(i, j) for i in range(9) for j in range(9) if i < j])


def make_robot(x, d):
    Rt, dpsi, gamma, beta, l, h, dpsi2, alt = x
    geom = build_geometry(d, l, Rt, gamma, beta, h, dpsi, dpsi2, alt)
    r = RSS9(geom, DUMMY_INERTIA)
    fix_branch(r, beta, h)
    return r


def pose_metrics(r, x, d, v, home=None):
    """All quantities of one pose; None if the IK fails."""
    h, beta, l = x[5], x[3], x[4]
    p, Q1, Q2 = pose_from_params(h, v)
    try:
        th = r.ik(p, Q1, Q2)
    except ValueError:
        return None
    kn = r.kin(p, Q1, Q2, th)
    H = kn['J'] / kn['K'][:, None]
    H[:, :3] *= L_CHAR
    sv = np.linalg.svd(H, compute_uv=False)
    out = dict(icond=sv[-1] / sv[0], th=th)
    out['travel'] = np.max(np.abs((th - beta + np.pi) % (2 * np.pi) - np.pi))
    out['trans'] = np.min(np.abs(kn['K']) / (l * d))
    leg = kn['m'] / l
    # cones: leg direction in the crank frame and in the platform frame vs home
    # crank side: misalignment of the rod end (bolt along v_i)
    out['mis'] = np.max(np.arcsin(np.clip(np.abs(np.einsum('ij,ij->i', leg, r.v)), 0, 1)))
    # platform side: leg direction in the platform frame (cone evaluated over all poses)
    Qs = np.where((r.body == 1)[:, None, None], Q1[None], Q2[None])
    out['lp'] = np.einsum('kji,kj->ki', Qs, leg)
    out['leg'] = leg
    tips = r.c + kn['D']
    anchors = tips + kn['m']
    tr = LIMITS['trim'] / l
    L0, L1 = tips + tr * kn['m'], anchors - tr * kn['m']
    i, j = PAIRS.T
    out['leg_leg'] = seg_dist(L0[i], L1[i], L0[j], L1[j]).min()
    # leg i against crank j (j != i)
    ii, jj = np.where(~np.eye(9, dtype=bool))
    out['leg_crank'] = seg_dist(L0[ii], L1[ii], r.c[jj], tips[jj]).min()
    out['crank_crank'] = seg_dist(r.c[i], tips[i], r.c[j], tips[j]).min()
    out['tips'], out['anchors'] = tips, anchors
    return out


def evaluate(x, d, W, details=False):
    try:
        r = make_robot(x, d)
        p, Q1, Q2 = home_pose(x[5])
        r.ik(p, Q1, Q2)
    except ValueError:
        return 1e3 if not details else None
    worst = dict(icond=np.inf, trans=np.inf, leg_leg=np.inf, leg_crank=np.inf,
                 crank_crank=np.inf, travel=0, mis=0, cone=0)
    fails = 0
    allm, LP = [], []
    for v in W:
        m = pose_metrics(r, x, d, v)
        if m is None:
            fails += 1
            continue
        allm.append(m); LP.append(m['lp'])
        for k in ('icond', 'trans', 'leg_leg', 'leg_crank', 'crank_crank'):
            worst[k] = min(worst[k], m[k])
        for k in ('travel', 'mis'):
            worst[k] = max(worst[k], m[k])
    axis = None
    if LP:
        LP = np.array(LP)                                   # (poses, 9, 3)
        axis = LP.mean(0); axis /= np.linalg.norm(axis, axis=1)[:, None]
        worst['cone'] = float(np.arccos(np.clip(np.einsum('pij,ij->pi', LP, axis), -1, 1)).max())
    if details:
        return dict(worst=worst, fails=fails, robot=r, all=allm, socket_axis=axis)
    pen = 10.0 * fails
    pen += 10 * max(0, worst['travel'] - LIMITS['travel'])
    pen += 10 * max(0, LIMITS['trans'] - worst['trans'])
    pen += 10 * max(0, worst['mis'] - LIMITS['mis'])
    pen += 10 * max(0, worst['cone'] - LIMITS['cone'])
    for k in ('leg_leg', 'leg_crank', 'crank_crank'):
        pen += 100 * max(0, LIMITS[k] - worst[k])
    if OBJ == 'core':
        pen += 100 * max(0, ICOND_FLOOR - worst['icond'])
        if pen > 0:
            return 1.0 + pen
        ic = []
        for v in CORE:
            m = pose_metrics(r, x, d, v)
            if m is None:
                return 1.0 + 10
            ic.append(m['icond'])
        return -min(ic)
    if pen > 0:
        return 1.0 + pen
    return -worst['icond']


CORE = core_workspace()


BOUNDS = [(0.04, 0.12), (-80 * DEG, 80 * DEG), (-180 * DEG, 180 * DEG), (-60 * DEG, 60 * DEG),
          (0.06, 0.18), (0.04, 0.18), (-80 * DEG, 80 * DEG), (0, 40 * DEG)]
NAMES = ['Rt', 'dpsi', 'gamma', 'beta', 'l', 'h', 'dpsi2', 'alt']


def optimise(d, seed=0, maxiter=150, popsize=15):
    W = workspace()
    it = [0]

    def cb(xk, convergence=None):
        it[0] += 1
        if it[0] % 10 == 0:
            print(f'   gen {it[0]}: f = {evaluate(xk, d, W):.5f}', flush=True)
    res = differential_evolution(evaluate, BOUNDS, args=(d, W), seed=seed, maxiter=maxiter,
                                 popsize=popsize, tol=1e-8, polish=False, workers=WORKERS,
                                 updating='deferred', init='sobol', callback=cb)
    return res, W


if __name__ == '__main__':
    ds = [float(a) * MM for a in sys.argv[1:]] or [25 * MM, 35 * MM]
    for d in ds:
        best = None
        for seed in [int(s) for s in os.environ.get('SEEDS', '0').split(',')]:
            res, W = optimise(d, seed)
            print(f'd = {d / MM:.0f} mm seed {seed}: f = {res.fun:.5f}  x = {np.round(res.x, 4)}', flush=True)
            if best is None or res.fun < best.fun:
                best = res
        from scipy.optimize import minimize
        loc = minimize(evaluate, best.x, args=(d, W), method='Nelder-Mead',
                       options=dict(maxiter=1500, xatol=1e-6, fatol=1e-8))
        print(f'   Nelder-Mead polish: {best.fun:.6f} -> {loc.fun:.6f}', flush=True)
        if loc.fun < best.fun:
            best.x, best.fun = loc.x, loc.fun
        det = evaluate(best.x, d, W, details=True)
        w = det['worst']
        out = dict(d=d, x=dict(zip(NAMES, best.x.tolist())), objective=-best.fun,
                   worst={k: float(v) for k, v in w.items()}, fails=det['fails'],
                   limits={k: float(v) for k, v in LIMITS.items()}, L_char=L_CHAR)
        from geometry import PSCALE
        out['platform_scale'] = PSCALE
        tag = f'd{d / MM:.0f}' + ('' if PSCALE == 1 else f'_s{PSCALE:g}'.replace('.', '')) + ('' if OBJ == 'worst' else f'_{OBJ}')
        out['objective_kind'] = OBJ
        json.dump(out, open(os.path.join(HERE, 'results', f'design_{tag}.json'), 'w'), indent=1)
        print(json.dumps(out, indent=1), flush=True)
