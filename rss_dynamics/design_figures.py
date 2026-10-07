"""Figures of the dimensional synthesis.

  figures/design_geometry.png    the two designs at home (3D and top view)
  figures/conditioning_maps.png  1/kappa over translation and rotation slices,
                                 RSS d = 25 / 35 mm and the original PSS robot
  results/design_summary.json    dimensions, constraint margins, conditioning stats
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'ninedof_kinematics'))
from ninedof_kinematics.kinematics import NineDofKinematics, xyz_from_rot

from design import make_robot, NAMES, workspace, evaluate, pose_metrics, L_CHAR, LIMITS, HERE, MM, DEG
from geometry import pose_from_params, GEOM_YAML
import plotstyle as ps

FIG = os.path.join(HERE, 'figures')
PSS = NineDofKinematics.from_yaml(GEOM_YAML)


def load(name):
    D = json.load(open(os.path.join(HERE, 'results', f'design_{name}.json')))
    return D, np.array([D['x'][k] for k in NAMES])


def icond_rss(r, x, d, v):
    m = pose_metrics(r, x, d, v)
    return np.nan if m is None else m['icond']


def icond_pss(v):
    p, Q1, Q2 = pose_from_params(PSS.home[2], v)
    xx = np.r_[p, xyz_from_rot(Q1), xyz_from_rot(Q2)]
    try:
        q = PSS.inverse(xx)
    except ValueError:
        return np.nan
    if np.abs(q).max() > PSS.stroke:
        return np.nan
    J, K = PSS.jacobians(xx, q)
    H = np.linalg.solve(K, J); H[:, :3] *= L_CHAR
    s = np.linalg.svd(H, compute_uv=False)
    return s[-1] / s[0]


def main():
    ps.style()
    designs = {n: load(n) for n in ('d25', 'd35')}
    summary = {}
    W = workspace()
    # ------------------------------------------------------------ summary
    for n, (D, x) in designs.items():
        det = evaluate(x, D['d'], W, details=True)
        ic = np.array([m['icond'] for m in det['all']])
        r = det['robot']
        p0 = np.array([0, 0, x[5]])
        th0 = r.ik(p0, np.eye(3), np.eye(3))
        tips = r.c + r.crank(th0)[0]
        summary[n] = dict(
            crank_d_mm=D['d'] / MM, link_l_mm=x[4] / MM, home_height_h_mm=x[5] / MM,
            tip_radius_Rt_mm=x[0] / MM, pivot_radius_mm=float(np.hypot(r.c[:, 0], r.c[:, 1]).max() / MM),
            gamma_deg=x[2] / DEG, beta_deg=x[3] / DEG, dpsi1_deg=x[1] / DEG, dpsi2_deg=x[6] / DEG, alt_deg=x[7] / DEG,
            icond_home=float(ic[0]), icond_min=float(ic.min()), icond_median=float(np.median(ic)),
            kappa_home=float(1 / ic[0]), kappa_max=float(1 / ic.min()),
            worst={k: float(v) for k, v in det['worst'].items()}, fails=det['fails'],
            crank_axes=r.v.tolist(), pivots_mm=(r.c / MM).tolist(), tips_home_mm=(tips / MM).tolist(),
            socket_axes_platform=det['socket_axis'].tolist())
    icp = np.array([icond_pss(v) for v in W])
    summary['pss_original'] = dict(icond_home=float(icp[0]), icond_min=float(np.nanmin(icp)),
                                   icond_median=float(np.nanmedian(icp)), kappa_home=float(1 / icp[0]))
    json.dump(summary, open(os.path.join(HERE, 'results', 'design_summary.json'), 'w'), indent=1)
    # ------------------------------------------------------------ geometry
    fig = plt.figure(figsize=(15, 7))
    for k, (n, (D, x)) in enumerate(designs.items()):
        r = make_robot(x, D['d'])
        a3 = fig.add_subplot(1, 4, 2 * k + 1, projection='3d')
        at = fig.add_subplot(1, 4, 2 * k + 2)
        for v, alpha in ((np.zeros(7), 1.0), (np.array([0, 0, 0, 0, 0, 0, 15 * DEG]), 0.25)):
            p, Q1, Q2 = pose_from_params(x[5], v)
            th = r.ik(p, Q1, Q2)
            kn = r.kin(p, Q1, Q2, th)
            tips = r.c + kn['D']; A = p + kn['Qa']
            for i in range(9):
                col = ps.C9[i]
                a3.plot(*np.c_[r.c[i], tips[i]] * 1e3, color='k', lw=3, alpha=alpha)
                a3.plot(*np.c_[tips[i], A[i]] * 1e3, color=col, lw=2, alpha=alpha)
                if alpha == 1:
                    at.plot(*np.c_[r.c[i, :2], tips[i, :2]] * 1e3, color='k', lw=3)
                    at.plot(*np.c_[tips[i, :2], A[i, :2]] * 1e3, color=col, lw=2)
                    at.plot(*r.c[i, :2] * 1e3, 'ko', ms=4)
                    at.text(*(tips[i, :2] * 1e3 * 1.08), str(i + 1), fontsize=8, ha='center')
            for kk, cc in ((1, ps.C9[0]), (2, ps.C9[5])):
                idx = np.flatnonzero(r.body == kk)
                poly = np.r_[A[idx], [p]]
                a3.plot_trisurf(poly[:, 0] * 1e3, poly[:, 1] * 1e3, poly[:, 2] * 1e3, color=cc, alpha=0.35 * alpha)
        a3.set_title(f'd = {D["d"] / MM:.0f} mm,  l = {x[4] / MM:.0f} mm,  h = {x[5] / MM:.0f} mm', fontsize=10)
        a3.set_box_aspect((1, 1, 0.9)); a3.view_init(22, -60)
        a3.set_xlabel('x'); a3.set_ylabel('y'); a3.set_zlabel('z (mm)')
        at.set_aspect('equal'); at.set_title(f'vista superior (manivelas en negro)', fontsize=10)
        at.set_xlabel('x (mm)'); at.set_ylabel('y (mm)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'design_geometry.png'), dpi=140); plt.close(fig)
    # ------------------------------------------------------------ conditioning maps
    gx = np.linspace(-35, 35, 36) * MM
    ga = np.linspace(-30, 30, 37) * DEG
    gyaw = np.linspace(-60, 60, 37) * DEG
    cases = [('PSS original (graspability)', None)] + [(f'RSS d = {D["d"] / MM:.0f} mm', (D, x)) for D, x in designs.values()]
    fig, ax = plt.subplots(3, 3, figsize=(14, 12))
    for col, (lab, dd) in enumerate(cases):
        if dd is not None:
            D, x = dd; r = make_robot(x, D['d'])
            f = lambda v: icond_rss(r, x, D['d'], v)
        else:
            f = icond_pss
        maps = []
        M1 = np.array([[f(np.array([a, b, 0, 0, 0, 0, 0])) for a in gx] for b in gx])
        M2 = np.array([[f(np.array([0, 0, 0, a, b, 0, 0])) for a in ga] for b in ga])
        M3 = np.array([[f(np.array([0, 0, 0, 0, 0, a, j])) for a in gyaw] for j in np.linspace(0, 20, 21) * DEG])
        for row, (M, ext, xl, yl) in enumerate(((M1, [-35, 35, -35, 35], 'x (mm)', 'y (mm)'),
                                                 (M2, [-30, 30, -30, 30], 'giro x (grados)', 'giro y (grados)'),
                                                 (M3, [-60, 60, 0, 20], 'giro z (grados)', 'pinza: medio ángulo (grados)'))):
            a = ax[row, col]
            im = a.imshow(np.log10(M), origin='lower', extent=ext, aspect='auto', vmin=-4, vmax=-1, cmap='viridis')
            cs = a.contour(np.log10(M), levels=[-3, -2.5, -2], colors='w', linewidths=0.8, origin='lower', extent=ext)
            a.clabel(cs, fmt=lambda v: f'1/κ=1e{v:.1f}', fontsize=7)
            a.set_xlabel(xl); a.set_ylabel(yl)
            if row == 0:
                a.set_title(lab)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, label='log10(1/κ)  (blanco: sin solución / fuera de carrera)')
    fig.suptitle('Condicionamiento 1/κ(K$^{-1}$J), L = 37.5 mm.  Fila 1: traslación en z = home.  '
                 'Fila 2: giro conjunto.  Fila 3: giro z con pinza abierta', fontsize=11)
    fig.savefig(os.path.join(FIG, 'conditioning_maps.png'), dpi=130); plt.close(fig)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in summary.items()}, indent=1))


if __name__ == '__main__':
    main()
