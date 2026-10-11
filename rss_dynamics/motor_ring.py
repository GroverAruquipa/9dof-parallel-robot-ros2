"""Nine vertical MPX-40C4WA motors on one circle under the base (as in the 3_unlimited_rotation
robot), as small as the motors allow.  Each crank gets a vertical transfer shaft with a 1:1
mitre (bevel) pair at the crank axis; motor -> transfer shaft by a horizontal GT2 belt
16T -> 64T (N = 4).  Motor envelope: body D 51 mm, square flange 67 mm, 140 mm long.

    python3 motor_ring.py d35_core_R100   -> results/motor_ring_<name>.json, figures/motor_ring_<name>.png
"""
import json, os, sys, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
from design import make_robot, NAMES, HERE
from params import pulley

FLANGE, BODY, CLEAR = 67.0, 51.0, 3.0           # mm
R_BIG, R_SMALL = pulley(64)['r'] * 1e3, pulley(16)['r'] * 1e3
n = 9
LEVEL = 25.0                                    # mm between flange levels (flange + belt)


def square(cx, cy, ang, s=FLANGE):
    h = s / 2; P = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return P @ R.T + [cx, cy]


def sq_gap(A, B):                                 # separating-axis distance of two squares
    best = -1e9
    for P in (A, B):
        for k in range(4):
            e = P[(k + 1) % 4] - P[k]; nr = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            a, b = A @ nr, B @ nr; best = max(best, b.min() - a.max(), a.min() - b.max())
    return best


def pt_sq(p, P):
    return min(np.linalg.norm(P[k] + np.clip((p - P[k]) @ (P[(k + 1) % 4] - P[k]) / ((P[(k + 1) % 4] - P[k]) @ (P[(k + 1) % 4] - P[k])), 0, 1)
                              * (P[(k + 1) % 4] - P[k]) - p) for k in range(4))


def min_ring(levels):
    """Smallest ring radius (flange sides facing radially): flanges on the same level must not
    touch; a flange must clear the bodies of motors on other levels."""
    for R in np.arange(60, 200, 0.25):
        ok = True
        for k in range(1, n):
            a = 2 * np.pi * k / n
            A, B = square(R, 0, 0), square(R * np.cos(a), R * np.sin(a), a)
            if (k % levels == 0 or (n - k) % levels == 0 and levels > 1 and False) or levels == 1:
                ok &= sq_gap(A, B) >= CLEAR
            else:
                ok &= pt_sq(np.array([R * np.cos(a), R * np.sin(a)]), A) >= BODY / 2 + CLEAR
        if ok:
            return R


R_same = min_ring(1)
R_stag = min_ring(3)

def layout(name, R, ring_offset, LEVELS=1):
    D = json.load(open(os.path.join(HERE, 'results', f'design_{name}.json')))
    x = np.array([D['x'][k] for k in NAMES]); r = make_robot(x, D['d'])
    # transfer shaft on the crank axis line, slid along v_i within +-25 mm to meet the belt best
    piv = r.c[:, :2] * 1e3; v = r.v[:, :2]
    ang = ring_offset + np.arange(n) * 2 * np.pi / n
    motors = R * np.c_[np.cos(ang), np.sin(ang)]
    order = np.argsort(np.arctan2(piv[:, 1], piv[:, 0]) % (2 * np.pi))
    best = None
    for rot in range(n):                      # keep the circular order, choose the rotation
        assign = {order[k]: (k + rot) % n for k in range(n)}
        C, shafts = [], []
        for i in range(n):
            m = motors[assign[i]]
            s_best = min(np.linspace(-25, 25, 51), key=lambda s: np.linalg.norm(piv[i] + s * v[i] - m))
            sh = piv[i] + s_best * v[i]; shafts.append(sh); C.append(np.linalg.norm(sh - m))
        C = np.array(C); shafts_a = np.array(shafts)
        if min(C) < R_BIG + R_SMALL + 2:      # pulleys would touch
            continue
        dd = np.linalg.norm(shafts_a[:, None] - shafts_a[None], axis=2) + 1e3 * np.eye(n)
        if np.any(dd < 22):
            continue                          # transfer shafts (and their mitre gears) too close
        clash = dd < 2 * R_BIG + 2            # 64T pulleys that would touch on one plane
        if LEVELS > 1:
            lev = np.array([assign[i] % LEVELS for i in range(n)])
            if np.any(clash & (lev[:, None] == lev[None])):
                continue
        else:                                 # one motor level: two belt planes (pulley hubs), 2-colouring
            lev = -np.ones(n, int)
            okc = True
            for st in range(n):
                if lev[st] >= 0:
                    continue
                lev[st] = 0; stack = [st]
                while stack:
                    a = stack.pop()
                    for b in np.flatnonzero(clash[a]):
                        if lev[b] < 0:
                            lev[b] = 1 - lev[a]; stack.append(b)
                        elif lev[b] == lev[a]:
                            okc = False
            if not okc:
                continue
        if best is None or C.max() < best[0].max():
            best = (C, shafts, assign, lev)
    if best is None:
        return None
    C, shafts, assign, lev = best
    belts = 2 * C + np.pi * (R_BIG + R_SMALL) + (R_BIG - R_SMALL) ** 2 / C
    return r, motors, np.array(shafts), assign, C, belts, lev

name = sys.argv[1] if len(sys.argv) > 1 else 'd35_core_R100'
out = {}
fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
for ax, (lab, R, stag) in zip(axs, (('mismo nivel', R_same, False), ('en 3 niveles', R_stag, True))):
    best = None
    for off in np.linspace(0, 2 * np.pi / n, 40, endpoint=False):
        res = layout(name, R, off, 3 if stag else 1)
        if res is not None and (best is None or res[4].max() < best[4].max()):
            best = res + (off,)
    r, motors, shafts, assign, C, belts, lev, off = best
    if best is None:
        out[lab] = dict(ring_radius_mm=R, feasible=False); continue
    corner = np.hypot(R + FLANGE / 2, FLANGE / 2)
    out[lab] = dict(ring_radius_mm=R, outer_radius_flange_corners_mm=corner, outer_radius_body_mm=R + BODY / 2,
                    belt_centre_distance_mm=C.round(1).tolist(), belt_teeth=np.round(belts / 2).astype(int).tolist(),
                    motor_azimuth_deg=np.degrees(off + np.arange(n) * 2 * np.pi / n).round(1).tolist(),
                    leg_to_motor={int(i + 1): int(assign[i] + 1) for i in range(n)},
                    transfer_shaft_xy_mm=shafts.round(1).tolist(),
                    belt_plane={int(i + 1): int(lev[i]) for i in range(n)},
                    depth_extra_mm=(2 * LEVEL if stag else 0))
    ang = off + np.arange(n) * 2 * np.pi / n
    for k, (mx, my) in enumerate(motors):
        sq = Rectangle((mx - FLANGE / 2, my - FLANGE / 2), FLANGE, FLANGE, fc=['0.80', '0.62', '0.45'][k % 3] if stag else '0.75', ec='0.3', alpha=0.55,
                       transform=Affine2D().rotate_around(mx, my, ang[k]) + ax.transData)
        ax.add_patch(sq); ax.add_patch(Circle((mx, my), BODY / 2, fc='0.35', ec='k', alpha=0.8))
        ax.add_patch(Circle((mx, my), R_SMALL, fc='gold', ec='k'))
        ax.text(mx * 1.27, my * 1.27, f'M{k + 1}' + (f' (nivel {k % 3})' if stag else ''), ha='center', va='center', fontsize=8)
    tips = r.c + r.crank(r.ik(np.array([0, 0, 0.13626]), np.eye(3), np.eye(3)))[0]
    for i in range(n):
        ax.add_patch(Circle(shafts[i], R_BIG, fc='none', ec=['tab:blue', 'tab:green', 'tab:purple'][lev[i]], lw=1.5))
        m = motors[assign[i]]; ax.plot(*np.c_[shafts[i], m], color='tab:blue', lw=1)
        ax.plot(*np.c_[r.c[i, :2], tips[i, :2]] * 1e3, 'r-', lw=3)
        ax.text(*(tips[i, :2] * 1e3 * 1.12), str(i + 1), color='r', fontsize=9, ha='center')
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(corner * np.cos(th), corner * np.sin(th), 'k--', lw=0.8)
    ax.plot(100 * np.cos(th), 100 * np.sin(th), 'r:', lw=0.8)
    ax.set_aspect('equal'); ax.set_xlim(-160, 160); ax.set_ylim(-160, 160); ax.grid(alpha=0.3)
    ax.set_title(f'Motores verticales {lab}: anillo R {R:.0f} mm, exterior R {corner:.0f} mm\n'
                 f'correas 16T-64T, distancia entre centros {C.min():.0f}-{C.max():.0f} mm', fontsize=11)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
fig.suptitle('\n' + 'Vista superior. Rojo: manivelas (radio barrido 100 mm, punteado). Azul: polea de 64T en el eje vertical '
             'de transferencia, con par cónico 1:1 a la manivela. Gris: brida y cuerpo del motor; amarillo: polea 16T', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); p = os.path.join(HERE, 'figures', f'motor_ring_{name}.png'); fig.savefig(p, dpi=120)
json.dump(out, open(os.path.join(HERE, 'results', f'motor_ring_{name}.json'), 'w'), indent=1)
print(json.dumps(out, indent=1)); print(p)
