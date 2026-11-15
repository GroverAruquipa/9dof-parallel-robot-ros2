"""Export the CCToMM geometry of the 9-DoF robot for Isaac Sim.

Reads the kinematic model and the CAD parts of the private repository robot_papers_9dof_6p1_4dof
(sim/nine_dof: kin.py, anim_model.py, meshes/parts.json) and writes, next to this script:

  geometry_cctomm.yaml       joint points, leg lengths, body groups and estimated mass properties
  reference_trajectory.npz   showcase trajectory (same script as nine_dof_motions.mp4) solved with the
                             paper inverse kinematics: t, p, Q1, Q2 and the nine actuator positions q

All lengths in metres, CAD world frame (z up), which is also the Isaac Sim world frame.
Reference pose (all joint coordinates zero): the CAD pose, q_i = 0, Q1 = Q2 = I, p = P0.

Usage:  python3 export_cctomm.py --cad-dir /path/to/robot_papers_9dof_6p1_4dof/sim/nine_dof
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import trimesh
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Densities (kg/m^3) by render material of the CAD part. ESTIMATES: replace the body masses with the
# SolidWorks Mass Properties when available.
DENSITY = {'white': 1240.0,   # printed supports and central joint parts (PLA)
           'light': 2700.0,   # linear stages (aluminium)
           'mid': 1240.0,     # platforms, brackets (printed)
           'dark': 7800.0,    # tweezers grippers (steel)
           'carbon': 1600.0,  # distal tubes
           'metal': 7800.0,   # rods, pins, screws, bearings
           'ruby': 4000.0}    # ruby balls
MIN_INERTIA = 1e-8            # kg m^2, lower bound on the principal moments (PhysX stability)


def load_cad(cad_dir):
    sys.path.insert(0, cad_dir)
    sys.path.insert(0, os.path.dirname(cad_dir))
    sys.modules.setdefault('mujoco', types.ModuleType('mujoco'))   # only needed for rendering there
    import kin
    from anim_model import assign_groups
    from sheetlib import style_of
    from rules import RULES
    parts = json.load(open(os.path.join(cad_dir, 'meshes', 'parts.json')))
    return kin, parts, assign_groups(parts), lambda name: style_of(name, RULES)


def body_of_group(group):
    """CAD group -> rigid body. The cardan of the central joint is visual only for now: yoke O and
    the core move with platform 1, yoke U with platform 2 (the physics uses one spherical joint)."""
    if group in ('sj_yokeO', 'sj_core'):
        return 'plat1'
    if group == 'sj_yokeU':
        return 'plat2'
    return group


def mass_properties(cad_dir, parts, groups, material, origins):
    """Mass, centre of mass and inertia (about the COM, world axes) of every body, from the CAD meshes."""
    acc = {}
    for p, g in zip(parts, groups):
        b = body_of_group(g)
        m = trimesh.load(os.path.join(cad_dir, 'meshes', p['file']))
        if not m.is_watertight:
            m = m.convex_hull                      # hollow or open shells: hull, an upper bound
        m.density = DENSITY[material(p['name'])]
        mp = m.mass_properties
        a = acc.setdefault(b, [0.0, np.zeros(3), []])
        a[0] += mp['mass']; a[1] += mp['mass'] * mp['center_mass']
        a[2].append((mp['mass'], np.array(mp['center_mass']), np.array(mp['inertia'])))
    out = {}
    for b, (M, mc, items) in acc.items():
        c = mc / M
        I = np.zeros((3, 3))
        for mi, ci, Ii in items:                   # parallel axis theorem to the body COM
            d = ci - c
            I += Ii + mi * (d @ d * np.eye(3) - np.outer(d, d))
        out[b] = {'mass': float(M),
                  'com': [float(x) for x in c - origins[b]],      # in the body frame (origin offset)
                  'inertia': [[float(x) for x in row] for row in I]}
    return out


def showcase(kin):
    """Same motion script as video.py, sampled at 100 Hz, with 0.4 s holds between segments."""
    I3 = np.eye(3); P0 = kin.P0
    ease = lambda u: 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))
    swing = lambda u: np.sin(2 * np.pi * ease(u))
    EX, EY, EZ = (1, 0, 0), (0, 1, 0), (0, 0, 1)

    def trans(ax, A):
        return lambda u: (P0 + A * swing(u) * np.array(ax, float), I3, I3)

    def rigid(ax, A):
        return lambda u: (P0, kin.rot(ax, A * swing(u)), kin.rot(ax, A * swing(u)))

    def cone(tilt, turns):
        def f(u):
            ramp = ease(u / 0.15) * ease((1 - u) / 0.15); psi = 2 * np.pi * turns * ease(u)
            R = kin.rot((np.cos(psi), np.sin(psi), 0), tilt * ramp)
            return P0, R, R
        return f

    def relative(ax, keys):
        ks = np.array([0] + list(keys) + [0], float)
        def f(u):
            x = u * (len(ks) - 1); i = min(int(x), len(ks) - 2)
            a = ks[i] + (ks[i + 1] - ks[i]) * ease(x - i)
            return P0, kin.rot(ax, a), kin.rot(ax, -a)
        return f

    hold = lambda u: (P0, I3, I3)
    script = [('hold', 2.0, hold),
              ('translation x', 4.0, trans(EX, 15)), ('translation y', 4.0, trans(EY, 15)),
              ('translation z', 4.0, trans(EZ, 15)),
              ('rigid rotation x', 4.5, rigid(EX, 30)), ('rigid rotation y', 4.5, rigid(EY, 30)),
              ('rigid rotation z', 4.5, rigid(EZ, 30)), ('cone', 7.0, cone(30, 2)),
              ('relative y (grasp)', 6.0, relative(EY, [25, -5, 25, -5])),
              ('relative x', 4.5, relative(EX, [30, -30])), ('relative z', 4.5, relative(EZ, [30, -30])),
              ('hold', 2.0, hold)]
    rate, gap = 100, 0.4
    rows, seg = [], []
    for k, (name, dur, f) in enumerate(script):
        for i in range(int(round(dur * rate))):
            rows.append(f(i / (dur * rate - 1))); seg.append(k)
        for _ in range(int(gap * rate)):
            rows.append(f(1.0)); seg.append(k)
    return rows, np.array(seg), [s[0] for s in script], rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cad-dir', required=True, help='robot_papers_9dof_6p1_4dof/sim/nine_dof')
    a = ap.parse_args()
    cad_dir = os.path.abspath(a.cad_dir)
    kin, parts, groups, material = load_cad(cad_dir)

    mm = 1e-3
    names = list(kin.LEGS)                                     # platform 1 legs first, then platform 2
    legs = []
    for i, k in enumerate(names, 1):
        legs.append({'name': f'leg{i}', 'cad_instance': k, 'platform': int(kin.PLAT[k]),
                     'b0': [round(float(x) * mm, 7) for x in kin.B0[k]],
                     'a0': [round(float(x) * mm, 7) for x in kin.A0[k]],
                     'length': round(float(kin.L[k]) * mm, 7)})
    P0 = kin.P0 * mm
    origins = {'base': np.zeros(3), 'plat1': P0, 'plat2': P0}
    for k in names:
        origins[f'carriage_{k}'] = kin.B0[k] * mm
        origins[f'leg_{k}'] = kin.B0[k] * mm
    masses = mass_properties(cad_dir, parts, groups, material, origins)

    rows, seg, seg_names, rate = showcase(kin)
    n = len(rows)
    t = np.arange(n) / rate
    p = np.zeros((n, 3)); Q1 = np.zeros((n, 3, 3)); Q2 = np.zeros((n, 3, 3)); q = np.zeros((n, 9))
    for j, (pj, Q1j, Q2j) in enumerate(rows):
        sol = kin.ik(pj, Q1j, Q2j)
        p[j] = pj * mm; Q1[j] = Q1j; Q2[j] = Q2j
        q[j] = [sol[k][0] * mm for k in names]
    np.savez_compressed(os.path.join(HERE, 'reference_trajectory.npz'), t=t, p=p, Q1=Q1, Q2=Q2, q=q,
                        segment=seg, segment_names=np.array(seg_names), legs=np.array(names))

    geo = {
        'source': '9_dof_step_cctomm.STEP (robot_papers_9dof_6p1_4dof/sim/nine_dof/kin.py)',
        'units': 'metre, kilogram; CAD world frame, z up',
        'reference_pose': 'all joint coordinates zero = CAD pose: q_i = 0, Q1 = Q2 = I, p = p0',
        'p0': [round(float(x), 7) for x in P0],
        'actuator_axis': [0.0, 0.0, 1.0],
        'actuator_limits': [round(float(q.min()) - 0.003, 4), round(float(q.max()) + 0.003, 4)],
        'legs': legs,
        'central_joint': {'type': 'spherical', 'note': 'CAD: cardan with two swivels (4 concurrent axes, '
                          'one idle roll); one spherical joint in the physics, cardan parts are visual'},
        'bodies': {b: {'origin': [round(float(x), 7) for x in origins[b]], **{k: v for k, v in
                   masses[b].items()}} for b in origins},
        'mass_note': 'ESTIMATES from the CAD meshes and assumed densities ' + json.dumps(DENSITY)
                     + '; open shells use their convex hull',
        'min_inertia': MIN_INERTIA,
        'drive': {'stiffness': 5.0e4, 'damping': 44.7, 'max_force': 20.0, 'armature': 0.01,
                  'note': 'same servo as the MuJoCo model (kp 5e4 N/m, critical damping with 0.01 kg)'},
    }
    with open(os.path.join(HERE, 'geometry_cctomm.yaml'), 'w') as f:
        f.write('# Generated by export_cctomm.py. Do not edit by hand.\n')
        yaml.safe_dump(geo, f, sort_keys=False, width=120)
    print(f'{len(legs)} legs, {n} trajectory samples ({t[-1]:.1f} s), q range '
          f'[{q.min() * 1000:.2f}, {q.max() * 1000:.2f}] mm')
    for b in sorted(masses):
        print(f'  {b:18s} {masses[b]["mass"] * 1000:8.2f} g')


if __name__ == '__main__':
    main()
