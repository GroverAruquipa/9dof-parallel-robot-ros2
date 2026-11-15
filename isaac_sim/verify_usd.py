"""Offline checks of the generated USD (no Isaac Sim needed, only usd-core).

1. Every joint: the anchor seen from body 0 and from body 1 is the same world point in the CAD pose
   (otherwise PhysX pulls the bodies together at the first step).
2. Articulation tree: exactly one root (fixed base), the 8 excluded joints close loops.
3. Masses and inertias are positive.
4. With --cad-dir: the ruby balls of the CAD lie on the ball-joint centres b_i0 and a_i0.

Usage: python3 verify_usd.py ninedof_cctomm_simple.usda [--cad-dir .../sim/nine_dof]
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import yaml
from pxr import Usd, UsdGeom, UsdPhysics

HERE = os.path.dirname(os.path.abspath(__file__))


def world_point(stage, body_path, local):
    if not body_path:
        return np.asarray(local, float)
    x = UsdGeom.Xformable(stage.GetPrimAtPath(body_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(x.Transform(tuple(float(c) for c in local)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('usd'); ap.add_argument('--cad-dir')
    ap.add_argument('--geometry', default=os.path.join(HERE, 'geometry_cctomm.yaml'))
    a = ap.parse_args()
    stage = Usd.Stage.Open(a.usd)
    ok = True

    worst = 0.0; n = 0; excl = 0; tree_edges = []
    for p in stage.Traverse():
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p); n += 1
        b0 = j.GetBody0Rel().GetTargets(); b1 = j.GetBody1Rel().GetTargets()
        w0 = world_point(stage, b0[0] if b0 else None, j.GetLocalPos0Attr().Get())
        w1 = world_point(stage, b1[0], j.GetLocalPos1Attr().Get())
        worst = max(worst, float(np.linalg.norm(w0 - w1)))
        if j.GetExcludeFromArticulationAttr().Get():
            excl += 1
        else:
            tree_edges.append((str(b0[0]) if b0 else 'world', str(b1[0])))
    bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    parents = {}
    for u, v in tree_edges:
        parents.setdefault(v, []).append(u)
    multi = [b for b, ps in parents.items() if len(ps) > 1]
    orphan = [b for b in bodies if b not in parents]
    print(f'joints: {n}, anchor mismatch max {worst * 1e6:.3f} um')
    print(f'tree: {len(tree_edges)} edges for {len(bodies)} bodies, {excl} loop closures, '
          f'bodies with 2 parents {multi}, without parent {orphan}')
    ok &= worst < 1e-6 and not multi and not orphan and len(tree_edges) == len(bodies)

    bad = []
    for b in bodies:
        m = UsdPhysics.MassAPI(stage.GetPrimAtPath(b))
        if m.GetMassAttr().Get() is None or m.GetMassAttr().Get() <= 0 or min(m.GetDiagonalInertiaAttr().Get()) <= 0:
            bad.append(b)
    print(f'mass properties: {"ok" if not bad else bad}')
    ok &= not bad

    if a.cad_dir:
        cad = os.path.abspath(a.cad_dir)
        sys.path.insert(0, cad); sys.path.insert(0, os.path.dirname(cad))
        sys.modules.setdefault('mujoco', types.ModuleType('mujoco'))
        import trimesh
        from anim_model import assign_groups
        geo = yaml.safe_load(open(a.geometry))
        parts = json.load(open(os.path.join(cad, 'meshes', 'parts.json')))
        balls = [trimesh.load(os.path.join(cad, 'meshes', p['file'])) for p in parts if '_sphere' in p['name']]
        centres = np.array([m.bounding_sphere.primitive.center for m in balls])
        errs = []
        for leg in geo['legs']:
            for key in ('b0', 'a0'):
                errs.append(float(np.min(np.linalg.norm(centres - np.array(leg[key]), axis=1))))
        print(f'CAD ruby balls vs joint centres: {len(balls)} balls, max distance {max(errs) * 1000:.3f} mm')
        ok &= max(errs) < 0.2e-3
    print('ALL CHECKS PASSED' if ok else 'CHECK FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
