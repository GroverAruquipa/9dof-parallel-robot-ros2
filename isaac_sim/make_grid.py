"""Grid of N copies of the 9-DoF robot in one stage, each with its own servo stiffness.

Every copy references /World/ninedof of the robot USD under /World/envs/env_KK/ninedof (relationship targets are
remapped by the reference, so joints point to the bodies of their own copy). The fixed joint of each base is
overridden with the offset of its copy. Stiffness is log-spaced from kp_min (copy nearest to the camera) to kp_max
(farthest copy); damping is critical with the same armature as the single robot.

Used by grid_sim.py (physics) and render_grid.py (render); needs only usd-core, so it is checked offline.
"""
import math
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ROBOT_IN_FILE = '/World/ninedof'
CAM_AZ, CAM_EL = -128.0, -24.0                    # degrees, MuJoCo free-camera convention, same azimuth as the videos


def camera_forward(az=CAM_AZ, el=CAM_EL):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def layout(n, spacing, kp_min, kp_max):
    """Offsets of the copies on a square grid and their stiffness, softest nearest to the camera."""
    side = int(math.ceil(math.sqrt(n)))
    offs = [np.array([(c - (side - 1) / 2) * spacing, (r - (side - 1) / 2) * spacing, 0.0])
            for r in range(side) for c in range(side)][:n]
    fwd = camera_forward()
    order = np.argsort([o @ fwd for o in offs])              # nearest first (smallest depth along the view)
    kps = np.empty(n)
    kps[order] = np.geomspace(kp_min, kp_max, n) if n > 1 else [kp_min]
    return offs, kps


def build(path, robot_usd, n=16, spacing=0.25, kp_min=2e4, kp_max=2e6, armature=0.01, max_force=200.0, hz=1000):
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, '/World'); stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, '/World/physicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene.CreateGravityMagnitudeAttr(9.81)
    sp = scene.GetPrim(); sp.AddAppliedSchema('PhysxSceneAPI')
    sp.CreateAttribute('physxScene:solverType', Sdf.ValueTypeNames.Token).Set('TGS')
    sp.CreateAttribute('physxScene:timeStepsPerSecond', Sdf.ValueTypeNames.UInt).Set(int(hz))
    sp.CreateAttribute('physxScene:enableGPUDynamics', Sdf.ValueTypeNames.Bool).Set(False)
    UsdGeom.Scope.Define(stage, '/World/envs')

    rel = os.path.relpath(os.path.abspath(robot_usd), os.path.dirname(os.path.abspath(path)))
    offs, kps = layout(n, spacing, kp_min, kp_max)
    envs = []
    for k, (o, kp) in enumerate(zip(offs, kps)):
        env = f'/World/envs/env_{k:02d}'
        UsdGeom.Xform.Define(stage, env).AddTranslateOp().Set(Gf.Vec3d(*o))
        robot = stage.DefinePrim(f'{env}/ninedof')
        robot.GetReferences().AddReference(rel, ROBOT_IN_FILE)
        fix = UsdPhysics.Joint(stage.GetPrimAtPath(f'{env}/ninedof/joints/fixed_base'))
        base0 = np.array(fix.GetLocalPos0Attr().Get(), float)         # base origin in the robot file (world frame)
        fix.GetLocalPos0Attr().Set(Gf.Vec3f(*(base0 + o)))
        for i in range(1, 10):
            d = UsdPhysics.DriveAPI(stage.GetPrimAtPath(f'{env}/ninedof/joints/q{i}'), 'linear')
            d.GetStiffnessAttr().Set(float(kp)); d.GetDampingAttr().Set(float(2 * math.sqrt(kp * armature)))
            d.GetMaxForceAttr().Set(float(max_force))
        envs.append({'path': f'{env}/ninedof', 'offset': o, 'kp': float(kp)})
    stage.GetRootLayer().Save()
    return stage, envs


if __name__ == '__main__':                               # offline check with usd-core
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'results', 'grid_check.usda')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stage, envs = build(out, os.path.join(here, 'ninedof_cctomm_simple.usda'), n=int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    worst, n_joints, n_bodies, wrong_env = 0.0, 0, 0, 0
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            n_bodies += 1
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p); n_joints += 1
        env = str(p.GetPath()).split('/')[3]
        pts = []
        for rel_, lp in ((j.GetBody0Rel(), j.GetLocalPos0Attr()), (j.GetBody1Rel(), j.GetLocalPos1Attr())):
            t = rel_.GetTargets()
            if not t:
                pts.append(np.array(lp.Get(), float)); continue
            if str(t[0]).split('/')[3] != env:
                wrong_env += 1
            M = UsdGeom.Xformable(stage.GetPrimAtPath(t[0])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pts.append(np.array(M.Transform(Gf.Vec3d(*[float(c) for c in lp.Get()]))))
        worst = max(worst, float(np.linalg.norm(pts[0] - pts[1])))
    print(f'{len(envs)} copies, {n_bodies} bodies, {n_joints} joints, joints pointing to another copy: {wrong_env}, '
          f'anchor mismatch max {worst * 1e6:.3f} um')
    print('stiffness (N/m), nearest first:', ' '.join(f'{e["kp"]:.0f}' for e in sorted(envs, key=lambda e: e['kp'])))
