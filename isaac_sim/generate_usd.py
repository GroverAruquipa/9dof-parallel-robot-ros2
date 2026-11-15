"""Build the USD (Isaac Sim / PhysX) model of the 9-DoF 5PSS-S-4PSS robot from geometry_cctomm.yaml.

Closed loops in PhysX: one articulation tree plus loop-closing joints excluded from it.
  base (fixed to the world)
   +- carriage_i   prismatic along z at b_i0       (actuator q_i, position drive)        i = 1..9
       +- leg_i    spherical at b_i0               (lower ruby ball joint)
   leg1 -> plat1   spherical at a_10               (upper ball joint of leg 1, inside the tree)
   plat1 -> plat2  spherical at p0                 (central joint)
   leg_i -> plat   spherical at a_i0, i = 2..9     (8 loop closures, excludeFromArticulation)
Every body frame is parallel to the world frame in the CAD pose, with its origin on a joint centre,
so all joint coordinates are zero in the CAD pose.

Visuals:
  --visual simple   cylinders, spheres and boxes from the geometry only (public, no CAD needed)
  --visual cad      the 238 meshes of 9_dof_step_cctomm.STEP, one merged mesh per body and material
                    (needs --cad-dir; the CAD is private, keep that USD out of the public repository)

Usage:
  python3 generate_usd.py --visual simple -o ninedof_cctomm_simple.usda
  python3 generate_usd.py --visual cad --cad-dir .../robot_papers_9dof_6p1_4dof/sim/nine_dof -o ninedof_cctomm.usdc
Needs only usd-core (pip install usd-core), numpy, pyyaml and, for --visual cad, trimesh.
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = '/World/ninedof'
COLORS = {'white': (.95, .95, .94), 'light': (.80, .80, .80), 'mid': (.50, .50, .52), 'dark': (.33, .33, .35),
          'carbon': (.27, .27, .29), 'metal': (.70, .71, .73), 'ruby': (.80, .07, .10),
          'plat1': (.80, .25, .20), 'plat2': (.20, .40, .80)}
METALLIC = {'metal': .9, 'ruby': .1, 'carbon': .3}


def v3(x):
    return Gf.Vec3f(*[float(c) for c in x])


def make_materials(stage):
    mats = {}
    for name, rgb in COLORS.items():
        path = f'{ROBOT}/Looks/{name}'
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + '/shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.35 if name in METALLIC else 0.6)
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(METALLIC.get(name, 0.0))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        mats[name] = mat
    return mats


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def add_mass(prim, props, min_inertia):
    """MassAPI from mass, COM (body frame) and inertia about the COM (world = body axes)."""
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(props['mass'])
    m.CreateCenterOfMassAttr(v3(props['com']))
    w, V = np.linalg.eigh(np.array(props['inertia']))
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]
    w = np.maximum(w, min_inertia)
    m.CreateDiagonalInertiaAttr(v3(w))
    m.CreatePrincipalAxesAttr(Gf.Quatf(*quat_wxyz(V)))     # columns of V = principal axes in the body frame


def quat_wxyz(R):
    """Unit quaternion (w, x, y, z) of a proper rotation matrix (columns = rotated axes)."""
    R = np.asarray(R, float); t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(1 + t); q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4; q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q); q /= np.linalg.norm(q)
    return [float(c) for c in q]


def make_body(stage, name, origin, props, min_inertia, dynamic=True):
    xf = UsdGeom.Xform.Define(stage, f'{ROBOT}/{name}')
    xf.AddTranslateOp().Set(Gf.Vec3d(*origin))
    prim = xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if props is not None:
        add_mass(prim, props, min_inertia)
    return prim


def joint_common(j, body0, body1, pos0, pos1):
    if body0:
        j.CreateBody0Rel().SetTargets([Sdf.Path(f'{ROBOT}/{body0}')])
    j.CreateBody1Rel().SetTargets([Sdf.Path(f'{ROBOT}/{body1}')])
    j.CreateLocalPos0Attr(v3(pos0)); j.CreateLocalPos1Attr(v3(pos1))
    j.CreateLocalRot0Attr(Gf.Quatf(1)); j.CreateLocalRot1Attr(Gf.Quatf(1))


def spherical(stage, name, body0, body1, pos0, pos1, exclude=False):
    j = UsdPhysics.SphericalJoint.Define(stage, f'{ROBOT}/joints/{name}')
    joint_common(j, body0, body1, pos0, pos1)
    j.CreateAxisAttr('Z')
    if exclude:
        j.CreateExcludeFromArticulationAttr(True)
    return j


# ---------------------------------------------------------------- visuals
def cylinder_between(stage, path, a, b, r, mat):
    a = np.asarray(a, float); b = np.asarray(b, float); d = b - a; h = float(np.linalg.norm(d))
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(r); c.CreateHeightAttr(h); c.CreateAxisAttr('Z')
    z = d / h; q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*z)).GetQuat()
    c.AddTranslateOp().Set(Gf.Vec3d(*((a + b) / 2)))
    c.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()), *[float(x) for x in q.GetImaginary()]))
    bind(c.GetPrim(), mat)


def sphere(stage, path, centre, r, mat):
    s = UsdGeom.Sphere.Define(stage, path); s.CreateRadiusAttr(r)
    s.AddTranslateOp().Set(Gf.Vec3d(*centre)); bind(s.GetPrim(), mat)


def box(stage, path, centre, size, mat):
    c = UsdGeom.Cube.Define(stage, path); c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*centre)); c.AddScaleOp().Set(Gf.Vec3f(*size)); bind(c.GetPrim(), mat)


def simple_visuals(stage, geo, mats):
    """Body-frame primitives: base disc, carriages, distal links, ball joints and two platform discs."""
    p0 = np.array(geo['p0'])
    b0 = np.array([l['b0'] for l in geo['legs']])
    c = b0[:, :2].mean(0); r_base = float(np.max(np.linalg.norm(b0[:, :2] - c, axis=1))) + 0.015
    base = UsdGeom.Cylinder.Define(stage, f'{ROBOT}/base/visual/plate')
    base.CreateRadiusAttr(r_base); base.CreateHeightAttr(0.006)
    base.AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), float(b0[:, 2].min()) - 0.045))
    bind(base.GetPrim(), mats['light'])
    for leg in geo['legs']:
        n = leg['name']; b = np.array(leg['b0']); a = np.array(leg['a0'])
        car = f'carriage_{leg["cad_instance"]}'; lk = f'leg_{leg["cad_instance"]}'
        box(stage, f'{ROBOT}/{car}/visual/stage', (0, 0, -0.022), (0.012, 0.012, 0.040), mats['light'])
        rail_top = b[2] - 0.042 - (b0[:, 2].min() - 0.042)
        box(stage, f'{ROBOT}/base/visual/rail_{n}', (b[0], b[1], b0[:, 2].min() - 0.042 + rail_top / 2 - 0.0),
            (0.008, 0.008, max(rail_top, 0.004)), mats['white'])
        sphere(stage, f'{ROBOT}/{car}/visual/ball', (0, 0, 0), 0.0025, mats['ruby'])
        cylinder_between(stage, f'{ROBOT}/{lk}/visual/tube', (0, 0, 0), a - b, 0.0025, mats['carbon'])
        sphere(stage, f'{ROBOT}/{lk}/visual/ball', a - b, 0.0025, mats['ruby'])
    for k in (1, 2):
        pts = np.array([l['a0'] for l in geo['legs'] if l['platform'] == k]) - p0
        ctr = pts.mean(0); rad = float(np.max(np.linalg.norm(pts[:, :2] - ctr[:2], axis=1))) + 0.004
        d = UsdGeom.Cylinder.Define(stage, f'{ROBOT}/plat{k}/visual/disc')
        d.CreateRadiusAttr(rad); d.CreateHeightAttr(0.004)
        d.AddTranslateOp().Set(Gf.Vec3d(*ctr)); bind(d.GetPrim(), mats[f'plat{k}'])
        cylinder_between(stage, f'{ROBOT}/plat{k}/visual/arm', ctr, (0, 0, 0), 0.002, mats[f'plat{k}'])
        cylinder_between(stage, f'{ROBOT}/plat{k}/visual/finger', (0, 0, 0), (0, 0, 0.045), 0.0015,
                         mats['dark'])
    sphere(stage, f'{ROBOT}/plat1/visual/central', (0, 0, 0), 0.004, mats['white'])


def cad_visuals(stage, geo, mats, cad_dir):
    """CAD meshes merged per (body, material); vertices moved to the body frame (origin offset)."""
    import trimesh
    sys.path.insert(0, cad_dir); sys.path.insert(0, os.path.dirname(cad_dir))
    sys.modules.setdefault('mujoco', types.ModuleType('mujoco'))
    from anim_model import assign_groups
    from sheetlib import style_of
    from rules import RULES
    sys.path.insert(0, HERE)
    from export_cctomm import body_of_group
    parts = json.load(open(os.path.join(cad_dir, 'meshes', 'parts.json')))
    merged = {}
    for p, g in zip(parts, assign_groups(parts)):
        b = body_of_group(g); mat = style_of(p['name'], RULES)
        merged.setdefault((b, mat), []).append(trimesh.load(os.path.join(cad_dir, 'meshes', p['file'])))
    for (b, mat), ms in sorted(merged.items()):
        m = trimesh.util.concatenate(ms)
        o = np.array(geo['bodies'][b]['origin'])
        mesh = UsdGeom.Mesh.Define(stage, f'{ROBOT}/{b}/visual/{mat}')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((m.vertices - o).astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.faces), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m.faces.astype(np.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr('none')
        bind(mesh.GetPrim(), mats[mat])
    return len(merged)


# ---------------------------------------------------------------- model
def build(geo, out, visual, cad_dir=None):
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, '/World'); stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, '/World/physicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene.CreateGravityMagnitudeAttr(9.81)
    sp = scene.GetPrim(); sp.AddAppliedSchema('PhysxSceneAPI')
    sp.CreateAttribute('physxScene:solverType', Sdf.ValueTypeNames.Token).Set('TGS')
    sp.CreateAttribute('physxScene:timeStepsPerSecond', Sdf.ValueTypeNames.UInt).Set(2000)
    sp.CreateAttribute('physxScene:enableGPUDynamics', Sdf.ValueTypeNames.Bool).Set(False)

    robot = UsdGeom.Xform.Define(stage, ROBOT).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    robot.AddAppliedSchema('PhysxArticulationAPI')
    robot.CreateAttribute('physxArticulation:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(64)
    robot.CreateAttribute('physxArticulation:solverVelocityIterationCount', Sdf.ValueTypeNames.Int).Set(4)
    robot.CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool).Set(False)
    mats = make_materials(stage)
    UsdGeom.Scope.Define(stage, f'{ROBOT}/joints')

    B = geo['bodies']; mi = float(geo['min_inertia'])
    make_body(stage, 'base', B['base']['origin'], B['base'], mi)
    fix = UsdPhysics.FixedJoint.Define(stage, f'{ROBOT}/joints/fixed_base')
    joint_common(fix, None, 'base', B['base']['origin'], (0, 0, 0))

    dr = geo['drive']; lo, hi = geo['actuator_limits']; p0 = np.array(geo['p0'])
    make_body(stage, 'plat1', p0, B['plat1'], mi); make_body(stage, 'plat2', p0, B['plat2'], mi)
    for i, leg in enumerate(geo['legs']):
        k = leg['cad_instance']; b = np.array(leg['b0']); a = np.array(leg['a0'])
        car, lk = f'carriage_{k}', f'leg_{k}'
        make_body(stage, car, b, B[car], mi); make_body(stage, lk, b, B[lk], mi)
        # actuator: prismatic along z, position drive (same servo as the MuJoCo model)
        j = UsdPhysics.PrismaticJoint.Define(stage, f'{ROBOT}/joints/q{i + 1}')
        joint_common(j, 'base', car, b - np.array(B['base']['origin']), (0, 0, 0))
        j.CreateAxisAttr('Z'); j.CreateLowerLimitAttr(lo); j.CreateUpperLimitAttr(hi)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), 'linear')
        d.CreateTypeAttr('force'); d.CreateTargetPositionAttr(0.0)
        d.CreateStiffnessAttr(dr['stiffness']); d.CreateDampingAttr(dr['damping'])
        d.CreateMaxForceAttr(dr['max_force'])
        jp = j.GetPrim(); jp.AddAppliedSchema('PhysxJointAPI')
        jp.CreateAttribute('physxJoint:armature', Sdf.ValueTypeNames.Float).Set(dr['armature'])
        # lower ball joint and upper ball joint (loop closure except for leg 1)
        spherical(stage, f'{leg["name"]}_lower', car, lk, (0, 0, 0), (0, 0, 0))
        plat = f'plat{leg["platform"]}'
        spherical(stage, f'{leg["name"]}_upper', lk, plat, a - b, a - p0, exclude=(i > 0))
    spherical(stage, 'central', 'plat1', 'plat2', (0, 0, 0), (0, 0, 0))

    n_vis = None
    if visual == 'simple':
        simple_visuals(stage, geo, mats)
    else:
        n_vis = cad_visuals(stage, geo, mats, os.path.abspath(cad_dir))
    stage.GetRootLayer().Save()
    return stage, n_vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geometry', default=os.path.join(HERE, 'geometry_cctomm.yaml'))
    ap.add_argument('--visual', choices=('simple', 'cad'), default='simple')
    ap.add_argument('--cad-dir')
    ap.add_argument('-o', '--out')
    a = ap.parse_args()
    if a.visual == 'cad' and not a.cad_dir:
        ap.error('--visual cad needs --cad-dir')
    out = a.out or os.path.join(HERE, 'ninedof_cctomm.usdc' if a.visual == 'cad' else 'ninedof_cctomm_simple.usda')
    geo = yaml.safe_load(open(a.geometry))
    if os.path.exists(out):
        os.remove(out)
    stage, n_vis = build(geo, out, a.visual, a.cad_dir)
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    excl = [p for p in joints if UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()]
    print(f'{out}: {len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)])} bodies, '
          f'{len(joints)} joints ({len(excl)} loop closures), visual={a.visual}'
          + (f' ({n_vis} merged CAD meshes)' if n_vis else ''))


if __name__ == '__main__':
    main()
