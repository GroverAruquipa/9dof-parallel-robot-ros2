"""Placement of the robot copies of a grid from recorded poses (shared by render_grid.py and the offline check).

Needs np, Gf, Usd, UsdGeom, UsdPhysics in the namespace. Bodies of a copy live under an env Xform that carries the
copy offset, so the local translation written on a body is its world position minus that offset.
"""


def rot_between(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v); c = np.cross(u, v); s = np.linalg.norm(c)
    if s < 1e-12:
        return np.eye(3)
    k = c / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]); t = np.arctan2(s, u @ v)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-3:
        return Gf.Quatd(w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))
    q = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()
    return Gf.Quatd(q.GetReal(), q.GetImaginary())


class Body:
    """Body prim driven by translate + orient ops; origin = world position in the CAD pose of its copy."""

    def __init__(self, stage, path):
        prim = stage.GetPrimAtPath(path)
        x = UsdGeom.Xformable(prim)
        tc = Usd.TimeCode.Default()
        self.origin = np.array(x.ComputeLocalToWorldTransform(tc).ExtractTranslation())
        self.parent = np.array(UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(tc).ExtractTranslation())
        x.ClearXformOpOrder()
        self.t = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        self.o = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    def set(self, p, R=np.eye(3)):
        self.t.Set(Gf.Vec3d(*[float(c) for c in np.asarray(p) - self.parent])); self.o.Set(quat(R))


def build_copies(stage, robot_paths):
    """Collect each copy's legs from its joints, then remove every physics schema so nothing moves the bodies."""
    copies = []
    for rp in robot_paths:
        legs = []
        for p in Usd.PrimRange(stage.GetPrimAtPath(f'{rp}/joints')):
            if p.IsA(UsdPhysics.SphericalJoint) and p.GetName().endswith('_upper'):
                j = UsdPhysics.Joint(p)
                lk = j.GetBody0Rel().GetTargets()[0].name; plat = j.GetBody1Rel().GetTargets()[0].name
                k = lk[len('leg_'):]
                legs.append({'k': k, 'leg': Body(stage, f'{rp}/{lk}'), 'car': Body(stage, f'{rp}/carriage_{k}'),
                             'plat': plat, 'ab': np.array(j.GetLocalPos0Attr().Get(), float),
                             'ap': np.array(j.GetLocalPos1Attr().Get(), float)})
        copies.append({'legs': legs, 'plat1': Body(stage, f'{rp}/plat1'), 'plat2': Body(stage, f'{rp}/plat2')})
    phys = stage.GetPrimAtPath('/World/physicsScene')
    if phys:
        phys.SetActive(False)
    for rp in robot_paths:
        r = stage.GetPrimAtPath(rp)
        r.RemoveAPI(UsdPhysics.ArticulationRootAPI); r.RemoveAppliedSchema('PhysxArticulationAPI')
        for prim in list(Usd.PrimRange(r)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI); prim.RemoveAPI(UsdPhysics.MassAPI)
        stage.GetPrimAtPath(f'{rp}/joints').SetActive(False)
    return copies


def place_copy(c, p1, R1, p2, R2, q, col):
    P = {'plat1': (p1, R1), 'plat2': (p2, R2)}
    for name, (p, R) in P.items():
        c[name].set(p, R)
    for L in c['legs']:
        b = L['car'].origin + np.array([0, 0, q[col[L['k']]]])
        p, R = P[L['plat']]
        a = p + R @ L['ap']
        L['car'].set(b); L['leg'].set(b, rot_between(L['ab'], a - b))
