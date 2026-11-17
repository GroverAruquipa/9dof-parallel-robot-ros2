"""Render the PhysX run of the 9-DoF robot with the Isaac Sim RTX renderer (headless), one PNG per frame.

The poses come from a run of run_validation.py (default results/isaac_gravity.npz: platforms p1, R1, p2, R2
and actuators q at 100 Hz); the carriages and distal links are placed from them, physics is switched off.
House style of the MuJoCo videos (sim/STYLE.md): 1920x1080, 30 fps, white background, no shadows, fixed
wide camera (lookat (-0.040, -0.040, 0.105) m, distance 0.39 m, azimuth -128 deg, elevation -18 deg,
vertical field of view 45 deg), caption in the top-left corner.

Run in the container:  ./python.sh /work/isaac_sim/render_video.py [--usd ...] [--frames 3]
then encode on the VM: ffmpeg -framerate 30 -i results/frames/%05d.png -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT = '/World/ninedof'
FPS, W, H = 30, 1920, 1080
CAM = ([-0.040, -0.040, 0.105], 0.39, -128.0, -18.0)        # lookat, distance, azimuth, elevation (MuJoCo)
TITLES = {'hold': 'Reference pose', 'translation x': 'Translation along x', 'translation y': 'Translation along y',
          'translation z': 'Translation along z', 'rigid rotation x': 'Rigid rotation about x',
          'rigid rotation y': 'Rigid rotation about y', 'rigid rotation z': 'Rigid rotation about z',
          'cone': 'Tilt about a turning axis', 'relative y (grasp)': 'Relative rotation about y: grasp',
          'relative x': 'Relative rotation about x', 'relative z': 'Relative rotation about z'}

ap = argparse.ArgumentParser()
cad = os.path.join(HERE, 'ninedof_cctomm.usdc')
ap.add_argument('--usd', default=cad if os.path.exists(cad) else os.path.join(HERE, 'ninedof_cctomm_simple.usda'))
ap.add_argument('--run', default=os.path.join(HERE, 'results', 'isaac_gravity.npz'))
ap.add_argument('--subtitle', default='Isaac Sim 6.1, PhysX simulation with gravity, RTX render')
ap.add_argument('--frames', type=int, help='only N frames spread over the run (look check)')
ap.add_argument('--out', default=os.path.join(HERE, 'results', 'frames'))
args = ap.parse_args()

from isaacsim import SimulationApp                         # noqa: E402
app = SimulationApp({'headless': True, 'width': W, 'height': H})

import omni.replicator.core as rep                         # noqa: E402
import omni.usd                                            # noqa: E402
from PIL import Image, ImageDraw, ImageFont                 # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(os.path.abspath(args.usd))
for _ in range(10):
    app.update()
stage = ctx.get_stage()
stage.GetPrimAtPath('/World/physicsScene').SetActive(False)          # kinematic playback of the PhysX run


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
    q = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()   # rare large-angle branch
    return Gf.Quatd(q.GetReal(), q.GetImaginary())


class Body:
    """Rigid body prim driven by translate + orient ops; origin = CAD-pose translation."""

    def __init__(self, name):
        x = UsdGeom.Xformable(stage.GetPrimAtPath(f'{ROBOT}/{name}'))
        self.origin = np.array(x.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
        x.ClearXformOpOrder()
        self.t = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        self.o = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    def set(self, p, R=np.eye(3)):
        self.t.Set(Gf.Vec3d(*[float(c) for c in p])); self.o.Set(quat(R))


# geometry from the USD itself: leg i hangs from carriage i, its upper joint names the platform
legs = []
for p in stage.Traverse():
    if p.IsA(UsdPhysics.SphericalJoint) and p.GetName().endswith('_upper'):
        j = UsdPhysics.Joint(p)
        lk = j.GetBody0Rel().GetTargets()[0].name; plat = j.GetBody1Rel().GetTargets()[0].name
        legs.append({'k': lk[len('leg_'):], 'leg': Body(lk), 'car': Body('carriage_' + lk[len('leg_'):]),
                     'plat': plat, 'ab': np.array(j.GetLocalPos0Attr().Get(), float),
                     'ap': np.array(j.GetLocalPos1Attr().Get(), float)})
plats = {'plat1': Body('plat1'), 'plat2': Body('plat2')}

ref = np.load(os.path.join(HERE, 'reference_trajectory.npz'))
run = np.load(args.run)
col = {str(k): i for i, k in enumerate(ref['legs'])}
seg_names = [str(s) for s in ref['segment_names']]


def pose_at(j):
    P = {'plat1': (run['p1'][j], run['R1'][j]), 'plat2': (run['p2'][j], run['R2'][j])}
    for name, (p, R) in P.items():
        plats[name].set(p, R)
    for L in legs:
        b = L['car'].origin + np.array([0, 0, run['q'][j][col[L['k']]]])
        p, R = P[L['plat']]
        a = p + R @ L['ap']
        L['car'].set(b); L['leg'].set(b, rot_between(L['ab'], a - b))


# scene: white floor under the base, white sky, no shadows
lo = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(
    stage.GetPrimAtPath(f'{ROBOT}/base')).ComputeAlignedRange().GetMin()
floor = UsdGeom.Cube.Define(stage, '/World/floor'); floor.CreateSizeAttr(1.0)
floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, lo[2] - 0.0025)); floor.AddScaleOp().Set(Gf.Vec3f(4, 4, 0.005))
floor.CreateDisplayColorAttr([(0.97, 0.97, 0.97)])
dome = UsdLux.DomeLight.Define(stage, '/World/sky'); dome.CreateIntensityAttr(900); dome.CreateColorAttr((1, 1, 1))
key = UsdLux.DistantLight.Define(stage, '/World/key'); key.CreateIntensityAttr(1800); key.CreateAngleAttr(2.0)
key.AddRotateXYZOp().Set(Gf.Vec3f(-40, 25, 30))
for lt in (dome, key):
    UsdLux.ShadowAPI.Apply(lt.GetPrim()).CreateShadowEnableAttr(False)

# camera: MuJoCo free-camera convention, vertical field of view 45 deg
la, dist, az, el = CAM
a, e = np.radians(az), np.radians(el)
fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
pos = np.array(la) - dist * fwd
hap = 20.955; focal = (hap / 2) / np.tan(np.arctan(np.tan(np.radians(22.5)) * W / H))
cam = rep.create.camera(position=tuple(pos), look_at=tuple(la), focal_length=focal, horizontal_aperture=hap,
                        clipping_range=(0.01, 100.0))
rp = rep.create.render_product(cam, (W, H))
rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([rp])


def font(size):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


F = (font(40), font(30))
os.makedirs(args.out, exist_ok=True)
n_run = len(run['t']); t_end = run['t'][-1]
frames = np.arange(int(t_end * FPS) + 1)
if args.frames:
    frames = np.linspace(0, frames[-1], args.frames).astype(int)
pose_at(0)
for _ in range(20):                                        # warm-up: shaders and textures
    rep.orchestrator.step(rt_subframes=4)
for i, f in enumerate(frames):
    j = min(int(round(f / FPS * 100)), n_run - 1)
    pose_at(j)
    rep.orchestrator.step(rt_subframes=4)
    img = Image.fromarray(np.asarray(rgb.get_data())[:, :, :3])
    dr = ImageDraw.Draw(img)
    dr.text((70, 58), TITLES.get(seg_names[ref['segment'][j]], ''), fill=(20, 20, 20), font=F[0])
    dr.text((70, 112), args.subtitle, fill=(90, 90, 90), font=F[1])
    dr.line((70, 160, 470, 160), fill=(20, 20, 20), width=2)
    img.save(os.path.join(args.out, f'{i:05d}.png'))
    if i % 150 == 0:
        print(f'  frame {i}/{len(frames)}', flush=True)
print('wrote', len(frames), 'frames to', args.out)
app.close()
