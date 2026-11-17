"""RTX video of the grid run (grid_sim.py): N copies of the robot, PhysX poses played back, physics removed.

Same pipeline as render_video.py (verified): every body is placed from the recorded data, the robot is cut out with
the depth map and laid on white, key and fill lights, caption in the top-left corner. One fixed camera on the
grid, same azimuth as the robot videos; the softest servo is the copy nearest to the camera.

Run in the container:  ./python.sh /work/isaac_sim/render_grid.py [--frames 4]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FPS, W, H = 30, 1920, 1080
TITLES = {'hold': 'Reference pose', 'translation x': 'Translation along x', 'translation y': 'Translation along y',
          'translation z': 'Translation along z', 'rigid rotation x': 'Rigid rotation about x',
          'rigid rotation y': 'Rigid rotation about y', 'rigid rotation z': 'Rigid rotation about z',
          'cone': 'Tilt about a turning axis', 'relative y (grasp)': 'Relative rotation about y: grasp',
          'relative x': 'Relative rotation about x', 'relative z': 'Relative rotation about z'}

ap = argparse.ArgumentParser()
cad = os.path.join(HERE, 'ninedof_cctomm.usdc')
ap.add_argument('--robot', default=cad if os.path.exists(cad) else os.path.join(HERE, 'ninedof_cctomm_simple.usda'))
ap.add_argument('--run', default=os.path.join(HERE, 'results', 'grid_run.npz'))
ap.add_argument('--frames', type=int, help='only N frames spread over the run (look check)')
ap.add_argument('--distance', type=float, default=1.3, help='camera distance to the grid centre (m)')
ap.add_argument('--sky', type=float, default=250.0)
ap.add_argument('--key', type=float, default=900.0)
ap.add_argument('--fill', type=float, default=350.0)
ap.add_argument('--out', default=os.path.join(HERE, 'results', 'grid_frames'))
args = ap.parse_args()

from isaacsim import SimulationApp                         # noqa: E402
app = SimulationApp({'headless': True, 'width': W, 'height': H})

import omni.replicator.core as rep                         # noqa: E402
import omni.timeline                                       # noqa: E402
import omni.usd                                            # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont   # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics       # noqa: E402

import make_grid                                           # noqa: E402

run = np.load(args.run)
offsets = run['offsets']; kps = run['kp']; N = len(kps)
spacing = float(np.min([np.linalg.norm(a - b) for i, a in enumerate(offsets) for b in offsets[i + 1:]])) if N > 1 else 0.25
grid_usd = os.path.join(HERE, 'results', 'grid_render.usda')
_, envs = make_grid.build(grid_usd, args.robot, n=N, spacing=spacing, kp_min=float(kps.min()), kp_max=float(kps.max()))
assert np.allclose([e['offset'] for e in envs], offsets), 'grid layout differs from the recorded run'

ctx = omni.usd.get_context()
ctx.open_stage(grid_usd)
for _ in range(10):
    app.update()
stage = ctx.get_stage()
exec(open(os.path.join(HERE, 'grid_place.py')).read())    # rot_between, quat, Body, build_copies, place_copy

copies = build_copies(stage, [e['path'] for e in envs])   # noqa: F821  (reads joints, then removes all physics)
omni.timeline.get_timeline_interface().stop()

ref = np.load(os.path.join(HERE, 'reference_trajectory.npz'))
col = {str(k): i for i, k in enumerate(ref['legs'])}
seg_names = [str(s) for s in ref['segment_names']]


def pose_at(j):
    for k, c in enumerate(copies):
        place_copy(c, run['p1'][j, k], run['R1'][j, k], run['p2'][j, k], run['R2'][j, k], run['q'][j, k], col)  # noqa: F821


def distant(path, direction, intensity):
    lt = UsdLux.DistantLight.Define(stage, path); lt.CreateIntensityAttr(intensity); lt.CreateAngleAttr(1.0)
    q = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*np.asarray(direction, float))).GetQuat()
    lt.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()), *[float(c) for c in q.GetImaginary()]))
    return lt


dome = UsdLux.DomeLight.Define(stage, '/World/sky'); dome.CreateIntensityAttr(args.sky); dome.CreateColorAttr((1, 1, 1))
for lt in (dome, distant('/World/key', (-.5, .7, -.8), args.key), distant('/World/fill', (.7, -.3, -.4), args.fill)):
    UsdLux.ShadowAPI.Apply(lt.GetPrim()).CreateShadowEnableAttr(False)

centre = np.array(ref['p'][0]) + offsets.mean(0) - np.array([0, 0, 0.09])     # framing checked offline
fwd = make_grid.camera_forward()
pos = centre - args.distance * fwd
hap = 20.955; focal = (hap / 2) / np.tan(np.arctan(np.tan(np.radians(22.5)) * W / H))
cam = rep.create.camera(position=tuple(pos), look_at=tuple(centre), focal_length=focal, horizontal_aperture=hap,
                        clipping_range=(0.01, 100.0))
rp = rep.create.render_product(cam, (W, H))
rgb = rep.AnnotatorRegistry.get_annotator('rgb'); rgb.attach([rp])
depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'); depth.attach([rp])


def on_white(color, d):
    d = np.squeeze(np.asarray(d, float))
    mask = Image.fromarray(((np.isfinite(d) & (d > 0) & (d < 50)) * 255).astype(np.uint8))
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.8)), float)[:, :, None] / 255.0
    return Image.fromarray((color * m + 255.0 * (1 - m)).astype(np.uint8))


def font(size):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


F = (font(40), font(30))
title = f'{N} robots simulated in parallel'
sub = f'Isaac Sim 6.1 PhysX, servo stiffness from {kps.min():.0e} (front) to {kps.max():.0e} N/m (back)'.replace('e+0', 'e')
os.makedirs(args.out, exist_ok=True)
n_run = len(run['t'])
frames = np.arange(int(run['t'][-1] * FPS) + 1)
if args.frames:
    frames = np.linspace(0, frames[-1], args.frames).astype(int)
pose_at(0)
for _ in range(20):
    rep.orchestrator.step(rt_subframes=4)
for i, f in enumerate(frames):
    j = min(int(round(f / FPS * 100)), n_run - 1)
    pose_at(j)
    rep.orchestrator.step(rt_subframes=4)
    if i < 3 or i % 300 == 0:                              # placement check: nothing moves the bodies after placing
        M = UsdGeom.Xformable(stage.GetPrimAtPath(f'{envs[0]["path"]}/plat1')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        drift = np.linalg.norm(np.array(M.ExtractTranslation()) - run['p1'][j, 0]) * 1000
        if drift > 0.01:
            print(f'DIAG frame {i}: platform 1 of copy 0 moved after placement by {drift:.3f} mm', flush=True)
    img = on_white(np.asarray(rgb.get_data())[:, :, :3].astype(float), depth.get_data())
    dr = ImageDraw.Draw(img)
    dr.text((70, 58), title, fill=(20, 20, 20), font=F[0])
    dr.text((70, 112), f'{TITLES.get(seg_names[ref["segment"][j]], "")}.  {sub}', fill=(90, 90, 90), font=F[1])
    dr.line((70, 160, 470, 160), fill=(20, 20, 20), width=2)
    img.save(os.path.join(args.out, f'{i:05d}.png'))
    if i % 150 == 0:
        print(f'  frame {i}/{len(frames)}', flush=True)
print('wrote', len(frames), 'frames to', args.out)
app.close()
