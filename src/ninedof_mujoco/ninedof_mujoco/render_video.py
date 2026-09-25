"""Render the recorded pick-and-place (qpos) offline to an mp4 video.

    MUJOCO_GL=osmesa ros2 run ninedof_mujoco render_video \
        pick_place_output/pick_place_qpos.npz -o pick_place.mp4

1920x1080, 30 fps, H.264 (ffmpeg), fixed camera "grasp_cam" of
pick_place_scene.xml. The video focuses on the geometry: each leg is drawn as
a line coloured like the platform it drives, the ball joints and the central
spherical joint are marked, the block shows its heading, and an overlay gives
the current step, the jaw angle and the yaw of the gripper and of the block.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')   # headless rendering (libosmesa6)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

RED = np.array([0.85, 0.25, 0.2, 1.0])
BLUE = np.array([0.2, 0.45, 0.9, 1.0])
GREEN = np.array([0.1, 0.8, 0.3, 1.0])
WHITE = np.array([1.0, 1.0, 1.0, 1.0])
ORANGE = np.array([1.0, 0.6, 0.1, 1.0])


def default_model():
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('ninedof_description')
    except Exception:  # running from the source tree
        share = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description')
    return os.path.join(share, 'mujoco', 'pick_place_scene.xml')


def add_geom(scene, gtype, size, pos, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def add_line(scene, a, b, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


class GeometryOverlay:
    """3D annotations: legs, joints and the heading of the block."""

    def __init__(self, model):
        self.m = model
        self.legs = []
        for i in range(model.nbody):
            name = model.body(i).name
            if name.endswith('_distal'):
                leg = name[:-len('_distal')]
                eq = model.equality(f'{leg}_upper_joint')
                platform = model.body(model.eq_obj2id[eq.id]).name
                self.legs.append((i, model.site(f'{leg}_A').id,
                                  RED if platform == 'platform_1' else BLUE))
        self.p1 = model.body('platform_1').id
        self.block = model.body('block').id

    def draw(self, data, scene):
        for body, site, rgba in self.legs:
            B, A = data.xpos[body], data.site_xpos[site]
            add_line(scene, B, A, 0.0032, rgba)
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0035, 0, 0], B, np.eye(3), ORANGE)
            add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0048, 0, 0], A, np.eye(3), rgba)
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0045, 0, 0], data.xpos[self.p1],
                 np.eye(3), WHITE)
        # Heading of the block: arrow along its marked (+x) face.
        R = data.xmat[self.block].reshape(3, 3)
        top = data.xpos[self.block] + R[:, 2] * 0.0125
        add_line(scene, top, top + R[:, 0] * 0.022, 0.0015, GREEN)


def text_overlay(img, lines, font):
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im, 'RGBA')
    w = max(draw.textlength(s, font=font) for s in lines) + 40
    h = len(lines) * (font.size + 10) + 24
    draw.rounded_rectangle([24, 24, 24 + w, 24 + h], radius=14, fill=(0, 0, 0, 150))
    y = 36
    for s in lines:
        draw.text((44, y), s, font=font, fill=(255, 255, 255, 255))
        y += font.size + 10
    return np.asarray(im)


def load_font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def yaw_deg(R):
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('recording', help='pick_place_qpos.npz written by pick_place_demo')
    parser.add_argument('-o', '--output', default='pick_place.mp4')
    parser.add_argument('--model', default=default_model())
    parser.add_argument('--camera', default='grasp_cam')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args(argv)

    rec = np.load(args.recording, allow_pickle=False)
    t, qpos = rec['time'], rec['qpos']
    names, spans = list(rec['step_names']), rec['step_times']
    model = mujoco.MjModel.from_xml_path(args.model)
    if qpos.shape[1] != model.nq:
        sys.exit(f'qpos has {qpos.shape[1]} values, the model {model.nq}: wrong model?')
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    overlay = GeometryOverlay(model)
    opt = mujoco.MjvOption()
    font = load_font(34)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        sys.exit('ffmpeg not found (sudo apt-get install ffmpeg)')
    proc = subprocess.Popen(
        [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{args.width}x{args.height}', '-r', str(args.fps), '-i', '-',
         '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
         args.output], stdin=subprocess.PIPE)

    b1, b2, blk = (model.body(n).id for n in ('platform_1', 'platform_2', 'block'))
    mount = model.body('robot_mount').id
    half_h = model.geom('block').size[2]
    frame_times = np.arange(t[0], t[-1], 1.0 / args.fps)
    for n, ft in enumerate(frame_times):
        data.qpos[:] = qpos[min(np.searchsorted(t, ft), len(t) - 1)]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=args.camera, scene_option=opt)
        overlay.draw(data, renderer.scene)
        img = renderer.render()

        step = next((nm for nm, (a, b) in zip(names, spans) if a <= ft < b), '')
        Rm = data.xmat[mount].reshape(3, 3)
        Q1 = Rm.T @ data.xmat[b1].reshape(3, 3)
        Q2 = Rm.T @ data.xmat[b2].reshape(3, 3)
        rel = Q1.T @ Q2
        jaw = math.degrees(math.atan2(-rel[1, 2], rel[2, 2])) / -2.0
        grip_yaw = yaw_deg(Q1)
        lines = [
            f'9-DoF parallel robot 5PSS-S-4PSS - pick and place    t = {ft - t[0]:5.2f} s',
            f'step: {step}',
            f'jaw (platform 1 / 2 about x): {jaw:+5.1f} deg    gripper yaw: {grip_yaw:+6.1f} deg',
            f'block yaw: {yaw_deg(data.xmat[blk].reshape(3, 3)):+6.1f} deg    '
            f'block height: {(data.xpos[blk][2] - half_h) * 1000:5.1f} mm',
            'legs: red -> platform 1 (5 legs), blue -> platform 2 (4 legs)',
        ]
        proc.stdin.write(text_overlay(img, lines, font).tobytes())
        if n % 60 == 0:
            print(f'\rframe {n + 1}/{len(frame_times)}', end='', flush=True)
    proc.stdin.close()
    proc.wait()
    print(f'\nWrote {args.output} ({len(frame_times)} frames, {args.fps} fps)')


if __name__ == '__main__':
    main()
