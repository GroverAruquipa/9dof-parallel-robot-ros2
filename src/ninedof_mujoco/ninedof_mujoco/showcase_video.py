"""Render the showcase of the 9-DoF parallel robot to an mp4 video.

    MUJOCO_GL=osmesa ros2 run ninedof_mujoco showcase_video -o showcase.mp4
    ros2 run ninedof_mujoco showcase_video --check      # only verify the trajectory

Kinematic animation: every frame shows the exact pose of the trajectory in
ninedof_kinematics.showcase (inverse kinematics for the actuators, closed
chains assembled), so the video shows the geometry of the motion without
dynamic effects. 1920x1080, 30 fps, H.264, fixed camera; the legs are coloured
like the platform they drive.
"""

import argparse
import os
import shutil
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')   # headless rendering (libosmesa6)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ninedof_kinematics.kinematics import NineDofKinematics  # noqa: E402
from ninedof_kinematics.showcase import Showcase  # noqa: E402
from ninedof_mujoco.sim import set_robot_pose  # noqa: E402

RED = np.array([0.82, 0.24, 0.20, 1.0], np.float32)
BLUE = np.array([0.20, 0.42, 0.85, 1.0], np.float32)


def share(*path):
    try:
        from ament_index_python.packages import get_package_share_directory
        root = get_package_share_directory('ninedof_description')
    except Exception:  # running from the source tree
        root = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description')
    return os.path.join(root, *path)


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    for d in ('/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/dejavu'):
        if os.path.exists(os.path.join(d, name)):
            return ImageFont.truetype(os.path.join(d, name), size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def colour_legs(model, data, scene):
    """Draw each leg (B_i -> A_i) in the colour of the platform it drives."""
    for i in range(model.neq):
        eq = model.equality(i)
        if not eq.name.endswith('_upper_joint'):
            continue
        leg = eq.name[:-len('_upper_joint')]
        platform = model.body(model.eq_obj2id[i]).name
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), RED if platform == 'platform_1' else BLUE)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0032,
                             data.xpos[model.body(f'{leg}_distal').id],
                             data.site_xpos[model.site(f'{leg}_A').id])
        scene.ngeom += 1


def caption(img, title, label, value, fonts):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    h = im.height
    d.text((60, 50), title, font=fonts[0], fill=(90, 90, 95))
    d.text((60, h - 150), label, font=fonts[1], fill=(35, 35, 40))
    d.text((60, h - 88), value, font=fonts[2], fill=(90, 90, 95))
    return np.asarray(im)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-o', '--output', default='showcase.mp4')
    parser.add_argument('--model', default=share('mujoco', 'showcase_scene.xml'))
    parser.add_argument('--geometry', default=share('config', 'geometry.yaml'))
    parser.add_argument('--camera', default='showcase_cam')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--check', action='store_true', help='only verify the trajectory')
    args = parser.parse_args(argv)

    kin = NineDofKinematics.from_yaml(args.geometry)
    show = Showcase(kin.home)
    worst = show.check(kin)
    print(f'Trajectory {show.duration:.1f} s, inside the workspace '
          f'(max actuator travel {worst * 1000:.1f} of {kin.stroke * 1000:.0f} mm)')
    if args.check:
        return

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        sys.exit('ffmpeg not found (sudo apt-get install ffmpeg)')
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    fonts = (font(30), font(56, bold=True), font(38))
    title = '9-DoF parallel robot  5PSS-S-4PSS'
    proc = subprocess.Popen(
        [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{args.width}x{args.height}', '-r', str(args.fps), '-i', '-',
         '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
         args.output], stdin=subprocess.PIPE)
    times = np.arange(0.0, show.duration, 1.0 / args.fps)
    for n, t in enumerate(times):
        label, value, _, x = show.sample(t)
        set_robot_pose(model, data, kin, x)
        renderer.update_scene(data, camera=args.camera)
        colour_legs(model, data, renderer.scene)
        proc.stdin.write(caption(renderer.render(), title, label, value, fonts).tobytes())
        if n % 90 == 0:
            print(f'\rframe {n + 1}/{len(times)}', end='', flush=True)
    proc.stdin.close()
    proc.wait()
    print(f'\nWrote {args.output} ({len(times)} frames, {args.fps} fps)')


if __name__ == '__main__':
    main()
