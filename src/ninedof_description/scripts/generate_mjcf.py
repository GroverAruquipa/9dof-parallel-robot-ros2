#!/usr/bin/env python3
"""Generate the MuJoCo model (MJCF) of the 9-DoF parallel robot.

Reads config/geometry.yaml, config/dynamics.yaml and config/pick_place.yaml
and writes mujoco/ninedof.xml (robot), mujoco/scene.xml (robot + floor) and
mujoco/pick_place_scene.xml (robot hanging above a table with a block).

Unlike URDF, MJCF can close the kinematic loops: each distal link hangs from
its actuator with a ball joint and its upper ball A_i is tied to the platform
with an equality "connect" constraint. The model is written in its home
configuration so that qpos0 is an assembled robot.

    python3 scripts/generate_mjcf.py            # from the package directory
"""

import math
import os

import numpy as np
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Finger plates of each platform (platform frame, metres), measured on the
# meshes: base quad (z = 5 mm), bend (z ~ 18-19 mm) and tip (z ~ 36-40 mm).
FINGER = {
    1: {'base': [[-17.82, 6.09, 5.0], [-9.67, 7.61, 5.0], [-11.9, 3.14, 5.0], [-15.59, 10.57, 5.0]],
        'bend': [[-17.21, 5.79, 17.73], [-14.98, 10.26, 17.73],
                 [-23.67, 9.01, 19.0], [-21.44, 13.48, 19.0]],
        'tip': [[1.35, 2.12, 35.97], [-0.89, -2.35, 35.97],
                [-2.45, 4.02, 40.21], [-4.68, -0.46, 40.21]]},
    2: {'base': [[11.9, -3.14, 5.0], [15.59, -10.57, 5.0], [9.67, -7.61, 5.0], [17.82, -6.09, 5.0]],
        'bend': [[17.21, -5.79, 17.73], [14.98, -10.26, 17.73],
                 [23.67, -9.01, 19.0], [21.44, -13.48, 19.0]],
        'tip': [[-1.35, -2.12, 35.97], [0.89, 2.35, 35.97],
                [2.45, -4.02, 40.21], [4.68, 0.46, 40.21]]},
}


def fmt(v):
    return ' '.join(f'{x:.6g}' for x in np.atleast_1d(v))


def quat_z_to(u):
    """Quaternion (w x y z) rotating +Z onto the unit vector u."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, u)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    angle = math.atan2(s, float(z @ u))
    axis /= s
    return np.r_[math.cos(angle / 2), axis * math.sin(angle / 2)]


def rot_x_matrix(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def disk_inertia(mass, radius, height):
    ixx = mass * (3 * radius ** 2 + height ** 2) / 12
    return [ixx, ixx, mass * radius ** 2 / 2]


def rod_inertia(mass, length):
    """Slender rod along Z, centre of mass at its middle."""
    ixx = mass * length ** 2 / 12
    return [ixx, ixx, 1e-9]


def robot_parts(geometry, dynamics, finger_friction=None, indent='    ', mount=None):
    """MJCF fragments of the robot: base bodies (base at the local origin),
    platform bodies (top level, since they carry a free joint), equality
    constraints, actuators and sensors. mount = (pos, euler) places the platforms
    in the world when the base is attached to a mount body."""
    g = geometry['ninedof']
    d = dynamics['ninedof_dynamics']
    l = g['distal_length']
    ball_h = g['slider_ball_height']
    stroke = g['actuator_stroke']
    home = np.array([0.0, 0.0, g['home_z']])
    act = d['actuator']
    m = d['mass']
    legs = g['legs']

    b = []

    def w(line):
        b.append(indent + line)

    w('<body name="base_link">')
    w(f'  <geom class="visual" mesh="base" material="base_grey" '
      f'pos="0 0 {g["base_mesh_offset_z"]}"/>')
    # Legs: slider on a slide joint, distal link on a ball joint at B_i.
    for leg in legs:
        n = leg['name']
        A = home + np.array(leg['a'])
        B = np.array([leg['base_x'], leg['base_y'], leg['base_z0']])
        u = (A - B) / np.linalg.norm(A - B)
        w(f'  <body name="{n}_slider" pos="{fmt([B[0], B[1], B[2] - ball_h])}">')
        w(f'    <joint name="{n}_actuator_joint" type="slide" axis="0 0 1" '
          f'range="{-stroke} {stroke}" armature="{act["armature"]}" '
          f'actuatorfrcrange="{-act["max_force"]} {act["max_force"]}"/>')
        w(f'    <inertial pos="0 0 {ball_h / 2:.6g}" mass="{m["slider"]}" '
          f'diaginertia="{fmt(rod_inertia(m["slider"], ball_h))}"/>')
        w('    <geom class="visual" mesh="slider" material="rod_white"/>')
        w(f'    <body name="{n}_distal" pos="0 0 {ball_h}" quat="{fmt(quat_z_to(u))}">')
        w(f'      <joint name="{n}_lower_joint" type="ball" '
          f'damping="{d["passive_joint_damping"]}"/>')
        w(f'      <inertial pos="0 0 {l / 2}" mass="{m["distal_link"]}" '
          f'diaginertia="{fmt(rod_inertia(m["distal_link"], l))}"/>')
        w('      <geom class="visual" mesh="distal_link" material="rod_white"/>')
        w(f'      <site name="{n}_A" pos="0 0 {l}" size="0.002"/>')
        w('    </body>')
        w('  </body>')
    w('</body>')
    base = b
    b = []
    indent = '    '
    # Platforms: platform 1 floats (its pose is imposed by legs 1-5), platform 2
    # hangs from it through the central spherical joint.
    if mount:
        R = rot_x_matrix(mount[1][0])
        platform_pose = f'pos="{fmt(np.asarray(mount[0]) + R @ home)}" euler="{fmt(mount[1])}"'
    else:
        platform_pose = f'pos="{fmt(home)}"'
    friction = ('' if finger_friction is None
                else f' friction="{finger_friction} 0.02 0.0001"')
    w(f'  <body name="platform_1" {platform_pose}>')
    w('    <freejoint name="platform_1_free"/>')
    w(f'    <inertial pos="0 0 0.005" mass="{m["platform_1"]}" '
      f'diaginertia="{fmt(disk_inertia(m["platform_1"], 0.035, 0.01))}"/>')
    w('    <geom class="visual" mesh="platform_1" material="platform_1_red"/>')
    w(f'    <geom class="finger" name="finger_1_lower" mesh="finger_1_lower"{friction}/>')
    w(f'    <geom class="finger" name="finger_1_upper" mesh="finger_1_upper"{friction}/>')
    w('    <body name="platform_2">')
    w(f'      <joint name="central_sphere_joint" type="ball" '
      f'damping="{d["passive_joint_damping"]}"/>')
    w(f'      <inertial pos="0 0 0.005" mass="{m["platform_2"]}" '
      f'diaginertia="{fmt(disk_inertia(m["platform_2"], 0.035, 0.01))}"/>')
    w('      <geom class="visual" mesh="platform_2" material="platform_2_blue"/>')
    w(f'      <geom class="finger" name="finger_2_lower" mesh="finger_2_lower"{friction}/>')
    w(f'      <geom class="finger" name="finger_2_upper" mesh="finger_2_upper"{friction}/>')
    w('    </body>')
    w('  </body>')

    # Close the loops: upper ball A_i of each distal link on its platform.
    eq = [f'    <connect name="{leg["name"]}_upper_joint" body1="{leg["name"]}_distal" '
          f'body2="platform_{leg["platform"]}" anchor="0 0 {l}"/>' for leg in legs]
    # Position servos named like the ros2_control joints (mujoco_ros2_control).
    actuators = [f'    <position name="{leg["name"]}_actuator_joint" '
                 f'joint="{leg["name"]}_actuator_joint" kp="{act["kp"]}" '
                 f'dampratio="{act["damping_ratio"]}" ctrlrange="{-stroke} {stroke}" '
                 f'forcerange="{-act["max_force"]} {act["max_force"]}"/>' for leg in legs]
    sensors = [f'    <actuatorfrc name="{leg["name"]}_force" '
               f'actuator="{leg["name"]}_actuator_joint"/>' for leg in legs]
    return base, b, eq, actuators, sensors


def compose(geometry, dynamics, model_name, mount=None, finger_friction=None,
            extra_assets=(), extra_world=(), extra_after_world=(), extra_equality=(),
            extra_contact=()):
    """Complete MJCF file. With mount = (pos, euler) the robot is attached to a
    fixed body at that pose (e.g. hanging upside down)."""
    d = dynamics['ninedof_dynamics']
    out = []
    w = out.append
    w('<!-- Generated by scripts/generate_mjcf.py from the files in config/.')
    w('     Do not edit by hand. -->')
    w(f'<mujoco model="{model_name}">')
    w('  <compiler angle="radian" meshdir="../meshes" autolimits="true"/>')
    # Stiff loop closures (ideal joints) need a small time step.
    w('  <option timestep="0.0005" integrator="implicitfast"/>')
    w('')
    w('  <default>')
    w('    <default class="visual">')
    w('      <geom type="mesh" contype="0" conaffinity="0" group="2" mass="0"/>')
    w('    </default>')
    w('    <default class="finger">')
    # Fingers only touch objects whose conaffinity has bit 2 (not the table).
    w(f'      <geom type="mesh" contype="2" conaffinity="0" condim="4" group="3" mass="0" '
      f'friction="{d["finger_friction"]} 0.02 0.0001" rgba="0.9 0.6 0.1 0.4"/>')
    w('    </default>')
    w('    <equality solref="0.001 1" solimp="0.99 0.999 0.0001"/>')
    w('  </default>')
    w('')
    w('  <asset>')
    for name in ('base', 'slider', 'distal_link', 'platform_1', 'platform_2'):
        w(f'    <mesh name="{name}" file="{name}.stl"/>')
    for k, f in FINGER.items():
        lower = np.array(f['base'] + f['bend']) / 1000.0
        upper = np.array(f['bend'] + f['tip']) / 1000.0
        w(f'    <mesh name="finger_{k}_lower" vertex="{fmt(lower.ravel())}"/>')
        w(f'    <mesh name="finger_{k}_upper" vertex="{fmt(upper.ravel())}"/>')
    w('    <material name="base_grey" rgba="0.75 0.75 0.75 1"/>')
    w('    <material name="rod_white" rgba="0.92 0.92 0.92 1"/>')
    w('    <material name="platform_1_red" rgba="0.80 0.25 0.20 1"/>')
    w('    <material name="platform_2_blue" rgba="0.20 0.40 0.80 1"/>')
    out.extend(extra_assets)
    w('  </asset>')
    w('')
    base, platforms, eq, actuators, sensors = robot_parts(
        geometry, dynamics, finger_friction, indent='      ' if mount else '    ', mount=mount)
    w('  <worldbody>')
    out.extend(extra_world)
    if mount:
        w(f'    <body name="robot_mount" pos="{fmt(mount[0])}" euler="{fmt(mount[1])}">')
        out.extend(base)
        w('    </body>')
    else:
        out.extend(base)
    out.extend(platforms)
    w('  </worldbody>')
    w('')
    w('  <equality>')
    out.extend(eq)
    out.extend(extra_equality)
    w('  </equality>')
    w('')
    w('  <contact>')
    w('    <exclude body1="platform_1" body2="platform_2"/>')
    out.extend(extra_contact)
    w('  </contact>')
    w('')
    w('  <actuator>')
    out.extend(actuators)
    w('  </actuator>')
    w('')
    w('  <sensor>')
    out.extend(sensors)
    w('  </sensor>')
    out.extend(extra_after_world)
    w('</mujoco>')
    return '\n'.join(out) + '\n'


def generate(geometry, dynamics):
    """Robot alone, base at the origin (included by scene.xml)."""
    return compose(geometry, dynamics, 'ninedof')


def look_at_xyaxes(pos, target, up=(0.0, 0.0, 1.0)):
    """MJCF camera xyaxes for a camera at pos looking at target."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return np.r_[x, y]


def generate_pick_place(geometry, dynamics, pick_place):
    """Pick-and-place scene: robot hanging upside down above a table, a free
    block, the target zone and a fixed camera on the grasp."""
    pp = pick_place['pick_place']
    table, block, target = pp['table'], pp['block'], pp['target']
    bx, by, bz = (s / 2 for s in block['size'])
    tz = table['top_z']
    tx, ty, tt = table['size']
    frame_z = pp['mount_height'] + 0.03   # top of the base ring
    cam_pos = [0.125, -0.19, 0.105]
    cam_target = [0.0, 0.0, 0.045]

    assets = [
        '    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.55" rgb2="0.05 0.05 0.08"'
        ' width="512" height="3072"/>',
        '    <texture type="2d" name="floor_tex" builtin="checker" mark="edge" rgb1="0.25 0.27 0.3"'
        ' rgb2="0.2 0.22 0.25" markrgb="0.5 0.5 0.5" width="300" height="300"/>',
        '    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="8 8"/>',
        '    <material name="table_mat" rgba="0.55 0.42 0.30 1"/>',
        '    <material name="frame_mat" rgba="0.25 0.25 0.28 1"/>',
        '    <material name="block_mat" rgba="0.95 0.80 0.15 1"/>',
        '    <material name="marker_mat" rgba="0.10 0.10 0.10 1"/>',
        '    <material name="target_mat" rgba="0.15 0.75 0.30 0.45"/>',
    ]
    world = [
        '    <light name="key" pos="0.3 -0.4 0.8" dir="-0.3 0.4 -0.8" diffuse="0.8 0.8 0.8"'
        ' castshadow="true"/>',
        '    <light name="fill" pos="-0.4 0.3 0.6" dir="0.4 -0.3 -0.6" diffuse="0.35 0.35 0.35"'
        ' castshadow="false"/>',
        f'    <geom name="floor" type="plane" size="1 1 0.05" pos="0 0 {tz - tt - 0.7}"'
        ' material="floor_mat" contype="1" conaffinity="1"/>',
        f'    <geom name="table" type="box" size="{tx / 2} {ty / 2} {tt / 2}"'
        f' pos="0 0 {tz - tt / 2}" material="table_mat" contype="1" conaffinity="1"'
        ' friction="0.6 0.005 0.0001"/>',
        # Frame holding the robot upside down (visual only).
        f'    <geom name="frame_top" type="box" size="0.146 0.106 0.006" pos="0 0 {frame_z + 0.006}"'
        ' material="frame_mat" contype="0" conaffinity="0"/>',
    ]
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        h = (frame_z - tz) / 2
        world.append(f'    <geom name="frame_post_{"p" if sx > 0 else "m"}{"p" if sy > 0 else "m"}"'
                     f' type="box" size="0.006 0.006 {h:.4f}"'
                     f' pos="{0.14 * sx} {0.10 * sy} {tz + h:.4f}" material="frame_mat"'
                     ' contype="0" conaffinity="0"/>')
    world.append(f'    <geom name="frame_beam_p" type="box" size="0.146 0.006 0.006"'
                 f' pos="0 0.10 {frame_z}" material="frame_mat" contype="0" conaffinity="0"/>')
    world.append(f'    <geom name="frame_beam_m" type="box" size="0.146 0.006 0.006"'
                 f' pos="0 -0.10 {frame_z}" material="frame_mat" contype="0" conaffinity="0"/>')
    # Target zone: tolerance square + outline of the block at the target yaw.
    tol = target['tolerance_xy']
    world.append(f'    <geom name="target_zone" type="box" size="{bx + tol} {by + tol} 0.0005"'
                 f' pos="{target["xy"][0]} {target["xy"][1]} {tz + 0.0005}"'
                 f' euler="0 0 {target["yaw"]}" material="target_mat" contype="0" conaffinity="0"/>')
    # Free block with a dark marker on its +x face (makes the 90 deg turn visible).
    world += [
        f'    <body name="block" pos="{block["pick_xy"][0]} {block["pick_xy"][1]} {tz + bz}"'
        f' euler="0 0 {block["pick_yaw"]}">',
        '      <freejoint name="block_free"/>',
        f'      <geom name="block" type="box" size="{bx} {by} {bz}" mass="{block["mass"]}"'
        f' material="block_mat" friction="{block["friction"]} 0.02 0.0001" condim="4"'
        ' contype="1" conaffinity="3"/>',
        f'      <geom name="block_marker" type="box" size="0.0005 {by * 0.6} {bz * 0.6}"'
        f' pos="{bx + 0.0005} 0 0" material="marker_mat" contype="0" conaffinity="0" mass="0"/>',
        '    </body>',
        f'    <camera name="grasp_cam" mode="fixed" pos="{fmt(cam_pos)}"'
        f' xyaxes="{fmt(look_at_xyaxes(cam_pos, cam_target))}" fovy="45"/>',
    ]
    after = [
        '',
        '  <visual>',
        '    <global offwidth="1920" offheight="1080"/>',
        '    <quality shadowsize="4096"/>',
        '    <headlight ambient="0.25 0.25 0.25" diffuse="0.3 0.3 0.3" specular="0 0 0"/>',
        '  </visual>',
        '  <statistic center="0 0 0.12" extent="0.35"/>',
    ]
    mount = ([0.0, 0.0, pp['mount_height']], [math.pi, 0.0, 0.0])
    # The grasp is a virtual spring-damper applied by the SimStatePublisher plugin
    # (ninedof_mujoco): friction grasping with these small hook-shaped fingers is
    # not reliable, so the demo focuses on geometry and control accuracy. The
    # finger/block contacts are disabled; the descent around the block is
    # collision free by design.
    equality = []
    contact = ['    <exclude body1="platform_1" body2="block"/>',
               '    <exclude body1="platform_2" body2="block"/>']
    return compose(geometry, dynamics, 'ninedof_pick_place', mount=mount,
                   finger_friction=pp['finger_friction'], extra_assets=assets,
                   extra_world=world, extra_after_world=after, extra_equality=equality,
                   extra_contact=contact)


SCENE = """<!-- MuJoCo scene of the 9-DoF parallel robot. -->
<mujoco model="ninedof_scene">
  <include file="ninedof.xml"/>

  <statistic center="0 0 0.08" extent="0.3"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="135" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
              reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" pos="0 0 -0.03" size="0 0 0.05" type="plane" material="groundplane"
          contype="1" conaffinity="1"/>
  </worldbody>
</mujoco>
"""


def load(name):
    with open(os.path.join(PKG, 'config', name)) as f:
        return yaml.safe_load(f)


def main():
    geometry, dynamics = load('geometry.yaml'), load('dynamics.yaml')
    os.makedirs(os.path.join(PKG, 'mujoco'), exist_ok=True)
    files = {
        'ninedof.xml': generate(geometry, dynamics),
        'scene.xml': SCENE,
        'pick_place_scene.xml': generate_pick_place(geometry, dynamics, load('pick_place.yaml')),
    }
    for name, text in files.items():
        with open(os.path.join(PKG, 'mujoco', name), 'w') as f:
            f.write(text)
    print('Wrote ' + ', '.join('mujoco/' + n for n in files))


if __name__ == '__main__':
    main()
