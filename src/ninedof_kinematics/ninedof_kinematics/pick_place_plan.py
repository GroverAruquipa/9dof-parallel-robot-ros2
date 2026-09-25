"""Pick-and-place with reorientation for the 9-DoF parallel robot (pure numpy).

The gripper is formed by the two moving platforms: rotating platform 1 by +theta
and platform 2 by -theta about the robot x axis opens the jaw (the finger tips
separate), and turning both platforms by psi about the robot z axis turns the
held object. A pose of the sequence is therefore

    Q1 = Rz(psi) Rx(theta),  Q2 = Rz(psi) Rx(-theta),  p = [x, y, home_z + dz].

The robot hangs upside down above the table (config/pick_place.yaml), so the
robot frame is the world frame rotated by pi about X:
(x, y, z)_world = (x_r, -y_r, mount_height - z_r).
"""

from dataclasses import dataclass
import math

import numpy as np
import yaml

from ninedof_kinematics.kinematics import (
    NineDofKinematics, UnreachablePose, rot_x, rot_z, xyz_from_rot)


@dataclass
class Waypoint:
    name: str
    pose: np.ndarray      # [x, y, z, a1, a2, a3, b1, b2, b3] (robot frame)
    hold: float = 0.5     # seconds to wait once the pose is reached


def load_pick_place(path):
    with open(path) as f:
        return yaml.safe_load(f)['pick_place']


def gripper_pose(kin, xy, dz, psi, theta):
    Q1 = rot_z(psi) @ rot_x(theta)
    Q2 = rot_z(psi) @ rot_x(-theta)
    return np.r_[xy[0], xy[1], kin.home[2] + dz, xyz_from_rot(Q1), xyz_from_rot(Q2)]


def world_to_robot_xy(xy_world):
    return np.array([xy_world[0], -xy_world[1]])


def jaw_direction(tip_centroid, theta):
    """Unit horizontal vector (robot frame, psi = 0) from the platform-2 tip to
    the platform-1 tip: the direction along which the jaw squeezes."""
    t = np.asarray(tip_centroid, dtype=float)
    d = rot_x(theta) @ t - rot_x(-theta) @ (-t * np.array([1.0, 1.0, -1.0]))
    d[2] = 0.0
    return d / np.linalg.norm(d)


def block_yaw_for(psi, tip_centroid, theta):
    """World yaw the block must have so that two of its faces are normal to the
    jaw when the platforms are turned by psi (robot frame)."""
    d = jaw_direction(tip_centroid, theta)
    jaw_angle_r = psi + math.atan2(d[1], d[0])
    # Block local +y along the jaw (robot frame), then mirror to the world.
    yaw_r = jaw_angle_r - math.pi / 2
    yaw_w = -yaw_r
    return math.atan2(math.sin(yaw_w), math.cos(yaw_w))


def plan(kin, pp):
    """Waypoints: approach, open, descend, close, lift, rotate, place, open."""
    g = pp['gripper']
    pick = world_to_robot_xy(pp['block']['pick_xy'])
    place = world_to_robot_xy(pp['target']['xy'])
    op, cl = g['jaw_open'], g['jaw_closed']
    zs, zl, zg = g['z_start'], g['z_lift'], g['z_grasp']
    yp, yl = g['yaw_pick'], g['yaw_place']
    return [
        Waypoint('start', gripper_pose(kin, (0.0, 0.0), zs, 0.0, 0.0), 0.5),
        Waypoint('approach', gripper_pose(kin, pick, zs, yp, 0.0)),
        # The jaw opens while the gripper comes down to z_open (tips stay above
        # the object), then the open fingers straddle it on the way down.
        Waypoint('open', gripper_pose(kin, pick, g['z_open'], yp, op)),
        Waypoint('descend', gripper_pose(kin, pick, zg, yp, op)),
        Waypoint('close', gripper_pose(kin, pick, zg, yp, cl), 1.0),
        Waypoint('lift', gripper_pose(kin, pick, zl, yp, cl)),
        Waypoint('rotate', gripper_pose(kin, pick, zl, yl, cl)),
        Waypoint('place', gripper_pose(kin, place, zg, yl, cl), 1.0),
        Waypoint('release', gripper_pose(kin, place, zg, yl, op), 1.0),
        Waypoint('retreat', gripper_pose(kin, place, g['z_retreat'], yl, op), 1.0),
    ]


def segment(x0, x1, v_lin, v_ang, dt):
    """Poses the cartesian_pose_controller goes through from x0 to x1 (same
    interpolation: straight line in [p, Euler angles], velocity limited)."""
    delta = np.asarray(x1) - np.asarray(x0)
    duration = max(np.linalg.norm(delta[:3]) / v_lin, np.abs(delta[3:]).max() / v_ang)
    n = max(1, int(math.ceil(duration / dt)))
    return [np.asarray(x0) + delta * (i / n) for i in range(1, n + 1)], duration


def check(kin, waypoints, v_lin, v_ang, stroke_margin=0.98, dt=0.01):
    """Verify every waypoint and every interpolated pose between them: real IK
    solution and actuators within stroke_margin of their stroke.

    Returns a list of (waypoint name, ok, worst |q| [m], message)."""
    report = []
    prev = None
    for wp in waypoints:
        poses = [wp.pose] if prev is None else segment(prev, wp.pose, v_lin, v_ang, dt)[0]
        worst, msg = 0.0, ''
        for x in poses:
            try:
                q = kin.inverse(x)
            except UnreachablePose as e:
                msg = str(e)
                worst = math.inf
                break
            worst = max(worst, float(np.abs(q).max()))
        ok = worst <= kin.stroke * stroke_margin
        if not ok and not msg:
            msg = f'actuator travel {worst * 1000:.1f} mm > {kin.stroke * stroke_margin * 1000:.1f} mm'
        report.append((wp.name, ok, worst, msg))
        prev = wp.pose
    return report


def evaluate_block(pos, quat_wxyz, pp):
    """Compare the final block pose (world) with the target zone."""
    t = pp['target']
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    tilt = math.acos(np.clip(R[2, 2], -1.0, 1.0))
    yaw = math.atan2(R[1, 0], R[0, 0])
    yaw_err = math.atan2(math.sin(yaw - t['yaw']), math.cos(yaw - t['yaw']))
    xy_err = float(np.linalg.norm(np.asarray(pos[:2]) - np.asarray(t['xy'])))
    on_table = abs(pos[2] - (pp['table']['top_z'] + pp['block']['size'][2] / 2)) < 0.005
    checks = {
        'position_error_mm': (xy_err * 1000, xy_err <= t['tolerance_xy']),
        'yaw_error_deg': (math.degrees(yaw_err), abs(yaw_err) <= t['tolerance_yaw']),
        'tilt_deg': (math.degrees(tilt), tilt <= t['tolerance_tilt']),
        'resting_on_table': (float(pos[2]), on_table),
    }
    return all(ok for _, ok in checks.values()), checks


def load_kinematics(geometry_file):
    return NineDofKinematics.from_yaml(geometry_file)
