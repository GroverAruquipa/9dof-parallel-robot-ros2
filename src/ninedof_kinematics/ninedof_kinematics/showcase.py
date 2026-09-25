"""Showcase trajectory of the 9-DoF parallel robot (pure numpy).

Shows the degrees of freedom one family at a time, then two smooth
trajectories, always starting and ending at home:

    translations X, Y, Z -> rotations roll, pitch, yaw (both platforms together)
    -> relative rotation (jaw) -> circle -> cone

A pose is built from v = (x, y, z, roll, pitch, yaw, jaw):
    p  = home + [x, y, z]
    Q1 = R Rx(+jaw),  Q2 = R Rx(-jaw),  R = Rz(yaw) Ry(pitch) Rx(roll)
Every piece is eased with smootherstep: zero velocity and acceleration at the
start and end of each piece.
"""

import math

import numpy as np

from ninedof_kinematics.kinematics import rot_x, rot_y, rot_z, xyz_from_rot

MM, DEG = 1e-3, math.pi / 180.0
KEYS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'jaw')


def params(**kw):
    return np.array([kw.get(k, 0.0) for k in KEYS], dtype=float)


def pose_from_params(home, v):
    x, y, z, roll, pitch, yaw, jaw = v
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    return np.r_[np.asarray(home[:3]) + [x, y, z], xyz_from_rot(R @ rot_x(jaw)),
                 xyz_from_rot(R @ rot_x(-jaw))]


def smootherstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (6 * s - 15) + 10)


def _line(v0, v1):
    return lambda s: v0 + (v1 - v0) * s


def _swing(key, amplitude, duration):
    """0 -> +A -> -A -> 0 on one parameter."""
    a, z = params(**{key: amplitude}), params()
    return [(_line(z, a), duration / 4), (_line(a, -a), duration / 2), (_line(-a, z), duration / 4)]


def _loop(point, duration):
    """Ease from home to point(0), go once around point(angle), ease back."""
    z = params()
    return [(_line(z, point(0.0)), 1.2),
            (lambda s: point(2 * math.pi * s), duration),
            (_line(point(0.0), z), 1.2)]


def segments():
    """(label, value text, pieces); a piece is (f(s) -> params, duration), s in [0, 1]."""
    r, tilt = 25 * MM, 20 * DEG
    return [
        ('Home', lambda v: '', [(_line(params(), params()), 1.0)]),
        ('Translation X', lambda v: f'x = {v[0] / MM:+5.1f} mm', _swing('x', 30 * MM, 4.5)),
        ('Translation Y', lambda v: f'y = {v[1] / MM:+5.1f} mm', _swing('y', 30 * MM, 4.5)),
        ('Translation Z', lambda v: f'z = {v[2] / MM:+5.1f} mm', _swing('z', 15 * MM, 4.5)),
        ('Rotation about X', lambda v: f'roll = {v[3] / DEG:+5.1f} deg',
         _swing('roll', 20 * DEG, 4.5)),
        ('Rotation about Y', lambda v: f'pitch = {v[4] / DEG:+5.1f} deg',
         _swing('pitch', 20 * DEG, 4.5)),
        ('Rotation about Z', lambda v: f'yaw = {v[5] / DEG:+5.1f} deg',
         _swing('yaw', 45 * DEG, 5.0)),
        ('Relative rotation (jaw)', lambda v: f'{2 * v[6] / DEG:4.1f} deg between platforms',
         [(_line(params(), params(jaw=30 * DEG)), 1.5),
          (_line(params(jaw=30 * DEG), params()), 1.5)] * 2),
        ('Circle trajectory',
         lambda v: f'r = 25 mm    x = {v[0] / MM:+5.1f}   y = {v[1] / MM:+5.1f} mm',
         _loop(lambda a: params(x=r * math.cos(a), y=r * math.sin(a)), 5.0)),
        ('Cone trajectory',
         lambda v: f'tilt 20 deg    roll = {v[3] / DEG:+5.1f}   pitch = {v[4] / DEG:+5.1f} deg',
         _loop(lambda a: params(roll=tilt * math.cos(a), pitch=tilt * math.sin(a)), 5.0)),
        ('Home', lambda v: '', [(_line(params(), params()), 1.0)]),
    ]


class Showcase:
    """Sample the showcase trajectory at any time t (seconds)."""

    def __init__(self, home):
        self.home = np.asarray(home, dtype=float)
        self.pieces = []   # (t0, t1, f, label, formatter)
        t = 0.0
        for label, fmt, pieces in segments():
            for f, duration in pieces:
                self.pieces.append((t, t + duration, f, label, fmt))
                t += duration
        self.duration = t

    def sample(self, t):
        """(label, value text, params, pose) at time t."""
        for t0, t1, f, label, fmt in self.pieces:
            if t <= t1:
                break
        v = f(smootherstep((t - t0) / (t1 - t0)))
        return label, fmt(v), v, pose_from_params(self.home, v)

    def check(self, kin, dt=0.01, stroke_margin=0.9):
        """Largest actuator travel along the whole trajectory (m). Raises if a
        pose has no IK solution or needs more than stroke_margin of the stroke."""
        worst = 0.0
        for t in np.arange(0.0, self.duration + dt / 2, dt):
            label, _, _, x = self.sample(t)
            worst = max(worst, float(np.abs(kin.inverse(x)).max()))
            if worst > kin.stroke * stroke_margin:
                raise ValueError(f'{label} at t = {t:.2f} s: actuator travel {worst * 1000:.1f} mm')
        return worst
