"""Trajectories with exact velocities and accelerations.

A pose follows the convention of the grasping robot (showcase.py):
    v = (x, y, z, roll, pitch, yaw, jaw)
    p = home + (x, y, z),  R = Rz(yaw) Ry(pitch) Rx(roll),
    Q1 = R Rx(+jaw),  Q2 = R Rx(-jaw)
and c_dot = [p_dot, w1, w2], c_dd = [p_dd, w1_dot, w2_dot] are computed
analytically (no finite differences), so they can be used to validate the model.
"""
import math
import numpy as np

from model import rot_x, rot_y, rot_z

MM, DEG = 1e-3, math.pi / 180
EX, EY, EZ = np.eye(3)


def pose_and_rates(h, v, vd, vdd):
    x, y, z, roll, pitch, yaw, jaw = v
    Rz, Ry, Rx = rot_z(yaw), rot_y(pitch), rot_x(roll)
    R = Rz @ Ry @ Rx
    # w_R = yaw' z + pitch' Rz y + roll' Rz Ry x  (each about its current axis)
    a1, a2, a3 = EZ, Rz @ EY, Rz @ Ry @ EX
    w12 = vd[5] * a1                       # angular velocity of the frame carrying a2
    w123 = vd[5] * a1 + vd[4] * a2         # ... carrying a3
    wR = vd[5] * a1 + vd[4] * a2 + vd[3] * a3
    wRd = (vdd[5] * a1 + vdd[4] * a2 + vd[4] * np.cross(w12, a2)
           + vdd[3] * a3 + vd[3] * np.cross(w123, a3))
    out = []
    rx = R @ EX
    for s in (1, -1):
        Q = R @ rot_x(s * jaw)
        w = wR + s * vd[6] * rx
        wd = wRd + s * vdd[6] * rx + s * vd[6] * np.cross(wR, rx)
        out.append((Q, w, wd))
    (Q1, w1, w1d), (Q2, w2, w2d) = out
    p = np.array([x, y, h + z])
    return p, Q1, Q2, np.r_[vd[:3], w1, w2], np.r_[vdd[:3], w1d, w2d]


def smootherstep(t, T):
    """s, s', s'' of 6r^5 - 15r^4 + 10r^3 on [0, T]."""
    if t <= 0:
        return 0.0, 0.0, 0.0
    if t >= T:
        return 1.0, 0.0, 0.0
    r = t / T
    return (r ** 3 * (r * (6 * r - 15) + 10), 30 * r ** 2 * (r - 1) ** 2 / T,
            60 * r * (2 * r - 1) * (r - 1) / T ** 2)


def params(**kw):
    keys = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'jaw')
    return np.array([kw.get(k, 0.0) for k in keys], float)


class Line:
    def __init__(self, v0, v1):
        self.v0, self.v1 = v0, v1

    def __call__(self, s):
        return self.v0 + (self.v1 - self.v0) * s, self.v1 - self.v0, np.zeros(7)


class Loop:
    """point(angle) = c + A cos(angle) + B sin(angle), angle = 2 pi s."""
    def __init__(self, A, B):
        self.A, self.B = A, B

    def __call__(self, s):
        a = 2 * np.pi * s
        k = 2 * np.pi
        return (self.A * np.cos(a) + self.B * np.sin(a),
                k * (-self.A * np.sin(a) + self.B * np.cos(a)),
                -k * k * (self.A * np.cos(a) + self.B * np.sin(a)))


class Showcase:
    """The showcase of the grasping robot (same motions and amplitudes), time-scaled
    by `speed` (1 = original timing)."""

    def __init__(self, h, speed=1.0, amp=1.0):
        self.h = h
        z = params()
        A = amp
        pieces = [('Home', Line(z, z), 1.0, False)]

        def swing(label, key, a, T):
            p = params(**{key: a})
            return [(label, Line(z, p), T / 4, True), (label, Line(p, -p), T / 2, True),
                    (label, Line(-p, z), T / 4, True)]
        pieces += swing('Traslacion X', 'x', 30 * MM * A, 4.5)
        pieces += swing('Traslacion Y', 'y', 30 * MM * A, 4.5)
        pieces += swing('Traslacion Z', 'z', 15 * MM * A, 4.5)
        pieces += swing('Rotacion X', 'roll', 20 * DEG * A, 4.5)
        pieces += swing('Rotacion Y', 'pitch', 20 * DEG * A, 4.5)
        pieces += swing('Rotacion Z', 'yaw', 45 * DEG * A, 5.0)
        j = params(jaw=15 * DEG * A)
        pieces += [('Pinza (rotacion relativa)', Line(z, j), 1.5, True), ('Pinza (rotacion relativa)', Line(j, z), 1.5, True)] * 2
        c0 = params(x=25 * MM * A)
        pieces += [('Circulo r = 25 mm', Line(z, c0), 1.2, True),
                   ('Circulo r = 25 mm', Loop(params(x=25 * MM * A), params(y=25 * MM * A)), 5.0, 'loop'),
                   ('Circulo r = 25 mm', Line(c0, z), 1.2, True)]
        k0 = params(roll=20 * DEG * A)
        pieces += [('Cono 20 deg', Line(z, k0), 1.2, True),
                   ('Cono 20 deg', Loop(params(roll=20 * DEG * A), params(pitch=20 * DEG * A)), 5.0, 'loop'),
                   ('Cono 20 deg', Line(k0, z), 1.2, True)]
        pieces += [('Home', Line(z, z), 1.0, False)]
        self.pieces = []
        t = 0.0
        for label, f, T, ease in pieces:
            T = T / speed
            self.pieces.append((t, t + T, f, label, ease))
            t += T
        self.duration = t

    def label(self, t):
        for t0, t1, f, label, ease in self.pieces:
            if t <= t1:
                return label
        return self.pieces[-1][3]

    def params(self, t):
        for t0, t1, f, label, ease in self.pieces:
            if t <= t1:
                break
        T = t1 - t0
        if ease == 'loop':
            # constant angular rate with smootherstep blends at the ends would
            # break C2 continuity with the line pieces; use a smootherstep in s
            s, sd, sdd = smootherstep(t - t0, T)
        else:
            s, sd, sdd = smootherstep(t - t0, T)
        v, dv, ddv = f(s)
        return v, dv * sd, ddv * sd ** 2 + dv * sdd

    def __call__(self, t):
        v, vd, vdd = self.params(t)
        return pose_and_rates(self.h, v, vd, vdd)


class MultiSine:
    """Exciting trajectory: every parameter is a sum of two sines, faded in and out."""

    def __init__(self, h, T=4.0, amp=1.0, seed=3):
        self.h, self.T = h, T
        rng = np.random.default_rng(seed)
        base = np.array([15 * MM, 15 * MM, 8 * MM, 12 * DEG, 12 * DEG, 25 * DEG, 8 * DEG]) * amp
        self.A = base[:, None] * np.array([0.7, 0.3])[None, :]
        self.w = 2 * np.pi * rng.uniform(0.4, 1.6, (7, 2))
        self.ph = rng.uniform(0, 2 * np.pi, (7, 2))
        self.Tf = 0.6

    def params(self, t):
        s = self.A * np.sin(self.w * t + self.ph)
        sd = self.A * self.w * np.cos(self.w * t + self.ph)
        sdd = -self.A * self.w ** 2 * np.sin(self.w * t + self.ph)
        g, gd, gdd = self._fade(t)
        v = g * s.sum(1)
        vd = gd * s.sum(1) + g * sd.sum(1)
        vdd = gdd * s.sum(1) + 2 * gd * sd.sum(1) + g * sdd.sum(1)
        v[6] += 8 * DEG * g      # keep the jaw open
        vd[6] += 8 * DEG * gd
        vdd[6] += 8 * DEG * gdd
        return v, vd, vdd

    def _fade(self, t):
        a, ad, add = smootherstep(t, self.Tf)
        b, bd, bdd = smootherstep(self.T - t, self.Tf)
        return a * b, ad * b - a * bd, add * b - 2 * ad * bd + a * bdd

    def __call__(self, t):
        v, vd, vdd = self.params(t)
        return pose_and_rates(self.h, v, vd, vdd)
