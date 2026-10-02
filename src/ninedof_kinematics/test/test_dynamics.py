"""Checks of the analytic inverse dynamics that use only positions: the
potential energy through the forward kinematics, and the energy balance along
a trajectory with velocities from finite differences of the poses."""

import os

import numpy as np
import pytest

from ninedof_kinematics.dynamics import (
    NineDofDynamics, euler_rates, harmonic_trajectory, platform_twist)
from ninedof_kinematics.kinematics import rot_xyz

CONFIG = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description', 'config')
DEG = np.pi / 180.0
AMPLITUDE = np.array([0.012, -0.012, 0.012, 12 * DEG, -12 * DEG, 15 * DEG,
                      -12 * DEG, 12 * DEG, -15 * DEG])
FREQUENCY = np.array([1.3, 1.1, 1.7, 0.9, 1.5, 1.2, 1.4, 0.8, 1.6])


@pytest.fixture(scope='module')
def dyn():
    return NineDofDynamics.from_yaml(os.path.join(CONFIG, 'geometry.yaml'),
                                     os.path.join(CONFIG, 'dynamics.yaml'))


def trajectory(dyn, t):
    return harmonic_trajectory(dyn.kin.home, AMPLITUDE, FREQUENCY, t)


def bodies(dyn, x):
    """Centre of mass and orientation of every moving body for the pose x:
    (mass, com, R, inertia in the body frame or None for the rods)."""
    kin = dyn.kin
    q = kin.inverse(x)
    p, Q1, Q2 = kin.split(x)
    out = [(dyn.m_plat[k], p + Q @ dyn.c_plat[k], Q, dyn.I_plat[k])
           for k, Q in ((1, Q1), (2, Q2))]
    A, B = kin.platform_points(x), kin.base_points(q)
    for i in range(len(q)):
        out.append((dyn.m_slider, B[i], None, None))
        out.append((dyn.m_rod, B[i] + dyn.rod_com * (A[i] - B[i]), (A[i] - B[i]) / kin.l, None))
    return out, q


def potential(dyn, x):
    return -sum(m * (dyn.g @ c) for m, c, _, _ in bodies(dyn, x)[0])


def test_euler_rates_match_finite_differences(dyn):
    x, xd, xdd = trajectory(dyn, 0.83)
    h = 1e-6
    for s in (slice(3, 6), slice(6, 9)):
        R0 = rot_xyz(trajectory(dyn, 0.83 - h)[0][s])
        R1 = rot_xyz(trajectory(dyn, 0.83 + h)[0][s])
        W = (R1 - R0) / (2 * h) @ rot_xyz(x[s]).T
        E, _ = euler_rates(x[s], xd[s])
        np.testing.assert_allclose(E @ xd[s], [W[2, 1], W[0, 2], W[1, 0]], atol=1e-6)


def test_static_forces_are_the_gradient_of_the_potential(dyn):
    """Virtual work: f_i = dU/dq_i with the pose from the forward kinematics."""
    kin = dyn.kin
    for t in (0.0, 0.7, 1.9):
        x = trajectory(dyn, t)[0]
        q = kin.inverse(x)
        h = 1e-6
        grad = np.empty(9)
        for i in range(9):
            dq = np.zeros(9)
            dq[i] = h
            up = potential(dyn, _forward(kin, q + dq, x))
            um = potential(dyn, _forward(kin, q - dq, x))
            grad[i] = (up - um) / (2 * h)
        # Forces of a few newtons; the finite differences of the forward
        # kinematics near the weak direction are good to ~1e-4 N.
        np.testing.assert_allclose(dyn.gravity_forces(x), grad, atol=2e-4)


def _forward(kin, q, x0):
    """Forward kinematics converged to machine precision (Newton steps with
    the analytic Jacobian of the constraints)."""
    x = np.array(x0, dtype=float)
    for _ in range(20):
        f = kin.constraints(x, q)
        D = np.empty((9, 9))
        for k in range(9):
            dx = np.zeros(9)
            dx[k] = 1e-7
            D[:, k] = (kin.constraints(x + dx, q) - kin.constraints(x - dx, q)) / 2e-7
        step = np.linalg.solve(D, f)
        x -= step
        if np.abs(step).max() < 1e-15:
            break
    return x


def test_total_weight_is_carried(dyn):
    """The actuators (along Z) are the only vertical supports of the moving
    parts, so at rest their forces add up to the total weight."""
    weight = -dyn.g[2] * (sum(dyn.m_plat.values()) + 9 * (dyn.m_slider + dyn.m_rod))
    for t in (0.0, 0.7, 1.9):
        assert dyn.gravity_forces(trajectory(dyn, t)[0]).sum() == pytest.approx(weight, rel=1e-9)


def kinetic_energy(dyn, t, h=1e-6):
    """Kinetic energy with every velocity from central differences."""
    (b0, q0), (b1, q1) = (bodies(dyn, trajectory(dyn, t + s)[0]) for s in (-h, h))
    b, _ = bodies(dyn, trajectory(dyn, t)[0])
    T = 0.5 * dyn.armature * np.sum(((q1 - q0) / (2 * h)) ** 2)
    w_rel = []
    for (m, c0, R0, I), (_, c1, R1, _), (_, _, R, _) in zip(b0, b1, b):
        T += 0.5 * m * np.sum(((c1 - c0) / (2 * h)) ** 2)
        if I is not None:      # platform: R is the rotation matrix
            W = (R1 - R0) / (2 * h) @ R.T
            w = np.array([W[2, 1], W[0, 2], W[1, 0]])
            T += 0.5 * w @ (R @ I @ R.T) @ w
            w_rel.append(w)
        elif R is not None:    # rod: R is the unit axis, no spin
            w = np.cross(R, (R1 - R0) / (2 * h))
            T += 0.5 * dyn.I_rod_perp * w @ w
            w_rel.append(w)
    w1, w2, rods = w_rel[0], w_rel[1], w_rel[2:]
    dissipation = dyn.damping * (np.sum((w2 - w1) ** 2) + sum(w @ w for w in rods))
    return T, dissipation


def test_energy_balance(dyn):
    """f . q_dot = d(T + U)/dt + power dissipated in the ball joints."""
    for t in np.linspace(0.05, 3.0, 12):
        x, xd, xdd = trajectory(dyn, t)
        f, mo = dyn.inverse_dynamics(x, xd, xdd)
        H = 1e-4
        Tp, _ = kinetic_energy(dyn, t + H)
        Tm, _ = kinetic_energy(dyn, t - H)
        _, P_damp = kinetic_energy(dyn, t)
        dE = (Tp - Tm + potential(dyn, trajectory(dyn, t + H)[0])
              - potential(dyn, trajectory(dyn, t - H)[0])) / (2 * H)
        assert f @ mo['qd'] == pytest.approx(dE + P_damp, rel=1e-5, abs=1e-7)


def test_twist_matches_platform_points(dyn):
    """q_dot from the model equals the finite difference of the inverse kinematics."""
    t, h = 1.37, 1e-6
    x, xd, xdd = trajectory(dyn, t)
    cd, cdd = platform_twist(x, xd, xdd)
    mo = dyn.leg_motion(x, cd, cdd)
    q = [dyn.kin.inverse(trajectory(dyn, t + s)[0]) for s in (-h, 0.0, h)]
    np.testing.assert_allclose(mo['qd'], (q[2] - q[0]) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(mo['qdd'], (q[2] - 2 * q[1] + q[0]) / h ** 2, atol=2e-3)
