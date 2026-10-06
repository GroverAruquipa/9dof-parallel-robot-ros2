"""Checks of the virtual-work dynamics (dynamics.py).

The reference is an independent Newton-Euler model: every body (two platforms,
the distal links and the sliders) gets its own free-body diagram with unknown
joint reactions, and all velocities and accelerations come from finite
differences of the position-level inverse kinematics.  Nothing of J, K, the
leg-rate formulas or the end-point equivalence of the links is used there.
"""

import os

import numpy as np
import pytest
import yaml

from ninedof_kinematics.dynamics import TwoPlatformDynamics, rot_axis, skew
from ninedof_kinematics.kinematics import NineDofKinematics, rot_xyz

CONFIG = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description', 'config')

# Leg subsets and joint axes used to obtain the 8- and 7-DoF variants from the
# geometry of the 9-DoF prototype (4+4 and 4+3 legs).
VARIANTS = {
    'spherical': dict(legs=None, axes=None),
    'universal': dict(legs=[0, 1, 2, 3, 5, 6, 7, 8], axes=([1, 0, 0], [0, 1, 0])),
    'revolute': dict(legs=[0, 1, 2, 3, 5, 6, 7], axes=([1, 0, 0],)),
}


def make(joint):
    kin = NineDofKinematics.from_yaml(os.path.join(CONFIG, 'geometry.yaml'))
    with open(os.path.join(CONFIG, 'dynamics.yaml')) as f:
        dyn = yaml.safe_load(f)
    return TwoPlatformDynamics(kin, dyn, joint, **VARIANTS[joint])


def pose(model, t):
    """A smooth, fast motion (about 2 Hz) that respects the internal joint."""
    w = 2 * np.pi * 2.0
    p = np.array([0.004 * np.sin(w * t), 0.003 * np.cos(1.3 * w * t),
                  0.14668 + 0.005 * np.sin(0.7 * w * t)])
    Q1 = rot_xyz([0.10 * np.sin(w * t), 0.08 * np.cos(0.9 * w * t), 0.15 * np.sin(0.6 * w * t)])
    if model.joint == 'spherical':
        R = rot_xyz([0.25 * np.sin(1.1 * w * t), 0.1 * np.sin(0.8 * w * t), 0.12 * np.cos(w * t)])
    elif model.joint == 'universal':
        R = rot_axis(model.axes[0], 0.2 * np.sin(1.1 * w * t)) @ \
            rot_axis(model.axes[1], 0.1 * np.cos(0.8 * w * t))
    else:
        R = rot_axis(model.axes[0], 0.3 * np.sin(1.1 * w * t))
    return p, Q1, Q1 @ R


def d_dt(f, t, h=1e-4):
    """Five-point central difference of an array-valued function of time."""
    return (f(t - 2 * h) - 8 * f(t - h) + 8 * f(t + h) - f(t + 2 * h)) / (12 * h)


def ang_vel(Qf):
    def w(t):
        W = d_dt(Qf, t) @ Qf(t).T
        return np.array([W[2, 1], W[0, 2], W[1, 0]])
    return w


def motion(model, t):
    """p, Q1, Q2, c_dot, c_ddot of the test motion, by finite differences."""
    w1 = ang_vel(lambda s: pose(model, s)[1])
    w2 = ang_vel(lambda s: pose(model, s)[2])
    pd = lambda s: d_dt(lambda u: pose(model, u)[0], s)
    c_dot = np.r_[pd(t), w1(t), w2(t)]
    c_ddot = np.r_[d_dt(pd, t), d_dt(w1, t), d_dt(w2, t)]
    return (*pose(model, t), c_dot, c_ddot)


def newton_euler(model, t, wrench=None):
    """Actuator forces and internal-joint moments from free-body diagrams."""
    n, l, g = model.n, model.l, model.g
    P = lambda s: pose(model, s)
    q = lambda s: model.inverse_kinematics(*P(s))
    A = lambda s: np.array([P(s)[0] + model.Qk(i, *P(s)[1:]) @ model.a[i] for i in range(n)])
    Bp = lambda s: model.b0 + q(s)[:, None] * model.e
    u = lambda s: (A(s) - Bp(s)) / l
    qdd = d_dt(lambda s: d_dt(q, s), t)
    aA, aB = d_dt(lambda s: d_dt(A, s), t), d_dt(lambda s: d_dt(Bp, s), t)
    wl = lambda s: np.cross(u(s), d_dt(u, s))             # link angular velocity
    al = d_dt(wl, t)
    p, Q1, Q2, c_dot, c_ddot = motion(model, t)
    _, B = model.joint_axes(Q1, Q2)
    nb = B.shape[1]

    # Unknowns: [F_A1, F_B1, ..., F_An, F_Bn, f_J, mu]; F_A (F_B) acts on the
    # link at A (B), f_J and B mu act on platform 2 (minus on platform 1).
    nu = 6 * n + 3 + nb
    rows, rhs = [], []
    uu, A0, B0, ww = u(t), A(t), Bp(t), wl(t)
    for i in range(n):
        G = (1 - model.s_link) * B0[i] + model.s_link * A0[i]
        IG = model.Jt_link * (np.eye(3) - np.outer(uu[i], uu[i]))
        aG = (1 - model.s_link) * aB[i] + model.s_link * aA[i]
        R = np.zeros((6, nu))
        R[:3, 6 * i:6 * i + 3] = np.eye(3)
        R[:3, 6 * i + 3:6 * i + 6] = np.eye(3)
        R[3:, 6 * i:6 * i + 3] = skew(A0[i] - G)
        R[3:, 6 * i + 3:6 * i + 6] = skew(B0[i] - G)
        rows.append(R)
        rhs.append(np.r_[model.m_link * (aG - g), IG @ al[i] + np.cross(ww[i], IG @ ww[i])])
    w = (c_dot[3:6], c_dot[6:9])
    wd = (c_ddot[3:6], c_ddot[6:9])
    ext = np.zeros(9) if wrench is None else np.asarray(wrench, float)
    for k, Q in enumerate((Q1, Q2)):
        c = Q @ model.c0[k]
        IG = Q @ model.I0[k] @ Q.T
        aG = c_ddot[:3] + np.cross(wd[k], c) + np.cross(w[k], np.cross(w[k], c))
        sgn = -1.0 if k == 0 else 1.0
        R = np.zeros((6, nu))
        for i in np.flatnonzero(model.body == k + 1):
            R[:3, 6 * i:6 * i + 3] = -np.eye(3)
            R[3:, 6 * i:6 * i + 3] = -skew(A0[i] - p)        # moments about O_p
        R[:3, 6 * n:6 * n + 3] = sgn * np.eye(3)
        R[3:, 6 * n + 3:] = sgn * B
        rows.append(R)
        f_ext = ext[:3] if k == 0 else np.zeros(3)            # total force on body 1
        rhs.append(np.r_[model.m_body[k] * (aG - g) - f_ext,
                         IG @ wd[k] + np.cross(w[k], IG @ w[k])
                         + model.m_body[k] * np.cross(c, aG - g) - ext[3 + 3 * k:6 + 3 * k]])
    M, b = np.vstack(rows), np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(M, b, rcond=None)
    assert np.linalg.norm(M @ x - b) < 1e-8 * max(1.0, np.linalg.norm(b))
    FB = x[:6 * n].reshape(n, 6)[:, 3:]
    # Slider: (m_s + m_rotor) q_ddot = tau - e.F_B + m_s g.e
    tau = ((model.m_slider + model.m_rotor) * qdd - model.m_slider * model.e @ g
           + np.einsum('ij,ij->i', model.e, FB))
    return tau, x[6 * n + 3:]


@pytest.mark.parametrize('joint', ['spherical', 'universal', 'revolute'])
@pytest.mark.parametrize('t', [0.03, 0.17, 0.41])
def test_virtual_work_matches_newton_euler(joint, t):
    model = make(joint)
    p, Q1, Q2, cd, cdd = motion(model, t)
    wrench = np.r_[0.05, -0.02, 0.1, 1e-3, 0.0, -2e-3, 0.0, 1e-3, 5e-4]
    res = model.inverse_dynamics(p, Q1, Q2, cd, cdd, wrench, full=True)
    tau_ne, mu_ne = newton_euler(model, t, wrench)
    scale = np.max(np.abs(tau_ne))
    np.testing.assert_allclose(res['tau'], tau_ne, atol=1e-6 * scale)
    if res['mu'].size:
        np.testing.assert_allclose(res['mu'], mu_ne, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize('joint', ['spherical', 'universal', 'revolute'])
def test_reduced_model_is_consistent(joint):
    """Gamma^T tau = M_nu nu_dot + eta, M_nu symmetric positive definite, and
    the forward dynamics inverts the inverse dynamics."""
    model = make(joint)
    p, Q1, Q2, cd, cdd = motion(model, 0.23)
    T = model.complement(Q1, Q2)
    nu = np.linalg.lstsq(T, cd, rcond=None)[0]
    np.testing.assert_allclose(T @ nu, cd, atol=1e-10)
    rng = np.random.default_rng(0)
    nu_dot = rng.normal(size=nu.size)
    c_dot, c_ddot = model.nu_to_c(Q1, Q2, nu, nu_dot)
    tau = model.inverse_dynamics(p, Q1, Q2, c_dot, c_ddot)
    M, eta, G = model.reduced_model(p, Q1, Q2, nu)
    np.testing.assert_allclose(M, M.T, rtol=1e-9, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(M) > 0)
    np.testing.assert_allclose(G.T @ tau, M @ nu_dot + eta, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(model.forward_dynamics(p, Q1, Q2, nu, tau), nu_dot,
                               rtol=1e-7, atol=1e-9)


def test_power_balance_spherical():
    """tau^T q_dot = W*^T c_dot is the virtual-work statement itself."""
    model = make('spherical')
    p, Q1, Q2, cd, cdd = motion(model, 0.11)
    r = model.inverse_dynamics(p, Q1, Q2, cd, cdd, full=True)
    np.testing.assert_allclose((r['tau'] - r['tau0']) @ r['qd'], r['W'] @ cd, rtol=1e-10)


def test_static_load_at_home():
    """At rest only gravity acts: the nine forces carry the total weight."""
    model = make('spherical')
    kin = NineDofKinematics.from_yaml(os.path.join(CONFIG, 'geometry.yaml'))
    p, Q1, Q2 = kin.split(kin.home)
    tau = model.inverse_dynamics(p, Q1, Q2, np.zeros(9), np.zeros(9))
    weight = 9.81 * (model.m_body.sum() + model.n * (model.m_link + model.m_slider))
    np.testing.assert_allclose(tau.sum(), weight, rtol=1e-9)
