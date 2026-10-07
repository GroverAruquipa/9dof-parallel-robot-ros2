"""Forward simulation with the document's direct dynamics (eq. directa).

State: p, Q1, Q2 (rotation matrices, advanced with Q_dot = Omega Q through the
exponential map, no Euler angles) and c_dot = [p_dot, w1, w2].  The crank
angles are not states: they follow from the analytic inverse kinematics.
Integrator: classical RK4 on (p, c_dot) with the rotations advanced by the
RK4-weighted angular velocity (a Lie-group RK4 of Crouch-Grossman type would
be more accurate for long runs; at dt = 1e-4 s the difference is negligible).
"""
import numpy as np

from model import expm_so3


def rk4_step(robot, p, Q1, Q2, cd, tau_fn, t, dt, f_ext_fn=None):
    def acc(pp, q1, q2, cdd_, tt):
        th = robot.ik(pp, q1, q2)
        kn_tau = tau_fn(tt, pp, q1, q2, cdd_, th)
        fe = None if f_ext_fn is None else f_ext_fn(tt, pp, q1, q2, cdd_)
        return robot.direct_dynamics(pp, q1, q2, cdd_, kn_tau, th=th, f_ext=fe)

    def adv(cd_, h):
        return p + h * cd_[:3], expm_so3(h * cd_[3:6]) @ Q1, expm_so3(h * cd_[6:9]) @ Q2

    k1v = cd
    k1a = acc(p, Q1, Q2, cd, t)
    s2 = adv(k1v, dt / 2); k2v = cd + dt / 2 * k1a; k2a = acc(*s2, k2v, t + dt / 2)
    s3 = adv(k2v, dt / 2); k3v = cd + dt / 2 * k2a; k3a = acc(*s3, k3v, t + dt / 2)
    s4 = adv(k3v, dt);     k4v = cd + dt * k3a;     k4a = acc(*s4, k4v, t + dt)
    v = (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    a = (k1a + 2 * k2a + 2 * k3a + k4a) / 6
    p_n = p + dt * v[:3]
    Q1n = expm_so3(dt * v[3:6]) @ Q1
    Q2n = expm_so3(dt * v[6:9]) @ Q2
    # re-orthonormalise
    for Q in (Q1n, Q2n):
        U, _, Vt = np.linalg.svd(Q); Q[:] = U @ Vt
    return p_n, Q1n, Q2n, cd + dt * a


def simulate(robot, p, Q1, Q2, cd, tau_fn, T, dt=1e-4, every=10, f_ext_fn=None):
    out = []
    t = 0.0
    nsteps = int(round(T / dt))
    for k in range(nsteps + 1):
        if k % every == 0:
            out.append((t, p.copy(), Q1.copy(), Q2.copy(), cd.copy()))
        if k == nsteps:
            break
        try:
            p, Q1, Q2, cd = rk4_step(robot, p, Q1, Q2, cd, tau_fn, t, dt, f_ext_fn)
        except ValueError:          # left the workspace (open loop): stop here
            break
        t += dt
    return out


def rot_err(Qa, Qb):
    """Angle (rad) of Qa^T Qb."""
    R = Qa.T @ Qb
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
