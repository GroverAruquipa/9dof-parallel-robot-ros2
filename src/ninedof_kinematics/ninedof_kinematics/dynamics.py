"""Inverse dynamics of the 9-DoF 5PSS-S-4PSS parallel robot (principle of
virtual work, pure numpy).

The moving bodies are the two platforms, the nine sliders (actuated prismatic
joints) and the nine distal links (SS rods). With the twist of the platforms
c_dot = [p_dot, w1, w2] and the kinematics J c_dot = K q_dot of
kinematics.py, q_dot = H c_dot with H = K^-1 J, and the virtual power of the
actuator forces balances that of the inertia, gravity and damping wrenches:

    f^T q_dot = sum_b (F_b . v_b + N_b . w_b)      for every c_dot
    F_b = m_b (a_b - g),   N_b = I_b dw_b + w_b x I_b w_b (+ joint damping)

Splitting the sum into the sliders (whose velocity is q_dot_i e_i) and the
other bodies (velocity Jacobians Jv_b, Jw_b in c_dot):

    f = K J^-T tau_c + f_slider,    tau_c = sum_b (Jv_b^T F_b + Jw_b^T N_b)
    f_slider,i = (m_s + m_a) q_ddot_i - m_s g . e_i

The distal links are uniform about their axis u_i (B_i -> A_i) and do not
spin about it (an SS rod cannot transmit a moment about its axis):
w_i = m_i x m_dot_i / l^2 and dw_i = m_i x m_ddot_i / l^2, m_i = A_i - B_i.
The actuator acceleration follows from d^2/dt^2 |m_i|^2 = 0:
q_ddot_i = (|m_dot_i|^2 + m_i . a_Ai) / (m_i . e_i).

The pose x = [p, alpha, beta] uses the Euler angles of kinematics.py, so
w = E(angles) angles_dot and dw = E angles_ddot + E_dot angles_dot.
"""

import numpy as np
import yaml

from ninedof_kinematics.kinematics import NineDofKinematics, rot_x, rot_y

GRAVITY = np.array([0.0, 0.0, -9.81])
E_X, E_Y, E_Z = np.eye(3)


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def full_inertia(v):
    """3x3 tensor from [ixx, iyy, izz, ixy, ixz, iyz]."""
    ixx, iyy, izz, ixy, ixz, iyz = v
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def euler_rates(angles, rates):
    """E and E_dot of w = E(angles) rates for Q = Qx Qy Qz."""
    Rx = rot_x(angles[0])
    c2, c3 = Rx @ E_Y, Rx @ rot_y(angles[1]) @ E_Z
    w1 = rates[0] * E_X
    w12 = w1 + rates[1] * c2
    E = np.column_stack((E_X, c2, c3))
    E_dot = np.column_stack((np.zeros(3), np.cross(w1, c2), np.cross(w12, c3)))
    return E, E_dot


def platform_twist(x, xd, xdd):
    """c_dot = [p_dot, w1, w2] and c_ddot from the pose and its derivatives."""
    x, xd, xdd = (np.asarray(v, dtype=float) for v in (x, xd, xdd))
    cd, cdd = np.empty(9), np.empty(9)
    cd[:3], cdd[:3] = xd[:3], xdd[:3]
    for s in (slice(3, 6), slice(6, 9)):
        E, E_dot = euler_rates(x[s], xd[s])
        cd[s] = E @ xd[s]
        cdd[s] = E @ xdd[s] + E_dot @ xd[s]
    return cd, cdd


class NineDofDynamics:

    def __init__(self, kinematics, dynamics, gravity=GRAVITY):
        d = dynamics['ninedof_dynamics'] if 'ninedof_dynamics' in dynamics else dynamics
        self.kin = kinematics
        self.g = np.asarray(gravity, dtype=float)
        m = d['mass']
        self.m_slider = float(m['slider'])
        self.m_rod = float(m['distal_link'])
        self.armature = float(d['actuator'].get('armature', 0.0))
        self.damping = float(d.get('passive_joint_damping', 0.0))
        rod = d['inertia']['distal_link']
        Irod = full_inertia(rod['inertia'])
        self.rod_com = float(rod['com'][2]) / self.kin.l   # fraction of l from B_i
        self.I_rod_perp = float(Irod[0, 0])   # about the centre of mass, normal to u
        self.I_rod_axial = float(Irod[2, 2])
        # Platform k: mass, centre of mass and inertia in its frame (origin S).
        self.m_plat = {k: float(m[f'platform_{k}']) for k in (1, 2)}
        self.c_plat = {k: np.asarray(d['inertia'][f'platform_{k}']['com'], dtype=float)
                       for k in (1, 2)}
        self.I_plat = {k: full_inertia(d['inertia'][f'platform_{k}']['inertia'])
                       for k in (1, 2)}

    @classmethod
    def from_yaml(cls, geometry_path, dynamics_path, **kw):
        with open(dynamics_path) as f:
            dyn = yaml.safe_load(f)
        return cls(NineDofKinematics.from_yaml(geometry_path), dyn, **kw)

    # ----------------------------------------------------------- kinematics
    def leg_motion(self, x, cd, cdd, q=None):
        """q, q_dot, q_ddot and the vectors of each leg for the platform twist
        cd and its derivative cdd."""
        kin = self.kin
        if q is None:
            q = kin.inverse(x)
        p, Q1, Q2 = kin.split(x)
        J, K = kin.jacobians(x, q)
        qd = np.linalg.solve(K, J @ cd)
        n = len(q)
        r, vA, aA = np.empty((n, 3)), np.empty((n, 3)), np.empty((n, 3))
        for i in range(n):
            k = kin.platform[i]
            w, dw = (cd[3:6], cdd[3:6]) if k == 1 else (cd[6:9], cdd[6:9])
            r[i] = (Q1 if k == 1 else Q2) @ kin.a[i]
            vA[i] = cd[:3] + np.cross(w, r[i])
            aA[i] = cdd[:3] + np.cross(dw, r[i]) + np.cross(w, np.cross(w, r[i]))
        m = kin.platform_points(x) - kin.base_points(q)
        md = vA - qd[:, None] * kin.e
        qdd = (np.einsum('ij,ij->i', md, md) + np.einsum('ij,ij->i', m, aA)) \
            / np.einsum('ij,ij->i', m, kin.e)
        mdd = aA - qdd[:, None] * kin.e
        return dict(q=q, qd=qd, qdd=qdd, J=J, K=K, r=r, m=m, md=md, mdd=mdd, aA=aA)

    # ------------------------------------------------------------- dynamics
    def inverse_dynamics(self, x, xd, xdd, q=None, gravity=True, damping=True):
        """Actuator forces f (9,) for the trajectory point (x, x_dot, x_ddot).

        Returns (f, motion) where motion holds q, q_dot, q_ddot of the legs."""
        cd, cdd = platform_twist(x, xd, xdd)
        return self.inverse_dynamics_twist(x, cd, cdd, q, gravity, damping)

    def inverse_dynamics_twist(self, x, cd, cdd, q=None, gravity=True, damping=True):
        kin = self.kin
        g = self.g if gravity else np.zeros(3)
        c = self.damping if damping else 0.0
        mo = self.leg_motion(x, cd, cdd, q)
        J, K = mo['J'], mo['K']
        H = np.linalg.solve(K, J)       # q_dot = H c_dot
        p, Q1, Q2 = kin.split(x)
        tau = np.zeros(9)
        w_rel = cd[6:9] - cd[3:6]       # central spherical joint, platform 2 vs 1

        # Platforms (origin S, centre of mass r = Q c).
        for k, Q, s in ((1, Q1, slice(3, 6)), (2, Q2, slice(6, 9))):
            w, dw = cd[s], cdd[s]
            r = Q @ self.c_plat[k]
            a = cdd[:3] + np.cross(dw, r) + np.cross(w, np.cross(w, r))
            I = Q @ self.I_plat[k] @ Q.T
            F = self.m_plat[k] * (a - g)
            N = I @ dw + np.cross(w, I @ w) + (c * w_rel if k == 2 else -c * w_rel)
            tau[:3] += F
            tau[s] += np.cross(r, F) + N     # Jv = [I, -[r]x], Jw = [0, I]

        # Distal links: centre of mass at B_i + rod_com m_i, no spin.
        l2 = kin.l ** 2
        for i in range(len(kin.a)):
            s = slice(3, 6) if kin.platform[i] == 1 else slice(6, 9)
            JA = np.zeros((3, 9))
            JA[:, :3] = np.eye(3)
            JA[:, s] = -skew(mo['r'][i])
            JB = np.outer(kin.e[i], H[i])
            Jv = (1 - self.rod_com) * JB + self.rod_com * JA
            Jw = skew(mo['m'][i]) @ (JA - JB) / l2
            aB = mo['qdd'][i] * kin.e[i]
            a = (1 - self.rod_com) * aB + self.rod_com * mo['aA'][i]
            w = np.cross(mo['m'][i], mo['md'][i]) / l2
            dw = np.cross(mo['m'][i], mo['mdd'][i]) / l2
            # w is normal to the axis, so I w = I_perp w and w x I w = 0.
            F = self.m_rod * (a - g)
            N = self.I_rod_perp * dw + c * w
            tau += Jv.T @ F + Jw.T @ N

        f = K @ np.linalg.solve(J.T, tau)
        f += (self.m_slider + self.armature) * mo['qdd'] - self.m_slider * (kin.e @ g)
        return f, mo

    def gravity_forces(self, x):
        """Static actuator forces holding the pose x."""
        z = np.zeros(9)
        return self.inverse_dynamics(x, z, z)[0]


def harmonic_trajectory(home, amplitude, frequency, t):
    """x(t) = home + amplitude (1 - cos(2 pi frequency t)) / 2 on each pose
    coordinate, with its exact first and second derivatives. Starts at rest."""
    w = 2 * np.pi * np.asarray(frequency, dtype=float)
    a = np.asarray(amplitude, dtype=float) / 2
    x = np.asarray(home, dtype=float) + a * (1 - np.cos(w * t))
    return x, a * w * np.sin(w * t), a * w ** 2 * np.cos(w * t)
