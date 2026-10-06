"""Inverse and forward dynamics of the two-platform PSS parallel robots.

Virtual-work model of docs/dynamics/dynamics_virtual_work.tex. The same code
covers the three architectures of the kinematic note through the internal
joint between the platforms:

    spherical  5PSS-S-4PSS   d_p = 3   n = 9
    universal  4PSS-U-4PSS   d_p = 2   n = 8
    revolute   4PSS-R-3PSS   d_p = 1   n = 7

Kinematics (identical to kinematics.py):

    m_i = p + Q_k a_i - b_i0 - q_i e_i,   m_i^T m_i = l^2
    J c_dot = K q_dot,                    c_dot = [p_dot, w1, w2]  (9,)

Dynamics (the result of the note):

    J^T lam = W*(c, c_dot, c_ddot)        one 9x9 solve
    tau     = tau0 + K^T lam              tau_i = tau0_i + K_ii lam_i

where W* gathers the inertial-minus-applied wrenches of both platforms about
the shared point O_p plus the end-point forces equivalent to the distal links,
tau0 holds the slider (and rotor) inertia, and the last 3 - d_p entries of lam
are the reaction moments of the internal joint.  Only numpy is required.
"""

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])
JOINT_DOF = {'spherical': 3, 'universal': 2, 'revolute': 1}


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot_axis(axis, t):
    """Rodrigues rotation of angle t about the unit vector axis."""
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    S = skew(k)
    return np.eye(3) + np.sin(t) * S + (1.0 - np.cos(t)) * S @ S


def _unit_normal(v):
    """Any vector not parallel to v (used as the auxiliary w of the note)."""
    k = int(np.argmin(np.abs(v)))
    w = np.zeros(3)
    w[k] = 1.0
    return w


class TwoPlatformDynamics:
    """Rigid-body dynamics of the two-platform robot.

    Parameters
    ----------
    kin : NineDofKinematics
        Provides b0, a, e, l and the platform index of every leg.
    dynamics : dict
        The ``ninedof_dynamics`` block of config/dynamics.yaml, optionally
        extended with ``com`` and ``inertia`` (see ``from_config``).
    joint : 'spherical' | 'universal' | 'revolute'
    axes : tuple of body-frame axes of the internal joint
        universal: (e1 in platform 1, e2 in platform 2);
        revolute:  (e_r in platform 1,).  Ignored for the spherical joint.
    legs : indices of the legs that are kept (default: all of kin).
    """

    def __init__(self, kin, dynamics, joint='spherical', axes=None, legs=None,
                 gravity=GRAVITY):
        idx = np.arange(len(kin.a)) if legs is None else np.asarray(legs)
        self.body = kin.platform[idx].astype(int)          # 1 or 2 for every leg
        self.b0 = kin.b0[idx]
        self.a = kin.a[idx]
        self.e = kin.e[idx]
        self.l = float(kin.l)
        self.n = len(idx)
        self.joint = joint
        self.dp = JOINT_DOF[joint]
        if self.n != 6 + self.dp:
            raise ValueError(f'{joint} joint needs {6 + self.dp} legs, got {self.n}')
        self.axes = None if axes is None else [np.asarray(v, float) / np.linalg.norm(v)
                                               for v in axes]
        if joint == 'revolute':
            er = self.axes[0]
            self.w0 = _unit_normal(er)                     # auxiliary w, body 1
        self.g = np.asarray(gravity, dtype=float)

        d = dynamics['ninedof_dynamics'] if 'ninedof_dynamics' in dynamics else dynamics
        ms = d['mass']
        self.m_body = np.array([ms['platform_1'], ms['platform_2']], dtype=float)
        self.c0 = [np.asarray(c, float) for c in d.get('com', [[0, 0, 0.005]] * 2)]
        self.I0 = [np.asarray(I, float) for I in d.get(
            'inertia', [np.diag([4.01094e-06, 4.01094e-06, 7.80938e-06]),
                        np.diag([3.90083e-06, 3.90083e-06, 7.595e-06])])]
        self.m_link = float(ms['distal_link'])
        self.s_link = float(d.get('link_com_ratio', 0.5))       # |B G| / l
        self.Jt_link = float(d.get('link_transverse_inertia',
                                   self.m_link * self.l ** 2 / 12.0))
        self.m_slider = float(ms['slider'])
        self.m_rotor = float(d.get('actuator', {}).get('armature', 0.0))

    # ------------------------------------------------------------ geometry
    def Qk(self, i, Q1, Q2):
        return Q1 if self.body[i] == 1 else Q2

    def inverse_kinematics(self, p, Q1, Q2):
        q = np.empty(self.n)
        for i in range(self.n):
            d = p + self.Qk(i, Q1, Q2) @ self.a[i] - self.b0[i]
            de = d @ self.e[i]
            disc = de ** 2 - (d @ d - self.l ** 2)
            if disc < 0.0:
                raise ValueError(f'leg {i}: no real inverse-kinematics solution')
            q[i] = de - np.sqrt(disc)
        return q

    def joint_axes(self, Q1, Q2):
        """D_p (3 x d_p) and B (3 x (3 - d_p)) in the fixed frame."""
        if self.joint == 'spherical':
            return np.eye(3), np.zeros((3, 0))
        if self.joint == 'universal':
            e1, e2 = Q1 @ self.axes[0], Q2 @ self.axes[1]
            return np.c_[e1, e2], np.cross(e1, e2)[:, None]
        er = Q1 @ self.axes[0]
        b1 = np.cross(er, Q1 @ self.w0)
        return er[:, None], np.c_[b1, np.cross(er, b1)]

    def joint_axes_rate(self, Q1, Q2, w1, w2):
        """D_p_dot and B_dot (every axis is fixed in one of the platforms)."""
        D, B = self.joint_axes(Q1, Q2)
        if self.joint == 'spherical':
            return np.zeros((3, 3)), B
        if self.joint == 'universal':
            e1, e2 = D[:, 0], D[:, 1]
            e1d, e2d = np.cross(w1, e1), np.cross(w2, e2)
            return np.c_[e1d, e2d], (np.cross(e1d, e2) + np.cross(e1, e2d))[:, None]
        return np.cross(w1, D[:, 0])[:, None], np.cross(w1, B.T).T

    def jacobians(self, p, Q1, Q2, q=None):
        """J (9 x 9) and K (9 x n) of J c_dot = K q_dot, plus m_i and r_i."""
        q = self.inverse_kinematics(p, Q1, Q2) if q is None else q
        J = np.zeros((9, 9))
        K = np.zeros((9, self.n))
        r = np.array([self.Qk(i, Q1, Q2) @ self.a[i] for i in range(self.n)])
        m = p + r - self.b0 - q[:, None] * self.e
        for i in range(self.n):
            col = 3 if self.body[i] == 1 else 6
            J[i, :3] = m[i]
            J[i, col:col + 3] = np.cross(r[i], m[i])
            K[i, i] = m[i] @ self.e[i]
        _, B = self.joint_axes(Q1, Q2)
        J[self.n:, 3:6] = -B.T
        J[self.n:, 6:9] = B.T
        return J, K, m, r

    def complement(self, Q1, Q2):
        """T (9 x n) with c_dot = T nu, nu = [p_dot, w1, phi_dot] (natural
        orthogonal complement of the joint rows: [0 -B^T B^T] T = 0)."""
        D, _ = self.joint_axes(Q1, Q2)
        T = np.zeros((9, 6 + self.dp))
        T[:6, :6] = np.eye(6)
        T[6:9, 3:6] = np.eye(3)
        T[6:9, 6:] = D
        return T

    # ------------------------------------------------------------ dynamics
    def inverse_dynamics(self, p, Q1, Q2, c_dot, c_ddot, wrench=None, full=False):
        """Actuator forces tau (n,) for the motion (c_dot, c_ddot).

        c_dot = [p_dot, w1, w2], c_ddot = [p_ddot, w1_dot, w2_dot] must satisfy
        the internal-joint constraint.  wrench = [f, n1, n2] (9,) is the external
        wrench applied to the platforms about O_p (f is the total force).
        """
        g = self.g
        pd, w = c_dot[:3], (c_dot[3:6], c_dot[6:9])
        pdd, wd = c_ddot[:3], (c_ddot[3:6], c_ddot[6:9])
        q = self.inverse_kinematics(p, Q1, Q2)
        J, K, m, r = self.jacobians(p, Q1, Q2, q)
        Kd = np.diag(K)
        Ws = np.zeros(9)

        # Platforms: inertial-minus-applied wrench about O_p.
        for k, Q in enumerate((Q1, Q2)):
            c = Q @ self.c0[k]
            IG = Q @ self.I0[k] @ Q.T
            aG = pdd + np.cross(wd[k], c) + np.cross(w[k], np.cross(w[k], c))
            Ws[:3] += self.m_body[k] * (aG - g)
            Ws[3 + 3 * k:6 + 3 * k] += (IG @ wd[k] + np.cross(w[k], IG @ w[k])
                                        + self.m_body[k] * np.cross(c, aG - g))

        # Legs: rates, then the distal link replaced by two end-point forces.
        qd, qdd, tau0 = np.empty(self.n), np.empty(self.n), np.empty(self.n)
        for i in range(self.n):
            k = self.body[i] - 1
            wk, wdk, ri, mi, ei = w[k], wd[k], r[i], m[i], self.e[i]
            ni = np.cross(ri, mi)
            vA = pd + np.cross(wk, ri)
            qd[i] = (mi @ pd + ni @ wk) / Kd[i]
            md = vA - qd[i] * ei
            cen = np.cross(wk, np.cross(wk, ri))
            aA = pdd + np.cross(wdk, ri) + cen
            qdd[i] = (mi @ pdd + ni @ wdk + mi @ cen + md @ md) / Kd[i]
            mdd = aA - qdd[i] * ei
            u = mi / self.l
            wl = np.cross(mi, md) / self.l ** 2
            al = np.cross(mi, mdd) / self.l ** 2
            Il = self.Jt_link * (np.eye(3) - np.outer(u, u))
            fl = self.m_link * ((1 - self.s_link) * qdd[i] * ei + self.s_link * aA - g)
            nl = Il @ al + np.cross(wl, Il @ wl)
            FA = self.s_link * fl + np.cross(nl, u) / self.l
            FB = (1 - self.s_link) * fl - np.cross(nl, u) / self.l
            Ws[:3] += FA
            Ws[3 + 3 * k:6 + 3 * k] += np.cross(ri, FA)
            tau0[i] = ((self.m_slider + self.m_rotor) * qdd[i]
                       - self.m_slider * g @ ei + FB @ ei)

        if wrench is not None:
            Ws -= np.asarray(wrench, dtype=float)
        lam = np.linalg.solve(J.T, Ws)
        tau = tau0 + Kd * lam[:self.n]
        if not full:
            return tau
        return dict(tau=tau, lam=lam[:self.n], mu=lam[self.n:], q=q, qd=qd, qdd=qdd,
                    J=J, K=K, W=Ws, tau0=tau0,
                    leg_force=lam[:self.n] * self.l)   # |lam_i m_i|, axial, N

    # ----------------------------------------- minimal (reduced) coordinates
    def nu_to_c(self, Q1, Q2, nu, nu_dot):
        """c_dot = T nu and c_ddot = T nu_dot + T_dot nu."""
        T = self.complement(Q1, Q2)
        c_dot = T @ nu
        Dd, _ = self.joint_axes_rate(Q1, Q2, c_dot[3:6], c_dot[6:9])
        c_ddot = T @ nu_dot
        c_ddot[6:9] += Dd @ nu[6:]
        return c_dot, c_ddot

    def gamma(self, p, Q1, Q2):
        """Gamma (n x n): q_dot = Gamma nu.  Square, invertible off singularities."""
        J, K, _, _ = self.jacobians(p, Q1, Q2)
        Jr = J[:self.n] @ self.complement(Q1, Q2)
        return Jr / np.diag(K)[:, None]

    def reduced_model(self, p, Q1, Q2, nu, wrench=None):
        """M_nu and eta of  M_nu nu_dot + eta = Gamma^T tau  (unit-acceleration
        method: each column of M_nu is one call to the inverse dynamics)."""
        G = self.gamma(p, Q1, Q2)
        z = np.zeros(6 + self.dp)
        g_save, self.g = self.g, np.zeros(3)
        M = np.empty((6 + self.dp, 6 + self.dp))
        for j in range(6 + self.dp):
            cd, cdd = self.nu_to_c(Q1, Q2, z, np.eye(6 + self.dp)[j])
            M[:, j] = G.T @ self.inverse_dynamics(p, Q1, Q2, cd, cdd)
        self.g = g_save
        cd, cdd = self.nu_to_c(Q1, Q2, nu, z)
        eta = G.T @ self.inverse_dynamics(p, Q1, Q2, cd, cdd, wrench)
        return M, eta, G

    def forward_dynamics(self, p, Q1, Q2, nu, tau, wrench=None):
        """nu_dot for the actuator forces tau."""
        M, eta, G = self.reduced_model(p, Q1, Q2, nu, wrench)
        return np.linalg.solve(M, G.T @ tau - eta)
