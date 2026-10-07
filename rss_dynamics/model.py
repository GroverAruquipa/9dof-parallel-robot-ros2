"""Kinematics and dynamics of the 9-DoF 5RSS-S-4RSS robot.

This file implements, equation by equation, the model of the document
"Modelo dinamico de las arquitecturas de 9, 8 y 7 grados de libertad"
for the spherical case (D_p = I_3, no joint rows, lambda = sigma):

    c_dot = [p_dot, w1, w2]                      (eq. maestra)
    m_i = p - b_i + Q_k a_i,  b_i = c_i + d_i    (eq. mi, bi)
    n_i = Q_k a_i x m_i,  K_ii = m_i^T (v_i x d_i)
    J c_dot = K theta_dot                         (eq. legrow, maestra)
    J c_dd + u = K theta_dd                       (eq. qdd, maestraacc)
    J^T sigma = Mc c_dd + hc - gc                 (eq. NEmat, deflam)
    tau = Im theta_dd - gm + K sigma              (eq. fi, fvec)
    Ma, ca, ga                                    (eq. fq)
    direct dynamics in minimal coordinates        (eq. directa)

Units: SI.  Crank angle theta_i is measured about v_i from the unit vector
u_i:  d_i = d (cos(theta) u_i + sin(theta) w_i),  w_i = v_i x u_i,
so that d_dot = theta_dot v_i x d_i (eq. bdot).
Only numpy is required.
"""
import numpy as np

G_DEFAULT = np.array([0.0, 0.0, -9.81])


# ----------------------------------------------------------------- algebra
def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_xyz(a):
    return rot_x(a[0]) @ rot_y(a[1]) @ rot_z(a[2])


def expm_so3(w):
    """Rotation matrix of the rotation vector w (Rodrigues)."""
    t = np.linalg.norm(w)
    if t < 1e-14:
        return np.eye(3) + skew(w)
    k = skew(w / t)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def euler_xyz_rates(a, ad, add):
    """Angular velocity and acceleration (world frame) of Q = Qx(a1) Qy(a2) Qz(a3).

    w = a1' e_x + a2' Qx e_y + a3' Qx Qy e_z  (each partial rotation about its
    current axis, as eq. O1 of the document for ZYZ).
    """
    ex, ey, ez = np.eye(3)
    Rx = rot_x(a[0])
    e2 = Rx @ ey
    e3 = Rx @ rot_y(a[1]) @ ez
    w = ad[0] * ex + ad[1] * e2 + ad[2] * e3
    w12 = ad[0] * ex + ad[1] * e2            # angular velocity of the frame carrying e3
    wd = (add[0] * ex + add[1] * e2 + ad[1] * np.cross(ad[0] * ex, e2)
          + add[2] * e3 + ad[2] * np.cross(w12, e3))
    return w, wd


# ---------------------------------------------------------------- the robot
class RSS9:
    """5RSS-S-4RSS robot.

    geom: dict with
        d (9,) crank lengths, l (9,) distal lengths,
        body (9,) 1 or 2, a (9,3) platform points in the body frame,
        c (9,3) crank pivots, v (9,3) crank axes, u (9,3) zero directions,
        branch (9,) +1/-1 inverse-kinematics branch.
    inert: dict with
        M (2,), s0 (2,3) centre of mass in the body frame (from O_p),
        IC0 (2,3,3) inertia about the centre of mass in the body frame,
        Im (9,) crank-side inertia about v_i (crank + lumped half link +
                reflected rotor and pulleys),
        mm (9,) crank-side mass, rc (9,) distance of its centre of mass along d_i.
    """

    def __init__(self, geom, inert, g=G_DEFAULT):
        self.d = np.asarray(geom['d'], float)
        self.l = np.asarray(geom['l'], float)
        self.body = np.asarray(geom['body'], int)
        self.a = np.asarray(geom['a'], float)
        self.c = np.asarray(geom['c'], float)
        self.v = np.asarray(geom['v'], float)
        self.v /= np.linalg.norm(self.v, axis=1)[:, None]
        u = np.asarray(geom['u'], float)
        u = u - np.einsum('ij,ij->i', u, self.v)[:, None] * self.v
        self.u = u / np.linalg.norm(u, axis=1)[:, None]
        self.w = np.cross(self.v, self.u)
        self.branch = np.asarray(geom['branch'], float)
        self.n = len(self.d)
        self.set_inertia(inert)
        self.g = np.asarray(g, float)
        self.slot = np.where(self.body == 1, 3, 6)

    def set_inertia(self, inert):
        self.M = np.asarray(inert['M'], float)
        self.s0 = np.asarray(inert['s0'], float)
        self.IC0 = np.asarray(inert['IC0'], float)
        self.Im = np.asarray(inert['Im'], float)
        self.mm = np.asarray(inert['mm'], float)
        self.rc = np.asarray(inert['rc'], float)
        # extension (not in the document): 2/3 of the tube mass at the middle of
        # each link (thin-rod 3-mass equivalent m/6 - 2m/3 - m/6); zero = document
        self.m_mid = np.asarray(inert.get('m_mid', np.zeros(len(self.Im))), float)

    # ------------------------------------------------------------ geometry
    def crank(self, th):
        """d_i (9,3) and the centre-of-mass vector d_ci (9,3)."""
        e = np.cos(th)[:, None] * self.u + np.sin(th)[:, None] * self.w
        return self.d[:, None] * e, self.rc[:, None] * e

    def Qa(self, Q1, Q2):
        out = np.empty((self.n, 3))
        for i in range(self.n):
            out[i] = (Q1 if self.body[i] == 1 else Q2) @ self.a[i]
        return out

    def ik(self, p, Q1, Q2, branch=None):
        """Crank angles (eq. mi solved for theta).  Raises ValueError if unreachable."""
        br = self.branch if branch is None else branch
        e = p + self.Qa(Q1, Q2) - self.c
        A = np.einsum('ij,ij->i', e, self.u)
        B = np.einsum('ij,ij->i', e, self.w)
        E = np.einsum('ij,ij->i', e, e) - np.einsum('ij,ij->i', e, self.v) ** 2  # in-plane part
        vz = np.einsum('ij,ij->i', e, self.v)
        C = (A ** 2 + B ** 2 + vz ** 2 + self.d ** 2 - self.l ** 2) / (2 * self.d)
        rho = np.hypot(A, B)
        r = C / rho
        if np.any(np.abs(r) > 1):
            raise ValueError('unreachable: legs %s' % np.flatnonzero(np.abs(r) > 1))
        return np.arctan2(B, A) + br * np.arccos(r)

    # ----------------------------------------------------------- velocity
    def kin(self, p, Q1, Q2, th):
        """m (9,3), n (9,3), Qa, d_i, K (9,), J (9,9) of eq. maestra."""
        D, _ = self.crank(th)
        Qa = self.Qa(Q1, Q2)
        m = p + Qa - self.c - D
        nn = np.cross(Qa, m)
        K = np.einsum('ij,ij->i', m, np.cross(self.v, D))
        J = np.zeros((self.n, 9))
        J[:, :3] = m
        for i in range(self.n):
            J[i, self.slot[i]:self.slot[i] + 3] = nn[i]
        return dict(m=m, n=nn, Qa=Qa, D=D, K=K, J=J)

    def omega_of(self, cd):
        return np.where((self.body == 1)[:, None], cd[3:6], cd[6:9])

    def joint_rates(self, kn, cd):
        """theta_dot (eq. qinv)."""
        return kn['J'] @ cd / kn['K']

    def accel_terms(self, kn, cd, thd):
        """u_i of eq. qdd."""
        m, Qa, D = kn['m'], kn['Qa'], kn['D']
        w = self.omega_of(cd)
        md = cd[:3] - thd[:, None] * np.cross(self.v, D) + np.cross(w, Qa)
        return (np.einsum('ij,ij->i', md, md)
                + np.einsum('ij,ij->i', m, np.cross(w, np.cross(w, Qa)))
                + thd ** 2 * np.einsum('ij,ij->i', m, D))

    # ------------------------------------------------------------- inertia
    def bodies(self, Q1, Q2):
        """s_k, IC_k, IO_k in the fixed frame (eqs. datos, IO)."""
        out = []
        for k, Q in enumerate((Q1, Q2)):
            s = Q @ self.s0[k]
            IC = Q @ self.IC0[k] @ Q.T
            S = skew(s)
            IO = IC - self.M[k] * S @ S
            out.append((s, S, IC, IO))
        return out

    def Mc_hc_gc(self, Q1, Q2, cd):
        """Eqs. Mc and hcgc."""
        (s1, S1, _, IO1), (s2, S2, _, IO2) = self.bodies(Q1, Q2)
        M1, M2 = self.M
        w1, w2 = cd[3:6], cd[6:9]
        Mc = np.zeros((9, 9))
        Mc[:3, :3] = (M1 + M2) * np.eye(3)
        Mc[:3, 3:6] = -M1 * S1
        Mc[:3, 6:9] = -M2 * S2
        Mc[3:6, :3] = M1 * S1
        Mc[3:6, 3:6] = IO1
        Mc[6:9, :3] = M2 * S2
        Mc[6:9, 6:9] = IO2
        hc = np.r_[M1 * np.cross(w1, np.cross(w1, s1)) + M2 * np.cross(w2, np.cross(w2, s2)),
                   np.cross(w1, IO1 @ w1), np.cross(w2, IO2 @ w2)]
        g = self.g
        gc = np.r_[(M1 + M2) * g, M1 * np.cross(s1, g), M2 * np.cross(s2, g)]
        return Mc, hc, gc

    def gm(self, th):
        """g_m of eq. fvec: m_mi g^T (v_i x d_ci)."""
        _, Dc = self.crank(th)
        return self.mm * (np.cross(self.v, Dc) @ self.g)

    # ------------------------------------------------------- inverse model
    def inverse_dynamics(self, p, Q1, Q2, cd, cdd, th=None):
        """Eqs. qinv, laminv and fi.  Returns a dict with theta, rates, sigma, tau."""
        th = self.ik(p, Q1, Q2) if th is None else th
        kn = self.kin(p, Q1, Q2, th)
        thd = self.joint_rates(kn, cd)
        uu = self.accel_terms(kn, cd, thd)
        thdd = (kn['J'] @ cdd + uu) / kn['K']
        Mc, hc, gc = self.Mc_hc_gc(Q1, Q2, cd)
        rhs = Mc @ cdd + hc - gc
        tau_mid = np.zeros(self.n)
        if np.any(self.m_mid > 0):
            Fh, wrench = self.mid_loads(kn, cd, cdd, thd, thdd)
            rhs = rhs + wrench
            tau_mid = np.einsum('ij,ij->i', Fh, np.cross(self.v, kn['D']))
        sig = np.linalg.solve(kn['J'].T, rhs)
        tau = self.Im * thdd - self.gm(th) + kn['K'] * sig + tau_mid
        return dict(th=th, thd=thd, thdd=thdd, sigma=sig, tau=tau, K=kn['K'], J=kn['J'],
                    m=kn['m'], u=uu, leg_force=sig * self.l)

    def mid_loads(self, kn, cd, cdd, thd, thdd):
        """Extension: inertia-minus-weight F_i = m_mid (a_mid - g) of the middle mass of
        each link, a_mid = (a_b + a_A)/2.  By virtual work it acts as F_i/2 at the
        crank ball and F_i/2 at the platform ball.  Returns F_i/2 (9,3) and its
        generalised force on c (9,), to be added to Mc c_dd + hc - gc."""
        w = self.omega_of(cd); wd = self.omega_of(cdd)
        Qa, D = kn['Qa'], kn['D']
        aA = cdd[:3] + np.cross(wd, Qa) + np.cross(w, np.cross(w, Qa))
        ab = thdd[:, None] * np.cross(self.v, D) - (thd ** 2)[:, None] * D
        Fh = 0.5 * self.m_mid[:, None] * (0.5 * (aA + ab) - self.g)
        wrench = np.zeros(9)
        wrench[:3] = Fh.sum(0)
        tq = np.cross(Qa, Fh)
        wrench[3:6] = tq[self.body == 1].sum(0)
        wrench[6:9] = tq[self.body == 2].sum(0)
        return Fh, wrench

    # ------------------------------------------------- actuator-space form
    def actuator_space(self, p, Q1, Q2, cd, th=None):
        """Ma, ca, ga of eq. fq and G = J^-1 K."""
        th = self.ik(p, Q1, Q2) if th is None else th
        kn = self.kin(p, Q1, Q2, th)
        thd = self.joint_rates(kn, cd)
        uu = self.accel_terms(kn, cd, thd)
        Mc, hc, gc = self.Mc_hc_gc(Q1, Q2, cd)
        Jinv = np.linalg.inv(kn['J'])
        G = Jinv * kn['K'][None, :]
        Ma = np.diag(self.Im) + G.T @ Mc @ G
        ca = G.T @ (hc - Mc @ Jinv @ uu)
        ga = self.gm(th) + G.T @ gc
        return dict(Ma=Ma, ca=ca, ga=ga, G=G, th=th, thd=thd)

    # ---------------------------------------------------- direct dynamics
    E9 = np.block([[np.eye(3), np.zeros((3, 6))],
                   [np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                   [np.zeros((3, 3)), np.eye(3), np.eye(3)]])

    def direct_dynamics(self, p, Q1, Q2, cd, tau, th=None, f_ext=None):
        """Eq. directa with D_p = I (E_dot = 0).  Returns c_dd.

        f_ext (optional, 9,) is an external generalized force on c (same
        coordinates as g_c), added to g_c.
        """
        th = self.ik(p, Q1, Q2) if th is None else th
        kn = self.kin(p, Q1, Q2, th)
        thd = self.joint_rates(kn, cd)
        uu = self.accel_terms(kn, cd, thd)
        Mc, hc, gc = self.Mc_hc_gc(Q1, Q2, cd)
        if f_ext is not None:
            gc = gc + f_ext
        if np.any(self.m_mid > 0):
            return self._direct_affine(p, Q1, Q2, cd, tau, th, kn, thd, f_ext)
        E = self.E9
        Jx = kn['J'] @ E
        Mx, cx, gx = E.T @ Mc @ E, E.T @ hc, E.T @ gc
        Kinv = 1.0 / kn['K']
        W = self.Im * Kinv ** 2
        A = Mx + Jx.T @ (W[:, None] * Jx)
        b = Jx.T @ (Kinv * (tau + self.gm(th))) - Jx.T @ (W * uu) - cx + gx
        xdd = np.linalg.solve(A, b)
        return E @ xdd

    def _direct_affine(self, p, Q1, Q2, cd, tau, th, kn, thd, f_ext):
        """Direct dynamics with the middle masses: the inverse dynamics is affine in
        c_dd, so tau(c_dd) = A c_dd + b is built from 10 inverse-dynamics calls."""
        def tau_of(cdd):
            out = self.inverse_dynamics(p, Q1, Q2, cd, cdd, th=th)
            t = out['tau']
            if f_ext is not None:          # external generalized force on c
                t = t - out['K'] * np.linalg.solve(kn['J'].T, f_ext)
            return t
        E = self.E9
        b = tau_of(np.zeros(9))
        A = np.column_stack([tau_of(E[:, j]) - b for j in range(9)])
        return E @ np.linalg.solve(A, tau - b)

    # ------------------------------------------------- forward kinematics
    def fk(self, th, p, Q1, Q2, iters=30, tol=1e-11):
        """Pose from the crank angles, Newton on J dc = K dtheta from a nearby pose."""
        p, Q1, Q2 = p.copy(), Q1.copy(), Q2.copy()
        thc = self.ik(p, Q1, Q2)
        for _ in range(iters):
            err = (th - thc + np.pi) % (2 * np.pi) - np.pi
            e0 = np.max(np.abs(err))
            if e0 < tol:
                break
            kn = self.kin(p, Q1, Q2, thc)
            dc = np.linalg.solve(kn['J'], kn['K'] * err)
            step = 1.0
            while step > 1e-3:              # backtracking: stay where the IK exists
                pn = p + step * dc[:3]
                Q1n = expm_so3(step * dc[3:6]) @ Q1
                Q2n = expm_so3(step * dc[6:9]) @ Q2
                try:
                    thn = self.ik(pn, Q1n, Q2n)
                    en = np.max(np.abs((th - thn + np.pi) % (2 * np.pi) - np.pi))
                    if en < e0:
                        break
                except ValueError:
                    pass
                step *= 0.5
            else:
                break
            p, Q1, Q2, thc = pn, Q1n, Q2n, thn
        return p, Q1, Q2

    # --------------------------------------------------------------- energy
    def energy(self, p, Q1, Q2, cd, th, thd):
        """Kinetic and potential energy, computed body by body (eqs. Eck, Ep)."""
        Ec, Ep = 0.0, 0.0
        for k, Q in enumerate((Q1, Q2)):
            s = Q @ self.s0[k]
            w = cd[3 + 3 * k:6 + 3 * k]
            vk = cd[:3] + np.cross(w, s)
            IC = Q @ self.IC0[k] @ Q.T
            Ec += 0.5 * self.M[k] * vk @ vk + 0.5 * w @ IC @ w
            Ep -= self.M[k] * self.g @ (p + s)
        _, Dc = self.crank(th)
        Ec += 0.5 * np.sum(self.Im * thd ** 2)
        Ep -= np.sum(self.mm * ((self.c + Dc) @ self.g))
        return Ec, Ep


# ---------------------------------------------------------- trajectories
class EulerTrajectory:
    """x(t) = [p (3), alpha (3), beta (3)] with Q1 = Qxyz(alpha), Q2 = Qxyz(beta).

    f(t) must return x, x_dot, x_ddot (each (9,)).
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, t):
        x, xd, xdd = self.f(t)
        Q1, Q2 = rot_xyz(x[3:6]), rot_xyz(x[6:9])
        w1, w1d = euler_xyz_rates(x[3:6], xd[3:6], xdd[3:6])
        w2, w2d = euler_xyz_rates(x[6:9], xd[6:9], xdd[6:9])
        return x[:3], Q1, Q2, np.r_[xd[:3], w1, w2], np.r_[xdd[:3], w1d, w2d]


def quintic(t, T):
    """s, s', s'' of the 0->1 quintic blend on [0, T]."""
    if t <= 0:
        return 0.0, 0.0, 0.0
    if t >= T:
        return 1.0, 0.0, 0.0
    r = t / T
    return (10 * r**3 - 15 * r**4 + 6 * r**5,
            (30 * r**2 - 60 * r**3 + 30 * r**4) / T,
            (60 * r - 180 * r**2 + 120 * r**3) / T**2)
