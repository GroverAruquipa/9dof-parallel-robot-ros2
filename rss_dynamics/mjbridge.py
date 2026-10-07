"""State transfer between the document's model and the MuJoCo model, and the
two independent MuJoCo computations used to validate the model:

  mujoco_inverse(...)  joint torques and loop forces from MuJoCo's own
                       recursive Newton-Euler on the spanning tree (mj_inverse
                       with the loops cut) plus the loop-closure forces solved
                       with MuJoCo's own constraint Jacobians:
                           M q_dd + c(q, q_d) = S^T tau + Jc^T f
  mujoco_forward(...)  accelerations from MuJoCo's forward dynamics with the
                       loops closed by its (very stiff) soft constraints.
"""
import numpy as np
import mujoco

from mjcf import quat_of


def rot_axis(n, t):
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def min_rot(a, b):
    """Smallest rotation taking unit a to unit b."""
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-14:
        return np.eye(3)
    return rot_axis(v, np.arctan2(s, c))


class Bridge:
    def __init__(self, robot, model, th0, h):
        self.r = robot
        self.m = model
        self.d = mujoco.MjData(model)
        self.th0 = th0
        self.h = h
        n = robot.n
        jid = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        self.qa_crank = np.array([model.jnt_qposadr[jid(f'th{k + 1}')] for k in range(n)])
        self.va_crank = np.array([model.jnt_dofadr[jid(f'th{k + 1}')] for k in range(n)])
        self.qa_link = np.array([model.jnt_qposadr[jid(f's{k + 1}')] for k in range(n)])
        self.va_link = np.array([model.jnt_dofadr[jid(f's{k + 1}')] for k in range(n)])
        self.qa_free = model.jnt_qposadr[jid('platform_1')]
        self.va_free = model.jnt_dofadr[jid('platform_1')]
        self.qa_c = model.jnt_qposadr[jid('central')]
        self.va_c = model.jnt_dofadr[jid('central')]
        # home leg directions (link frames are world-aligned at home)
        p0 = np.array([0, 0, h])
        kn = robot.kin(p0, np.eye(3), np.eye(3), th0)
        self.u0 = kn['m'] / np.linalg.norm(kn['m'], axis=1)[:, None]
        self.bid_link = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'link_{k + 1}') for k in range(n)]
        self.bid_plat = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'platform_{b}') for b in robot.body]
        self.bid_crank = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'crank_{k + 1}') for k in range(n)]

    # ------------------------------------------------------------------
    def full_state(self, p, Q1, Q2, cd, cdd=None, th=None, thd=None, thdd=None):
        """qpos, qvel (and qacc if cdd is given) of MuJoCo for a model state."""
        r = self.r
        th = r.ik(p, Q1, Q2) if th is None else th
        kn = r.kin(p, Q1, Q2, th)
        thd = r.joint_rates(kn, cd) if thd is None else thd
        qpos = np.zeros(self.m.nq); qvel = np.zeros(self.m.nv); qacc = np.zeros(self.m.nv)
        qpos[self.qa_crank] = th - self.th0
        qvel[self.va_crank] = thd
        w = r.omega_of(cd)
        md = cd[:3] - thd[:, None] * np.cross(r.v, kn['D']) + np.cross(w, kn['Qa'])
        if cdd is not None:
            if thdd is None:
                uu = r.accel_terms(kn, cd, thd)
                thdd = (kn['J'] @ cdd + uu) / kn['K']
            qacc[self.va_crank] = thdd
            wd = r.omega_of(cdd)
            bdd = thdd[:, None] * np.cross(r.v, kn['D']) - (thd ** 2)[:, None] * kn['D']
            mdd = cdd[:3] - bdd + np.cross(wd, kn['Qa']) + np.cross(w, np.cross(w, kn['Qa']))
        for i in range(r.n):
            Rc = rot_axis(r.v[i], th[i] - self.th0[i])            # crank world rotation
            u = kn['m'][i] / np.linalg.norm(kn['m'][i])
            Rl = min_rot(self.u0[i], u)                           # link world rotation (spin-free choice)
            qpos[self.qa_link[i]:self.qa_link[i] + 4] = quat_of(Rc.T @ Rl)
            l2 = kn['m'][i] @ kn['m'][i]
            wl = np.cross(kn['m'][i], md[i]) / l2                 # link angular velocity, no spin
            wc = thd[i] * r.v[i]
            qvel[self.va_link[i]:self.va_link[i] + 3] = Rl.T @ (wl - wc)
            if cdd is not None:
                al = np.cross(kn['m'][i], mdd[i]) / l2
                ac = thdd[i] * r.v[i]
                qacc[self.va_link[i]:self.va_link[i] + 3] = Rl.T @ (al - ac - np.cross(wl, wl - wc))
        qpos[self.qa_free:self.qa_free + 3] = p
        qpos[self.qa_free + 3:self.qa_free + 7] = quat_of(Q1)
        qvel[self.va_free:self.va_free + 3] = cd[:3]
        qvel[self.va_free + 3:self.va_free + 6] = Q1.T @ cd[3:6]
        qpos[self.qa_c:self.qa_c + 4] = quat_of(Q1.T @ Q2)
        qvel[self.va_c:self.va_c + 3] = Q2.T @ (cd[6:9] - cd[3:6])
        if cdd is not None:
            qacc[self.va_free:self.va_free + 3] = cdd[:3]
            qacc[self.va_free + 3:self.va_free + 6] = Q1.T @ cdd[3:6]
            qacc[self.va_c:self.va_c + 3] = Q2.T @ (cdd[6:9] - cdd[3:6] - np.cross(cd[6:9], cd[6:9] - cd[3:6]))
        return qpos, qvel, qacc, th, thd

    def model_state(self):
        """(p, Q1, Q2, c_dot) from the MuJoCo state."""
        d = self.d
        p = d.qpos[self.qa_free:self.qa_free + 3].copy()
        Q1 = np.zeros(9); mujoco.mju_quat2Mat(Q1, d.qpos[self.qa_free + 3:self.qa_free + 7]); Q1 = Q1.reshape(3, 3)
        Qr = np.zeros(9); mujoco.mju_quat2Mat(Qr, d.qpos[self.qa_c:self.qa_c + 4]); Qr = Qr.reshape(3, 3)
        Q2 = Q1 @ Qr
        w1 = Q1 @ d.qvel[self.va_free + 3:self.va_free + 6]
        w2 = w1 + Q2 @ d.qvel[self.va_c:self.va_c + 3]
        return p, Q1, Q2, np.r_[d.qvel[self.va_free:self.va_free + 3], w1, w2]

    def model_accel(self):
        d = self.d
        p, Q1, Q2, cd = self.model_state()
        w1d = Q1 @ d.qacc[self.va_free + 3:self.va_free + 6]
        w2d = w1d + Q2 @ d.qacc[self.va_c:self.va_c + 3] + np.cross(cd[6:9], cd[6:9] - cd[3:6])
        return np.r_[d.qacc[self.va_free:self.va_free + 3], w1d, w2d]

    def set(self, qpos, qvel, qacc=None):
        self.d.qpos[:] = qpos; self.d.qvel[:] = qvel
        if qacc is not None:
            self.d.qacc[:] = qacc

    # ------------------------------------------------------------------
    def loop_jacobian(self):
        """Jc (3n x nv): velocity of link end minus velocity of the platform point."""
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        Jc = np.zeros((3 * self.r.n, m.nv))
        jp1 = np.zeros((3, m.nv)); jp2 = np.zeros((3, m.nv))
        for i in range(self.r.n):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'link_end_{i + 1}')
            pt = d.site_xpos[sid].copy()
            mujoco.mj_jac(m, d, jp1, None, pt, self.bid_link[i])
            mujoco.mj_jac(m, d, jp2, None, pt, self.bid_plat[i])
            Jc[3 * i:3 * i + 3] = jp1 - jp2
        return Jc

    def mujoco_inverse(self, qpos, qvel, qacc):
        """tau (9,), loop forces f (9,3) on the link ends, residual."""
        m, d = self.m, self.d
        self.set(qpos, qvel, qacc)
        flags = m.opt.disableflags
        m.opt.disableflags = flags | (mujoco.mjtDisableBit.mjDSBL_EQUALITY | mujoco.mjtDisableBit.mjDSBL_SPRING
                                     | mujoco.mjtDisableBit.mjDSBL_DAMPER | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
        mujoco.mj_inverse(m, d)
        qf = d.qfrc_inverse.copy()
        m.opt.disableflags = flags
        Jc = self.loop_jacobian()
        S = np.zeros((self.r.n, m.nv)); S[np.arange(self.r.n), self.va_crank] = 1
        A = np.c_[S.T, Jc.T]
        sol, *_ = np.linalg.lstsq(A, qf, rcond=None)
        res = np.linalg.norm(A @ sol - qf) / max(1e-12, np.linalg.norm(qf))
        return sol[:self.r.n], sol[self.r.n:].reshape(-1, 3), res, Jc

    def mujoco_forward(self, qpos, qvel, tau, xfrc=None):
        m, d = self.m, self.d
        self.set(qpos, qvel)
        d.qfrc_applied[:] = 0
        d.qfrc_applied[self.va_crank] = tau
        d.ctrl[:] = 0
        if xfrc is not None:
            d.xfrc_applied[:] = xfrc
        mujoco.mj_forward(m, d)
        return d.qacc.copy()


def _kkt_parts(br, qpos, qvel, h=1e-6):
    """MuJoCo's M, bias c, loop Jacobian Jc and Jc_dot q_dot (central difference)."""
    m, d = br.m, br.d
    br.set(qpos, qvel)
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M)
    c = d.qfrc_bias.copy()
    Jc = br.loop_jacobian()
    qp = qpos.copy(); qm = qpos.copy()
    mujoco.mj_integratePos(m, qp, qvel, h)
    mujoco.mj_integratePos(m, qm, qvel, -h)
    br.set(qp, qvel); Jp = br.loop_jacobian()
    br.set(qm, qvel); Jm = br.loop_jacobian()
    br.set(qpos, qvel); mujoco.mj_forward(m, d)
    Jdqd = (Jp - Jm) @ qvel / (2 * h)
    return M, c, Jc, Jdqd


def mujoco_rigid_forward(br, qpos, qvel, tau, gen_force=None):
    """Forward dynamics with MuJoCo's own M, c and Jc and RIGID loop closure:
        [M  -Jc^T] [qdd]   [S^T tau - c]
        [Jc   0  ] [ f ] = [ -Jc_dot qd]
    (the soft-constraint regularisation of MuJoCo is not used)."""
    m = br.m
    M, c, Jc, Jdqd = _kkt_parts(br, qpos, qvel)
    rhs_q = -c.copy()
    rhs_q[br.va_crank] += tau
    if gen_force is not None:
        rhs_q += gen_force
    nc = Jc.shape[0]
    A = np.block([[M, -Jc.T], [Jc, np.zeros((nc, nc))]])
    sol = np.linalg.lstsq(A, np.r_[rhs_q, -Jdqd], rcond=None)[0]
    return sol[:m.nv], sol[m.nv:]
