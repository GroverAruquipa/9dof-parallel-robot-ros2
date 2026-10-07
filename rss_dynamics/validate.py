"""Validation of the dynamic model of the document, without bias.

Every check compares the model against something computed in a different way:

 V1  J, K, u                 vs finite differences of the inverse kinematics
 V2  Mc (energy)             vs kinetic energy summed body by body;
     power balance           tau^T theta_dot vs d(Ec + Ep)/dt by finite differences
 V3  inverse dynamics        vs MuJoCo: recursive Newton-Euler on the spanning
                             tree + loop forces from MuJoCo's constraint Jacobian
                             ('ideal' MuJoCo model = the document's hypotheses)
 V4  leg forces sigma_i l_i  vs MuJoCo loop forces
 V5  the real robot          MuJoCo 'real' model (carbon tubes with distributed
                             mass): error of the document's m_b/2 lumping and of
                             the 3-mass extension, in torque and in acceleration
 V6  direct dynamics         eq. directa integrated (RK4) vs MuJoCo's own
                             simulation (soft constraints, RK4) under the same
                             open-loop torques

    python3 validate.py d25        # figures/validation_*.png, results/validation_d25.json
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco

from robot_setup import setup, HERE
from trajectories import MultiSine, Showcase
from mjbridge import mujoco_rigid_forward
from sim_model import simulate, rot_err
from params import lumped_inertia
import plotstyle as ps

name = sys.argv[1] if len(sys.argv) > 1 else 'd25'
PAYLOAD = (0.020, 0.020)
T_OPEN = float(os.environ.get("T_OPEN", "1.5"))
FIG = os.path.join(HERE, 'figures')
RES = {}

S_ideal = setup(name, payload=PAYLOAD, mode='ideal')
S_real = setup(name, payload=PAYLOAD, mode='real')
r = S_ideal['robot']
h = S_ideal['h']
traj = MultiSine(h, T=4.0)
ts = np.linspace(0, traj.T, 401)

# --------------------------------------------------------------- V1 kinematics
def theta_of(t):
    p, Q1, Q2, *_ = traj(t)
    return r.ik(p, Q1, Q2)

e_vel, e_acc, n_vel, n_acc = [], [], [], []
for t in ts[1:-1]:
    p, Q1, Q2, cd, cdd = traj(t)
    ID = r.inverse_dynamics(p, Q1, Q2, cd, cdd)
    hh, h2 = 1e-5, 3e-5
    thp, thm, th0 = theta_of(t + hh), theta_of(t - hh), ID['th']
    fd_v = (thp - thm) / (2 * hh)
    fd_a = (theta_of(t + h2) - 2 * th0 + theta_of(t - h2)) / h2 ** 2
    e_vel.append(np.abs(fd_v - ID['thd']).max()); n_vel.append(np.abs(ID['thd']).max())
    e_acc.append(np.abs(fd_a - ID['thdd']).max()); n_acc.append(np.abs(ID['thdd']).max())
e_vel = np.array(e_vel) / max(n_vel); e_acc = np.array(e_acc) / max(n_acc)
RES['V1'] = dict(max_rel_err_theta_dot=float(e_vel.max()), max_rel_err_theta_dd=float(e_acc.max()),
                 note='central differences, h = 1e-5 s (velocity) and 3e-5 s (acceleration): the error shown is the finite-difference error')

# ------------------------------------------------------------- V2 energy/power
def total_energy(t):
    p, Q1, Q2, cd, cdd = traj(t)
    th = r.ik(p, Q1, Q2)
    kn = r.kin(p, Q1, Q2, th)
    thd = r.joint_rates(kn, cd)
    Ec, Ep = r.energy(p, Q1, Q2, cd, th, thd)
    Mc, _, _ = r.Mc_hc_gc(Q1, Q2, cd)
    Ec_mat = 0.5 * cd @ Mc @ cd + 0.5 * np.sum(r.Im * thd ** 2)
    return Ec, Ep, Ec_mat, thd

P_tau, P_dE, eEc = [], [], []
for t in ts[1:-1]:
    p, Q1, Q2, cd, cdd = traj(t)
    ID = r.inverse_dynamics(p, Q1, Q2, cd, cdd)
    hh = 1e-5
    Ecp, Epp, _, _ = total_energy(t + hh)
    Ecm, Epm, _, _ = total_energy(t - hh)
    Ec, Ep, Ecm_, thd = total_energy(t)
    P_tau.append(ID['tau'] @ ID['thd'])
    P_dE.append(((Ecp + Epp) - (Ecm + Epm)) / (2 * hh))
    eEc.append(abs(Ec - Ecm_) / max(Ec, 1e-12))
P_tau, P_dE = np.array(P_tau), np.array(P_dE)
RES['V2'] = dict(max_rel_err_Ec_matrix_vs_bodies=max(eEc),
                 max_abs_power_residual_W=float(np.abs(P_tau - P_dE).max()),
                 max_power_W=float(np.abs(P_tau).max()))

# ------------------------------------------------- V3/V4 inverse dynamics vs MuJoCo
def id_compare(S, rod_model):
    rr = S['robot']; b = S['bridge']
    rr.set_inertia(lumped_inertia(S['P'], rr.a, rod_model))
    T_mod, T_mj, F_mod, F_mj, A_err, res, A_ref = [], [], [], [], [], [], []
    for t in ts:
        p, Q1, Q2, cd, cdd = traj(t)
        ID = rr.inverse_dynamics(p, Q1, Q2, cd, cdd)
        qpos, qvel, qacc, th, thd = b.full_state(p, Q1, Q2, cd, cdd)
        tau_mj, fl, rs, _ = b.mujoco_inverse(qpos, qvel, qacc)
        T_mod.append(ID['tau']); T_mj.append(tau_mj); res.append(rs)
        F_mod.append(ID['sigma'] * np.linalg.norm(ID['m'], axis=1))
        # loop force on the platform = -f (f acts on the link); sign along m_i
        mhat = ID['m'] / np.linalg.norm(ID['m'], axis=1)[:, None]
        # in the 'ideal' MuJoCo model the platform half of the link mass sits on the
        # link at the ball: the rod force of the document is then -f + m_pt (a_A - g)
        w = rr.omega_of(cd); wd = rr.omega_of(cdd); Qa = p + rr.Qa(Q1, Q2) - p
        aA = cdd[:3] + np.cross(wd, Qa) + np.cross(w, np.cross(w, Qa))
        m_pt = S['P']['rod']['m'] / 2 if S is S_ideal else 0.0
        F_mj.append(np.einsum('ij,ij->i', -fl + m_pt * (aA - rr.g), mhat))
        # forward: accelerations from MuJoCo's rigid KKT with the model torques
        qdd, _ = mujoco_rigid_forward(b, qpos, qvel, ID['tau'])
        b.d.qacc[:] = qdd
        A_err.append(np.abs(b.model_accel() - cdd)); A_ref.append(np.abs(cdd))
    A_ref = np.array(A_ref)
    A_err = np.array(A_err) / np.r_[np.full(3, A_ref[:, :3].max()), np.full(6, A_ref[:, 3:].max())]
    return [np.array(a) for a in (T_mod, T_mj, F_mod, F_mj, A_err, res)]

T_mod, T_mj, F_mod, F_mj, A_err_i, res_i = id_compare(S_ideal, 'doc')
scale = np.abs(T_mj).max()
RES['V3'] = dict(max_abs_tau_err_Nm=float(np.abs(T_mod - T_mj).max()), max_tau_Nm=float(scale),
                 rel=float(np.abs(T_mod - T_mj).max() / scale),
                 lsq_residual_tree=float(np.max(res_i)),
                 max_rel_acc_err_rigid_forward=float(A_err_i.max()))
RES['V4'] = dict(max_abs_leg_force_err_N=float(np.abs(F_mod - F_mj).max()),
                 max_leg_force_N=float(np.abs(F_mj).max()))

Tr_doc, Tr_mj, _, _, Ar_doc, _ = id_compare(S_real, 'doc')
Tr_3m, _, _, _, Ar_3m, _ = id_compare(S_real, '3mass')
sc = np.abs(Tr_mj).max()
RES['V5'] = dict(doc_tau_max_rel=float(np.abs(Tr_doc - Tr_mj).max() / sc),
                 mass3_tau_max_rel=float(np.abs(Tr_3m - Tr_mj).max() / sc),
                 doc_acc_max_rel=float(Ar_doc.max()), mass3_acc_max_rel=float(Ar_3m.max()),
                 doc_acc_platform_rms_rel=float(np.sqrt((Ar_doc[:, :3] ** 2).mean())),
                 mass3_acc_platform_rms_rel=float(np.sqrt((Ar_3m[:, :3] ** 2).mean())))

# ---------------------------------------------------------- figures V1-V5
ps.style()
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
a = ax[0, 0]
a.semilogy(ts[1:-1], e_vel, label=r'$\dot\theta$: $\mathbf{K}^{-1}\mathbf{J}\dot{\mathbf{c}}$ vs dif. finitas')
a.semilogy(ts[1:-1], e_acc, label=r'$\ddot\theta$: $\mathbf{K}^{-1}(\mathbf{J}\ddot{\mathbf{c}}+\mathbf{u})$ vs dif. finitas')
a.set_title('V1  Cinemática (ecs. legrow, qdd)'); a.set_ylabel('error relativo'); a.set_xlabel('t (s)'); a.legend(fontsize=8)
a = ax[0, 1]
a.plot(ts[1:-1], P_tau, lw=2, label=r'$\boldsymbol{\tau}^T\dot{\boldsymbol{\theta}}$ (modelo)')
a.plot(ts[1:-1], P_dE, '--', lw=1.5, label=r'$d(E_c+E_p)/dt$ (dif. finitas)')
a.set_title('V2  Balance de potencias'); a.set_ylabel('W'); a.set_xlabel('t (s)'); a.legend(fontsize=8)
a2 = a.twinx(); a2.semilogy(ts[1:-1], np.abs(P_tau - P_dE) + 1e-16, color='0.6', lw=0.8); a2.set_ylabel('|residuo| (W)', color='0.5')
a = ax[1, 0]
for i in range(9):
    a.plot(ts, T_mod[:, i] * 1e3, color=ps.C9[i], lw=1.6)
    a.plot(ts[::8], T_mj[::8, i] * 1e3, 'o', ms=3.5, mfc='none', color=ps.C9[i])
a.set_title('V3  Par en las manivelas: modelo (línea) vs MuJoCo RNE+lazos (círculos)')
a.set_ylabel('τ (mN·m)'); a.set_xlabel('t (s)')
a = ax[1, 1]
a.semilogy(ts, np.abs(T_mod - T_mj).max(1) / scale + 1e-18, label='inversa: |Δτ|/max|τ| (ideal)')
a.semilogy(ts, A_err_i.max(1) + 1e-18, label=r'directa: |Δ$\ddot{\mathbf{c}}$| rel. (ideal, KKT rígido)')
a.semilogy(ts, np.abs(F_mod - F_mj).max(1) / np.abs(F_mj).max() + 1e-18, label=r'fuerza en bielas $\sigma_i l_i$')
a.set_title('V3/V4  Modelo vs MuJoCo con las mismas hipótesis'); a.set_ylabel('error relativo'); a.set_xlabel('t (s)')
a.set_ylim(1e-14, 1); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, f'validation_ideal_{name}.png'), dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
a = ax[0]
for i in range(9):
    a.plot(ts, F_mj[:, i], color=ps.C9[i], lw=1.4)
    a.plot(ts[::8], F_mod[::8, i], 'o', ms=3, mfc='none', color=ps.C9[i])
a.set_title(r'V4  Fuerza en bielas $\sigma_i l_i$: MuJoCo (línea) vs modelo (círculos)', fontsize=10)
a.set_ylabel('N (+ tracción)'); a.set_xlabel('t (s)')
a = ax[1]
a.semilogy(ts, np.abs(Tr_doc - Tr_mj).max(1) / sc, label='documento (m$_b$/2 + m$_b$/2)')
a.semilogy(ts, np.abs(Tr_3m - Tr_mj).max(1) / sc, label='extensión 3 masas')
a.set_title('V5  Robot real (tubos de carbono): error en par', fontsize=10); a.set_ylabel('|Δτ| / max|τ|'); a.set_xlabel('t (s)'); a.legend(fontsize=8)
a = ax[2]
a.semilogy(ts, Ar_doc[:, :3].max(1), label=r'documento: $\ddot{\mathbf{p}}$')
a.semilogy(ts, Ar_doc[:, 3:].max(1), label=r'documento: $\dot{\boldsymbol{\omega}}_{1,2}$')
a.semilogy(ts, Ar_3m[:, :3].max(1), label=r'3 masas: $\ddot{\mathbf{p}}$')
a.semilogy(ts, Ar_3m[:, 3:].max(1), label=r'3 masas: $\dot{\boldsymbol{\omega}}_{1,2}$')
a.set_title('V5  Robot real: error de la dinámica directa', fontsize=10); a.set_ylabel('error relativo'); a.set_xlabel('t (s)'); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, f'validation_real_{name}.png'), dpi=150); plt.close(fig)

# --------------------------------------------- V6 forward simulation, open loop
def open_loop(S, rod_model, T=None, dt=1e-4):
    T = T_OPEN if T is None else T
    rr = S['robot']; b = S['bridge']; m = S['model']
    rr.set_inertia(lumped_inertia(S['P'], rr.a, rod_model))
    # open-loop torques: the model's inverse dynamics of the reference trajectory
    # evaluated on the reference (not on the simulated state), tabulated
    tt = np.arange(0, T + dt, 5 * dt)
    TT = np.array([rr.inverse_dynamics(*traj(t))['tau'] for t in tt])
    tau_fn = lambda t, *a: np.array([np.interp(t, tt, TT[:, i]) for i in range(9)])
    p, Q1, Q2, cd, cdd = traj(0.0)
    out = simulate(rr, p, Q1, Q2, cd, tau_fn, T, dt=dt, every=20)
    qpos, qvel, _, _, _ = b.full_state(p, Q1, Q2, cd)
    m.opt.timestep = dt
    # the document's model has no friction: switch MuJoCo's motor friction off here
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS | mujoco.mjtDisableBit.mjDSBL_DAMPER
    mujoco.mj_resetData(m, b.d)
    b.set(qpos, qvel)
    mj = []
    nsteps = int(round(T / dt))
    for k in range(nsteps + 1):
        if k % 20 == 0:
            mj.append(b.model_state())
        if k == nsteps:
            break
        b.d.qfrc_applied[:] = 0
        b.d.qfrc_applied[b.va_crank] = tau_fn(b.d.time)
        mujoco.mj_step(m, b.d)
    m.opt.disableflags = 0
    ref = [traj(o[0]) for o in out]
    dp = np.array([np.linalg.norm(o[1] - q[0]) for o, q in zip(out, mj)])
    dq = np.array([max(rot_err(o[2], q[1]), rot_err(o[3], q[2])) for o, q in zip(out, mj)])
    dref = np.array([np.linalg.norm(o[1] - rf[0]) for o, rf in zip(out, ref)])
    t_out = np.array([o[0] for o in out])
    return t_out, dp, dq, dref, out, mj

V6 = {}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for S, rm, lab, col in ((S_ideal, 'doc', 'MuJoCo ideal vs modelo (documento)', ps.C[0]),
                        (S_real, 'doc', 'MuJoCo real vs modelo (documento)', ps.C[1]),
                        (S_real, '3mass', 'MuJoCo real vs modelo (3 masas)', ps.C[2])):
    t_o, dp, dq, dref, out, mj = open_loop(S, rm)
    V6[lab] = dict(max_dp_mm=float(dp.max() * 1e3), max_dQ_deg=float(np.degrees(dq.max())),
                   drift_from_reference_mm=float(dref.max() * 1e3))
    ax[0].semilogy(t_o, dp * 1e3 + 1e-12, color=col, label=lab)
    ax[1].semilogy(t_o, np.degrees(dq) + 1e-12, color=col, label=lab)
ax[0].semilogy(t_o, dref * 1e3, 'k:', label='modelo vs trayectoria de referencia (lazo abierto)')
ax[0].set_title('V6  Simulación directa en lazo abierto: |Δp|'); ax[0].set_ylabel('mm'); ax[0].set_xlabel('t (s)')
ax[1].set_title(r'V6  ángulo de $\mathbf{Q}_{k,\mathrm{modelo}}^T\mathbf{Q}_{k,\mathrm{MuJoCo}}$'); ax[1].set_ylabel('grados'); ax[1].set_xlabel('t (s)')
ax[0].legend(fontsize=7.5); ax[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, f'validation_forward_{name}.png'), dpi=150); plt.close(fig)
RES['V6'] = V6

json.dump(RES, open(os.path.join(HERE, 'results', f'validation_{name}.json'), 'w'), indent=1)
print(json.dumps(RES, indent=1))
