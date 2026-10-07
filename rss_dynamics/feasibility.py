"""Feasibility of the Matsushita MPX-40C4WA motors with a GT2 belt stage of ratio N.

For each crank length (25 / 35 mm) and each N:
  - crank torque along the showcase (1x, 2x, 3x speed), with and without a payload,
    split into the rotor part N^2 J theta_dd and the rest (model with 3-mass links)
  - motor current I = (tau / (N eta) + friction) / Kt for both Kt bounds
  - force the platform can hold (worst direction, worst pose) with the continuous current
  - pHRI transparency: apparent mass at O_p, Coulomb friction felt when backdriving,
    encoder resolution at the platform

    python3 feasibility.py          # figures/feasibility_*.png, results/feasibility.json
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from robot_setup import setup, HERE
from trajectories import Showcase
from params import MOTOR, lumped_inertia, build as build_params
from design import workspace, core_workspace
from geometry import pose_from_params
import plotstyle as ps

ETA = 0.95                     # one GT2 stage
DESIGNS = [(n, ls) for n, ls in zip(os.environ.get('DESIGNS', 'd25,d35').split(','), ('-', '--', ':'))]
NS = np.array([1, 1.5, 2, 2.5, 3, 3.75, 4.5, 6, 8])
FIG = os.path.join(HERE, 'figures')


def showcase_profile(S, speed, payload, dt=0.01):
    """theta_dot, theta_dd and crank torque without the rotor (N-independent part)."""
    r = S['robot']
    P = build_params(S['d'], S['x'][4], 1.0, payload=payload, J_rotor=0.0)
    r.set_inertia(lumped_inertia(P, r.a, '3mass'))
    # remove the motor pulley as well: it is reflected with N^2 below
    r.Im = r.Im - P['pulley_motor']['J']
    tr = Showcase(S['h'], speed=speed)
    ts = np.arange(0, tr.duration, dt)
    TH, THD, THDD, TAU = [], [], [], []
    for t in ts:
        ID = r.inverse_dynamics(*tr(t))
        TH.append(ID['th']); THD.append(ID['thd']); THDD.append(ID['thdd']); TAU.append(ID['tau'])
    return ts, np.array(TH), np.array(THD), np.array(THDD), np.array(TAU)


def motor_current(tau_c, thd, thdd, N, Kt):
    """Motor current for crank torque tau_c (without rotor), crank rates."""
    Jm = MOTOR['J_rotor'] + 5.0e-8          # rotor + motor pulley
    tau_m = tau_c / (N * ETA) + Jm * N * thdd \
        + MOTOR['tau_coulomb'] * np.sign(thd) + MOTOR['b_visc'] * N * thd
    return tau_m / Kt


def static_capacity(S, N, Kt, I, W):
    """Largest force (N) and moment (N m) at O_p that the robot can hold in every
    direction and every pose of W, with current I, against gravity and friction."""
    r = S['robot']
    Fmin, Mmin = np.inf, np.inf
    tau_av = N * ETA * Kt * I - N * MOTOR['tau_coulomb']
    dirs = np.random.default_rng(0).normal(size=(400, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    for v in W:
        p, Q1, Q2 = pose_from_params(S['h'], v)
        A = r.actuator_space(p, Q1, Q2, np.zeros(9))
        G = A['G']
        tau_g = -A['ga']                       # holding torque (zero velocity)
        margin = tau_av - np.abs(tau_g)
        if np.any(margin <= 0):
            return 0.0, 0.0
        Tf = np.abs(G[:3].T @ dirs.T)           # torque per unit force at O_p (9, ndir)
        Fmin = min(Fmin, np.min(np.min(margin[:, None] / Tf, axis=0)))
        Tm = np.abs(G[3:6].T @ dirs.T) + 0 * Tf  # moment on platform 1 ...
        Tm2 = np.abs((G[3:6] + G[6:9]).T @ dirs.T)  # ... and on both together
        Mmin = min(Mmin, np.min(np.min(margin[:, None] / Tm2, axis=0)))
    return Fmin, Mmin


def transparency(S, N, W):
    """Apparent translational mass at O_p (rotations free), friction force felt when
    pushing O_p (both platforms translating), platform resolution of one count."""
    r = S['robot']
    P = build_params(S['d'], S['x'][4], N, payload=(0, 0))
    r.set_inertia(lumped_inertia(P, r.a, '3mass'))
    mapp, ffric, res = [], [], []
    for v in W:
        p, Q1, Q2 = pose_from_params(S['h'], v)
        A = r.actuator_space(p, Q1, Q2, np.zeros(9))
        G = A['G']
        Ginv = np.linalg.inv(G)                # theta_dot = Ginv c_dot
        Mtot = Ginv.T @ A['Ma'] @ Ginv         # inertia in c (rotors + cranks + bodies)
        Lam = np.linalg.inv(np.linalg.inv(Mtot)[:3, :3])
        ev = np.linalg.eigvalsh(Lam)
        mapp.append([ev.min(), ev.max()])
        for e in np.eye(3):
            thd = Ginv @ np.r_[e, 0, 0, 0, 0, 0, 0]
            ffric.append(np.sum(N * MOTOR['tau_coulomb'] * np.abs(thd)))
        dth = 2 * np.pi / (MOTOR['cpr'] * N)
        res.append([np.abs(G[:3]).sum(1).max() * dth, np.abs(G[3:]).sum(1).max() * dth])
    return np.array(mapp), np.array(ffric), np.array(res)


def main():
    out = {}
    ps.style()
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.6))
    for name, ls in DESIGNS:
        S = setup(name, mode='ideal')
        dlab = f'd = {S["d"] * 1e3:.0f} mm'
        res = {}
        prof = {}
        for speed in (1, 2, 3):
            for pl in (0.0, 0.05):
                prof[(speed, pl)] = showcase_profile(S, speed, (pl, pl))
        # torque decomposition at 2x speed, 50 g per platform
        ts, TH, THD, THDD, TAU = prof[(2, 0.05)]
        res['showcase_2x_50g'] = dict(
            crank_torque_without_rotor_peak_Nm=float(np.abs(TAU).max()),
            crank_speed_peak_rad_s=float(np.abs(THD).max()),
            crank_acc_peak_rad_s2=float(np.abs(THDD).max()))
        Ipk = {k: [] for k in ('low', 'high')}
        Irms = {k: [] for k in ('low', 'high')}
        for N in NS:
            for kk, Kt in (('low', MOTOR['Kt_low']), ('high', MOTOR['Kt_high'])):
                I = motor_current(TAU, THD, THDD, N, Kt)
                Ipk[kk].append(np.abs(I).max())
                Irms[kk].append(np.sqrt((I ** 2).mean(0)).max())
        res['I_peak_A_Ktlow'] = dict(zip(NS.tolist(), Ipk['low']))
        res['I_rms_A_Ktlow'] = dict(zip(NS.tolist(), Irms['low']))
        res['I_peak_A_Kthigh'] = dict(zip(NS.tolist(), Ipk['high']))
        a = ax[0, 0]
        a.plot(NS, Ipk['low'], ls, color=ps.C[0], label=f'{dlab}: pico, Kt bajo')
        a.plot(NS, Irms['low'], ls, color=ps.C[1], label=f'{dlab}: RMS, Kt bajo')
        a.plot(NS, Ipk['high'], ls, color=ps.C[0], alpha=0.4, label=f'{dlab}: pico, Kt alto')
        # capacity and transparency, on the core workspace (pHRI) and on the whole showcase
        for wname, W in (('core', core_workspace()), ('full', workspace()[:45])):
            F, M, mlo, mhi, ff, rs, rr_ = [], [], [], [], [], [], []
            for N in NS:
                S2 = setup(name, mode='ideal', N=N)
                Fc, Mc = static_capacity(S2, N, MOTOR['Kt_low'], MOTOR['I_cont'], W)
                F.append(Fc); M.append(Mc)
                mp, fr, rr = transparency(S2, N, W)
                mlo.append(mp[:, 0].min()); mhi.append(mp[:, 1].max()); ff.append(fr.max())
                rs.append(rr[:, 0].max()); rr_.append(rr[:, 1].max())
            res[wname] = dict(force_capacity_N=dict(zip(NS.tolist(), F)), moment_capacity_Nm=dict(zip(NS.tolist(), M)),
                              apparent_mass_kg_min=dict(zip(NS.tolist(), mlo)), apparent_mass_kg_max=dict(zip(NS.tolist(), mhi)),
                              coulomb_force_N_max=dict(zip(NS.tolist(), ff)),
                              resolution_mm=dict(zip(NS.tolist(), (np.array(rs) * 1e3).tolist())),
                              resolution_deg=dict(zip(NS.tolist(), np.degrees(rr_).tolist())))
            al = 1.0 if wname == 'core' else 0.35
            lab = f'{dlab} ({"núcleo" if wname == "core" else "showcase completo"})'
            ax[0, 1].plot(NS, F, ls, color=ps.C[2], alpha=al, label=lab)
            ax[0, 2].plot(NS, np.array(mhi), ls, color=ps.C[4], alpha=al, label=lab + ': máx')
            ax[0, 2].plot(NS, np.array(mlo), ls, color=ps.C[3], alpha=al, label=lab + ': mín')
            ax[1, 0].plot(NS, ff, ls, color=ps.C[1], alpha=al, label=lab)
            ax[1, 1].plot(NS, np.array(rs) * 1e3, ls, color=ps.C[0], alpha=al, label=lab + ' (mm)')
            ax[1, 1].plot(NS, np.degrees(rr_), ls, color=ps.C[3], alpha=al, label=lab + ' (grados)')
        # torque split at N = 3 (example)
        if name == 'd25':
            N = 3.0
            Jm = MOTOR['J_rotor'] + 5e-8
            a = ax[1, 2]
            i = np.argmax(np.abs(TAU).max(0))
            a.plot(ts, TAU[:, i] * 1e3, color=ps.C[0], label='plataformas + bielas + manivela + gravedad')
            a.plot(ts, N ** 2 * Jm * THDD[:, i] * 1e3, color=ps.C[1], label=r'rotor reflejado $N^2 J_r\ddot\theta$')
            a.plot(ts, N * MOTOR['tau_coulomb'] * np.sign(THD[:, i]) * 1e3, color='0.5', lw=0.8, label=r'fricción Coulomb $N\tau_c$ (estimada)')
            a.set_title(f'Par en la manivela {i + 1}, showcase 2x, 50 g, N = 3, d = 25 mm', fontsize=10)
            a.set_xlabel('t (s)'); a.set_ylabel('mN·m'); a.legend(fontsize=7.5)
        out[name] = res
    ax[0, 0].axhline(MOTOR['I_cont'], color='k', lw=0.8, ls=':'); ax[0, 0].text(1, MOTOR['I_cont'] * 1.04, 'I continua (estimada)', fontsize=8)
    ax[0, 0].set_title('Corriente del motor, showcase 2x con 50 g por plataforma'); ax[0, 0].set_xlabel('N (correa)'); ax[0, 0].set_ylabel('A'); ax[0, 0].legend(fontsize=7)
    ax[0, 1].set_title('Fuerza sostenible en O$_p$ (peor dirección y pose), I = 2 A, Kt bajo'); ax[0, 1].set_xlabel('N'); ax[0, 1].set_ylabel('N'); ax[0, 1].legend(fontsize=8)
    ax[0, 2].set_title('Masa aparente en O$_p$ (rotaciones libres)'); ax[0, 2].set_xlabel('N'); ax[0, 2].set_ylabel('kg'); ax[0, 2].legend(fontsize=7.5)
    ax[1, 0].set_title('Fricción de Coulomb al empujar O$_p$ (motor sin compensar)'); ax[1, 0].set_xlabel('N'); ax[1, 0].set_ylabel('N'); ax[1, 0].legend(fontsize=8)
    ax[1, 1].set_title('Resolución (1 cuenta en cada encoder de motor)'); ax[1, 1].set_xlabel('N'); ax[1, 1].set_ylabel('mm  /  grados'); ax[1, 1].set_yscale('log'); ax[1, 1].legend(fontsize=6.5)
    for a in (ax[0, 1], ax[0, 2], ax[1, 0]):
        a.set_yscale('log')
    ax[0, 2].legend(fontsize=6); ax[0, 1].legend(fontsize=7); ax[1, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'feasibility.png'), dpi=150); plt.close(fig)
    json.dump(out, open(os.path.join(HERE, 'results', 'feasibility.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
