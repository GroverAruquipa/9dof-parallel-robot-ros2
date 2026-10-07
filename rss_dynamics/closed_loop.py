"""Closed-loop simulations in MuJoCo ('real' robot: carbon links with distributed
mass, rotor inertia, motor friction, encoder quantisation, torque saturation).

  tracking(...)  showcase trajectory with a soft joint PD, with and without the
                 model feedforward (inverse dynamics of the reference + friction)
  phri(...)      Cartesian impedance with model compensation; a 'human' pushes
                 and twists the platforms; the momentum observer estimates the
                 human wrench from the motor torques only

    python3 closed_loop.py d25
"""
import json
import os
import sys
import numpy as np
import mujoco

from robot_setup import setup, HERE
from trajectories import Showcase
from control import Encoders, friction_model, MomentumObserver, PoseEstimator, pose_error
from params import MOTOR, lumped_inertia
from model import rot_x

ETA = 0.95


def tau_limit(N):
    return N * ETA * MOTOR['Kt_low'] * MOTOR['I_peak']


def crank_angles(S):
    b = S['bridge']
    return b.d.qpos[b.qa_crank] + S['th0']


def tracking(S, speed=2.0, feedforward=True, Kt=600.0, Kr=0.5, zeta=0.8, T=None, fps=30,
             rod_model='3mass', quantise=True, fc=60.0):
    """Showcase tracking with a Cartesian PD (impedance) on c, from the encoders only:
        tau = tau_ff + G^T (K e + D e_dot),  tau_ff = inverse dynamics of the reference + friction.
    A joint PD is not used: the robot is far more compliant in some directions
    (yaw of each half), so equal joint gains give very unequal Cartesian stiffness."""
    r, b, m = S['robot'], S['bridge'], S['model']
    r.set_inertia(lumped_inertia(S['P'], r.a, rod_model))
    tr = Showcase(S['h'], speed=speed)
    T = tr.duration if T is None else T
    dt = m.opt.timestep
    dtc = 1e-3
    ctrl_every = int(round(dtc / dt))
    enc = Encoders(S['N'], dtc, quantise=quantise, fc=fc)
    mujoco.mj_resetData(m, b.d)
    p, Q1, Q2, cd, cdd = tr(0.0)
    qpos, qvel, *_ = b.full_state(p, Q1, Q2, cd)
    b.set(qpos, qvel)
    mujoco.mj_forward(m, b.d)
    est = PoseEstimator(r, p, Q1, Q2)
    Kd9 = np.r_[np.full(3, Kt), np.full(6, Kr)]
    Dd9 = 2 * zeta * np.sqrt(Kd9 * np.r_[np.full(3, 0.25), np.full(6, 2e-4)])
    tlim = tau_limit(S['N'])
    log = dict(t=[], tau=[], tau_ff=[], th=[], th_ref=[], ep=[], eR=[], label=[], qpos=[], tfr=[])
    tau = np.zeros(9)
    nsteps = int(round(T / dt))
    next_frame = 0.0
    for k in range(nsteps):
        t = k * dt
        if k % ctrl_every == 0:
            th_m, thd_m = enc.read(crank_angles(S))
            pe, Q1e, Q2e = est.update(th_m)
            ref = tr(t)
            ID = r.inverse_dynamics(*ref)
            tau_ff = ID['tau'] + friction_model(ID['thd'], S['N']) if feedforward else np.zeros(9)
            kn = r.kin(pe, Q1e, Q2e, th_m)
            G = np.linalg.solve(kn['J'], np.diag(kn['K']))
            cde = G @ thd_m
            w = Kd9 * pose_error(ref[0], ref[1], ref[2], pe, Q1e, Q2e) + Dd9 * (ref[3] - cde)
            tau = np.clip(tau_ff + G.T @ w, -tlim, tlim)
            pm, Q1m, Q2m, _ = b.model_state()
            ep = pose_error(ref[0], ref[1], ref[2], pm, Q1m, Q2m)
            log['t'].append(t); log['tau'].append(tau.copy()); log['tau_ff'].append(tau_ff.copy())
            log['th'].append(crank_angles(S)); log['th_ref'].append(ID['th'])
            log['ep'].append(np.linalg.norm(ep[:3])); log['eR'].append(max(np.linalg.norm(ep[3:6]), np.linalg.norm(ep[6:9])))
            log['label'].append(tr.label(t))
        if t >= next_frame - 1e-12:
            log['qpos'].append(b.d.qpos.copy()); log['tfr'].append(t); next_frame += 1.0 / fps
        b.d.qfrc_applied[:] = 0
        b.d.qfrc_applied[b.va_crank] = tau
        mujoco.mj_step(m, b.d)
    return {k: (np.array(v) if k != 'label' else v) for k, v in log.items()}


def human_force(t):
    """Force (N) at O_p and moment (N m) on platform 2 applied by the 'human'."""
    from trajectories import smootherstep
    def bump(t0, t1, T=0.4):
        a = smootherstep(t - t0, T)[0]; b_ = smootherstep(t - t1, T)[0]
        return a - b_
    f = np.array([6.0, 0, 0]) * bump(0.5, 2.0) + np.array([0, 0, -6.0]) * bump(2.8, 4.3) \
        + np.array([0, 5.0, 0]) * bump(5.1, 6.6)
    mo2 = np.array([0.06, 0, 0]) * bump(7.4, 8.9)       # twist platform 2 about x: opens the gripper
    return f, mo2


def phri(S, T=10.0, fps=30, Kt=(250.0, 250.0, 250.0), Kr=0.6, zeta=0.9, comp_friction=0.8,
         rod_model='3mass', obs_model='3mass', quantise=True):
    """Impedance: w = K e - D c_dot (9-D: translation, rotation 1, rotation 2), tau = G^T w + g + friction."""
    r, b, m = S['robot'], S['bridge'], S['model']
    r.set_inertia(lumped_inertia(S['P'], r.a, rod_model))
    dt = m.opt.timestep
    dtc = 1e-3
    ctrl_every = int(round(dtc / dt))
    enc = Encoders(S['N'], dtc, quantise=quantise)
    obs = MomentumObserver(KO=60.0)
    mujoco.mj_resetData(m, b.d)
    h = S['h']
    p_d, Q_d = np.array([0, 0, h]), np.eye(3)
    jaw0 = 8 * np.pi / 180
    Q1_d, Q2_d = rot_x(jaw0), rot_x(-jaw0)
    qpos, qvel, *_ = b.full_state(p_d, Q1_d, Q2_d, np.zeros(9))
    b.set(qpos, qvel); mujoco.mj_forward(m, b.d)
    est = PoseEstimator(r, p_d, Q1_d, Q2_d)
    Kd9 = np.r_[np.array(Kt), np.full(6, Kr)]
    tlim = tau_limit(S['N'])
    pid1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_1')
    pid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_2')
    log = dict(t=[], f_true=[], m2_true=[], w_est=[], dp=[], dR1=[], dR2=[], tau=[], qpos=[], tfr=[])
    tau = np.zeros(9)
    obs_inert = lumped_inertia(S['P'], r.a, obs_model)
    next_frame = 0.0
    for k in range(int(round(T / dt))):
        t = k * dt
        f_h, m2_h = human_force(t)
        if k % ctrl_every == 0:
            th_m, thd_m = enc.read(crank_angles(S))
            p, Q1, Q2 = est.update(th_m)
            A = r.actuator_space(p, Q1, Q2, np.zeros(9), th=th_m)
            G = A['G']
            cd = G @ thd_m
            A = r.actuator_space(p, Q1, Q2, cd, th=th_m)
            # damping from a modal mass of 0.25 kg / 2e-4 kg m^2 (critical-ish)
            Dd = 2 * zeta * np.sqrt(Kd9 * np.r_[np.full(3, 0.25), np.full(6, 2e-4)])
            w = Kd9 * pose_error(p_d, Q1_d, Q2_d, p, Q1, Q2) - Dd * cd
            tf_hat = friction_model(thd_m, S['N'])
            tau = np.clip(G.T @ w - A['ga'] + comp_friction * tf_hat, -tlim, tlim)
            # observer with its own inertial model
            r.set_inertia(obs_inert)
            Ao = r.actuator_space(p, Q1, Q2, cd, th=th_m)
            r.set_inertia(lumped_inertia(S['P'], r.a, rod_model))
            rr = obs.update(Ao['Ma'], Ao['ca'], Ao['ga'], thd_m, tau, tf_hat, dtc)
            w_est = np.linalg.solve(G.T, rr)
            pm, Q1m, Q2m, _ = b.model_state()
            log['t'].append(t); log['f_true'].append(f_h); log['m2_true'].append(m2_h)
            log['w_est'].append(w_est); log['tau'].append(tau.copy())
            e = pose_error(p_d, Q1_d, Q2_d, pm, Q1m, Q2m)
            log['dp'].append(-e[:3]); log['dR1'].append(-e[3:6]); log['dR2'].append(-e[6:9])
        if t >= next_frame - 1e-12:
            log['qpos'].append(b.d.qpos.copy()); log['tfr'].append(t); next_frame += 1.0 / fps
        # human: force at O_p (on platform 1) and a moment on platform 2
        b.d.xfrc_applied[:] = 0
        xO = b.d.xpos[pid1]
        b.d.xfrc_applied[pid1, :3] = f_h
        b.d.xfrc_applied[pid1, 3:] = np.cross(xO - b.d.xipos[pid1], f_h)
        b.d.xfrc_applied[pid2, 3:] = m2_h
        b.d.qfrc_applied[:] = 0
        b.d.qfrc_applied[b.va_crank] = tau
        mujoco.mj_step(m, b.d)
    return {k: np.array(v) for k, v in log.items()}


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'd25'
    S = setup(name, payload=(0.05, 0.05), mode='real', timestep=2e-4)
    ENC = S['D'].get('encoder', True)          # True: motor encoder; int: counts/turn on the crank
    out = dict(N=S['N'], encoder=ENC)
    for ff in (False, True):
        L = tracking(S, speed=2.0, feedforward=ff, quantise=ENC)
        np.savez_compressed(os.path.join(HERE, 'results', f'tracking_{name}_{"ff" if ff else "pd"}.npz'), **{k: v for k, v in L.items() if k != 'label'},
                            label=np.array(L['label']))
        out['ff' if ff else 'pd'] = dict(ep_max_mm=float(L['ep'].max() * 1e3), ep_rms_mm=float(np.sqrt((L['ep'] ** 2).mean()) * 1e3),
                                          eR_max_deg=float(np.degrees(L['eR'].max())), tau_max=float(np.abs(L['tau']).max()))
        print(out, flush=True)
    S = setup(name, payload=(0.0, 0.0), mode='real', timestep=2e-4)
    L = phri(S, quantise=ENC)
    e = L['w_est'][:, :3] - L['f_true']
    out['phri'] = dict(force_err_max_N=float(np.abs(e).max()), force_err_rms_N=float(np.sqrt((e ** 2).mean())),
                       disp_max_mm=float(np.abs(L['dp']).max() * 1e3), tau_max=float(np.abs(L['tau']).max()))
    print(out, flush=True)
    np.savez_compressed(os.path.join(HERE, 'results', f'phri_{name}.npz'), **L)
    json.dump(out, open(os.path.join(HERE, 'results', f'closed_loop_{name}.json'), 'w'), indent=1)
