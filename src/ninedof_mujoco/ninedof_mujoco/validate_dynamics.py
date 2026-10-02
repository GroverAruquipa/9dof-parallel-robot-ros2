"""Validation of the analytic inverse dynamics (ninedof_kinematics.dynamics)
against MuJoCo.

1. Rigid-body check. MuJoCo describes the robot as a tree (sliders, distal
   links on ball joints, platform 1 free, platform 2 on the central ball
   joint) plus 27 loop-closure rows. Along the trajectory the configurations
   are written with the inverse kinematics, the tree velocities and
   accelerations come from central differences, and the actuator forces f and
   the loop-closure forces lam solve
       M(qpos) qacc + c(qpos, qvel) = S^T f + G^T lam
   (M and c from MuJoCo, armature included, G the equality Jacobian). This is
   the exact rigid-body answer of MuJoCo's own multibody formulation; the
   residual of the least-squares solve measures the consistency of the motion.

2. Closed-loop simulation. The position servos are replaced by force motors
   driven by a computed-torque law built on the analytic model (measured
   platform state, 10 Hz task-space PD), and MuJoCo integrates the robot with
   its soft loop closures and joint damping. With an exact model the error
   dynamics are linear and the tracking error stays near zero; the same law
   with the placeholder parameters used before is the reference.
   A joint-space PD cannot be used here: near home the platforms have a weak
   direction (horizontal translation + rotation of platform 2 about Z, smallest
   singular value of K^-1 J ~ 1e-4) that the actuators barely observe.

    python3 -m ninedof_mujoco.validate_dynamics [--plot DIR]
"""

import argparse
import os

import numpy as np
import yaml

import mujoco

from ninedof_kinematics.dynamics import NineDofDynamics, harmonic_trajectory, platform_twist
from ninedof_kinematics.kinematics import NineDofKinematics, rot_xyz
from ninedof_mujoco.sim import measured_pose, set_robot_pose

DESC = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description')
DEG = np.pi / 180.0
# Nine incommensurate harmonics (pose coordinates [p, alpha, beta]): the
# platforms translate, rotate together and against each other. Peak values on
# 0-6 s: |q| = 19 mm of the 25 mm stroke, |f| = 14.6 N of the 20 N limit.
AMPLITUDE = np.array([0.012, -0.012, 0.012, 12 * DEG, -12 * DEG, 15 * DEG,
                      -12 * DEG, 12 * DEG, -15 * DEG])
FREQUENCY = np.array([1.3, 1.1, 1.7, 0.9, 1.5, 1.2, 1.4, 0.8, 1.6])


def load(model_file='ninedof.xml'):
    dyn = NineDofDynamics.from_yaml(os.path.join(DESC, 'config', 'geometry.yaml'),
                                    os.path.join(DESC, 'config', 'dynamics.yaml'))
    model = mujoco.MjModel.from_xml_path(os.path.join(DESC, 'mujoco', model_file))
    return model, mujoco.MjData(model), dyn


def trajectory(dyn, t):
    return harmonic_trajectory(dyn.kin.home, AMPLITUDE, FREQUENCY, t)


def actuated_dofs(model, kin):
    return np.array([model.jnt_dofadr[model.joint(f'{n}_actuator_joint').id]
                     for n in kin.names])


# ---------------------------------------------------------------- 1. rigid
def rigid_inverse_dynamics(model, data, kin, x_of_t, t, h=1e-3):
    """Actuator forces from MuJoCo's rigid multibody model at time t.

    Returns (f, relative residual of the least-squares solve)."""
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    qpos = []
    for s in (t - h, t, t + h):
        set_robot_pose(model, data, kin, x_of_t(s))
        qpos.append(data.qpos.copy())
    v_m, v_p = np.zeros(model.nv), np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, v_m, h, qpos[0], qpos[1])
    mujoco.mj_differentiatePos(model, v_p, h, qpos[1], qpos[2])
    data.qpos[:] = qpos[1]
    data.qvel[:] = (v_m + v_p) / 2
    qacc = (v_p - v_m) / h
    mujoco.mj_forward(model, data)
    tau = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, tau, qacc)
    tau += data.qfrc_bias
    eq = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    G = data.efc_J[:data.nefc * model.nv].reshape(data.nefc, model.nv)[eq]
    S = np.zeros((len(kin.names), model.nv))
    S[np.arange(len(kin.names)), actuated_dofs(model, kin)] = 1.0
    A = np.vstack((S, G)).T
    sol = np.linalg.lstsq(A, tau, rcond=None)[0]
    residual = np.linalg.norm(A @ sol - tau) / np.linalg.norm(tau)
    return sol[:len(kin.names)], residual


def rigid_check(times=np.linspace(0.0, 6.0, 241)):
    model, data, dyn = load()

    def x_of_t(t):
        return trajectory(dyn, t)[0]

    f_model, f_mj, res = [], [], []
    for t in times:
        # mj_mulM + qfrc_bias hold no passive forces: compare without damping.
        f_model.append(dyn.inverse_dynamics(*trajectory(dyn, t), damping=False)[0])
        f, r = rigid_inverse_dynamics(model, data, dyn.kin, x_of_t, t)
        f_mj.append(f)
        res.append(r)
    return np.asarray(times), np.array(f_model), np.array(f_mj), np.array(res)


# ------------------------------------------------------- 2. closed loop
def platform_state(model, data):
    """Measured pose x and twist c_dot = [p_dot, w1, w2] (world frame)."""
    cd = np.empty(9)
    v = np.zeros(6)
    for k, s in ((1, slice(3, 6)), (2, slice(6, 9))):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                 model.body(f'platform_{k}').id, v, 0)
        cd[s] = v[:3]
        if k == 1:
            cd[:3] = v[3:]
    return measured_pose(model, data), cd


def rotation_error(x_ref, x):
    """Small rotation vectors (world frame) taking the measured platforms to
    the reference ones."""
    e = []
    for s in (slice(3, 6), slice(6, 9)):
        R = rot_xyz(x_ref[s]) @ rot_xyz(x[s]).T
        e.append(0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]))
    return np.concatenate(e)


def simulate(dyn_ctrl=None, duration=6.0, bandwidth=10.0, stiff_loops=False,
             model_file='ninedof.xml'):
    """Computed-torque tracking of the trajectory with force motors.

    The motors apply f = ID(x, c_dot, c_ddot_ref + Kd (c_dot_ref - c_dot)
    + Kp e) with the measured platform state (exact in simulation); ID is the
    analytic model built with dyn_ctrl (default: the parameters of the
    simulated robot). With an exact model the closed loop is linear with the
    chosen bandwidth and the tracking error stays near zero. stiff_loops
    raises the impedance of the loop closures to MuJoCo's maximum (0.9999):
    the closures of the generated model (0.99) give a little under load."""
    model, data, dyn = load(model_file)
    if stiff_loops:
        model.eq_solimp[:, :2] = 0.9999
    ctrl_model = dyn if dyn_ctrl is None else dyn_ctrl
    kin = dyn.kin
    wn = 2 * np.pi * bandwidth
    kp, kd = wn ** 2, 2 * wn
    # Position servos -> force motors: force = ctrl, clipped to forcerange.
    model.actuator_gainprm[:, 0] = 1.0
    model.actuator_biasprm[:] = 0.0
    model.actuator_ctrllimited[:] = 0
    act = np.array([model.actuator(f'{n}_actuator_joint').id for n in kin.names])
    jnt = [model.joint(f'{n}_actuator_joint').id for n in kin.names]
    qadr = model.jnt_qposadr[jnt]
    mujoco.mj_resetData(model, data)
    set_robot_pose(model, data, kin, trajectory(dyn, 0.0)[0])
    data.qvel[:] = 0.0
    n = int(round(duration / model.opt.timestep))
    log = {k: [] for k in ('t', 'q_ref', 'q', 'f_model', 'f', 'pose_err', 'loop_err')}
    for _ in range(n):
        # mj_step1 updates the positions and velocities of the current state
        # (after mj_step, xpos/xmat would still be those of the previous step).
        mujoco.mj_step1(model, data)
        t = data.time
        x_ref, xd_ref, xdd_ref = trajectory(dyn, t)
        cd_ref, cdd_ref = platform_twist(x_ref, xd_ref, xdd_ref)
        x, cd = platform_state(model, data)
        e = np.r_[x_ref[:3] - x[:3], rotation_error(x_ref, x)]
        cdd = cdd_ref + kd * (cd_ref - cd) + kp * e
        data.ctrl[act] = ctrl_model.inverse_dynamics_twist(x, cd, cdd)[0]
        # Reference forces: the model of the simulated robot on the reference.
        f_ref, mo = dyn.inverse_dynamics_twist(x_ref, cd_ref, cdd_ref)
        log['t'].append(t)
        log['q_ref'].append(mo['q'])
        log['q'].append(data.qpos[qadr].copy())
        eq = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        log['loop_err'].append(np.abs(data.efc_pos[:data.nefc][eq]).max())
        mujoco.mj_step2(model, data)
        log['f_model'].append(f_ref)
        log['f'].append(data.actuator_force[act].copy())
        log['pose_err'].append(e)
    return {k: np.array(v) for k, v in log.items()}


def placeholder_model():
    """The model with the parameters used before this validation (masses from
    the bare PLA volumes, platform centres of mass on the axis, diagonal
    inertia): what a controller would use without the identified data."""
    with open(os.path.join(DESC, 'config', 'dynamics.yaml')) as f:
        d = yaml.safe_load(f)['ninedof_dynamics']
    d['mass'] = {'slider': 0.00174, 'distal_link': 0.00224,
                 'platform_1': 0.01275, 'platform_2': 0.0124}
    for k in (1, 2):
        m = d['mass'][f'platform_{k}']
        ixx = m * (3 * 0.035 ** 2 + 0.01 ** 2) / 12
        d['inertia'][f'platform_{k}'] = {'com': [0.0, 0.0, 0.005],
                                         'inertia': [ixx, ixx, m * 0.035 ** 2 / 2, 0, 0, 0]}
    d['inertia']['distal_link']['inertia'] = [d['mass']['distal_link'] * 0.12 ** 2 / 12] * 2 \
        + [1e-9, 0, 0, 0]
    kin = NineDofKinematics.from_yaml(os.path.join(DESC, 'config', 'geometry.yaml'))
    return NineDofDynamics(kin, d)


# ----------------------------------------------------------------- report
def summary(rigid, exact, stiff, placeholder):
    t, fm, fj, res = rigid
    err = fm - fj
    lines = ['1. Rigid-body check: model vs MuJoCo M qacc + c with the loop closures, '
             f'{len(t)} points on 0-{t[-1]:.0f} s',
             f'  max |f_model|            {np.abs(fm).max():9.4f} N',
             f'  max |f_model - f_mujoco| {np.abs(err).max():9.2e} N',
             f'  RMS  f_model - f_mujoco  {np.sqrt(np.mean(err ** 2)):9.2e} N',
             f'  max LSQ residual         {res.max():9.2e} (relative)',
             '', '2. Closed-loop simulation: computed torque with the model, 10 Hz bandwidth',
             '                         this model   + stiff loops   placeholder parameters']
    rows = (
        ('max position error S', lambda s: f'{1e6 * np.abs(s["pose_err"][:, :3]).max():8.2f} um'),
        ('max angle error', lambda s: f'{np.abs(s["pose_err"][:, 3:]).max() / DEG:8.4f} deg'),
        ('max actuator error', lambda s: f'{1e6 * np.abs(s["q"] - s["q_ref"]).max():8.2f} um'),
        ('RMS f - f_ref', lambda s: f'{np.sqrt(np.mean((s["f"] - s["f_model"]) ** 2)):8.4f} N'),
        ('max |f - f_ref|', lambda s: f'{np.abs(s["f"] - s["f_model"]).max():8.4f} N'),
        ('max loop closure gap', lambda s: f'{1e6 * s["loop_err"].max():8.2f} um'))
    for name, fn in rows:
        lines.append(f'  {name:22s} {fn(exact):>11s} {fn(stiff):>15s} {fn(placeholder):>16s}')
    return '\n'.join(lines)


def plot(rigid, exact, stiff, placeholder, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t, fm, fj, _ = rigid
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(exact['t'], exact['f'][:, i], color='0.7', lw=3.0,
                label='MuJoCo closed-loop simulation')
        ax.plot(exact['t'], exact['f_model'][:, i], color='tab:red', lw=1.0,
                label='Model')
        ax.plot(t, fm[:, i], color='tab:blue', lw=0.8, ls='--',
                label='Model without joint damping')
        ax.plot(t[::2], fj[::2, i], 'o', ms=3.0, mfc='none', color='tab:blue',
                label='MuJoCo rigid inverse dynamics (no damping)')
        ax.set_title(f'leg {i + 1} (platform {1 if i < 5 else 2})', fontsize=9)
        ax.grid(alpha=0.3)
        if i % 3 == 0:
            ax.set_ylabel('force [N]')
        if i >= 6:
            ax.set_xlabel('time [s]')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Actuator forces: analytic inverse dynamics vs MuJoCo')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dynamics_validation_forces.png'), dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].semilogy(t, np.maximum(np.abs(fm - fj).max(axis=1), 1e-12), '.', color='tab:blue')
    ax[0].set_ylabel('max |f_model - f_mujoco| [N]')
    ax[0].set_title('Rigid-body check', fontsize=10)
    for s, name, c in ((placeholder, 'placeholder parameters', 'tab:gray'),
                       (exact, 'this model', 'tab:red'),
                       (stiff, 'this model, stiff loops', 'tab:blue')):
        ax[1].semilogy(s['t'], 1e3 * np.linalg.norm(s['pose_err'][:, :3], axis=1),
                       color=c, lw=1, label=name)
        ax[2].semilogy(s['t'], np.abs(s['pose_err'][:, 3:]).max(axis=1) / DEG,
                       color=c, lw=1, label=name)
    ax[1].set_ylabel('|S - S_ref| [mm]')
    ax[2].set_ylabel('max angle error [deg]')
    ax[1].set_title('Computed-torque tracking', fontsize=10)
    ax[2].set_title('Computed-torque tracking', fontsize=10)
    for a in ax:
        a.set_xlabel('time [s]')
        a.grid(alpha=0.3, which='both')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dynamics_validation_errors.png'), dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--plot', metavar='DIR', help='write the figures to DIR')
    args = ap.parse_args()
    rigid = rigid_check()
    exact, stiff = simulate(), simulate(stiff_loops=True)
    placeholder = simulate(placeholder_model())
    print(summary(rigid, exact, stiff, placeholder))
    if args.plot:
        plot(rigid, exact, stiff, placeholder, args.plot)


if __name__ == '__main__':
    main()
