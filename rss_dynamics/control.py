"""Controller side: what the real robot would run, fed only by the encoders.

  Encoders    crank angle quantised to 2 pi / (cpr N), velocity by backward
              difference + first-order filter
  Pose        forward kinematics from the nine crank angles (model.fk)
  Control     joint PD (tracking) or Cartesian impedance on c = [p, rot1, rot2],
              plus model feedforward / compensation (gravity, inertia, friction)
  Observer    generalised-momentum observer in actuator space, gives the
              external torques, mapped to the wrench on the platforms with G^-T
"""
import numpy as np

from model import expm_so3
from params import MOTOR


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    t = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if t < 1e-8:
        return 0.5 * w
    return t / (2 * np.sin(t)) * w


class Encoders:
    """quantise: True -> motor encoder (cpr * N counts per crank turn);
    an int -> counts per crank turn of an encoder on the crank (e.g. 16384 for a
    14-bit magnetic encoder); False -> ideal."""
    def __init__(self, N, dt, fc=60.0, quantise=True):
        if quantise is True:
            self.q = 2 * np.pi / (MOTOR['cpr'] * N)
        elif quantise:
            self.q = 2 * np.pi / int(quantise)
        else:
            self.q = 0.0
        self.dt = dt
        self.alpha = 1 - np.exp(-2 * np.pi * fc * dt)
        self.prev = None
        self.vel = None

    def read(self, th_true):
        th = np.round(th_true / self.q) * self.q if self.q > 0 else th_true.copy()
        if self.prev is None:
            self.prev = th; self.vel = np.zeros_like(th)
        raw = (th - self.prev) / self.dt
        self.vel = self.vel + self.alpha * (raw - self.vel)
        self.prev = th
        return th, self.vel.copy()


def friction_model(thd, N, scale=1.0, vs=0.3):
    """Crank-side friction of the motor through the belt (smooth Coulomb + viscous)."""
    return scale * (N * MOTOR['tau_coulomb'] * np.tanh(thd / vs) + N ** 2 * MOTOR['b_visc'] * thd)


class MomentumObserver:
    """r -> tau_ext (actuator space), first order with gain K_O."""

    def __init__(self, KO=60.0):
        self.KO = KO
        self.integ = None
        self.r = None
        self.Ma_prev = None

    def update(self, Ma, ca, ga, thd, tau_cmd, tau_fric_hat, dt):
        p = Ma @ thd
        if self.integ is None:
            self.integ = p.copy(); self.r = np.zeros_like(p); self.Ma_prev = Ma
        Mdot_thd = (Ma - self.Ma_prev) / dt @ thd
        self.Ma_prev = Ma
        self.integ = self.integ + dt * (tau_cmd - tau_fric_hat - ca + ga + Mdot_thd + self.r)
        self.r = self.KO * (p - self.integ)
        return self.r.copy()


class PoseEstimator:
    def __init__(self, robot, p, Q1, Q2):
        self.r = robot
        self.p, self.Q1, self.Q2 = p.copy(), Q1.copy(), Q2.copy()

    def update(self, th):
        self.p, self.Q1, self.Q2 = self.r.fk(th, self.p, self.Q1, self.Q2)
        return self.p, self.Q1, self.Q2


def pose_error(p_d, Q1_d, Q2_d, p, Q1, Q2):
    return np.r_[p_d - p, log_so3(Q1_d @ Q1.T), log_so3(Q2_d @ Q2.T)]
