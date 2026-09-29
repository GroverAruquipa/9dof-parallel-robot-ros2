"""Learned initial guess for the forward kinematics (pure numpy).

The forward kinematics of a parallel robot has no closed form and several
solutions (assembly modes): Gauss-Newton converges to the one nearest to its
initial guess. Warm-starting it with the previous solution works while the
robot is tracked, but not at start-up or after the solver loses track. There,
a small multilayer perceptron trained on inverse-kinematics samples
(learning/train_fk.py) maps the actuator displacements q to an estimate of
the pose, and Gauss-Newton refines it to machine precision.

Only numpy is needed at run time; the weights are stored in an .npz file.
"""

import os

import numpy as np

from ninedof_kinematics.kinematics import rot_xyz, xyz_from_rot

DEG = np.pi / 180.0

# Operating workspace used for training and evaluation, as (x, y, z, roll,
# pitch, yaw, rel_x, rel_y, rel_z) half-ranges around home (see showcase.py
# for the parametrisation: both platforms turn by R, and against each other
# by +rel / -rel).
WORKSPACE = np.array([0.04, 0.04, 0.02, 25 * DEG, 25 * DEG, 60 * DEG,
                      20 * DEG, 20 * DEG, 40 * DEG])


def _batch_rot(axis, t):
    c, s = np.cos(t), np.sin(t)
    R = np.zeros(t.shape + (3, 3))
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    R[..., axis, axis] = 1.0
    R[..., i, i] = c
    R[..., j, j] = c
    sign = -1.0 if axis != 1 else 1.0
    R[..., i, j] = sign * s
    R[..., j, i] = -sign * s
    return R


def poses_from_params(home, v):
    """Batch version of showcase.pose_from_params: v (n, 9) -> x (n, 9)."""
    R = _batch_rot(2, v[:, 5]) @ _batch_rot(1, v[:, 4]) @ _batch_rot(0, v[:, 3])
    S = _batch_rot(2, v[:, 8]) @ _batch_rot(1, v[:, 7]) @ _batch_rot(0, v[:, 6])
    Q1, Q2 = R @ S, R @ np.swapaxes(S, 1, 2)
    return np.hstack([home[:3] + v[:, :3],
                      np.array([xyz_from_rot(Q) for Q in Q1]),
                      np.array([xyz_from_rot(Q) for Q in Q2])])


def batch_inverse(kin, x):
    """Inverse kinematics for a batch of poses; NaN where unreachable."""
    Q1 = np.array([rot_xyz(a) for a in x[:, 3:6]])
    Q2 = np.array([rot_xyz(b) for b in x[:, 6:9]])
    Q = np.where(kin.platform[None, :, None, None] == 1, Q1[:, None], Q2[:, None])
    A = x[:, None, :3] + np.einsum('nkij,kj->nki', Q, kin.a)
    n = A - kin.b0
    ne = n[..., 2]                                   # e = +Z for every leg
    disc = ne ** 2 - (np.einsum('nki,nki->nk', n, n) - kin.l ** 2)
    with np.errstate(invalid='ignore'):
        return ne - np.sqrt(disc)


def sample_workspace(kin, n, rng, workspace=WORKSPACE):
    """n poses drawn uniformly in the workspace box whose actuator
    displacements are within the stroke. Returns (x, q), each (n, 9)."""
    xs, qs, count = [], [], 0
    while count < n:
        v = rng.uniform(-workspace, workspace, size=(4 * n, 9))
        x = poses_from_params(kin.home, v)
        q = batch_inverse(kin, x)
        ok = np.all(np.abs(q) <= kin.stroke, axis=1)   # NaN compares False
        xs.append(x[ok])
        qs.append(q[ok])
        count += int(ok.sum())
    return np.vstack(xs)[:n], np.vstack(qs)[:n]


class LearnedForwardKinematics:
    """Multilayer perceptron q -> x, evaluated with numpy."""

    def __init__(self, weights, biases, q_scale, x_mean, x_scale):
        self.weights, self.biases = weights, biases
        self.q_scale, self.x_mean, self.x_scale = q_scale, x_mean, x_scale

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = default_model_path()
        d = np.load(path)
        n = int(d['n_layers'])
        return cls([d[f'W{i}'] for i in range(n)], [d[f'b{i}'] for i in range(n)],
                   float(d['q_scale']), d['x_mean'], d['x_scale'])

    def save(self, path):
        arrays = {f'W{i}': W for i, W in enumerate(self.weights)}
        arrays.update({f'b{i}': b for i, b in enumerate(self.biases)})
        np.savez(path, n_layers=len(self.weights), q_scale=self.q_scale,
                 x_mean=self.x_mean, x_scale=self.x_scale, **arrays)

    def __call__(self, q):
        """Pose estimate for q (9,) or a batch (n, 9)."""
        h = np.asarray(q, dtype=float) / self.q_scale
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ W.T + b)
        return (h @ self.weights[-1].T + self.biases[-1]) * self.x_scale + self.x_mean

    def forward(self, kin, q, **kwargs):
        """Exact forward kinematics: learned guess refined by Gauss-Newton."""
        return kin.forward(q, self(q), **kwargs)


def default_model_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        root = get_package_share_directory('ninedof_kinematics')
    except Exception:  # running from the source tree
        root = os.path.join(os.path.dirname(__file__), '..')
    return os.path.join(root, 'models', 'learned_fk.npz')
