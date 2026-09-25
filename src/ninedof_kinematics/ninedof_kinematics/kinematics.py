"""Kinematics of the 9-DoF 5PSS-S-4PSS parallel robot.

Notation follows Aruquipa, Lambert and Gosselin, "Kinematic Analysis and Design
of a Novel 9-DoF Parallel Robot with Grasping Capabilities":

    b_i = b_io + q_i e_i                  (actuated prismatic joint, e_i = +Z)
    n_i = p - b_io + Q a_i                (Q = Q1 for legs on platform 1, Q2 otherwise)
    q_i = n_i^T e_i - sqrt((n_i^T e_i)^2 - (n_i^T n_i - l_i^2))
    J c_dot = K q_dot,  c_dot = [p_dot, w1, w2]

The pose is x = [px, py, pz, alpha1, alpha2, alpha3, beta1, beta2, beta3] with
Q1 = Qx(alpha1) Qy(alpha2) Qz(alpha3) and Q2 = Qx(beta1) Qy(beta2) Qz(beta3).
Only numpy is required, so the module can be used without ROS.
"""

import numpy as np
import yaml

E_Z = np.array([0.0, 0.0, 1.0])


class UnreachablePose(ValueError):
    """Raised when the inverse kinematics has no real solution."""


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_xyz(angles):
    """Q = Qx(a1) Qy(a2) Qz(a3)."""
    return rot_x(angles[0]) @ rot_y(angles[1]) @ rot_z(angles[2])


def xyz_from_rot(R):
    """Inverse of rot_xyz (valid away from a2 = +/- 90 deg)."""
    a2 = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    a1 = np.arctan2(-R[1, 2], R[2, 2])
    a3 = np.arctan2(-R[0, 1], R[0, 0])
    return np.array([a1, a2, a3])


class NineDofKinematics:

    def __init__(self, geometry):
        g = geometry['ninedof'] if 'ninedof' in geometry else geometry
        legs = g['legs']
        self.names = [leg['name'] for leg in legs]
        self.platform = np.array([leg['platform'] for leg in legs])
        self.b0 = np.array([[leg['base_x'], leg['base_y'], leg['base_z0']] for leg in legs])
        self.a = np.array([leg['a'] for leg in legs], dtype=float)
        self.e = np.tile(E_Z, (len(legs), 1))
        self.l = float(g['distal_length'])
        self.stroke = float(g['actuator_stroke'])
        self.home = np.array([0.0, 0.0, float(g['home_z']), 0, 0, 0, 0, 0, 0], dtype=float)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls(yaml.safe_load(f))

    # ------------------------------------------------------------------ poses
    @staticmethod
    def split(x):
        x = np.asarray(x, dtype=float)
        return x[:3], rot_xyz(x[3:6]), rot_xyz(x[6:9])

    def platform_points(self, x):
        """World position of every A_i."""
        p, Q1, Q2 = self.split(x)
        A = np.empty_like(self.a)
        for i, a in enumerate(self.a):
            A[i] = p + (Q1 if self.platform[i] == 1 else Q2) @ a
        return A

    def base_points(self, q):
        """World position of every B_i."""
        return self.b0 + np.asarray(q, dtype=float)[:, None] * self.e

    # ------------------------------------------------------------ kinematics
    def inverse(self, x, check_stroke=False):
        """Actuator displacements q (9,) for the pose x (Eq. 18, lower branch)."""
        A = self.platform_points(x)
        n = A - self.b0
        ne = np.einsum('ij,ij->i', n, self.e)
        disc = ne ** 2 - (np.einsum('ij,ij->i', n, n) - self.l ** 2)
        if np.any(disc < 0.0):
            bad = [self.names[i] for i in np.flatnonzero(disc < 0.0)]
            raise UnreachablePose(f'no real IK solution for {", ".join(bad)}')
        q = ne - np.sqrt(disc)
        if check_stroke and np.any(np.abs(q) > self.stroke):
            bad = [self.names[i] for i in np.flatnonzero(np.abs(q) > self.stroke)]
            raise UnreachablePose(f'actuator stroke exceeded on {", ".join(bad)}')
        return q

    def constraints(self, x, q):
        """f_i = |A_i - B_i|^2 - l^2 (zero when the robot is assembled)."""
        m = self.platform_points(x) - self.base_points(q)
        return np.einsum('ij,ij->i', m, m) - self.l ** 2

    def jacobians(self, x, q):
        """Matrices J and K of Eqs. (13)-(15): J c_dot = K q_dot."""
        p, Q1, Q2 = self.split(x)
        m = self.platform_points(x) - self.base_points(q)
        J = np.zeros((len(self.a), 9))
        K = np.zeros((len(self.a), len(self.a)))
        for i in range(len(self.a)):
            Q = Q1 if self.platform[i] == 1 else Q2
            col = 3 if self.platform[i] == 1 else 6
            J[i, :3] = m[i]
            J[i, col:col + 3] = np.cross(Q @ self.a[i], m[i])
            K[i, i] = m[i] @ self.e[i]
        return J, K

    def condition_number(self, x):
        q = self.inverse(x)
        J, K = self.jacobians(x, q)
        return np.linalg.cond(np.linalg.solve(K, J))

    def forward(self, q, x0=None, tol=1e-12, max_iter=50):
        """Pose x from actuator displacements q (Gauss-Newton, as in the paper)."""
        q = np.asarray(q, dtype=float)
        x = np.array(self.home if x0 is None else x0, dtype=float)
        h = 1e-7
        for _ in range(max_iter):
            f = self.constraints(x, q)
            if np.max(np.abs(f)) < tol:
                return x
            D = np.empty((len(f), 9))
            for k in range(9):
                dx = np.zeros(9)
                dx[k] = h
                D[:, k] = (self.constraints(x + dx, q) - f) / h
            x = x - np.linalg.lstsq(D, f, rcond=None)[0]
        raise RuntimeError('forward kinematics did not converge')

    # ------------------------------------------------- URDF passive joints
    def joint_positions(self, x, q=None):
        """All URDF joint positions (actuated + passive) for the pose x.

        Returns a dict {joint_name: position} matching ninedof.urdf.xacro.
        """
        if q is None:
            q = self.inverse(x)
        p, Q1, Q2 = self.split(x)
        js = {}
        for name, v in zip(('base_x', 'base_y', 'base_z'), p):
            js[f'platform_{name}_joint'] = v
        for name, v in zip(('roll', 'pitch', 'yaw'), x[3:6]):
            js[f'platform_1_{name}_joint'] = v
        for name, v in zip(('x', 'y', 'z'), xyz_from_rot(Q1.T @ Q2)):
            js[f'central_sphere_{name}_joint'] = v
        u = self.platform_points(x) - self.base_points(q)
        u /= np.linalg.norm(u, axis=1)[:, None]
        for i, leg in enumerate(self.names):
            js[f'{leg}_actuator_joint'] = q[i]
            # Rx(a) Ry(b) e_z = u  ->  b = asin(ux), a = atan2(-uy, uz)
            js[f'{leg}_lower_x_joint'] = np.arctan2(-u[i, 1], u[i, 2])
            js[f'{leg}_lower_y_joint'] = np.arcsin(np.clip(u[i, 0], -1.0, 1.0))
        return js
