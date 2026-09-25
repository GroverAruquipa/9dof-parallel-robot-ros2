"""MuJoCo helpers for the 9-DoF parallel robot (no ROS).

* set_robot_pose(): writes an assembled configuration (actuators, ball joints,
  platforms) into qpos for a given pose, e.g. to build the start keyframe.
* ControllerReplica: same interpolation + IK as ninedof_controllers'
  CartesianPoseController, to run the demo offline without ROS.
"""

import numpy as np

import mujoco

from ninedof_kinematics.kinematics import xyz_from_rot
from ninedof_kinematics.pick_place_plan import segment


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).ravel())
    return q


def _rot_z_to(u):
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, u)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    a = axis / s
    t = np.arctan2(s, z @ u)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def robot_mount(model):
    """(R, t) of the robot base frame in the world (identity if not mounted)."""
    try:
        bid = model.body('robot_mount').id
    except KeyError:
        return np.eye(3), np.zeros(3)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.body_quat[bid])
    return R.reshape(3, 3), model.body_pos[bid].copy()


def set_robot_pose(model, data, kin, x):
    """Write the assembled robot configuration for pose x (robot frame) into
    data.qpos and data.ctrl. Returns the actuator displacements q."""
    q = kin.inverse(x)
    p, Q1, Q2 = kin.split(x)
    A = kin.platform_points(x)
    B = kin.base_points(q)
    Rm, tm = robot_mount(model)
    for i, name in enumerate(kin.names):
        data.qpos[model.jnt_qposadr[model.joint(f'{name}_actuator_joint').id]] = q[i]
        bid = model.body(f'{name}_distal').id
        R0 = np.zeros(9)
        mujoco.mju_quat2Mat(R0, model.body_quat[bid])
        adr = model.jnt_qposadr[model.joint(f'{name}_lower_joint').id]
        data.qpos[adr:adr + 4] = _quat(R0.reshape(3, 3).T @ _rot_z_to((A[i] - B[i]) / kin.l))
        data.ctrl[model.actuator(f'{name}_actuator_joint').id] = q[i]
    adr = model.jnt_qposadr[model.joint('platform_1_free').id]
    data.qpos[adr:adr + 3] = tm + Rm @ p
    data.qpos[adr + 3:adr + 7] = _quat(Rm @ Q1)
    adr = model.jnt_qposadr[model.joint('central_sphere_joint').id]
    data.qpos[adr:adr + 4] = _quat(Q1.T @ Q2)
    mujoco.mj_forward(model, data)
    return q


def measured_pose(model, data):
    """Pose of the platforms [p, alpha, beta] in the robot frame."""
    Rm, tm = robot_mount(model)
    b1, b2 = model.body('platform_1').id, model.body('platform_2').id
    Q1 = Rm.T @ data.xmat[b1].reshape(3, 3)
    Q2 = Rm.T @ data.xmat[b2].reshape(3, 3)
    return np.r_[Rm.T @ (data.xpos[b1] - tm), xyz_from_rot(Q1), xyz_from_rot(Q2)]


def block_pose(model, data, body='block'):
    bid = model.body(body).id
    return data.xpos[bid].copy(), data.xquat[bid].copy()


class ControllerReplica:
    """Offline twin of CartesianPoseController: interpolates the pose at
    v_lin / v_ang and sends the IK solution to the position servos every dt."""

    def __init__(self, model, data, kin, v_lin=0.02, v_ang=0.5, dt=0.01):
        self.m, self.d, self.kin = model, data, kin
        self.v_lin, self.v_ang, self.dt = v_lin, v_ang, dt
        self.ids = [model.actuator(f'{n}_actuator_joint').id for n in kin.names]

    def move(self, x0, x1, on_step=None):
        poses, _ = segment(x0, x1, self.v_lin, self.v_ang, self.dt)
        for x in poses:
            self.hold(x, self.dt, on_step)

    def hold(self, x, duration, on_step=None):
        self.d.ctrl[self.ids] = self.kin.inverse(x)
        n = max(1, int(round(duration / self.m.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            if on_step is not None:
                on_step()

