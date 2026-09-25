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

    def __init__(self, model, data, kin, v_lin=0.02, v_ang=0.5, dt=0.01, pre_step=None):
        self.m, self.d, self.kin = model, data, kin
        self.pre_step = pre_step
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
            if self.pre_step is not None:
                self.pre_step(self.d)
            mujoco.mj_step(self.m, self.d)
            if on_step is not None:
                on_step()



class VirtualGrasp:
    """Grasp as a stiff spring-damper holding the object at its pose relative to
    the holder when switched on (plus gravity compensation; reaction on the
    holder). Same law and gains as the SimStatePublisher plugin."""

    K_LIN, D_LIN, K_ROT, D_ROT = 4.5, 0.3, 2.7e-4, 1.8e-5
    PERIOD = 0.01   # the plugin runs at the ros2_control rate; wrench held in between

    def __init__(self, model, obj='block', holder='platform_1'):
        self.m = model
        self.o, self.h = model.body(obj).id, model.body(holder).id
        self.on = False
        self.last = -np.inf
        self.rel_pos, self.rel_quat = np.zeros(3), np.array([1.0, 0, 0, 0])

    def set(self, data, on):
        self.on = on
        if on:
            q_inv = np.zeros(4)
            mujoco.mju_negQuat(q_inv, data.xquat[self.h])
            mujoco.mju_rotVecQuat(self.rel_pos, data.xpos[self.o] - data.xpos[self.h], q_inv)
            mujoco.mju_mulQuat(self.rel_quat, q_inv, data.xquat[self.o])

    def apply(self, data):
        if data.time - self.last < self.PERIOD - 1e-9:
            return   # zero-order hold, like the plugin
        self.last = data.time
        data.xfrc_applied[[self.o, self.h]] = 0.0
        if not self.on:
            return
        m, o, h = self.m, self.o, self.h
        p_t = np.zeros(3)
        mujoco.mju_rotVecQuat(p_t, self.rel_pos, data.xquat[h])
        p_t += data.xpos[h]
        q_t = np.zeros(4)
        mujoco.mju_mulQuat(q_t, data.xquat[h], self.rel_quat)
        vo, vh = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, o, vo, 0)
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, h, vh, 0)
        v_t = vh[3:] + np.cross(vh[:3], p_t - data.xpos[h])
        force = self.K_LIN * (p_t - data.xpos[o]) + self.D_LIN * (v_t - vo[3:])
        force[2] -= m.body_subtreemass[o] * m.opt.gravity[2]
        q_inv, q_err, err = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(q_inv, data.xquat[o])
        mujoco.mju_mulQuat(q_err, q_t, q_inv)
        mujoco.mju_quat2Vel(err, q_err, 1.0)
        torque = self.K_ROT * err + self.D_ROT * (vh[:3] - vo[:3])
        data.xfrc_applied[o, :3] = force
        data.xfrc_applied[o, 3:] = torque + np.cross(data.xpos[o] - data.xipos[o], force)
        data.xfrc_applied[h, :3] = -force
        data.xfrc_applied[h, 3:] = -(torque + np.cross(data.xpos[o] - data.xipos[h], force))


def run_pick_place(model, data, kin, waypoints, v_lin=0.02, v_ang=0.5, on_step=None):
    """Offline pick and place (no ROS): same sequence, interpolation and grasp
    switching as pick_place_demo + CartesianPoseController + SimStatePublisher."""
    set_robot_pose(model, data, kin, waypoints[0].pose)
    grasp = VirtualGrasp(model)
    ctrl = ControllerReplica(model, data, kin, v_lin, v_ang, pre_step=grasp.apply)
    prev = waypoints[0].pose
    ctrl.hold(prev, waypoints[0].hold, on_step)
    for wp in waypoints[1:]:
        if wp.name == 'release':
            grasp.set(data, False)
        ctrl.move(prev, wp.pose, on_step)
        ctrl.hold(wp.pose, wp.hold, on_step)
        if wp.name == 'close':
            grasp.set(data, True)
        prev = wp.pose
    return block_pose(model, data)
