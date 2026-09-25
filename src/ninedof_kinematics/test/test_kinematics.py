import os

import numpy as np
import pytest

from ninedof_kinematics.kinematics import (
    NineDofKinematics, UnreachablePose, rot_x, rot_xyz, rot_y, xyz_from_rot)

GEOMETRY = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description',
                        'config', 'geometry.yaml')

# Pose measured on the exported SolidWorks assembly (model_6v3.STL): centre of
# the central sphere, platform orientations fitted on the ball joints A_i, and
# the actuator displacements read from the height of the balls B_i.
MEASURED_P = [-0.000184, -0.017543, 0.150456]
MEASURED_Q1 = [[0.981671, -0.132925, -0.136573],
               [0.115053, 0.984637, -0.131346],
               [0.151935, 0.113226, 0.981884]]
MEASURED_Q2 = [[0.955397, -0.243797, 0.166671],
               [0.293829, 0.841423, -0.45351],
               [-0.029676, 0.482255, 0.875528]]
MEASURED_Q = [0.010516, 0.005981, 0.001667, -0.002192, -0.004652,
              0.001669, 0.012577, -0.012929, -0.008713]


@pytest.fixture(scope='module')
def kin():
    return NineDofKinematics.from_yaml(GEOMETRY)


def measured_pose():
    return np.r_[MEASURED_P, xyz_from_rot(np.array(MEASURED_Q1)),
                 xyz_from_rot(np.array(MEASURED_Q2))]


def test_home_is_zero(kin):
    np.testing.assert_allclose(kin.inverse(kin.home), 0.0, atol=1e-3)


def test_ik_matches_cad_assembly(kin):
    q = kin.inverse(measured_pose())
    np.testing.assert_allclose(q, MEASURED_Q, atol=5e-4)


def test_fk_recovers_cad_pose(kin):
    x = kin.forward(MEASURED_Q)
    np.testing.assert_allclose(x, measured_pose(), atol=5e-3)


@pytest.mark.parametrize('seed', range(5))
def test_fk_inverts_ik(kin, seed):
    rng = np.random.default_rng(seed)
    x = kin.home + np.r_[rng.uniform(-0.01, 0.01, 3), rng.uniform(-0.2, 0.2, 6)]
    x_fk = kin.forward(kin.inverse(x))
    np.testing.assert_allclose(x_fk, x, atol=1e-8)


def test_jacobians_match_finite_differences(kin):
    x = measured_pose()
    q = kin.inverse(x)
    J, K = kin.jacobians(x, q)
    # Move along a Cartesian twist c_dot = [v, w1, w2] and compare q_dot.
    c_dot = np.array([0.01, -0.02, 0.015, 0.3, -0.2, 0.1, -0.1, 0.25, 0.2])
    dt = 1e-6
    p, Q1, Q2 = kin.split(x)

    def skew_exp(w):
        return np.eye(3) + np.cross(np.eye(3), w * dt)  # I + [w]x dt

    x2 = np.r_[p + c_dot[:3] * dt,
               xyz_from_rot(skew_exp(c_dot[3:6]) @ Q1),
               xyz_from_rot(skew_exp(c_dot[6:9]) @ Q2)]
    q_dot_fd = (kin.inverse(x2) - q) / dt
    np.testing.assert_allclose(np.linalg.solve(K, J @ c_dot), q_dot_fd, rtol=1e-3, atol=1e-6)


def test_unreachable_pose_raises(kin):
    x = kin.home.copy()
    x[0] += 0.2  # far outside the base: distal links cannot reach
    with pytest.raises(UnreachablePose):
        kin.inverse(x)


def test_rotation_round_trip():
    angles = np.array([0.3, -0.4, 1.1])
    np.testing.assert_allclose(xyz_from_rot(rot_xyz(angles)), angles)


def test_passive_joints_point_distal_links_at_platform(kin):
    x = measured_pose()
    js = kin.joint_positions(x)
    q = kin.inverse(x)
    B = kin.base_points(q)
    A = kin.platform_points(x)
    for i, leg in enumerate(kin.names):
        tip = B[i] + rot_x(js[f'{leg}_lower_x_joint']) @ rot_y(js[f'{leg}_lower_y_joint']) \
            @ np.array([0, 0, kin.l])
        np.testing.assert_allclose(tip, A[i], atol=1e-9)
