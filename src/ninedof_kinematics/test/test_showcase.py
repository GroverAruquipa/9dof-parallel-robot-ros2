import math
import os

import numpy as np
import pytest

from ninedof_kinematics.kinematics import NineDofKinematics
from ninedof_kinematics.showcase import Showcase, params, pose_from_params

GEOMETRY = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description',
                        'config', 'geometry.yaml')


@pytest.fixture(scope='module')
def setup():
    kin = NineDofKinematics.from_yaml(GEOMETRY)
    return kin, Showcase(kin.home)


def test_trajectory_inside_workspace(setup):
    kin, show = setup
    assert show.check(kin, dt=0.02) < 0.9 * kin.stroke


def test_starts_and_ends_at_home(setup):
    kin, show = setup
    np.testing.assert_allclose(show.sample(0.0)[3], kin.home, atol=1e-12)
    np.testing.assert_allclose(show.sample(show.duration)[3], kin.home, atol=1e-12)


def test_actuator_motion_is_smooth(setup):
    kin, show = setup
    dt = 0.005
    q = np.array([kin.inverse(show.sample(t)[3]) for t in np.arange(0, show.duration, dt)])
    acc = np.diff(q, 2, axis=0) / dt ** 2
    assert np.abs(acc).max() < 0.2   # m/s^2: no jumps between pieces


def test_covers_every_family_of_motion(setup):
    _, show = setup
    labels = {p[3] for p in show.pieces}
    for name in ('Translation X', 'Translation Y', 'Translation Z', 'Rotation about X',
                 'Rotation about Y', 'Rotation about Z', 'Relative rotation about X',
                 'Relative rotation about Y', 'Relative rotation about Z',
                 'Circle trajectory', 'Cone trajectory'):
        assert name in labels


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_relative_rotations_turn_the_platforms_against_each_other(setup, axis):
    kin, _ = setup
    a = 0.2
    v = params(**{('rel_x', 'rel_y', 'rel_z')[axis]: a})
    _, Q1, Q2 = kin.split(pose_from_params(kin.home, v))
    rel = Q1.T @ Q2          # rotation of platform 2 seen from platform 1
    angle = math.acos((np.trace(rel) - 1) / 2)
    assert angle == pytest.approx(2 * a)
    assert abs(Q1 @ Q2.T - np.eye(3)).max() > 0.1   # not a common rotation
    e = np.zeros(3)
    e[axis] = 1.0
    np.testing.assert_allclose(Q1 @ e, e, atol=1e-12)   # both turn about the same axis
    np.testing.assert_allclose(Q2 @ e, e, atol=1e-12)
