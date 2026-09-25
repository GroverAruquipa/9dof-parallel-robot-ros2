import os

import numpy as np
import pytest

from ninedof_kinematics.kinematics import NineDofKinematics
from ninedof_kinematics.showcase import Showcase

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
                 'Rotation about Y', 'Rotation about Z', 'Relative rotation (jaw)',
                 'Circle trajectory', 'Cone trajectory'):
        assert name in labels
