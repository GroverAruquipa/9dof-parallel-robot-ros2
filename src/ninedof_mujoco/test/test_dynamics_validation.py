"""The analytic inverse dynamics against MuJoCo (see validate_dynamics.py)."""

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from ninedof_mujoco import validate_dynamics as V  # noqa: E402


def test_model_matches_mujoco_rigid_inverse_dynamics():
    t, f_model, f_mujoco, residual = V.rigid_check(np.linspace(0.0, 2.0, 9))
    assert np.abs(f_model).max() > 5.0          # a demanding trajectory
    np.testing.assert_allclose(f_model, f_mujoco, atol=1e-4)
    assert residual.max() < 1e-6


def test_mujoco_assembly_matches_the_attachment_points():
    model, _, dyn = V.load()
    for i, name in enumerate(dyn.kin.names):
        anchor = model.equality(f'{name}_upper_joint').data[3:6]
        np.testing.assert_allclose(anchor, dyn.kin.a[i], atol=1e-9)


def test_computed_torque_tracks_the_trajectory():
    s = V.simulate(duration=1.0)
    assert np.abs(s['pose_err'][:, :3]).max() < 20e-6          # m
    assert np.abs(s['pose_err'][:, 3:]).max() < 0.05 * V.DEG   # rad
    assert np.abs(s['f'] - s['f_model']).max() < 0.2           # N
