import os

import numpy as np
import pytest

from ninedof_kinematics.kinematics import NineDofKinematics
from ninedof_kinematics.learned_fk import (
    batch_inverse, default_model_path, LearnedForwardKinematics, poses_from_params,
    sample_workspace, WORKSPACE)
from ninedof_kinematics.showcase import pose_from_params

GEOMETRY = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description',
                        'config', 'geometry.yaml')


@pytest.fixture(scope='module')
def kin():
    return NineDofKinematics.from_yaml(GEOMETRY)


@pytest.fixture(scope='module')
def model():
    return LearnedForwardKinematics.load(default_model_path())


def same_pose(kin, a, b):
    pa, Q1a, Q2a = kin.split(a)
    pb, Q1b, Q2b = kin.split(b)
    return max(np.abs(pa - pb).max(), np.abs(Q1a - Q1b).max(), np.abs(Q2a - Q2b).max()) < 1e-6


def test_batch_helpers_match_the_scalar_kinematics(kin):
    v = np.random.default_rng(0).uniform(-WORKSPACE, WORKSPACE, size=(40, 9))
    x = poses_from_params(kin.home, v)
    np.testing.assert_allclose(x, [pose_from_params(kin.home, vi) for vi in v], atol=1e-12)
    q = batch_inverse(kin, x)
    ok = ~np.isnan(q).any(axis=1)
    np.testing.assert_allclose(q[ok], [kin.inverse(xi) for xi in x[ok]], atol=1e-12)


def test_samples_are_within_the_stroke(kin):
    x, q = sample_workspace(kin, 200, np.random.default_rng(1))
    assert x.shape == q.shape == (200, 9)
    assert np.abs(q).max() <= kin.stroke
    np.testing.assert_allclose(q[:20], [kin.inverse(xi) for xi in x[:20]], atol=1e-12)


def test_single_and_batched_evaluation_agree(model, kin):
    _, q = sample_workspace(kin, 10, np.random.default_rng(2))
    np.testing.assert_allclose(model(q), [model(qi) for qi in q], atol=1e-12)


def test_save_and_load_round_trip(model, tmp_path):
    path = tmp_path / 'model.npz'
    model.save(path)
    q = np.linspace(-0.01, 0.01, 9)
    np.testing.assert_allclose(LearnedForwardKinematics.load(path)(q), model(q))


def test_learned_guess_beats_the_home_pose_as_cold_start(model, kin):
    xs, qs = sample_workspace(kin, 300, np.random.default_rng(3))

    def success_rate(guesses):
        ok = 0
        for x, q, g in zip(xs, qs, guesses):
            try:
                ok += same_pose(kin, kin.forward(q, g), x)
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                pass
        return ok / len(xs)

    with np.errstate(all='ignore'):
        home = success_rate([kin.home] * len(xs))
        learned = success_rate(model(qs))
    assert learned > 0.75
    assert learned > home + 0.2


def test_forward_is_exact_from_the_learned_guess(model, kin):
    xs, qs = sample_workspace(kin, 20, np.random.default_rng(4))
    for x, q in zip(xs, qs):
        try:
            sol = model.forward(kin, q)
        except RuntimeError:
            continue
        np.testing.assert_allclose(kin.constraints(sol, q), 0.0, atol=1e-11)
