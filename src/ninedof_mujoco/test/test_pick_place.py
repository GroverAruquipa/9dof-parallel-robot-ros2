import os

import pytest

from ninedof_kinematics import pick_place_plan as P
from ninedof_kinematics.kinematics import NineDofKinematics

DESC = os.path.join(os.path.dirname(__file__), '..', '..', 'ninedof_description')


@pytest.fixture(scope='module')
def setup():
    kin = NineDofKinematics.from_yaml(os.path.join(DESC, 'config', 'geometry.yaml'))
    pp = P.load_pick_place(os.path.join(DESC, 'config', 'pick_place.yaml'))
    return kin, pp


def test_sequence_inside_workspace(setup):
    kin, pp = setup
    report = P.check(kin, P.plan(kin, pp), 0.02, 0.5)
    assert [r[0] for r in report] == ['start', 'approach', 'open', 'descend', 'close', 'lift',
                                      'rotate', 'place', 'release', 'retreat']
    assert all(ok for _, ok, _, _ in report), report


def test_block_yaws_follow_the_jaw(setup):
    _, pp = setup
    g = pp['gripper']
    turn = pp['block']['pick_yaw'] - pp['target']['yaw']
    assert turn == pytest.approx(g['yaw_place'] - g['yaw_pick'], abs=1e-3)


def test_offline_simulation_succeeds(setup):
    mujoco = pytest.importorskip('mujoco')
    from ninedof_mujoco import sim
    kin, pp = setup
    model = mujoco.MjModel.from_xml_path(os.path.join(DESC, 'mujoco', 'pick_place_scene.xml'))
    data = mujoco.MjData(model)
    pos, quat = sim.run_pick_place(model, data, kin, P.plan(kin, pp))
    ok, checks = P.evaluate_block(pos, quat, pp)
    assert ok, checks
