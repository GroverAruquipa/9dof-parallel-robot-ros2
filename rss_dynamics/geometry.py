"""Geometry of the 5RSS-S-4RSS robot built on the CAD of the 9-DoF grasping robot.

The two platforms (anchors a_i, central spherical joint) are taken unchanged
from src/ninedof_description/config/geometry.yaml (SolidWorks model_6v3).  The
nine vertical prismatic actuators are replaced by nine horizontal-axis cranks
placed on the same base angles psi_i, so that at home each crank tip moves
vertically, as the slider it replaces.

Design variables (same for the nine legs):
    d      crank length ("eslabon bajo")
    l      distal (carbon-fibre) link length
    Rt     radius of the crank tips at home
    gamma  angle of the crank (seen from above) from the outward radial direction
    beta   elevation of the crank at home (theta_home)
    h      height of the centre of the spherical joint O_p above the crank axes
    dpsi   angular offset of the crank tips from the CAD base angles (tilts the
           legs tangentially, which is what holds x, y and the rotation about z)
"""
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM_YAML = os.path.join(HERE, '..', 'src', 'ninedof_description', 'config', 'geometry.yaml')
MESH_DIR = os.path.join(HERE, '..', 'src', 'ninedof_description', 'meshes')
MM, DEG = 1e-3, np.pi / 180
# scale of the CAD platforms (1 = the grasping robot as built); env PSCALE
PSCALE = float(os.environ.get('PSCALE', '1'))


_CAD = None


def cad_legs():
    global _CAD
    if _CAD is not None:
        return _CAD
    g = yaml.safe_load(open(GEOM_YAML))['ninedof']
    body = np.array([L['platform'] for L in g['legs']])
    a = np.array([L['a'] for L in g['legs']], float)
    psi = np.array([np.arctan2(L['base_y'], L['base_x']) for L in g['legs']])
    _CAD = (body, a, psi, g)
    return _CAD


def build_geometry(d, l, Rt, gamma, beta, h, dpsi=0.0, dpsi2=None, alt=0.0, **_):
    """Geometry dict for model.RSS9 and the home pose.

    dpsi, dpsi2: tangential offset of the tips of the legs of platform 1 and 2
    alt: extra offset +alt / -alt alternating along the legs of each platform
    (pairs of legs leaning in opposite senses, as in a Stewart platform)."""
    body, a, psi, _ = cad_legs()
    a = a * PSCALE
    n = len(psi)
    dpsi2 = dpsi if dpsi2 is None else dpsi2
    sgn = np.zeros(n)
    for k in (1, 2):
        idx = np.flatnonzero(body == k)
        idx = idx[np.argsort(psi[idx] % (2 * np.pi))]
        sgn[idx] = (-1.0) ** np.arange(len(idx))
    off = np.where(body == 1, dpsi, dpsi2) + alt * sgn
    ez = np.array([0, 0, 1.0])
    dirs = np.c_[np.cos(psi + off + gamma), np.sin(psi + off + gamma), np.zeros(n)]
    tips = np.c_[Rt * np.cos(psi + off), Rt * np.sin(psi + off), np.zeros(n)]
    crank_home = np.cos(beta) * dirs + np.sin(beta) * ez
    c = tips - d * crank_home
    v = np.cross(dirs, ez)                       # theta > 0 lifts the tip
    u = dirs                                     # theta = 0: crank horizontal
    geom = dict(d=np.full(n, d), l=np.full(n, l), body=body, a=a, c=c, v=v, u=u,
                branch=np.ones(n))
    return geom


def home_pose(h):
    return np.array([0, 0, h]), np.eye(3), np.eye(3)


def fix_branch(robot, beta, h):
    """Pick, leg by leg, the IK branch whose home angle is beta."""
    p, Q1, Q2 = home_pose(h)
    best = np.ones(robot.n)
    tp = robot.ik(p, Q1, Q2, branch=np.ones(robot.n))
    tm = robot.ik(p, Q1, Q2, branch=-np.ones(robot.n))
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    best[np.abs(wrap(tm - beta)) < np.abs(wrap(tp - beta))] = -1
    robot.branch = best
    return robot


def pose_from_params(h, v):
    """Graspability showcase convention: p = home + xyz, Q1 = R Rx(jaw), Q2 = R Rx(-jaw),
    R = Rz(yaw) Ry(pitch) Rx(roll)."""
    from model import rot_x, rot_y, rot_z
    x, y, z, roll, pitch, yaw, jaw = v
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    return np.array([x, y, h + z]), R @ rot_x(jaw), R @ rot_x(-jaw)
