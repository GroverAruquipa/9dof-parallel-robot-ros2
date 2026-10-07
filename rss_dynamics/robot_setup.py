"""Build the chosen robot (model + MuJoCo) from a design file."""
import json
import os
import numpy as np
import mujoco

from design import make_robot, NAMES
from params import build as build_params, lumped_inertia
from mjcf import build as build_mjcf
from mjbridge import Bridge

HERE = os.path.dirname(os.path.abspath(__file__))


def load_design(name='d25'):
    D = json.load(open(os.path.join(HERE, 'results', f'design_{name}.json')))
    x = np.array([D['x'][k] for k in NAMES])
    return D, x


def setup(name='d25', N=None, payload=(0.0, 0.0), rod_model='doc', mode='real', motors=None,
          timestep=2e-4, J_rotor=None):
    D, x = load_design(name)
    d = D['d']
    N = D.get('N', 3.0) if N is None else N
    r = make_robot(x, d)
    P = build_params(d, x[4], N, payload=payload, J_rotor=J_rotor)
    r.set_inertia(lumped_inertia(P, r.a, rod_model))
    if motors is None and mode == 'real':
        mf = os.path.join(HERE, 'results', f'motors_{name}.json')
        if os.path.exists(mf):
            motors = [{k: np.array(v) if isinstance(v, list) else v for k, v in mo.items()}
                      for mo in json.load(open(mf))]
    xml, th0 = build_mjcf(r, P, x[5], mode=mode, motors=motors, timestep=timestep)
    m = mujoco.MjModel.from_xml_string(xml)
    b = Bridge(r, m, th0, x[5])
    return dict(D=D, x=x, d=d, N=N, robot=r, P=P, model=m, bridge=b, xml=xml, th0=th0, h=x[5])
