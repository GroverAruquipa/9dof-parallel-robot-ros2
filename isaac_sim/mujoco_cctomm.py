"""MuJoCo twin of the USD model (same geometry_cctomm.yaml, masses and servos) and the reference run.

Builds an MJCF with the same bodies (platform 1 free, platform 2 on a ball joint at p0, 9 legs closed with
connect constraints), drives the nine actuators along reference_trajectory.npz and writes
result_mujoco.npz with the same fields as run_validation.py writes for Isaac Sim (t, p1, R1, p2, R2, q, f).

Usage: python3 mujoco_cctomm.py [--kp 5e4] [--no-gravity] [--dt 0.0005] [-o result_mujoco.npz]
"""
import argparse
import os

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def f3(x):
    return ' '.join(f'{float(c):.7g}' for c in x)


def inertial(b):
    I = np.array(b['inertia']); w = np.maximum(np.linalg.eigvalsh(I), 0)
    I = I + np.eye(3) * max(0.0, 1e-8 - w.min())
    full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    return f'<inertial pos="{f3(b["com"])}" mass="{b["mass"]:.7g}" fullinertia="{f3(full)}"/>'


def mjcf(geo, dr, gravity=True, dt=0.0005):
    B = geo['bodies']; lo, hi = geo['actuator_limits']; p0 = np.array(geo['p0'])
    legs, acts, eqs = [], [], []
    for i, l in enumerate(geo['legs'], 1):
        k = l['cad_instance']; b = np.array(l['b0']); a = np.array(l['a0'])
        legs.append(f'''
      <body name="carriage_{k}" pos="{f3(b)}">
        <joint name="q{i}" type="slide" axis="0 0 1" range="{lo} {hi}" armature="{dr['armature']}"/>
        {inertial(B['carriage_' + k])}
        <geom type="box" size=".006 .006 .02" pos="0 0 -.022" rgba=".8 .8 .8 1"/>
        <body name="leg_{k}">
          <joint name="{l['name']}_lower" type="ball" damping="1e-5"/>
          {inertial(B['leg_' + k])}
          <geom type="capsule" fromto="0 0 0 {f3(a - b)}" size=".0025" rgba=".27 .27 .29 1"/>
        </body>
      </body>''')
        acts.append(f'<position name="q{i}" joint="q{i}" kp="{dr["stiffness"]}" kv="{dr["damping"]}" '
                    f'forcerange="{-dr["max_force"]} {dr["max_force"]}"/>')
        plat = f'plat{l["platform"]}'
        eqs.append(f'<connect body1="leg_{k}" body2="{plat}" anchor="{f3(a - b)}"/>')
    return f'''<mujoco model="ninedof_cctomm">
  <option timestep="{dt}" integrator="implicitfast" gravity="0 0 {-9.81 if gravity else 0}"/>
  <default><geom contype="0" conaffinity="0"/><equality solref="0.001 1" solimp="0.99 0.999 0.0001"/></default>
  <worldbody>
    <body name="base">{''.join(legs)}
    </body>
    <body name="plat1" pos="{f3(p0)}">
      <freejoint/>
      {inertial(B['plat1'])}
      <geom type="sphere" size=".004" rgba=".8 .25 .2 1"/>
      <body name="plat2">
        <joint name="central" type="ball" damping="1e-5"/>
        {inertial(B['plat2'])}
        <geom type="sphere" size=".003" rgba=".2 .4 .8 1"/>
      </body>
    </body>
  </worldbody>
  <equality>
    {chr(10).join('    ' + e for e in eqs).strip()}
  </equality>
  <actuator>
    {chr(10).join('    ' + x for x in acts).strip()}
  </actuator>
</mujoco>'''


def mat_of_quat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); return R.reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kp', type=float); ap.add_argument('--no-gravity', action='store_true')
    ap.add_argument('--dt', type=float, default=0.0005); ap.add_argument('-o', '--out', default=os.path.join(HERE, 'results', 'mujoco_gravity.npz'))
    a = ap.parse_args()
    geo = yaml.safe_load(open(os.path.join(HERE, 'geometry_cctomm.yaml')))
    dr = dict(geo['drive'])
    if a.kp:                                   # stiffer servo, critical damping with the same armature
        dr['stiffness'] = a.kp; dr['damping'] = 2 * (a.kp * dr['armature']) ** 0.5; dr['max_force'] = 200.0
    xml = mjcf(geo, dr, gravity=not a.no_gravity, dt=a.dt)
    open(os.path.join(HERE, 'ninedof_cctomm_mujoco.xml'), 'w').write(xml)
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    ref = np.load(os.path.join(HERE, 'reference_trajectory.npz'))
    p0 = np.array(geo['p0'])
    sub = int(round((ref['t'][1] - ref['t'][0]) / m.opt.timestep))
    i1, i2 = m.body('plat1').id, m.body('plat2').id
    qadr = [m.jnt_qposadr[m.joint(f'q{i}').id] for i in range(1, 10)]
    for _ in range(int(0.5 / m.opt.timestep)):                                    # 0.5 s settling at the reference pose
        d.ctrl[:] = ref['q'][0]; mujoco.mj_step(m, d)
    out = {k: [] for k in ('p1', 'R1', 'p2', 'R2', 'q', 'f')}
    for j in range(len(ref['t'])):
        d.ctrl[:] = ref['q'][j]
        for _ in range(sub):
            mujoco.mj_step(m, d)
        R1 = mat_of_quat(d.xquat[i1]); R2 = mat_of_quat(d.xquat[i2])
        out['p1'].append(d.xpos[i1].copy()); out['R1'].append(R1)
        out['p2'].append(d.xpos[i2] + R2 @ np.zeros(3)); out['R2'].append(R2)
        out['q'].append(d.qpos[qadr].copy()); out['f'].append(d.actuator_force.copy())
    res = {k: np.array(v) for k, v in out.items()}
    np.savez_compressed(a.out, t=ref['t'], **res)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
