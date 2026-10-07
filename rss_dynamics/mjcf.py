"""MuJoCo model of the 5RSS-S-4RSS robot.

Body tree (every body frame is world-aligned at the home pose, so qpos = 0 is
the home configuration and the nine loops close exactly there):

    world
      base (static: plate, bearing blocks, motors, belts)
      crank_i   hinge th_i about v_i at the pivot c_i
        link_i  ball joint at the crank tip, site link_end_i at the other ball
      platform_1  free joint at O_p
        platform_2  ball joint at O_p (the central spherical joint)
    equality: connect link_i (end) <-> platform_k (anchor a_i)

mode = 'ideal'  : exactly the hypotheses of the document: links with m_b/2 at
                  each ball (the platform half as a point mass on the link at
                  the platform ball, dynamically the same as adding it to the
                  platform), crank inertia Im about v_i, no friction.
mode = 'real'   : the parts as built: PA12 platform meshes (MuJoCo computes
                  their inertia), carbon tubes with distributed mass and the
                  rod ends at the balls, crank arm/shaft/pulley as geoms, rotor
                  and motor pulley as armature N^2 J, motor friction reflected
                  through the belt (frictionloss, damping).
"""
import os
import numpy as np

from geometry import MESH_DIR, home_pose
from params import MOTOR, lumped_inertia, reflected_inertia, platform_body

f = lambda v: ' '.join(f'{x:.9g}' for x in np.atleast_1d(v))


def quat_of(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    if w < 1e-8:
        x = np.sqrt(max(0, (1 + R[0, 0] - R[1, 1] - R[2, 2]))) / 2
        y = np.sign(R[0, 1] + R[1, 0]) * np.sqrt(max(0, (1 - R[0, 0] + R[1, 1] - R[2, 2]))) / 2
        z = np.sign(R[0, 2] + R[2, 0]) * np.sqrt(max(0, (1 - R[0, 0] - R[1, 1] + R[2, 2]))) / 2
        return np.array([0.0, x, y, z])
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def frame_z_to(u):
    """Rotation whose z axis is u."""
    u = u / np.linalg.norm(u)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, u); x /= np.linalg.norm(x)
    return np.c_[x, np.cross(u, x), u]


def build(robot, P, h, mode='real', motors=None, payload_geoms=True, timestep=2e-4,
          servo=None, visual_extras=True):
    """MJCF string.  robot: model.RSS9 (with branch fixed), P: params.build(...).

    servo: None -> torque actuators (motor) on each crank, gear 1 (crank torque);
           (kp, kv) -> position servos instead.
    motors: list of dicts {pos, axis, len, r} for visual motors and belts.
    """
    ideal = mode == 'ideal'
    p0, Q1, Q2 = home_pose(h)
    th0 = robot.ik(p0, Q1, Q2)
    kn = robot.kin(p0, Q1, Q2, th0)
    tips = robot.c + kn['D']
    anchors = p0 + kn['Qa']
    li = lumped_inertia(P, robot.a)
    N = P['N']
    X = []
    w = X.append
    w(f'<mujoco model="rss9_{mode}">')
    w(f'  <compiler angle="radian" meshdir="{MESH_DIR}" autolimits="true" boundmass="1e-9" boundinertia="1e-15"/>')
    w(f'  <option timestep="{timestep}" integrator="RK4" gravity="{f(robot.g)}" iterations="100" '
      f'tolerance="1e-12" solver="Newton" cone="elliptic" jacobian="dense"/>')
    w('  <visual><global offwidth="1600" offheight="1200"/><quality shadowsize="4096"/>'
      '<headlight ambient=".35 .35 .35" diffuse=".5 .5 .5"/></visual>')
    w('  <default>')
    w('    <joint damping="0" armature="0" frictionloss="0"/>')
    w('    <geom contype="0" conaffinity="0"/>')
    w('    <equality solref="0.0004 1" solimp="0.99995 0.99999 0.0001 0.5 2"/>')
    w('  </default>')
    w('  <asset>')
    w('    <texture name="grid" type="2d" builtin="checker" rgb1=".94 .94 .94" rgb2=".87 .87 .88" width="512" height="512"/>')
    w('    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>')
    w('    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".80 .84 .90" width="512" height="512"/>')
    w('    <material name="pa12" rgba=".93 .93 .90 1"/>')
    w('    <material name="p1" rgba=".82 .27 .22 1"/>')
    w('    <material name="p2" rgba=".22 .42 .82 1"/>')
    w('    <material name="carbon" rgba=".10 .10 .11 1" specular=".6" shininess=".8"/>')
    w('    <material name="alu" rgba=".75 .76 .78 1" specular=".7"/>')
    w('    <material name="motor" rgba=".22 .22 .25 1" specular=".5"/>')
    w('    <mesh name="platform_1" file="platform_1.stl"/>')
    w('    <mesh name="platform_2" file="platform_2.stl"/>')
    w('  </asset>')
    w('  <worldbody>')
    zfloor = -0.20 if motors else -0.08
    w(f'    <geom type="plane" size="0.6 0.6 0.01" pos="0 0 {zfloor}" material="grid"/>')
    w('    <light pos="0.3 -0.4 0.8" dir="-0.3 0.4 -0.8" diffuse=".7 .7 .7" castshadow="true"/>')
    w('    <light pos="-0.4 0.3 0.6" dir="0.4 -0.3 -0.6" diffuse=".3 .3 .3" castshadow="false"/>')
    w(f'    <camera name="iso" pos="0.42 -0.42 {h + 0.20:.3f}" xyaxes="0.7071 0.7071 0 -0.33 0.33 0.884"/>')
    w(f'    <camera name="front" pos="0 -0.55 {h * 0.6:.3f}" xyaxes="1 0 0 0 0 1"/>')
    w(f'    <camera name="top" pos="0 0 0.7" xyaxes="1 0 0 0 1 0"/>')
    if visual_extras:
        # base plate below the cranks and a bearing block per crank (visual)
        zb = robot.c[:, 2].min() - 0.020
        rmax = np.linalg.norm(robot.c[:, :2], axis=1).max() + 0.03
        w(f'    <geom type="cylinder" pos="0 0 {zb - 0.004:.4f}" size="{rmax:.4f} 0.004" rgba=".85 .85 .86 1"/>')
        for i in range(robot.n):
            R = frame_z_to(robot.v[i])
            w(f'    <geom type="box" pos="{f(robot.c[i] - 0.012 * robot.v[i] - [0, 0, 0.0])}" quat="{f(quat_of(R))}" '
              f'size="0.009 0.009 0.004" material="pa12"/>')
            w(f'    <geom type="box" pos="{f(np.r_[robot.c[i, :2], (robot.c[i, 2] + zb) / 2] - 0.012 * robot.v[i])}" '
              f'size="0.004 0.004 {max(1e-3, (robot.c[i, 2] - zb) / 2):.4f}" material="pa12"/>')
        if motors:
            for i, mo in enumerate(motors):
                R = frame_z_to(mo['axis'])
                w(f'    <geom type="cylinder" pos="{f(mo["pos"])}" quat="{f(quat_of(R))}" '
                  f'size="{mo["r"]:.4f} {mo["len"] / 2:.4f}" material="motor"/>')
                # belt: two straight runs between the pulleys (pitch radii)
                pc = robot.c[i] + mo['belt_offset'] * robot.v[i]
                pm = mo['pulley_pos']
                e = pm - pc; L = np.linalg.norm(e); e /= L
                nrm = np.cross(robot.v[i], e)
                r1, r2 = P['pulley_crank']['r'], P['pulley_motor']['r']
                for s in (1, -1):
                    a0, a1 = pc + s * r1 * nrm, pm + s * r2 * nrm
                    Rb = frame_z_to(a1 - a0)
                    w(f'    <geom type="box" pos="{f((a0 + a1) / 2)}" quat="{f(quat_of(Rb))}" '
                      f'size="0.0006 0.003 {np.linalg.norm(a1 - a0) / 2:.4f}" rgba=".08 .08 .08 1"/>')
                Rm = frame_z_to(robot.v[i])
                w(f'    <geom type="cylinder" pos="{f(pm)}" quat="{f(quat_of(Rm))}" size="{r2:.4f} 0.0035" material="alu"/>')
    # ---------------------------------------------------------------- cranks
    cr = P['crank']
    for i in range(robot.n):
        k = i + 1
        ci, vi = robot.c[i], robot.v[i]
        dhat = kn['D'][i] / robot.d[i]
        R_arm = np.c_[dhat, np.cross(vi, dhat), vi]          # x along the crank, z along the axis
        w(f'    <body name="crank_{k}" pos="{f(ci)}">')
        arm = (f'armature="{reflected_inertia(P):.9g}" ' if not ideal else '')
        fr = '' if ideal else (f'frictionloss="{N * MOTOR["tau_coulomb"]:.6g}" '
                               f'damping="{N ** 2 * MOTOR["b_visc"]:.6g}" ')
        w(f'      <joint name="th{k}" type="hinge" axis="{f(vi)}" {arm}{fr}/>')
        if ideal:
            # mass mm at rc along d; inertia about the axis through the pivot = Im
            Iaxis_c = li['Im'][i] - li['mm'][i] * li['rc'][i] ** 2
            Rc = R_arm
            w(f'      <inertial pos="{f(li["rc"][i] * dhat)}" quat="{f(quat_of(Rc))}" mass="{li["mm"][i]:.9g}" '
              f'diaginertia="{f([Iaxis_c * 0.6, Iaxis_c * 0.6, Iaxis_c])}"/>')
        # arm (box), ball at the tip, shaft and pulley on the axis
        L = cr['L']
        dens = lambda m, vol: m / vol
        arm_vol = cr['width'] * cr['thick'] * L
        w(f'      <geom type="box" pos="{f((cr["x_arm"]) * dhat)}" quat="{f(quat_of(R_arm))}" '
          f'size="{L / 2:.5f} {cr["width"] / 2:.5f} {cr["thick"] / 2:.5f}" material="pa12" '
          f'{"mass=\"0\"" if ideal else f"density=\"{cr["m_arm"] / arm_vol:.3f}\""}/>')
        w(f'      <geom type="sphere" pos="{f(robot.d[i] * dhat)}" size="0.004" material="pa12" '
          f'mass="{0 if ideal else cr["m_ball"]:.9g}"/>')
        sh_from, sh_to = -0.030 * vi, 0.015 * vi
        w(f'      <geom type="cylinder" fromto="{f(sh_from)} {f(sh_to)}" size="0.003" material="alu" '
          f'mass="{0 if ideal else cr["m_shaft"]:.9g}"/>')
        pc = P['pulley_crank']
        w(f'      <geom type="cylinder" fromto="{f(-0.026 * vi)} {f(-0.017 * vi)}" size="{pc["r"]:.5f}" '
          f'material="pa12" mass="{0 if ideal else pc["m"]:.9g}"/>')
        # ---- distal link
        tip_rel = tips[i] - ci
        e = anchors[i] - tips[i]
        lk = np.linalg.norm(e)
        w(f'      <body name="link_{k}" pos="{f(tip_rel)}">')
        w(f'        <joint name="s{k}" type="ball"/>')
        rod = P['rod']
        if ideal:
            # m_b/2 as a point mass at the platform-side ball centre: it moves exactly
            # as the platform point a_i (the loop is closed there), so this is the
            # document's lumping, but it keeps MuJoCo's constraints well conditioned
            # (a massless link would make its soft constraints very soft)
            w(f'        <inertial pos="{f(e)}" mass="{P["rod"]["m"] / 2:.9g}" diaginertia="1e-13 1e-13 1e-13"/>')
            w(f'        <geom type="capsule" fromto="0 0 0 {f(e)}" size="{rod["od"] / 2:.4f}" material="carbon" mass="0"/>')
        else:
            # carbon tube between the two printed rod ends; the rod ends (m_end each)
            # are point masses at the two ball centres and move with the link
            u = e / lk
            m_t = rod['m_tube']
            ro, ri = rod['od'] / 2, rod['idd'] / 2
            Lt = rod['Lt']
            Ia = 0.5 * m_t * (ro ** 2 + ri ** 2)
            It = m_t * (3 * (ro ** 2 + ri ** 2) + Lt ** 2) / 12
            # end masses at both ball centres (Steiner about the link centre)
            me = rod['m_end']
            Itot = It + 2 * me * (lk / 2) ** 2
            mtot = m_t + 2 * me
            Rl = frame_z_to(u)
            w(f'        <inertial pos="{f(e / 2)}" quat="{f(quat_of(Rl))}" mass="{mtot:.9g}" '
              f'diaginertia="{f([Itot, Itot, Ia])}"/>')
            w(f'        <geom type="cylinder" fromto="{f(0.006 * u)} {f(e - 0.006 * u)}" size="{ro:.4f}" material="carbon" mass="0"/>')
            w(f'        <geom type="sphere" pos="0 0 0" size="0.0055" material="pa12" mass="0"/>')
            w(f'        <geom type="sphere" pos="{f(e)}" size="0.0055" material="pa12" mass="0"/>')
        w(f'        <site name="link_end_{k}" pos="{f(e)}" size="0.002"/>')
        w('      </body>')
        w('    </body>')
    # ------------------------------------------------------------- platforms
    w(f'    <body name="platform_1" pos="{f(p0)}">')
    w('      <freejoint name="platform_1"/>')

    def platform_inertial(k):
        # printed platform + payload (the links carry their own mass, also in
        # 'ideal', where m_b/2 sits on the link at the platform ball).  From the exact mass properties of the CAD mesh (trimesh): MuJoCo's
        # default mesh inertia uses the convex hull for this non-watertight mesh
        # and would add about 40 % of mass.
        Mk, sk, I = platform_body(P, k)
        ev, V = np.linalg.eigh(I)
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        return (f'      <inertial pos="{f(sk)}" quat="{f(quat_of(V))}" mass="{Mk:.9g}" '
                f'diaginertia="{f(ev)}"/>')

    w(platform_inertial(1))
    dens = P['platform'][0]['mass'] / P['platform'][0]['mesh'].volume
    w(f'      <geom type="mesh" mesh="platform_1" material="p1" mass="0"/>')
    pl = P['payload'][0]
    if pl['m'] > 0 and payload_geoms:
        w(f'      <geom type="sphere" pos="{f(pl["pos"])}" size="0.008" rgba=".95 .75 .15 1" mass="0"/>')
    w('      <site name="Op" pos="0 0 0" size="0.003"/>')
    w('      <body name="platform_2" pos="0 0 0">')
    w('        <joint name="central" type="ball"/>')
    w(platform_inertial(2).replace('      <', '        <'))
    dens = P['platform'][1]['mass'] / P['platform'][1]['mesh'].volume
    w(f'        <geom type="mesh" mesh="platform_2" material="p2" mass="0"/>')
    pl = P['payload'][1]
    if pl['m'] > 0 and payload_geoms:
        w(f'        <geom type="sphere" pos="{f(pl["pos"])}" size="0.008" rgba=".95 .75 .15 1" mass="0"/>')
    w('      </body>')
    w('    </body>')
    w('  </worldbody>')
    w('  <equality>')
    for i in range(robot.n):
        k = i + 1
        w(f'    <connect name="loop_{k}" body1="link_{k}" body2="platform_{robot.body[i]}" '
          f'anchor="{f(anchors[i] - tips[i])}"/>')
    w('  </equality>')
    w('  <contact><exclude body1="platform_1" body2="platform_2"/></contact>')
    w('  <actuator>')
    for i in range(robot.n):
        k = i + 1
        if servo is None:
            w(f'    <motor name="m{k}" joint="th{k}" gear="1"/>')
        else:
            w(f'    <position name="m{k}" joint="th{k}" kp="{servo[0]}" kv="{servo[1]}"/>')
    w('  </actuator>')
    w('</mujoco>')
    return '\n'.join(X), th0
