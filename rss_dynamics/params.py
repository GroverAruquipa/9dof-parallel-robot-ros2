"""Physical parameters of the 5RSS-S-4RSS prototype.

Materials (as the prototype is to be built):
    platforms, joints, cranks, crank pulleys : PA12 printed by powder-bed fusion (SLS/MJF)
    distal links                             : pultruded carbon-fibre tube
    shafts                                   : steel
Motors: Matsushita MPX-40C4WA of the 3_unlimited_rotation robot, one GT2 belt
stage per leg (motor pulley 16T, crank pulley 16*N teeth).

Every value marked ESTIMATE is not measured; the analysis shows how much each
one matters (see feasibility.py).

Two products:
    lumped_inertia(...)  -> the parameters of the document's model (rigid bodies
                            1 and 2, cranks; distal link split m_b/2 + m_b/2)
    parts(...)           -> the same robot as separate parts, for MuJoCo
"""
import os
import numpy as np
import trimesh

from geometry import MESH_DIR, MM

RHO = dict(pa12=950.0,        # kg/m^3, sintered PA12 (0.93-1.01 depending on process)
           carbon=1550.0,     # pultruded CF/epoxy
           steel=7850.0, alu=2700.0)

MOTOR = dict(
    name='Matsushita MPX-40C4WA',
    V=24.0, no_load_rpm=3580.0, no_load_A=0.30, cpr=2000,
    # torque constant: the datasheet is self-contradictory (see README):
    #   back-EMF bound   Kt = Ke <= 24 V / 375 rad/s = 0.064 N m/A
    #   slope of the four lab test points          = 0.12 N m/A
    Kt_low=0.064, Kt_high=0.12,
    I_cont=2.0,   # A, ESTIMATE (thermal, brushed 50 mm motor, no datasheet value)
    I_peak=5.0,   # A, driver limit used on the 3_unlimited robot
    J_rotor=1.0e-5,   # kg m^2, ESTIMATE (50 mm brushed DC motor, typ. 0.5-3e-5)
    tau_coulomb=0.015,  # N m at the motor, ESTIMATE (no-load current 0.3 A -> 0.019-0.036 N m at 3580 rpm)
    b_visc=2.7e-5,      # N m s/rad at the motor, ESTIMATE
)

GT2_PITCH = 2.0 * MM


def pulley(teeth, width=9 * MM, rho=RHO['pa12']):
    r = teeth * GT2_PITCH / (2 * np.pi)
    m = rho * np.pi * r ** 2 * width
    return dict(r=r, m=m, J=0.5 * m * r ** 2)


def platform_mesh(k):
    m = trimesh.load(os.path.join(MESH_DIR, f'platform_{k}.stl'))
    m.density = RHO['pa12']
    return dict(mass=m.mass, com=m.center_mass.copy(), I=m.moment_inertia.copy(), mesh=m)


def rod(l, od=6 * MM, idd=4 * MM, end_mass=1.2e-3):
    """Carbon tube + two printed rod ends (ESTIMATE 1.2 g each) at the joint centres."""
    Lt = l - 12 * MM                       # the rod ends take 6 mm at each ball
    m_tube = RHO['carbon'] * np.pi / 4 * (od ** 2 - idd ** 2) * Lt
    return dict(m_tube=m_tube, m_end=end_mass, m=m_tube + 2 * end_mass, od=od, idd=idd, Lt=Lt)


def crank(d, width=10 * MM, thick=8 * MM, hub=8 * MM, ball=0.5e-3):
    """PA12 crank arm from -hub to d+hub along its axis, a printed ball (0.5 g) at the tip,
    a steel shaft 6 x 50 mm on the pivot."""
    L = d + 2 * hub
    m_arm = RHO['pa12'] * width * thick * L
    x_arm = d / 2                                   # centre of the arm along d
    J_arm = m_arm * (L ** 2 + width ** 2) / 12 + m_arm * x_arm ** 2
    m_sh = RHO['steel'] * np.pi * (3 * MM) ** 2 * 50 * MM
    J_sh = 0.5 * m_sh * (3 * MM) ** 2
    return dict(m_arm=m_arm, x_arm=x_arm, J_arm=J_arm, m_ball=ball, m_shaft=m_sh, J_shaft=J_sh,
                L=L, width=width, thick=thick, hub=hub)


def build(d, l, N, payload=(0.0, 0.0), J_rotor=None, rod_kw=None):
    """All physical data for crank d, link l, belt ratio N.

    payload: extra point mass (kg) on platform 1 and 2, placed 25 mm above O_p
    on the axis of each half (a stand-in for the gripped object / test weights).
    """
    J_rotor = MOTOR['J_rotor'] if J_rotor is None else J_rotor
    P = dict(d=d, l=l, N=N)
    P['rod'] = rod(l, **(rod_kw or {}))
    P['crank'] = crank(d)
    P['pulley_crank'] = pulley(round(16 * N))
    P['pulley_motor'] = pulley(16, rho=RHO['alu'], width=7 * MM)
    P['J_rotor'] = J_rotor
    P['platform'] = [platform_mesh(1), platform_mesh(2)]
    P['payload'] = [dict(m=payload[k], pos=np.array([(-1) ** (k + 1) * 0.018, 0, 0.025]))
                    for k in range(2)]
    return P


def reflected_inertia(P):
    return P['N'] ** 2 * (P['J_rotor'] + P['pulley_motor']['J'])


def composite(masses, coms, Is):
    """Mass, centre of mass and inertia about it of a set of rigid parts."""
    masses = np.asarray(masses, float); coms = np.asarray(coms, float)
    Mk = masses.sum()
    sk = (masses[:, None] * coms).sum(0) / Mk
    I = np.zeros((3, 3))
    for mi, ci, Ii in zip(masses, coms, Is):
        r = ci - sk
        I += Ii + mi * (r @ r * np.eye(3) - np.outer(r, r))       # Steiner
    return Mk, sk, I


def platform_body(P, k):
    """Platform k as printed (mesh, PA12) plus its payload, without the links."""
    pm = P['platform'][k - 1]
    pl = P['payload'][k - 1]
    parts = [(pm['mass'], pm['com'], pm['I'])]
    if pl['m'] > 0:
        parts.append((pl['m'], pl['pos'], np.zeros((3, 3))))
    return composite(*zip(*parts))


def lumped_inertia(P, a, rod_model='doc'):
    """Parameters of the document's model (eq. datos and the crank data).

    a: (9,3) platform points in the body frames; body k gets legs with body index k.
    rod_model = 'doc'  : the distal link of mass m_b is split m_b/2 at each ball
                         centre, as in the document.
    rod_model = '3mass': thin-rod equivalent, m_end + m_tube/6 at each ball and
                         2 m_tube/3 at the middle (exact mass, centre of mass and
                         transverse inertia of the tube; extension of the document).
    """
    from geometry import cad_legs
    body = cad_legs()[0]
    rd = P['rod']
    # tube of length Lt centred between the balls: end masses m_t Lt^2/(6 l^2) keep its
    # transverse inertia m_t Lt^2/12 (Lt = l gives the classic m/6 - 2m/3 - m/6)
    me_t = rd['m_tube'] * rd['Lt'] ** 2 / (6 * P['l'] ** 2)
    mb = rd['m'] if rod_model == 'doc' else 2 * (rd['m_end'] + me_t)
    m_mid = 0.0 if rod_model == 'doc' else rd['m_tube'] - 2 * me_t
    M, s0, IC0 = [], [], []
    for k in (1, 2):
        pm = P['platform'][k - 1]
        masses = [pm['mass']]
        coms = [pm['com']]
        Is = [pm['I']]            # about own centre of mass
        for i in np.flatnonzero(body == k):
            masses.append(mb / 2); coms.append(a[i]); Is.append(np.zeros((3, 3)))
        pl = P['payload'][k - 1]
        if pl['m'] > 0:
            masses.append(pl['m']); coms.append(pl['pos']); Is.append(np.zeros((3, 3)))
        Mk, sk, I = composite(masses, coms, Is)
        M.append(Mk); s0.append(sk); IC0.append(I)
    cr = P['crank']
    pc = P['pulley_crank']
    m_tip = cr['m_ball'] + mb / 2
    mm = cr['m_arm'] + m_tip                       # shaft and pulley are on the axis
    rc = (cr['m_arm'] * cr['x_arm'] + m_tip * P['d']) / mm
    Im = cr['J_arm'] + m_tip * P['d'] ** 2 + cr['J_shaft'] + pc['J'] + reflected_inertia(P)
    n = len(a)
    return dict(M=np.array(M), s0=np.array(s0), IC0=np.array(IC0),
                Im=np.full(n, Im), mm=np.full(n, mm), rc=np.full(n, rc), m_mid=np.full(n, m_mid))


def summary(P, a):
    li = lumped_inertia(P, a)
    return {
        'platform_mass_g': [1e3 * P['platform'][0]['mass'], 1e3 * P['platform'][1]['mass']],
        'body_mass_g (with m_b/2 per leg and payload)': (1e3 * li['M']).tolist(),
        'rod_mass_g': 1e3 * P['rod']['m'],
        'crank_side_mass_g': 1e3 * li['mm'][0],
        'Im_total_kgm2': li['Im'][0],
        'Im_reflected_rotor_kgm2': reflected_inertia(P),
        'Im_crank_parts_kgm2': li['Im'][0] - reflected_inertia(P),
        'crank_pulley_teeth': round(16 * P['N']),
    }
