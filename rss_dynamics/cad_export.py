"""Dimensions for SolidWorks: every point and axis of a design at home, in mm and deg.

Frame: origin on the base axis, z up, z = 0 the design reference plane;
x, y as in geometry.yaml of the grasping robot.  Home: O_p = (0, 0, h), both
platforms with Q = I (gripper closed, as in the CAD).

    python3 cad_export.py d35_core       -> results/cad_<name>.md, results/cad_<name>.csv
"""
import json
import os
import sys
import numpy as np

from design import make_robot, evaluate, workspace, NAMES, HERE, MM, DEG
from params import pulley


def main(name):
    D = json.load(open(os.path.join(HERE, 'results', f'design_{name}.json')))
    x = np.array([D['x'][k] for k in NAMES]); d = D['d']; l = x[4]; h = x[5]
    r = make_robot(x, d)
    det = evaluate(x, d, workspace(), details=True)
    sock = det['socket_axis']
    p0 = np.array([0, 0, h])
    th = r.ik(p0, np.eye(3), np.eye(3))
    kn = r.kin(p0, np.eye(3), np.eye(3), th)
    tips = r.c + kn['D']
    A = p0 + kn['Qa']
    N = D.get('N', 4.0)
    pc = pulley(round(16 * N))
    f = lambda v: ', '.join(f'{c:8.2f}' for c in v)
    rows = []
    L = [f'# Cotas para CAD: diseño `{name}`', '',
         f'Manivela d = {d / MM:.1f} mm (centro del eje a centro de rótula), biela l = {l / MM:.2f} mm '
         f'(centro a centro de rótulas), centro de la rótula central O_p a z = {h / MM:.2f} mm, {(h - r.c[0, 2]) / MM:.2f} mm sobre los ejes de manivela.',
         f'Correa GT2: polea motor 16T, polea de manivela {round(16 * N)}T (diámetro primitivo {2 * pc["r"] / MM:.1f} mm).', '',
         'Origen en el eje vertical de la base, z hacia arriba; z = 0 es el plano de referencia del diseño (los ejes de manivela están todos a la misma altura c_z). '
         'En home las dos plataformas tienen Q = I (pinza cerrada, como el CAD de graspability).', '',
         '| pata | plataforma | eje de manivela c_i (mm) | dirección del eje v_i | ángulo de manivela en home: dirección d_i/|d_i| | '
         'punta b_i (centro rótula) (mm) | anclaje A_i en home (mm) | a_i en la plataforma (mm) | eje del asiento de rótula de plataforma (marco de plataforma) |',
         '|---|---|---|---|---|---|---|---|---|']
    for i in range(r.n):
        dh = kn['D'][i] / d
        L.append(f'| {i + 1} | {r.body[i]} | ({f(r.c[i] / MM)}) | ({f(r.v[i])}) | ({f(dh)}) | ({f(tips[i] / MM)}) | '
                 f'({f(A[i] / MM)}) | ({f(r.a[i] / MM)}) | ({f(sock[i])}) |')
        rows.append([i + 1, r.body[i], *(r.c[i] / MM), *r.v[i], *dh, *(tips[i] / MM), *(A[i] / MM), *(r.a[i] / MM), *sock[i]])
    az = np.degrees(np.arctan2(r.v[:, 1], r.v[:, 0]))
    elev = np.degrees(np.arcsin(np.clip(kn['D'][:, 2] / d, -1, 1)))     # crank above the horizontal at home
    L += ['', '| pata | radio del eje (mm) | azimut del eje (deg) | elevación de la manivela en home (deg) | radio de la punta (mm) |', '|---|---|---|---|---|']
    for i in range(r.n):
        L.append(f'| {i + 1} | {np.hypot(*r.c[i, :2]) / MM:.2f} | {az[i]:.2f} | {elev[i]:.2f} | {np.hypot(*tips[i, :2]) / MM:.2f} |')
    env = np.max(np.hypot(r.c[:, 0], r.c[:, 1])) + d
    L += ['', f'Radio barrido por las manivelas: {env / MM:.1f} mm. Altura de los ejes: z = {r.c[0, 2] / MM:.2f} mm.',
          f'Recorrido máximo de manivela en el showcase: ±{np.degrees(det["worst"]["travel"]):.0f} deg; '
          f'desalineación del rod-end: {np.degrees(det["worst"]["mis"]):.0f} deg; cono de rótula de plataforma: {np.degrees(det["worst"]["cone"]):.0f} deg.']
    open(os.path.join(HERE, 'results', f'cad_{name}.md'), 'w').write('\n'.join(L) + '\n')
    hdr = ('leg,platform,cx,cy,cz,vx,vy,vz,dx,dy,dz,bx,by,bz,Ax,Ay,Az,ax,ay,az,sx,sy,sz')
    np.savetxt(os.path.join(HERE, 'results', f'cad_{name}.csv'), np.array(rows, float), delimiter=',', header=hdr, comments='', fmt='%.4f')
    print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'd35_core')
