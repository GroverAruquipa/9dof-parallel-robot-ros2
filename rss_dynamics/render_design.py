"""Renders of a design in MuJoCo: iso, front and top at home, a pose with the gripper
open, and the same view of a reference design.   MUJOCO_GL=osmesa python3 render_design.py d35_core_R100 d35_core"""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, mujoco
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from robot_setup import setup, HERE
from trajectories import pose_and_rates

def cam(az, el, dist, look):
    c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = look; c.azimuth = az; c.elevation = el; c.distance = dist
    return c

def shot(S, v, c, W=900, H=900):
    m, b = S['model'], S['bridge']
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    p, Q1, Q2, cd, _ = pose_and_rates(S['h'], v, np.zeros(7), np.zeros(7))
    qpos, qvel, *_ = b.full_state(p, Q1, Q2, cd)
    b.set(qpos, qvel); mujoco.mj_forward(m, b.d)
    r = mujoco.Renderer(m, H, W); r.update_scene(b.d, c); img = r.render(); r.close()
    return img

name, ref = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
S = setup(name, mode='real', payload=(0.0, 0.0), motors=[])
h = S['h']; D = np.pi / 180
look = [0, 0, h * 0.45]
iso = lambda: cam(135, -20, 0.45, look)
top = lambda: cam(90, -62, 0.42, [0, 0, 0.02])
home = np.r_[0, 0, 0, 0, 0, 0, 8 * D]
imgs = [('Isométrica, home', shot(S, home, iso())),
        ('Frontal (x-z), home', shot(S, home, cam(90, -4, 0.45, look))),
        ('Pinza abierta 30°, giro x 15°, x +20 mm', shot(S, np.r_[0.02, 0, 0, 15 * D, 0, 0, 15 * D], iso())),
        ('Vista desde arriba: este diseño', shot(S, home, top()))]
if ref:
    R = setup(ref, mode='real', payload=(0.0, 0.0), motors=[])
    imgs += [(f'Diseño anterior ({ref}), misma cámara', shot(R, home, iso())),
             (f'Diseño anterior desde arriba, misma escala', shot(R, home, top()))]
n = len(imgs); cols = 3; rows = int(np.ceil(n / cols))
fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
for a in ax.ravel(): a.axis('off')
for a, (t, im) in zip(ax.ravel(), imgs):
    a.imshow(im); a.set_title(t, fontsize=13)
env = np.max(np.hypot(S['robot'].c[:, 0], S['robot'].c[:, 1])) + S['d']
fig.suptitle(f'{name}: manivela {S["d"]*1e3:.0f} mm, biela {S["x"][4]*1e3:.0f} mm, ejes a radio {np.hypot(*S["robot"].c[0,:2])*1e3:.0f} mm, '
             f'radio barrido {env*1e3:.0f} mm (sin motores)', fontsize=15)
fig.tight_layout(); out = os.path.join(HERE, 'figures', f'render_{name}.png'); fig.savefig(out, dpi=90); print(out)
