"""Videos of the MuJoCo simulations.

  videos/tracking_<name>.mp4  showcase at 2x speed, 50 g per platform, soft joint PD
                              + model feedforward; panels: crank torques applied in
                              MuJoCo vs the model's prediction, tracking error with
                              and without the model
  videos/phri_<name>.mp4      impedance mode; the 'human' force (red) and the force
                              estimated from the motor torques with the model (green)

    MUJOCO_GL=osmesa python3 animate.py d25
"""
import os
import sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np
import mujoco
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from robot_setup import setup, HERE
import plotstyle as ps

W, H = 1600, 900
RW, RH = 900, 900


def renderer_for(m):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, RW)
    m.vis.global_.offheight = max(m.vis.global_.offheight, RH)
    return mujoco.Renderer(m, RH, RW)


def cam(m, h):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = [0, 0, h * 0.42]
    c.distance = 0.78
    c.azimuth = 135
    c.elevation = -24
    return c


def add_arrow(scn, p0, p1, rgba, width=0.004):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def fig_to_array(fig):
    c = FigureCanvasAgg(fig)
    c.draw()
    return np.asarray(c.buffer_rgba())[..., :3].copy()


def tracking_video(name, speedup=1.0):
    S = setup(name, payload=(0.05, 0.05), mode='real')
    m, d = S['model'], S['bridge'].d
    ff = np.load(os.path.join(HERE, 'results', f'tracking_{name}_ff.npz'))
    pd = np.load(os.path.join(HERE, 'results', f'tracking_{name}_pd.npz'))
    rend = renderer_for(m)
    camera = cam(m, S['h'])
    out = os.path.join(HERE, 'videos', f'tracking_{name}.mp4')
    wr = imageio.get_writer(out, fps=30, codec='libx264', quality=6, macro_block_size=8)
    ps.style()
    t, tau, tff = ff['t'], ff['tau'], ff['tau_ff']
    legs = np.argsort(-np.abs(tff).max(0))[:3]
    kern = np.ones(25) / 25                      # 25 ms moving average (1 kHz log)
    tauf = np.column_stack([np.convolve(tau[:, j], kern, mode='same') for j in range(tau.shape[1])])
    labels = ff['label']
    for k, (tf, qpos) in enumerate(zip(ff['tfr'], ff['qpos'])):
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera)
        img = rend.render()
        i = min(np.searchsorted(t, tf), len(t) - 1)
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        axr = fig.add_axes([0, 0, RW / W, 1]); axr.imshow(img); axr.axis('off')
        axr.text(0.03, 0.96, f'{labels[i]}', transform=axr.transAxes, fontsize=16, weight='bold')
        axr.text(0.03, 0.92, f't = {tf:5.2f} s   (showcase x2, 50 g por plataforma)', transform=axr.transAxes, fontsize=11)
        axr.text(0.03, 0.03, 'MuJoCo "real": tubos de carbono, rotor, fricción, encoders de 14 bit en la manivela, N = 4\n'
                 'control: impedancia cartesiana + par del modelo dinámico (prealimentación)',
                 transform=axr.transAxes, fontsize=9.5, color='0.25')
        a1 = fig.add_axes([0.61, 0.56, 0.37, 0.38])
        w0 = max(0, tf - 4.0)
        sel = (t >= w0) & (t <= tf)
        for j, c in zip(legs, ps.C):
            a1.plot(t[sel], tau[sel, j] * 1e3, color=c, lw=0.6, alpha=0.25)
            a1.plot(t[sel], tauf[sel, j] * 1e3, color=c, lw=2.0, label=f'aplicado (filtrado 25 ms), manivela {j + 1}')
            a1.plot(t[sel], tff[sel, j] * 1e3, '--', color='k', lw=1.1)
        a1.plot([], [], '--', color='k', lw=1, label='predicho por el modelo')
        a1.set_xlim(w0, w0 + 4.0)
        lim = max(np.abs(tauf[:, legs]).max(), np.abs(tff[:, legs]).max()) * 1.4e3
        a1.set_ylim(-lim, lim)
        a1.set_title('Par en la manivela: total aplicado vs modelo', fontsize=11)
        a1.set_ylabel('mN·m'); a1.legend(fontsize=7.5, loc='upper left', ncol=2)
        a2 = fig.add_axes([0.61, 0.08, 0.37, 0.36])
        a2.plot(pd['t'][pd['t'] <= tf], pd['ep'][pd['t'] <= tf] * 1e3, color=ps.C[1], lw=1.4, label='solo PD')
        a2.plot(t[t <= tf], ff['ep'][t <= tf] * 1e3, color=ps.C[2], lw=1.4, label='PD + modelo dinámico')
        a2.set_xlim(0, t[-1]); a2.set_ylim(0, max(pd['ep'].max(), ff['ep'].max()) * 1.1e3)
        a2.set_title('Error de posición de O$_p$', fontsize=11); a2.set_xlabel('t (s)'); a2.set_ylabel('mm')
        a2.legend(fontsize=9, loc='upper right')
        wr.append_data(fig_to_array(fig)[:H, :W])
        plt.close(fig)
    wr.close()
    print('wrote', out)


def phri_video(name):
    S = setup(name, payload=(0.0, 0.0), mode='real')
    m, d = S['model'], S['bridge'].d
    L = np.load(os.path.join(HERE, 'results', f'phri_{name}.npz'))
    rend = renderer_for(m)
    camera = cam(m, S['h'])
    camera.azimuth = 120
    out = os.path.join(HERE, 'videos', f'phri_{name}.mp4')
    wr = imageio.get_writer(out, fps=30, codec='libx264', quality=6, macro_block_size=8)
    ps.style()
    t = L['t']
    pid1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_1')
    for tf, qpos in zip(L['tfr'], L['qpos']):
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera)
        i = min(np.searchsorted(t, tf), len(t) - 1)
        O = d.xpos[pid1].copy()
        sc = 0.012       # m per N
        f = L['f_true'][i]; fe = L['w_est'][i][:3]
        if np.linalg.norm(f) > 0.2:
            add_arrow(rend.scene, O - sc * f, O, [0.85, 0.1, 0.1, 1], 0.005)
        if np.linalg.norm(fe) > 0.2:
            add_arrow(rend.scene, O - sc * fe + [0, 0, 0.004], O + [0, 0, 0.004], [0.1, 0.7, 0.2, 0.9], 0.0035)
        img = rend.render()
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        axr = fig.add_axes([0, 0, RW / W, 1]); axr.imshow(img); axr.axis('off')
        axr.text(0.03, 0.96, 'Interacción física (pHRI): modo impedancia', transform=axr.transAxes, fontsize=16, weight='bold')
        axr.text(0.03, 0.92, f't = {tf:5.2f} s     rojo: fuerza del humano     verde: estimada con el modelo',
                 transform=axr.transAxes, fontsize=11)
        axr.text(0.03, 0.03, 'sin sensor de fuerza: observador de momento con M$_a$, c$_a$, g$_a$ del modelo\n'
                 'y solo los 9 encoders de 14 bit', transform=axr.transAxes, fontsize=9.5, color='0.25')
        a1 = fig.add_axes([0.61, 0.56, 0.37, 0.38])
        sel = t <= tf
        for j, (c, lab) in enumerate(zip(ps.C, 'xyz')):
            a1.plot(t[sel], L['f_true'][sel, j], color=c, lw=2, label=f'F{lab} humano')
            a1.plot(t[sel], L['w_est'][sel, j], '--', color=c, lw=1.2, label=f'F{lab} estimada')
        a1.set_xlim(0, t[-1]); a1.set_ylim(-8, 8)
        a1.set_title('Fuerza en O$_p$: aplicada vs estimada', fontsize=11); a1.set_ylabel('N')
        a1.legend(fontsize=7.5, ncol=3, loc='lower left')
        a2 = fig.add_axes([0.61, 0.08, 0.37, 0.36])
        for j, (c, lab) in enumerate(zip(ps.C, 'xyz')):
            a2.plot(t[sel], L['dp'][sel, j] * 1e3, color=c, lw=1.5, label=f'Δ{lab}')
        a2.plot(t[sel], np.degrees(np.linalg.norm(L['dR2'][sel] - L['dR1'][sel], axis=1)), color=ps.C[3], lw=1.5,
                label='apertura extra de la pinza (grados)')
        a2.set_xlim(0, t[-1])
        a2.set_title('Respuesta: desplazamiento de O$_p$ (mm) y pinza (grados)', fontsize=11); a2.set_xlabel('t (s)')
        a2.legend(fontsize=8, loc='lower left', ncol=2)
        wr.append_data(fig_to_array(fig)[:H, :W])
        plt.close(fig)
    wr.close()
    print('wrote', out)


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'd25'
    what = sys.argv[2] if len(sys.argv) > 2 else 'both'
    if what in ('both', 'tracking'):
        tracking_video(name)
    if what in ('both', 'phri'):
        phri_video(name)
