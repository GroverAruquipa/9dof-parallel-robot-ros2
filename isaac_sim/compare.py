"""Compare simulated platform poses with the paper inverse kinematics (reference_trajectory.npz).

For each result file (result_mujoco.npz, result_isaac.npz, ...) prints, per segment of the showcase, the
maximum position error of the central joint centre p (mm) and the maximum orientation errors of Q1 and Q2
(deg), plus the actuator tracking error; writes compare.png when matplotlib is available.

Usage: python3 compare.py result_mujoco.npz [result_isaac.npz]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def errors(res, ref):
    n = min(len(res['t']), len(ref['t']))
    ep = np.linalg.norm(res['p1'][:n] - ref['p'][:n], axis=1) * 1000
    e1 = np.array([angle(res['R1'][j], ref['Q1'][j]) for j in range(n)])
    e2 = np.array([angle(res['R2'][j], ref['Q2'][j]) for j in range(n)])
    eq = np.max(np.abs(res['q'][:n] - ref['q'][:n]), axis=1) * 1000
    return n, ep, e1, e2, eq


def main():
    ref = np.load(os.path.join(HERE, 'reference_trajectory.npz'))
    names = [str(s) for s in ref['segment_names']]
    rows = {}
    for path in sys.argv[1:]:
        res = np.load(path); n, ep, e1, e2, eq = errors(res, ref)
        seg = ref['segment'][:n]; label = os.path.basename(path).replace('result_', '').replace('.npz', '')
        rows[label] = (ref['t'][:n], ep, e1, e2, eq)
        print(f'\n{label}: max error over the whole run  p {ep.max():.3f} mm   Q1 {e1.max():.3f} deg   '
              f'Q2 {e2.max():.3f} deg   q tracking {eq.max():.3f} mm')
        print(f'  {"segment":22s} {"p (mm)":>8s} {"Q1 (deg)":>9s} {"Q2 (deg)":>9s} {"q (mm)":>8s}')
        for k, name in enumerate(names):
            s = seg == k
            if s.any():
                print(f'  {name:22s} {ep[s].max():8.3f} {e1[s].max():9.3f} {e2[s].max():9.3f} {eq[s].max():8.3f}')
    if len(sys.argv) > 2:                      # simulator against simulator (same model, same servo)
        base = np.load(sys.argv[1]); lb = os.path.basename(sys.argv[1])
        for path in sys.argv[2:]:
            res = np.load(path); n = min(len(base['t']), len(res['t'])); seg = ref['segment'][:n]
            dp = np.linalg.norm(res['p1'][:n] - base['p1'][:n], axis=1) * 1000
            d1 = np.array([angle(res['R1'][j], base['R1'][j]) for j in range(n)])
            d2 = np.array([angle(res['R2'][j], base['R2'][j]) for j in range(n)])
            print(f'\n{os.path.basename(path)} against {lb}: max difference  p {dp.max():.3f} mm   '
                  f'Q1 {d1.max():.3f} deg   Q2 {d2.max():.3f} deg')
            for k, name in enumerate(names):
                s = seg == k
                if s.any():
                    print(f'  {name:22s} {dp[s].max():8.3f} {d1[s].max():9.3f} {d2[s].max():9.3f}')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for label, (t, ep, e1, e2, eq) in rows.items():
        ax[0].plot(t, ep, label=label); ax[1].plot(t, e1, label=f'{label} Q1'); ax[1].plot(t, e2, '--', label=f'{label} Q2')
        ax[2].plot(t, eq, label=label)
    ax[0].set_ylabel('p error (mm)'); ax[1].set_ylabel('orientation error (deg)'); ax[2].set_ylabel('q tracking (mm)')
    ax[2].set_xlabel('time (s)')
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'compare.png'), dpi=120)
    print('\nwrote compare.png')


if __name__ == '__main__':
    main()
