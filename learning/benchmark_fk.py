"""Compare cold-start strategies for the forward kinematics.

    python3 learning/benchmark_fk.py [--plot docs/learned_fk.png]

For poses drawn in the operating workspace (never seen in training), solves
the forward kinematics from the actuator displacements alone, starting
Gauss-Newton from
    * home:    the robot's home pose (classical cold start),
    * learned: the estimate of the learned model,
and reports how often it recovers the true pose, how often it converges to
another assembly mode or diverges, the number of iterations and the time.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src', 'ninedof_kinematics'))

from ninedof_kinematics.kinematics import NineDofKinematics  # noqa: E402
from ninedof_kinematics.learned_fk import (  # noqa: E402
    LearnedForwardKinematics, sample_workspace)

GEOMETRY = os.path.join(ROOT, 'src', 'ninedof_description', 'config', 'geometry.yaml')
MODEL = os.path.join(ROOT, 'src', 'ninedof_kinematics', 'models', 'learned_fk.npz')


def gauss_newton(kin, q, x0, tol=1e-12, max_iter=50):
    """Same iteration as NineDofKinematics.forward, also returning the count."""
    x = np.array(x0, dtype=float)
    h = 1e-7
    for it in range(max_iter + 1):
        f = kin.constraints(x, q)
        if not np.all(np.isfinite(f)) or np.max(np.abs(f)) > 1.0:
            return None, it
        if np.max(np.abs(f)) < tol:
            return x, it
        if it == max_iter:
            break
        D = np.empty((len(f), 9))
        for k in range(9):
            dx = np.zeros(9)
            dx[k] = h
            D[:, k] = (kin.constraints(x + dx, q) - f) / h
        x = x - np.linalg.lstsq(D, f, rcond=None)[0]
    return None, max_iter


def same_pose(kin, a, b, tol=1e-6):
    pa, Q1a, Q2a = kin.split(a)
    pb, Q1b, Q2b = kin.split(b)
    return max(np.abs(pa - pb).max(), np.abs(Q1a - Q1b).max(), np.abs(Q2a - Q2b).max()) < tol


def evaluate(kin, xs, qs, guesses):
    out = {'correct': 0, 'other': 0, 'diverged': 0, 'iters': [], 'time': 0.0}
    for x, q, g in zip(xs, qs, guesses):
        t = time.perf_counter()
        sol, it = gauss_newton(kin, q, g)
        out['time'] += time.perf_counter() - t
        if sol is None:
            out['diverged'] += 1
        elif same_pose(kin, sol, x):
            out['correct'] += 1
            out['iters'].append(it)
        else:
            out['other'] += 1
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--plot', help='write a figure (needs matplotlib)')
    args = parser.parse_args(argv)

    kin = NineDofKinematics.from_yaml(GEOMETRY)
    model = LearnedForwardKinematics.load(args.model)
    xs, qs = sample_workspace(kin, args.samples, np.random.default_rng(args.seed))

    t = time.perf_counter()
    learned = model(qs)
    t_net = (time.perf_counter() - t) / len(qs)
    t = time.perf_counter()
    for q in qs[:200]:
        model(q)
    t_net_single = (time.perf_counter() - t) / 200
    err = learned - xs
    print(f'Learned model alone: position error {np.abs(err[:, :3]).mean() * 1e3:.2f} mm, '
          f'angle error {np.degrees(np.abs(err[:, 3:]).mean()):.2f} deg (mean); '
          f'{t_net_single * 1e6:.0f} us per call ({t_net * 1e6:.1f} us batched)')

    n = len(xs)
    results = {'home': evaluate(kin, xs, qs, [kin.home] * n),
               'learned': evaluate(kin, xs, qs, learned)}
    print(f'\n{n} random poses, forward kinematics from q only')
    print(f'{"initial guess":14s} {"true pose":>10s} {"other mode":>11s} {"diverged":>9s} '
          f'{"iterations":>11s} {"time":>9s}')
    for name, r in results.items():
        it = np.mean(r['iters']) if r['iters'] else float('nan')
        print(f'{name:14s} {100 * r["correct"] / n:9.1f}% {100 * r["other"] / n:10.1f}% '
              f'{100 * r["diverged"] / n:8.1f}% {it:11.1f} {r["time"] / n * 1e3:7.2f} ms')

    if args.plot:
        plot(args.plot, results, n)
    return results


def plot(path, results, n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['Home pose', 'Learned model']
    keys = ['home', 'learned']
    colours = {'correct': '#2f7d4f', 'other': '#d99a1e', 'diverged': '#c0392b'}
    names = {'correct': 'true pose', 'other': 'other assembly mode', 'diverged': 'diverged'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [3, 2]})
    left = np.zeros(2)
    for k in ('correct', 'other', 'diverged'):
        v = np.array([100 * results[s][k] / n for s in keys])
        ax1.barh(labels, v, left=left, color=colours[k], label=names[k])
        for i, (l0, w) in enumerate(zip(left, v)):
            if w > 6:
                ax1.text(l0 + w / 2, i, f'{w:.0f}%', ha='center', va='center', color='white',
                         fontsize=11, fontweight='bold')
        left += v
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('% of random poses')
    ax1.set_title('Gauss-Newton result, by initial guess')
    ax1.invert_yaxis()
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    for s in ('top', 'right'):
        ax1.spines[s].set_visible(False)

    bins = np.arange(0, 16) - 0.5
    for s, lab, c in zip(keys, labels, ('#8a8f98', '#2f6fd1')):
        ax2.hist(results[s]['iters'], bins=bins, alpha=0.75, color=c, label=lab)
    ax2.set_xlabel('iterations to converge (true pose)')
    ax2.set_ylabel('poses')
    ax2.set_title('Iterations')
    ax2.legend(frameon=False)
    for s in ('top', 'right'):
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
