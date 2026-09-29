"""Train the learned initial guess of the forward kinematics.

    pip install -r learning/requirements.txt
    python3 learning/train_fk.py            # writes the model shipped with ninedof_kinematics

Samples poses uniformly in the operating workspace, labels them with the
analytic inverse kinematics (q = IK(x), exact and cheap), and fits a
multilayer perceptron q -> x. The weights are exported to an .npz file read
with numpy by ninedof_kinematics.learned_fk, so the robot needs no PyTorch.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src', 'ninedof_kinematics'))

from ninedof_kinematics.kinematics import NineDofKinematics  # noqa: E402
from ninedof_kinematics.learned_fk import (  # noqa: E402
    LearnedForwardKinematics, sample_workspace)

GEOMETRY = os.path.join(ROOT, 'src', 'ninedof_description', 'config', 'geometry.yaml')
MODEL = os.path.join(ROOT, 'src', 'ninedof_kinematics', 'models', 'learned_fk.npz')


def mlp(width, depth):
    layers, n = [], 9
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.Tanh()]
        n = width
    return nn.Sequential(*layers, nn.Linear(n, 9))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-o', '--output', default=MODEL)
    parser.add_argument('--samples', type=int, default=400_000)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    kin = NineDofKinematics.from_yaml(GEOMETRY)

    t0 = time.time()
    x, q = sample_workspace(kin, args.samples + 20_000, rng)
    print(f'{len(x)} samples in {time.time() - t0:.0f} s')
    q_scale = kin.stroke
    x_mean, x_scale = x.mean(0), x.std(0)
    X = torch.tensor(q / q_scale, dtype=torch.float32)
    Y = torch.tensor((x - x_mean) / x_scale, dtype=torch.float32)
    X_val, Y_val, X, Y = X[:20_000], Y[:20_000], X[20_000:], Y[20_000:]

    net = mlp(args.width, args.depth)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps = args.epochs * (len(X) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        for i in range(len(X) // args.batch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            loss = nn.functional.mse_loss(net(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            with torch.no_grad():
                err = (net(X_val) - Y_val).numpy() * x_scale
            print(f'epoch {epoch + 1:3d}  val error: position {np.abs(err[:, :3]).mean() * 1e3:.3f} mm, '
                  f'angles {np.degrees(np.abs(err[:, 3:]).mean()):.3f} deg')

    linears = [m for m in net if isinstance(m, nn.Linear)]
    model = LearnedForwardKinematics(
        [m.weight.detach().double().numpy() for m in linears],
        [m.bias.detach().double().numpy() for m in linears], q_scale, x_mean, x_scale)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f'Wrote {args.output} ({os.path.getsize(args.output) / 1024:.0f} kB)')


if __name__ == '__main__':
    main()
