"""Demo motion: move the 9 DoF one after another with a sine wave."""

import math

import numpy as np

# Amplitude of each DoF: x, y, z [m] then alpha1..3, beta1..3 [rad]
AMPLITUDE = np.array([0.01, 0.01, 0.01] + [math.radians(10.0)] * 6)
PERIOD = 4.0  # seconds per DoF


def demo_pose(home, t):
    """Pose of the demo at time t [s], starting from the home pose."""
    k = int(t // PERIOD) % 9
    x = np.array(home, dtype=float)
    x[k] += AMPLITUDE[k] * math.sin(2.0 * math.pi * (t % PERIOD) / PERIOD)
    return x
