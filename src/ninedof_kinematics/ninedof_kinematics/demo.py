"""Demo motion: the showcase trajectory (ninedof_kinematics.showcase), looped."""

from ninedof_kinematics.showcase import Showcase

_cache = {}


def demo_pose(home, t):
    """Pose of the demo at time t [s], starting from the home pose."""
    key = tuple(home)
    if key not in _cache:
        _cache[key] = Showcase(home)
    show = _cache[key]
    return show.sample(t % show.duration)[3]
