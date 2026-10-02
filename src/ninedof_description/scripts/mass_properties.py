#!/usr/bin/env python3
"""Centre of mass and inertia tensor of the moving parts, from the meshes.

The meshes give the shape of the mass distribution; the masses come from
config/dynamics.yaml (the printed part plus its hardware: ball studs, screws,
sockets), so the tensor of each mesh is scaled to the chosen mass. Prints the
`inertia:` block of config/dynamics.yaml (body frames = mesh frames, metres,
kg m^2, about the centre of mass, MuJoCo fullinertia order
[ixx, iyy, izz, ixy, ixz, iyz]).

    python3 scripts/mass_properties.py       # needs trimesh
"""

import os

import trimesh
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = ('platform_1', 'platform_2', 'distal_link')


def mass_properties(mesh_file, mass):
    mesh = trimesh.load(mesh_file)
    mesh.density = 1.0
    com = mesh.center_mass
    # moment_inertia is about the centre of mass for unit density.
    I = mesh.moment_inertia * mass / mesh.volume
    return com, I


def main():
    with open(os.path.join(PKG, 'config', 'dynamics.yaml')) as f:
        mass = yaml.safe_load(f)['ninedof_dynamics']['mass']
    out = {}
    for part in PARTS:
        com, I = mass_properties(os.path.join(PKG, 'meshes', f'{part}.stl'), mass[part])
        out[part] = {
            'com': [round(float(v), 6) for v in com],
            'inertia': [float(f'{v:.4e}') for v in
                        (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])],
        }
    print(yaml.safe_dump({'inertia': out}, default_flow_style=None, sort_keys=False))


if __name__ == '__main__':
    main()
