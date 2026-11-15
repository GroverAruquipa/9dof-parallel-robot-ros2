# Isaac Sim (PhysX) model of the 9-DoF robot, CCToMM geometry

The robot of the CCToMM paper (geometry of `9_dof_step_cctomm.STEP`) as a closed-loop PhysX articulation,
with a MuJoCo twin built from the same file for a cross-simulator check.

## How the closed loops are modelled

| Body tree (articulation) | Joint |
|---|---|
| world → base | fixed |
| base → carriage *i* (*i* = 1…9) | prismatic along z, position drive (actuator *q<sub>i</sub>*) |
| carriage *i* → leg *i* | spherical at **b**<sub>*i*0</sub> |
| leg 1 → platform 1 | spherical at **a**<sub>10</sub> |
| platform 1 → platform 2 | spherical at **p**<sub>0</sub> (central joint) |
| leg *i* → platform, *i* = 2…9 | spherical at **a**<sub>*i*0</sub>, **excluded from the articulation** (8 loop closures) |

Every body frame is parallel to the world frame in the CAD pose, with its origin on a joint centre, so all
joint coordinates are zero in the CAD pose. The cardan of the central joint (four concurrent axes, one idle
roll) is a single spherical joint in the physics; its CAD parts are visual only.

## Files

| File | Content |
|---|---|
| `export_cctomm.py` | CAD kinematics → `geometry_cctomm.yaml` (joint points, estimated masses) and `reference_trajectory.npz` (showcase poses solved with the paper inverse kinematics) |
| `generate_usd.py` | `geometry_cctomm.yaml` → USD. `--visual simple` (primitives, public) or `--visual cad` (CAD meshes, private) |
| `verify_usd.py` | offline checks: joint anchors coincide, tree without cycles, masses, CAD ball centres on the joint centres |
| `mujoco_cctomm.py` | MuJoCo twin of the same model, runs the reference |
| `run_validation.py` | the same run in Isaac Sim (PhysX at 2000 Hz) |
| `compare.py` | errors against the inverse kinematics, and simulator against simulator |
| `run_container.sh` | starts the Isaac Sim container with this repository mounted at `/work` |

## Offline checks (no GPU)

```bash
pip install usd-core trimesh pyyaml mujoco
python3 generate_usd.py --visual simple && python3 verify_usd.py ninedof_cctomm_simple.usda
python3 mujoco_cctomm.py --no-gravity -o results/mujoco_nogravity.npz
python3 compare.py results/mujoco_nogravity.npz
```

`verify_usd.py`: 29 joints, anchor mismatch 0.004 µm, 21 bodies and 21 tree edges, 8 loop closures; the 19
ruby balls of the CAD lie within 0.007 mm of the joint centres.

## On the Isaac Sim VM

```bash
git clone -b isaac-sim https://github.com/GroverAruquipa/9dof-parallel-robot-ros2 ~/ninedof
~/ninedof/isaac_sim/run_container.sh validate      # headless runs + comparison with MuJoCo
~/ninedof/isaac_sim/run_container.sh               # shell; then ./runheadless.sh and open /work/isaac_sim/*.usda
```

## Conditioning of the geometry

Simulating the robot showed that the CAD geometry lies close to the coupled translational Type II singularity.
With the Jacobian of the inverse kinematics (actuator rates against the velocity of **p** and the angular
velocities of both platforms, mm and rad):

| Segment of the showcase | condition number | smallest singular value |
|---|---|---|
| CAD pose, translation along z | 4 105 | 0.0117 |
| translation along y | 2 368 to 12 534 | 0.0039 |
| rigid rotation about x | 1 334 to 326 706 | 0.0002 |
| relative rotation about x | 1 451 to 681 492 | 0.0001 |
| relative rotation about y (grasp) | 2 248 to 4 618 | 0.0107 |

At the CAD pose the softest direction is almost a pure translation of **p** along y: 1 mm of **p** changes the
actuators by 0.012 mm, so any compliance of the actuators is amplified about 85 times at the platforms. With
the position servo of the MuJoCo model (5·10⁴ N/m), gravity alone moves the platforms by about 3.7 mm. Both
simulators are therefore compared against each other first, and against the inverse kinematics only on the
well-conditioned segments. The geometry of `config/geometry.yaml` (model_6v3) has the same property
(condition number 18 008 at home).
