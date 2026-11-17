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
| `render_video.py` | RTX video of the showcase (inverse kinematics or a PhysX run) |
| `make_grid.py`, `grid_sim.py`, `grid_place.py`, `render_grid.py` | N copies of the robot with different servo stiffness: grid stage, PhysX run, placement, RTX video |
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

## Cross-simulator validation (Isaac Sim 6.1, PhysX, against MuJoCo)

Same model, same servos (5·10⁴ N/m), same 56 s showcase, 100 Hz targets, PhysX at 2000 Hz and MuJoCo at
10 kHz with stiff loop closures. Loop closures stay closed within 0.001 mm in PhysX and within 0.0001 mm in
MuJoCo during the whole run. Largest difference between the two simulators, per segment, with gravity:

| Segment | platform 1 centre (mm) | Q1 (deg) | Q2 (deg) |
|---|---|---|---|
| translation x | 0.705 | 1.678 | 1.596 |
| translation y | 0.091 | 0.215 | 0.205 |
| translation z | 0.064 | 0.153 | 0.146 |
| rigid rotation x | 0.051 | 0.120 | 0.116 |
| rigid rotation y | 0.058 | 0.134 | 0.166 |
| rigid rotation z | 0.056 | 0.123 | 0.132 |
| cone | 0.148 | 0.465 | 0.290 |
| relative rotation y (grasp) | 0.115 | 0.352 | 0.339 |
| relative rotation x | 0.118 | 0.283 | 0.280 |
| relative rotation z | 0.065 | 0.156 | 0.147 |

The initial settling differs by 1.9 mm (transient) and both simulators converge to the same sagged pose. In both,
gravity sags the platforms by about 5 mm away from the inverse kinematics while the actuators deflect only
0.1 mm: the amplification predicted by the conditioning below, reproduced by two independent physics engines.

Without gravity the two simulators drift apart (up to 15 mm in the cone segment) although each keeps its
actuators within 0.024 mm and its loops closed: with no load on the soft direction the forward problem is
nearly indeterminate, so numerical details of each solver decide where the platforms go. Gravity preloads
that direction and makes the response repeatable.

## Videos

| Command (on the VM) | Result |
|---|---|
| `run_container.sh video-test --kinematic` | 4 still frames to check the look |
| `run_container.sh video --kinematic` | `results/ninedof_isaac_sim.mp4`: the showcase with the CAD, RTX render of the paper inverse kinematics |
| `run_container.sh grid-test` | 4 robots for 3 s: checks the pipeline and prints how long the full grid would take |
| `run_container.sh grid` | `results/ninedof_grid.mp4`: 16 robots simulated at once in PhysX, servo stiffness from 2·10⁴ (front) to 2·10⁶ N/m (back) |

Rendering is a separate pass: the poses are computed first (inverse kinematics, or a PhysX run with `grid_sim.py`),
then every body is placed from them with all physics removed from the render stage (`grid_place.py`). Leaving the
rigid bodies in the stage let the renderer advance PhysX and the platforms spun freely on their ball joints.

## Servo stiffness sweep (16 robots in one PhysX scene)

`run_container.sh grid`: 16 copies of the robot with gravity, servo stiffness log-spaced from 2·10⁴ to 2·10⁶ N/m,
critical damping, first 20.5 s of the showcase (reference pose, translations, rigid rotation about x). Largest
errors of each copy against the inverse kinematics:

| Servo stiffness (N/m) | platform centre (mm) | Q1 (deg) | actuator tracking (mm) |
|---|---|---|---|
| 20 000 | 13.885 | 30.302 | 0.301 |
| 27 187 | 11.304 | 24.371 | 0.219 |
| 36 957 | 8.866 | 19.055 | 0.155 |
| 50 238 | 6.980 | 15.000 | 0.115 |
| 68 291 | 8.924 | 22.045 | 0.127 |
| 92 832 | 7.444 | 18.375 | 0.102 |
| 126 191 | 7.441 | 18.185 | 0.076 |
| 171 539 | 5.643 | 13.854 | 0.057 |
| 233 183 | 5.230 | 12.793 | 0.043 |
| 316 979 | 5.203 | 12.679 | 0.033 |
| 430 887 | 4.026 | 9.841 | 0.025 |
| 585 729 | 15.649 | 40.881 | 0.020 |
| 796 214 | 3.485 | 8.497 | 0.016 |
| 1 082 339 | 2.972 | 7.266 | 0.012 |
| 1 471 285 | 2.757 | 6.734 | 0.010 |
| 2 000 000 | 2.886 | 7.027 | 0.008 |

The actuators track within 0.3 mm even with the softest servo, while the platforms deviate up to 40 times more:
the amplification set by the conditioning below. A stiffer servo reduces it at the reference pose (0.4 mm and
0.9 deg for 2·10⁶ N/m over the first 3 s), but not in the rigid rotation about x, where the condition number
reaches 3·10⁵ and even the stiffest copy deviates 2.9 mm and 7 deg with 0.008 mm of actuator error. The copy at
5.9·10⁵ N/m leaves the trend (15.6 mm, 41 deg): near the singularity it moves to a different configuration.
The remedy is the geometry, not the servo gain.

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
