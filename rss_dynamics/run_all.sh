#!/bin/bash
# Regenerates every result of rss_dynamics (about 2 h on 4 cores).
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MUJOCO_GL=osmesa
OBJ=core python3 design.py 25 35        # results/design_d25_core.json, design_d35_core.json
python3 design_figures.py               # geometry, conditioning maps, design_summary.json
python3 feasibility.py                  # motors and belt ratio
python3 motor_layout.py d35_core        # motors and belts under the base
python3 validate.py d35_core            # V1-V6
python3 closed_loop.py d35_core         # tracking and pHRI in MuJoCo
python3 animate.py d35_core             # videos
