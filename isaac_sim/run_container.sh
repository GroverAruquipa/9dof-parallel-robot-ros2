#!/usr/bin/env bash
# Start the Isaac Sim container on the VM with this repository mounted at /work.
#   ./run_container.sh            interactive shell in the container (then ./runheadless.sh or ./python.sh ...)
#   ./run_container.sh validate   headless validation runs (gravity and no gravity), results in isaac_sim/results
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0}"
C=~/docker/isaac-sim
mkdir -p $C/{cache/main,cache/computecache,config,data,logs,pkg}
sudo chown -R 1234:1234 $C
chmod -R a+rwX "$REPO/isaac_sim"                     # the container user (1234) writes the results
ARGS=(--gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y --rm --network=host -u 1234:1234
  -v $C/cache/main:/isaac-sim/.cache:rw -v $C/cache/computecache:/isaac-sim/.nv/ComputeCache:rw
  -v $C/logs:/isaac-sim/.nvidia-omniverse/logs:rw -v $C/config:/isaac-sim/.nvidia-omniverse/config:rw
  -v $C/data:/isaac-sim/.local/share/ov/data:rw -v $C/pkg:/isaac-sim/.local/share/ov/pkg:rw
  -v "$REPO":/work:rw)
if [ "$1" = "validate" ]; then
  sudo docker run "${ARGS[@]}" --entrypoint bash "$IMAGE" -c "
    cd /isaac-sim &&
    ./python.sh /work/isaac_sim/run_validation.py --no-gravity -o /work/isaac_sim/results/isaac_nogravity.npz &&
    ./python.sh /work/isaac_sim/run_validation.py -o /work/isaac_sim/results/isaac_gravity.npz &&
    ./python.sh /work/isaac_sim/compare.py /work/isaac_sim/results/mujoco_nogravity.npz /work/isaac_sim/results/isaac_nogravity.npz &&
    ./python.sh /work/isaac_sim/compare.py /work/isaac_sim/results/mujoco_gravity.npz /work/isaac_sim/results/isaac_gravity.npz"
else
  sudo docker run -it "${ARGS[@]}" --entrypoint bash "$IMAGE"
fi
