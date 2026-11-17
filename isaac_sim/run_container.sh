#!/usr/bin/env bash
# Start the Isaac Sim container on the VM with this repository mounted at /work.
#   ./run_container.sh            interactive shell in the container (then ./runheadless.sh or ./python.sh ...)
#   ./run_container.sh quick      diagnostic run (6 s, prints only DIAG lines)
#   ./run_container.sh video-test 4 still frames;  ./run_container.sh video  full MP4 (results/ninedof_isaac_sim.mp4)
#   ./run_container.sh validate   headless validation runs (gravity and no gravity), results in isaac_sim/results
set -e
EXTRA="${*:2}"                                       # extra options for render_video.py, e.g. --key 1200
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.1.0}"
C=~/docker/isaac-sim
mkdir -p $C/{cache/main,cache/computecache,config,data,logs,pkg}
sudo chown -R 1234:1234 $C
sudo chmod -R a+rwX "$REPO/isaac_sim"                # the container user (1234) writes the results
trap 'sudo chown -R "$(id -u):$(id -g)" "$REPO/isaac_sim"' EXIT   # give the results back to the VM user
ARGS=(--gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y --rm --network=host -u 1234:1234
  -v $C/cache/main:/isaac-sim/.cache:rw -v $C/cache/computecache:/isaac-sim/.nv/ComputeCache:rw
  -v $C/logs:/isaac-sim/.nvidia-omniverse/logs:rw -v $C/config:/isaac-sim/.nvidia-omniverse/config:rw
  -v $C/data:/isaac-sim/.local/share/ov/data:rw -v $C/pkg:/isaac-sim/.local/share/ov/pkg:rw
  -v "$REPO":/work:rw)
if [ "$1" = "quick" ]; then              # diagnostic: 6 s of trajectory, no gravity, only the useful lines
  sudo docker run "${ARGS[@]}" --entrypoint bash "$IMAGE" -c "
    cd /isaac-sim && ./python.sh /work/isaac_sim/run_validation.py --no-gravity --samples 600 \
      -o /work/isaac_sim/results/isaac_quick.npz 2>&1 | grep -E 'DIAG|sample|note:|Error|error|Traceback|wrote' | grep -v -E 'OmniHub|hub-isaac' "
elif [ "$1" = "video-test" ]; then       # 4 still frames to check the look (results/frames/0000*.png)
  sudo rm -rf "$REPO/isaac_sim/results/frames"
  sudo docker run "${ARGS[@]}" --entrypoint bash "$IMAGE" -c "
    cd /isaac-sim && ./python.sh /work/isaac_sim/render_video.py --frames 4 $EXTRA 2>&1 | grep -E 'frame|wrote|Error|error|Traceback' | grep -v -E 'OmniHub|hub-isaac' "
elif [ "$1" = "video" ]; then            # full video: frames in the container, H.264 on the VM
  sudo rm -rf "$REPO/isaac_sim/results/frames"
  sudo docker run "${ARGS[@]}" --entrypoint bash "$IMAGE" -c "
    cd /isaac-sim && ./python.sh /work/isaac_sim/render_video.py $EXTRA 2>&1 | grep -E 'frame|wrote|Error|error|Traceback' | grep -v -E 'OmniHub|hub-isaac' "
  sudo docker run --rm -v "$REPO/isaac_sim/results":/r jrottenberg/ffmpeg:4.4-ubuntu \
    -y -loglevel error -framerate 30 -i /r/frames/%05d.png \
    -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p /r/ninedof_isaac_sim.mp4   # ffmpeg in Docker: apt mirror unreliable
  echo "video: $REPO/isaac_sim/results/ninedof_isaac_sim.mp4"
elif [ "$1" = "validate" ]; then
  sudo docker run "${ARGS[@]}" --entrypoint bash "$IMAGE" -c "
    cd /isaac-sim &&
    ./python.sh /work/isaac_sim/run_validation.py --no-gravity -o /work/isaac_sim/results/isaac_nogravity.npz &&
    ./python.sh /work/isaac_sim/run_validation.py -o /work/isaac_sim/results/isaac_gravity.npz &&
    ./python.sh /work/isaac_sim/compare.py /work/isaac_sim/results/mujoco_nogravity.npz /work/isaac_sim/results/isaac_nogravity.npz &&
    ./python.sh /work/isaac_sim/compare.py /work/isaac_sim/results/mujoco_gravity.npz /work/isaac_sim/results/isaac_gravity.npz"
else
  sudo docker run -it "${ARGS[@]}" --entrypoint bash "$IMAGE"
fi
