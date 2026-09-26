# ROS 2 Jazzy (LTS) desktop image (RViz, rqt, ...)
FROM osrf/ros:jazzy-desktop

ARG DEBIAN_FRONTEND=noninteractive

# ros2_control, controllers, Gazebo, and tools to render the videos
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo git nano less bash-completion \
      python3-colcon-common-extensions python3-rosdep python3-vcstool python3-pip \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-ros2-controllers-test-nodes \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros-gz \
      ros-jazzy-xacro \
      ros-jazzy-joint-state-publisher-gui \
      ros-jazzy-rqt-controller-manager \
      libosmesa6 ffmpeg fonts-dejavu-core python3-pil \
    && rm -rf /var/lib/apt/lists/*

# Python bindings of MuJoCo: offline rendering of the pick-and-place video
# (ros2 run ninedof_mujoco render_video). The simulation itself runs in
# mujoco_ros2_control (installed by rosdep from the package dependencies).
RUN pip3 install --no-cache-dir --break-system-packages mujoco

# The "ubuntu" user (uid 1000) already exists in the base image: passwordless sudo
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# Source ROS 2 (and the workspace, once built) in every terminal
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /home/ubuntu/.bashrc \
    && echo '[ -f "${CONTAINER_WORKSPACE_FOLDER:-/workspaces}/install/setup.bash" ] && source "${CONTAINER_WORKSPACE_FOLDER}/install/setup.bash"' >> /home/ubuntu/.bashrc \
    && echo 'source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash 2>/dev/null' >> /home/ubuntu/.bashrc

USER ubuntu
