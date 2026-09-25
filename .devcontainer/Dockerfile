# ROS 2 Jazzy (LTS) con escritorio (RViz, rqt, etc.)
FROM osrf/ros:jazzy-desktop

ARG DEBIAN_FRONTEND=noninteractive

# ros2_control, controladores y simulación con Gazebo
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo git nano less bash-completion \
      python3-colcon-common-extensions python3-rosdep python3-vcstool python3-pip \
      ros-jazzy-ros2-control \
      ros-jazzy-ros2-controllers \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-ros-gz \
      ros-jazzy-xacro \
      ros-jazzy-joint-state-publisher-gui \
      ros-jazzy-rqt-controller-manager \
    && rm -rf /var/lib/apt/lists/*

# El usuario "ubuntu" (uid 1000) ya existe en la imagen base; le damos sudo sin contraseña
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

# Cargar ROS 2 automáticamente en cada terminal
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /home/ubuntu/.bashrc \
    && echo '[ -f "${CONTAINER_WORKSPACE_FOLDER:-/workspaces}/install/setup.bash" ] && source "${CONTAINER_WORKSPACE_FOLDER}/install/setup.bash"' >> /home/ubuntu/.bashrc \
    && echo 'source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash 2>/dev/null' >> /home/ubuntu/.bashrc

USER ubuntu
