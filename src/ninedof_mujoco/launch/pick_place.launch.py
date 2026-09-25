"""Pick-and-place demo in MuJoCo with ros2_control.

    ros2 launch ninedof_mujoco pick_place.launch.py                 # MuJoCo viewer + RViz
    ros2 launch ninedof_mujoco pick_place.launch.py headless:=true gui:=false

Starts ninedof_bringup/ninedof.launch.py with the pick-and-place scene and the
SimStatePublisher plugin, then the pick_place_demo node. Results (qpos
recording and verdict) are written to output_dir; render the video with

    ros2 run ninedof_mujoco render_video <output_dir>/pick_place_qpos.npz
"""

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup = PathJoinSubstitution(
        [FindPackageShare('ninedof_bringup'), 'launch', 'ninedof.launch.py'])
    plugins = PathJoinSubstitution(
        [FindPackageShare('ninedof_mujoco'), 'config', 'mujoco_plugins.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('gui', default_value='true', description='Start RViz'),
        DeclareLaunchArgument('output_dir',
                              default_value=os.path.join(os.getcwd(), 'pick_place_output')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup),
            launch_arguments={
                'sim': 'mujoco',
                'mujoco_scene': 'pick_place_scene.xml',
                'mujoco_plugins': plugins,
                'headless': LaunchConfiguration('headless'),
                'gui': LaunchConfiguration('gui'),
                'demo': 'false',
            }.items()),
        # Give the controllers a few seconds to come up.
        TimerAction(period=6.0, actions=[
            Node(package='ninedof_mujoco', executable='pick_place_demo', output='screen',
                 parameters=[{'use_sim_time': True,
                              'output_dir': LaunchConfiguration('output_dir')}]),
        ]),
    ])
