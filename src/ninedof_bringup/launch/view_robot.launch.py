"""Show the 9-DoF parallel robot in RViz.

    ros2 launch ninedof_bringup view_robot.launch.py            # animated demo
    ros2 launch ninedof_bringup view_robot.launch.py demo:=false
    ros2 topic pub --once /pose_cmd std_msgs/msg/Float64MultiArray \
        "data: [0.0, 0.0, 0.1467, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3]"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    demo = LaunchConfiguration('demo')
    robot_description = ParameterValue(
        Command(['xacro ', PathJoinSubstitution(
            [FindPackageShare('ninedof_description'), 'urdf', 'ninedof.urdf.xacro'])]),
        value_type=str)
    rviz_config = PathJoinSubstitution([FindPackageShare('ninedof_bringup'), 'rviz', 'view_robot.rviz'])

    return LaunchDescription([
        DeclareLaunchArgument('demo', default_value='true',
                              description='Animate the 9 DoF one after another'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='ninedof_kinematics', executable='pose_to_joint_states',
             parameters=[{'demo': ParameterValue(demo, value_type=bool)}],
             output='screen'),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config]),
    ])
