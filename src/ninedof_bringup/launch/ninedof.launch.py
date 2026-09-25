"""Run the 9-DoF parallel robot with ros2_control.

    ros2 launch ninedof_bringup ninedof.launch.py              # mock hardware + demo motion
    ros2 launch ninedof_bringup ninedof.launch.py demo:=false
    ros2 topic pub --once /cartesian_pose_controller/pose_cmd std_msgs/msg/Float64MultiArray \
        "data: [0.0, 0.0, 0.1467, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3]"
    ros2 topic echo /platform_pose   # pose measured through the forward kinematics
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hardware_plugin = LaunchConfiguration('hardware_plugin')
    demo = LaunchConfiguration('demo')
    gui = LaunchConfiguration('gui')

    robot_description = ParameterValue(
        Command(['xacro ',
                 PathJoinSubstitution([FindPackageShare('ninedof_description'),
                                       'urdf', 'ninedof.urdf.xacro']),
                 ' hardware_plugin:=', hardware_plugin]),
        value_type=str)
    controllers = PathJoinSubstitution(
        [FindPackageShare('ninedof_bringup'), 'config', 'ninedof_controllers.yaml'])
    rviz_config = PathJoinSubstitution(
        [FindPackageShare('ninedof_bringup'), 'rviz', 'view_robot.rviz'])

    return LaunchDescription([
        DeclareLaunchArgument('hardware_plugin', default_value='mock_components/GenericSystem',
                              description='ros2_control hardware plugin of the actuators'),
        DeclareLaunchArgument('demo', default_value='true',
                              description='Send the demo motion to the controller'),
        DeclareLaunchArgument('gui', default_value='true', description='Start RViz'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='controller_manager', executable='ros2_control_node',
             parameters=[controllers],
             remappings=[('~/robot_description', '/robot_description')],
             output='screen'),
        Node(package='controller_manager', executable='spawner',
             arguments=['joint_state_broadcaster', 'cartesian_pose_controller',
                        '--param-file', controllers]),
        Node(package='ninedof_kinematics', executable='fk_joint_state_publisher'),
        Node(package='ninedof_kinematics', executable='demo_pose_publisher',
             remappings=[('pose_cmd', '/cartesian_pose_controller/pose_cmd')],
             condition=IfCondition(demo)),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config],
             condition=IfCondition(gui)),
    ])
