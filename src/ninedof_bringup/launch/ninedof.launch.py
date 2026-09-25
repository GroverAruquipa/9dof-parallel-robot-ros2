"""Run the 9-DoF parallel robot with ros2_control.

    ros2 launch ninedof_bringup ninedof.launch.py              # mock hardware + demo motion
    ros2 launch ninedof_bringup ninedof.launch.py sim:=mujoco  # MuJoCo physics (closed chains)
    ros2 launch ninedof_bringup ninedof.launch.py demo:=false
    ros2 topic pub --once /cartesian_pose_controller/pose_cmd std_msgs/msg/Float64MultiArray \
        "data: [0.0, 0.0, 0.1467, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3]"
    ros2 topic echo /platform_pose   # pose measured through the forward kinematics
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    sim = LaunchConfiguration('sim').perform(context)
    if sim not in ('mock', 'mujoco'):
        raise RuntimeError(f"sim must be 'mock' or 'mujoco', got '{sim}'")
    mujoco = sim == 'mujoco'
    hardware_plugin = ('mujoco_ros2_control/MujocoSystemInterface' if mujoco
                       else 'mock_components/GenericSystem')
    use_sim_time = {'use_sim_time': mujoco}
    demo = LaunchConfiguration('demo')
    gui = LaunchConfiguration('gui')

    robot_description = ParameterValue(
        Command(['xacro ',
                 PathJoinSubstitution([FindPackageShare('ninedof_description'),
                                       'urdf', 'ninedof.urdf.xacro']),
                 ' hardware_plugin:=', hardware_plugin,
                 ' headless:=', LaunchConfiguration('headless')]),
        value_type=str)
    controllers = PathJoinSubstitution(
        [FindPackageShare('ninedof_bringup'), 'config', 'ninedof_controllers.yaml'])
    rviz_config = PathJoinSubstitution(
        [FindPackageShare('ninedof_bringup'), 'rviz', 'view_robot.rviz'])

    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}, use_sim_time]),
        # mujoco_ros2_control ships its own ros2_control_node that also steps MuJoCo
        Node(package='mujoco_ros2_control' if mujoco else 'controller_manager',
             executable='ros2_control_node',
             parameters=[controllers, use_sim_time],
             remappings=[('~/robot_description', '/robot_description')],
             output='screen'),
        Node(package='controller_manager', executable='spawner',
             arguments=['joint_state_broadcaster', 'cartesian_pose_controller',
                        '--param-file', controllers],
             parameters=[use_sim_time]),
        Node(package='ninedof_kinematics', executable='fk_joint_state_publisher',
             parameters=[use_sim_time]),
        Node(package='ninedof_kinematics', executable='demo_pose_publisher',
             remappings=[('pose_cmd', '/cartesian_pose_controller/pose_cmd')],
             parameters=[use_sim_time], condition=IfCondition(demo)),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config],
             parameters=[use_sim_time], condition=IfCondition(gui)),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='mock',
                              description="'mock' (ideal actuators) or 'mujoco' (physics)"),
        DeclareLaunchArgument('headless', default_value='false',
                              description='MuJoCo without its viewer window'),
        DeclareLaunchArgument('demo', default_value='true',
                              description='Send the demo motion to the controller'),
        DeclareLaunchArgument('gui', default_value='true', description='Start RViz'),
        OpaqueFunction(function=launch_setup),
    ])
