"""Publish the demo motion as pose commands (for cartesian_pose_controller)."""

import os

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from ninedof_kinematics.demo import demo_pose
from ninedof_kinematics.kinematics import NineDofKinematics


class DemoPosePublisher(Node):

    def __init__(self):
        super().__init__('demo_pose_publisher')
        default_geometry = os.path.join(
            get_package_share_directory('ninedof_description'), 'config', 'geometry.yaml')
        geometry = self.declare_parameter('geometry_file', default_geometry).value
        rate = self.declare_parameter('rate', 30.0).value

        self.home = NineDofKinematics.from_yaml(geometry).home
        self.pub = self.create_publisher(Float64MultiArray, 'pose_cmd', 10)
        self.t0 = self.get_clock().now()
        self.create_timer(1.0 / rate, self.on_timer)

    def on_timer(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        self.pub.publish(Float64MultiArray(data=[float(v) for v in demo_pose(self.home, t)]))


def main(args=None):
    rclpy.init(args=args)
    node = DemoPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
