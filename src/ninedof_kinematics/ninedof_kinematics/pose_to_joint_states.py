"""Publish the joint states of the whole robot from a platform pose.

Subscribes to ``pose_cmd`` (std_msgs/Float64MultiArray, 9 values
[x, y, z, alpha1, alpha2, alpha3, beta1, beta2, beta3] in m and rad) and
publishes ``joint_states`` for every joint of ninedof.urdf.xacro, actuated
and passive, so robot_state_publisher and RViz can draw the closed chains.

With ``demo: true`` it animates the 9 DoF one after another instead.
"""

import os

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ninedof_kinematics.demo import demo_pose
from ninedof_kinematics.kinematics import NineDofKinematics, UnreachablePose


class PoseToJointStates(Node):

    def __init__(self):
        super().__init__('pose_to_joint_states')
        default_geometry = os.path.join(
            get_package_share_directory('ninedof_description'), 'config', 'geometry.yaml')
        geometry = self.declare_parameter('geometry_file', default_geometry).value
        self.demo = self.declare_parameter('demo', True).value
        rate = self.declare_parameter('rate', 30.0).value

        self.kin = NineDofKinematics.from_yaml(geometry)
        self.pose = self.kin.home.copy()
        self.t0 = self.get_clock().now()

        self.pub = self.create_publisher(JointState, 'joint_states', 10)
        self.create_subscription(Float64MultiArray, 'pose_cmd', self.on_pose_cmd, 10)
        self.create_timer(1.0 / rate, self.on_timer)
        self.get_logger().info(
            'Publishing joint states (%s). Send poses on %s as '
            '[x, y, z, a1, a2, a3, b1, b2, b3].'
            % ('demo' if self.demo else 'waiting for commands',
               self.resolve_topic_name('pose_cmd')))

    def on_pose_cmd(self, msg):
        if len(msg.data) != 9:
            self.get_logger().error('pose_cmd needs 9 values, got %d' % len(msg.data))
            return
        self.demo = False
        self.pose = np.array(msg.data, dtype=float)

    def on_timer(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        x = demo_pose(self.kin.home, t) if self.demo else self.pose
        try:
            q = self.kin.inverse(x, check_stroke=True)
        except UnreachablePose as e:
            self.get_logger().warn('Pose not reachable: %s' % e, throttle_duration_sec=2.0)
            return
        js = self.kin.joint_positions(x, q)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(js.keys())
        msg.position = [float(v) for v in js.values()]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseToJointStates()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
