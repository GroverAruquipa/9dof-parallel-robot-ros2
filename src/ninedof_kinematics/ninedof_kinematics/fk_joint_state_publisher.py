"""Complete the joint states of the closed chains from the actuator positions.

ros2_control only knows the 9 actuators. This node listens to their positions
on ``joint_states``, solves the forward kinematics (Gauss-Newton, warm-started
with the previous solution) and publishes on ``joint_states`` the positions of
the passive and virtual joints of the URDF, so robot_state_publisher can draw
the whole robot. It also publishes the measured pose of the platforms on
``platform_pose`` ([x, y, z, alpha1, alpha2, alpha3, beta1, beta2, beta3]).

When there is no previous solution (start-up, or after the solver failed) the
initial guess comes from the learned model of ninedof_kinematics.learned_fk
(parameter ``initial_guess: learned``, default) or is the home pose
(``initial_guess: home``).
"""

import os

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ninedof_kinematics.kinematics import NineDofKinematics
from ninedof_kinematics.learned_fk import default_model_path, LearnedForwardKinematics


class FkJointStatePublisher(Node):

    def __init__(self):
        super().__init__('fk_joint_state_publisher')
        default_geometry = os.path.join(
            get_package_share_directory('ninedof_description'), 'config', 'geometry.yaml')
        geometry = self.declare_parameter('geometry_file', default_geometry).value
        initial_guess = self.declare_parameter('initial_guess', 'learned').value
        model_file = self.declare_parameter('model_file', default_model_path()).value

        self.kin = NineDofKinematics.from_yaml(geometry)
        self.actuators = [f'{leg}_actuator_joint' for leg in self.kin.names]
        self.model = None
        if initial_guess == 'learned':
            self.model = LearnedForwardKinematics.load(model_file)
        elif initial_guess != 'home':
            raise ValueError(f"initial_guess must be 'learned' or 'home', not {initial_guess!r}")
        self.pose = None   # no previous solution yet

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.pose_pub = self.create_publisher(Float64MultiArray, 'platform_pose', 10)
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)

    def on_joint_states(self, msg):
        # Our own messages (passive joints only) are ignored here.
        index = {name: i for i, name in enumerate(msg.name)}
        if not all(name in index for name in self.actuators):
            return
        q = np.array([msg.position[index[name]] for name in self.actuators])
        try:
            self.pose = self.kin.forward(q, self.initial_guess(q))
        except RuntimeError:
            self.get_logger().warn('Forward kinematics did not converge',
                                   throttle_duration_sec=2.0)
            self.pose = None
            return

        joints = self.kin.joint_positions(self.pose, q)
        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = [name for name in joints if name not in self.actuators]
        out.position = [float(joints[name]) for name in out.name]
        self.joint_pub.publish(out)
        self.pose_pub.publish(Float64MultiArray(data=[float(v) for v in self.pose]))


    def initial_guess(self, q):
        if self.pose is not None:
            return self.pose
        return self.model(q) if self.model is not None else self.kin.home


def main(args=None):
    rclpy.init(args=args)
    node = FkJointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
