"""Pick-and-place with reorientation for the 9-DoF parallel robot (ROS 2 node).

Publishes the pose sequence of ninedof_kinematics.pick_place_plan on
/cartesian_pose_controller/pose_cmd: approach, open, descend, close, lift,
rotate, place, release (open) and retreat. Before sending a pose it checks that
the pose and the whole interpolated path to it are inside the workspace (real
inverse kinematics, actuators within their stroke).

While running with mujoco_ros2_control and the SimStatePublisher plugin it
  * switches the grasp on after closing and off before opening (/mujoco/grasp),
  * records every qpos published on /mujoco/qpos,
  * checks the final pose of the block against the target zone.

Outputs (output_dir): pick_place_qpos.npz (time, qpos, steps) and
pick_place_result.json. Offline check only:

    ros2 run ninedof_mujoco pick_place_demo --check
"""

import argparse
import json
import os
import sys

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
import rclpy.task
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64MultiArray

from ninedof_kinematics.kinematics import NineDofKinematics
from ninedof_kinematics import pick_place_plan as plan_mod


def default_file(name):
    return os.path.join(get_package_share_directory('ninedof_description'), 'config', name)


def check_plan(kin, waypoints, v_lin, v_ang, log=print):
    ok = True
    for name, good, worst, msg in plan_mod.check(kin, waypoints, v_lin, v_ang):
        log(f'  {name:9s} {"OK  " if good else "FAIL"} max actuator travel '
            f'{worst * 1000:5.1f} mm {msg}')
        ok &= good
    return ok


class PickPlaceDemo(Node):

    def __init__(self):
        super().__init__('pick_place_demo')
        self.declare_parameter('geometry_file', default_file('geometry.yaml'))
        self.declare_parameter('pick_place_file', default_file('pick_place.yaml'))
        # Must match cartesian_pose_controller (ninedof_controllers.yaml)
        self.declare_parameter('max_linear_velocity', 0.02)
        self.declare_parameter('max_angular_velocity', 0.5)
        self.declare_parameter('settle_time', 0.3)
        self.declare_parameter('output_dir', os.path.join(os.getcwd(), 'pick_place_output'))

        gp = self.get_parameter
        self.kin = NineDofKinematics.from_yaml(gp('geometry_file').value)
        self.pp = plan_mod.load_pick_place(gp('pick_place_file').value)
        self.v_lin = gp('max_linear_velocity').value
        self.v_ang = gp('max_angular_velocity').value
        self.settle = gp('settle_time').value
        self.output_dir = gp('output_dir').value
        self.waypoints = plan_mod.plan(self.kin, self.pp)

        self.pose_pub = self.create_publisher(
            Float64MultiArray, '/cartesian_pose_controller/pose_cmd', 10)
        self.grasp_pub = self.create_publisher(Bool, '/mujoco/grasp', 10)
        self.create_subscription(Float64MultiArray, '/mujoco/qpos', self.on_qpos, 100)
        self.create_subscription(PoseStamped, '/mujoco/object_pose', self.on_object, 10)

        self.samples = []            # (sim time, qpos)
        self.steps = []              # (name, sim time start, sim time end)
        self.object_pose = None
        self.index = -1
        self.deadline = None
        self.step_start = None
        self.prev = None
        self.done = False
        self.success = False
        self.finished = rclpy.task.Future()
        self.create_timer(0.02, self.on_timer)
        self.get_logger().info('Waiting for the simulation (/mujoco/qpos) and the controller...')

    # ------------------------------------------------------------ callbacks
    def on_qpos(self, msg):
        self.samples.append((msg.data[0], np.asarray(msg.data[1:], dtype=float)))

    def on_object(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        self.object_pose = (np.array([p.x, p.y, p.z]), np.array([q.w, q.x, q.y, q.z]))

    def sim_time(self):
        return self.samples[-1][0] if self.samples else None

    def on_timer(self):
        if self.done:
            return
        t = self.sim_time()
        if t is None or self.pose_pub.get_subscription_count() == 0:
            return
        if self.deadline is not None and t < self.deadline:
            return
        if self.index >= 0:
            wp = self.waypoints[self.index]
            self.steps.append((wp.name, self.step_start, t))
            if wp.name == 'close':
                self.grasp_pub.publish(Bool(data=True))
                self.get_logger().info('  grasp: attached')
        self.index += 1
        if self.index >= len(self.waypoints):
            self.finish()
            return
        wp = self.waypoints[self.index]
        # Workspace check of the pose and of the interpolated path to it.
        report = plan_mod.check(self.kin, ([] if self.prev is None else
                                           [plan_mod.Waypoint('prev', self.prev)]) + [wp],
                                self.v_lin, self.v_ang)
        name, ok, worst, msg = report[-1]
        if not ok:
            self.get_logger().error(f'{wp.name}: pose outside the workspace ({msg}); aborting')
            self.finish(aborted=True)
            return
        if wp.name == 'release':
            self.grasp_pub.publish(Bool(data=False))
            self.get_logger().info('  grasp: released')
        duration = 0.0 if self.prev is None else plan_mod.segment(
            self.prev, wp.pose, self.v_lin, self.v_ang, 0.01)[1]
        self.pose_pub.publish(Float64MultiArray(data=[float(v) for v in wp.pose]))
        self.get_logger().info(
            f'[{self.index + 1}/{len(self.waypoints)}] {wp.name:9s} max actuator travel '
            f'{worst * 1000:4.1f} mm, {duration:4.2f} s')
        self.prev = wp.pose
        self.step_start = t
        self.deadline = t + duration + self.settle + wp.hold

    # --------------------------------------------------------------- result
    def finish(self, aborted=False):
        self.done = True
        os.makedirs(self.output_dir, exist_ok=True)
        result = {'aborted': aborted, 'success': False}
        if not aborted and self.object_pose is not None:
            ok, checks = plan_mod.evaluate_block(*self.object_pose, self.pp)
            result.update({
                'success': bool(ok),
                'block_position_m': self.object_pose[0].tolist(),
                'block_quaternion_wxyz': self.object_pose[1].tolist(),
                'checks': {k: {'value': float(v), 'ok': bool(o)} for k, (v, o) in checks.items()},
            })
            for k, (v, o) in checks.items():
                self.get_logger().info(f'  {k:18s} {v:8.2f}  {"OK" if o else "FAIL"}')
        self.success = result['success']
        np.savez_compressed(
            os.path.join(self.output_dir, 'pick_place_qpos.npz'),
            time=np.array([s[0] for s in self.samples]),
            qpos=np.array([s[1] for s in self.samples]),
            step_names=np.array([s[0] for s in self.steps]),
            step_times=np.array([[s[1], s[2]] for s in self.steps]),
            waypoints=np.array([w.pose for w in self.waypoints]),
            waypoint_names=np.array([w.name for w in self.waypoints]))
        with open(os.path.join(self.output_dir, 'pick_place_result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        verdict = 'SUCCESS' if self.success else 'FAIL'
        self.get_logger().info(f'Pick and place: {verdict} (results in {self.output_dir}, '
                               f'{len(self.samples)} qpos samples)')
        self.finished.set_result(self.success)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='only check the sequence against the workspace and exit')
    args, ros_args = parser.parse_known_args(argv[1:])
    if args.check:
        kin = NineDofKinematics.from_yaml(default_file('geometry.yaml'))
        pp = plan_mod.load_pick_place(default_file('pick_place.yaml'))
        print('Workspace check of the pick-and-place sequence (poses and interpolated paths):')
        ok = check_plan(kin, plan_mod.plan(kin, pp), 0.02, 0.5)
        print('OK' if ok else 'The sequence leaves the workspace')
        sys.exit(0 if ok else 1)

    rclpy.init(args=[argv[0]] + ros_args)
    node = PickPlaceDemo()
    try:
        rclpy.spin_until_future_complete(node, node.finished)
    except KeyboardInterrupt:
        pass
    success = node.success
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

