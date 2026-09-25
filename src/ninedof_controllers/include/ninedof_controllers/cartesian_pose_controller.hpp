// Copyright 2026 Grover Aruquipa
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NINEDOF_CONTROLLERS__CARTESIAN_POSE_CONTROLLER_HPP_
#define NINEDOF_CONTROLLERS__CARTESIAN_POSE_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "ninedof_controllers/kinematics.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace ninedof_controllers
{

/// Receives the pose of the two moving platforms on ~/pose_cmd
/// (std_msgs/Float64MultiArray: [x, y, z, alpha1, alpha2, alpha3, beta1, beta2, beta3],
/// metres and radians) and moves towards it: the pose is interpolated with
/// max_linear_velocity / max_angular_velocity and the inverse kinematics gives the
/// position command of the 9 linear actuators at every cycle. Unreachable poses and
/// poses beyond the actuator stroke are rejected and the robot keeps its target.
class CartesianPoseController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  using PoseMsg = std_msgs::msg::Float64MultiArray;

  std::vector<std::string> joints_;
  Kinematics kinematics_;
  double max_linear_velocity_ = 0.02;
  double max_angular_velocity_ = 0.5;

  Pose current_pose_ = Pose::Zero();
  Pose target_pose_ = Pose::Zero();
  Eigen::VectorXd q_cmd_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<PoseMsg>> pose_cmd_buffer_;
  std::shared_ptr<PoseMsg> last_pose_cmd_;
  rclcpp::Subscription<PoseMsg>::SharedPtr pose_cmd_sub_;
};

}  // namespace ninedof_controllers

#endif  // NINEDOF_CONTROLLERS__CARTESIAN_POSE_CONTROLLER_HPP_
