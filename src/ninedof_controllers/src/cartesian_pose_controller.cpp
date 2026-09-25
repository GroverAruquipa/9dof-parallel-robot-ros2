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

#include "ninedof_controllers/cartesian_pose_controller.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

namespace ninedof_controllers
{

namespace
{
// Resolve "package://<pkg>/<path>" to an absolute path.
std::string resolve_package_path(const std::string & path)
{
  const std::string prefix = "package://";
  if (path.rfind(prefix, 0) != 0) {
    return path;
  }
  const auto slash = path.find('/', prefix.size());
  const auto package = path.substr(prefix.size(), slash - prefix.size());
  return ament_index_cpp::get_package_share_directory(package) +
         (slash == std::string::npos ? "" : path.substr(slash));
}
}  // namespace

using controller_interface::CallbackReturn;
using controller_interface::InterfaceConfiguration;
using controller_interface::interface_configuration_type;

CallbackReturn CartesianPoseController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", std::vector<std::string>());
    auto_declare<std::string>("geometry_file", "");
    auto_declare<double>("max_linear_velocity", 0.02);
    auto_declare<double>("max_angular_velocity", 0.5);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception in on_init: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

InterfaceConfiguration CartesianPoseController::command_interface_configuration() const
{
  InterfaceConfiguration config{interface_configuration_type::INDIVIDUAL, {}};
  for (const auto & joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  return config;
}

InterfaceConfiguration CartesianPoseController::state_interface_configuration() const
{
  InterfaceConfiguration config{interface_configuration_type::INDIVIDUAL, {}};
  for (const auto & joint : joints_) {
    config.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  return config;
}

CallbackReturn CartesianPoseController::on_configure(const rclcpp_lifecycle::State &)
{
  const auto logger = get_node()->get_logger();
  joints_ = get_node()->get_parameter("joints").as_string_array();
  max_linear_velocity_ = get_node()->get_parameter("max_linear_velocity").as_double();
  max_angular_velocity_ = get_node()->get_parameter("max_angular_velocity").as_double();
  std::string geometry_file = get_node()->get_parameter("geometry_file").as_string();

  try {
    geometry_file = resolve_package_path(geometry_file);
    kinematics_.load(geometry_file);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger, "Cannot load geometry_file '%s': %s", geometry_file.c_str(), e.what());
    return CallbackReturn::ERROR;
  }

  // joints[i] must drive legs[i] of the geometry file: check the names match.
  const auto & legs = kinematics_.legs();
  if (joints_.size() != legs.size()) {
    RCLCPP_ERROR(logger, "Expected %zu joints, got %zu", legs.size(), joints_.size());
    return CallbackReturn::ERROR;
  }
  for (size_t i = 0; i < legs.size(); ++i) {
    if (joints_[i].rfind(legs[i].name + "_", 0) != 0) {
      RCLCPP_ERROR(
        logger, "Joint '%s' does not belong to leg '%s' (joints must follow the order of %s)",
        joints_[i].c_str(), legs[i].name.c_str(), geometry_file.c_str());
      return CallbackReturn::ERROR;
    }
  }
  if (max_linear_velocity_ <= 0.0 || max_angular_velocity_ <= 0.0) {
    RCLCPP_ERROR(logger, "max_linear_velocity and max_angular_velocity must be positive");
    return CallbackReturn::ERROR;
  }
  q_cmd_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(legs.size()));

  pose_cmd_sub_ = get_node()->create_subscription<PoseMsg>(
    "~/pose_cmd", rclcpp::SystemDefaultsQoS(),
    [this](const std::shared_ptr<PoseMsg> msg) {
      if (msg->data.size() != 9) {
        RCLCPP_ERROR(
          get_node()->get_logger(), "pose_cmd needs 9 values, got %zu", msg->data.size());
        return;
      }
      pose_cmd_buffer_.writeFromNonRT(msg);
    });

  RCLCPP_INFO(
    logger, "Configured for %zu actuators, max velocity %.3f m/s and %.3f rad/s",
    joints_.size(), max_linear_velocity_, max_angular_velocity_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianPoseController::on_activate(const rclcpp_lifecycle::State &)
{
  // Start from the pose given by the measured actuator positions and hold it
  // until the first pose command arrives.
  for (size_t i = 0; i < joints_.size(); ++i) {
    const auto position = state_interfaces_[i].get_optional();
    if (!position.has_value() || !command_interfaces_[i].set_value(position.value())) {
      RCLCPP_ERROR(get_node()->get_logger(), "Cannot access joint '%s'", joints_[i].c_str());
      return CallbackReturn::ERROR;
    }
    q_cmd_[static_cast<Eigen::Index>(i)] = position.value();
  }
  current_pose_ = kinematics_.home();
  if (!kinematics_.forward(q_cmd_, current_pose_)) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "The actuator positions do not correspond to an assembled robot (forward kinematics "
      "did not converge)");
    return CallbackReturn::ERROR;
  }
  target_pose_ = current_pose_;
  pose_cmd_buffer_.reset();
  last_pose_cmd_.reset();
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianPoseController::on_deactivate(const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

controller_interface::return_type CartesianPoseController::update(
  const rclcpp::Time &, const rclcpp::Duration & period)
{
  const auto pose_cmd = *pose_cmd_buffer_.readFromRT();
  if (pose_cmd && pose_cmd != last_pose_cmd_) {
    last_pose_cmd_ = pose_cmd;
    Pose x;
    for (int k = 0; k < 9; ++k) {
      x[k] = pose_cmd->data[static_cast<size_t>(k)];
    }
    switch (kinematics_.inverse(x, q_cmd_)) {
      case IkStatus::OK:
        target_pose_ = x;
        break;
      case IkStatus::NO_REAL_SOLUTION:
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "Pose rejected: the distal links cannot reach the platforms");
        break;
      case IkStatus::STROKE_EXCEEDED:
        RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), 1000,
          "Pose rejected: actuator stroke (+/- %.3f m) exceeded", kinematics_.stroke());
        break;
    }
  }

  // Interpolate the pose (not the actuators): every intermediate command is then
  // an inverse-kinematics solution, i.e. compatible with the closed chains.
  const Pose delta = target_pose_ - current_pose_;
  const double dt = period.seconds();
  double scale = 1.0;
  const double linear = delta.head<3>().norm();
  const double angular = delta.tail<6>().cwiseAbs().maxCoeff();
  if (linear > max_linear_velocity_ * dt) {
    scale = std::min(scale, max_linear_velocity_ * dt / linear);
  }
  if (angular > max_angular_velocity_ * dt) {
    scale = std::min(scale, max_angular_velocity_ * dt / angular);
  }
  const Pose next = current_pose_ + scale * delta;
  if (kinematics_.inverse(next, q_cmd_) != IkStatus::OK) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000,
      "Path to the target leaves the workspace: stopping");
    target_pose_ = current_pose_;
    return controller_interface::return_type::OK;
  }
  current_pose_ = next;

  for (size_t i = 0; i < command_interfaces_.size(); ++i) {
    if (!command_interfaces_[i].set_value(q_cmd_[static_cast<Eigen::Index>(i)])) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "Could not write the command of joint '%s'", joints_[i].c_str());
    }
  }
  return controller_interface::return_type::OK;
}

}  // namespace ninedof_controllers

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  ninedof_controllers::CartesianPoseController, controller_interface::ControllerInterface)
