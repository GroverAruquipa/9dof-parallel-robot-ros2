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

#include <gmock/gmock.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "ninedof_controllers/cartesian_pose_controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using controller_interface::CallbackReturn;
using hardware_interface::CommandInterface;
using hardware_interface::LoanedCommandInterface;
using hardware_interface::LoanedStateInterface;
using hardware_interface::StateInterface;

namespace
{

constexpr size_t kLegs = 9;
constexpr double kDt = 0.01;

class CartesianPoseControllerTest : public ::testing::Test
{
protected:
  static void SetUpTestCase() {rclcpp::init(0, nullptr);}
  static void TearDownTestCase() {rclcpp::shutdown();}

  void SetUp() override
  {
    for (size_t i = 0; i < kLegs; ++i) {
      joints_.push_back("leg" + std::to_string(i + 1) + "_actuator_joint");
    }
    controller_ = std::make_unique<ninedof_controllers::CartesianPoseController>();
    ASSERT_EQ(
      controller_->init("cartesian_pose_controller", "", 100, "", controller_->define_custom_node_options()),
      controller_interface::return_type::OK);
    auto node = controller_->get_node();
    node->set_parameter({"joints", joints_});
    node->set_parameter(
      {"geometry_file", ament_index_cpp::get_package_share_directory("ninedof_description") +
        "/config/geometry.yaml"});
    node->set_parameter({"max_linear_velocity", 0.02});
    node->set_parameter({"max_angular_velocity", 0.5});
  }

  void assign_interfaces()
  {
    std::vector<LoanedCommandInterface> commands;
    std::vector<LoanedStateInterface> states;
    for (size_t i = 0; i < kLegs; ++i) {
      command_handles_.push_back(std::make_shared<CommandInterface>(
          joints_[i], hardware_interface::HW_IF_POSITION, &command_values_[i]));
      state_handles_.push_back(std::make_shared<StateInterface>(
          joints_[i], hardware_interface::HW_IF_POSITION, &state_values_[i]));
      commands.emplace_back(command_handles_.back(), nullptr);
      states.emplace_back(state_handles_.back(), nullptr);
    }
    controller_->assign_interfaces(std::move(commands), std::move(states));
  }

  void configure_and_activate()
  {
    ASSERT_EQ(controller_->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
    assign_interfaces();
    ASSERT_EQ(controller_->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  }

  void publish_pose(const std::vector<double> & pose)
  {
    auto node = std::make_shared<rclcpp::Node>("test_pose_publisher");
    auto pub = node->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/cartesian_pose_controller/pose_cmd", rclcpp::SystemDefaultsQoS());
    // Wait for the controller subscription to be matched.
    for (int i = 0; i < 100 && pub->get_subscription_count() == 0; ++i) {
      rclcpp::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_GT(pub->get_subscription_count(), 0u);
    std_msgs::msg::Float64MultiArray msg;
    msg.data = pose;
    pub->publish(msg);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(controller_->get_node()->get_node_base_interface());
    for (int i = 0; i < 20; ++i) {
      executor.spin_some(std::chrono::milliseconds(10));
    }
  }

  void run_cycles(int n)
  {
    for (int i = 0; i < n; ++i) {
      ASSERT_EQ(
        controller_->update(rclcpp::Time(0), rclcpp::Duration::from_seconds(kDt)),
        controller_interface::return_type::OK);
    }
  }

  std::vector<std::string> joints_;
  std::array<double, kLegs> command_values_{};
  std::array<double, kLegs> state_values_{};
  std::vector<CommandInterface::SharedPtr> command_handles_;
  std::vector<StateInterface::SharedPtr> state_handles_;
  std::unique_ptr<ninedof_controllers::CartesianPoseController> controller_;
};

}  // namespace

TEST_F(CartesianPoseControllerTest, ConfiguresWithMeasuredGeometry)
{
  ASSERT_EQ(controller_->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(controller_->command_interface_configuration().names.size(), kLegs);
  EXPECT_EQ(controller_->command_interface_configuration().names[0], "leg1_actuator_joint/position");
}

TEST_F(CartesianPoseControllerTest, ResolvesPackageUrl)
{
  controller_->get_node()->set_parameter(
    {"geometry_file", "package://ninedof_description/config/geometry.yaml"});
  EXPECT_EQ(controller_->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
}

TEST_F(CartesianPoseControllerTest, RejectsJointsInWrongOrder)
{
  std::swap(joints_[0], joints_[1]);
  controller_->get_node()->set_parameter({"joints", joints_});
  EXPECT_EQ(controller_->on_configure(rclcpp_lifecycle::State()), CallbackReturn::ERROR);
}

TEST_F(CartesianPoseControllerTest, RejectsMissingGeometry)
{
  controller_->get_node()->set_parameter({"geometry_file", "/does/not/exist.yaml"});
  EXPECT_EQ(controller_->on_configure(rclcpp_lifecycle::State()), CallbackReturn::ERROR);
}

TEST_F(CartesianPoseControllerTest, HoldsPositionWithoutCommand)
{
  state_values_.fill(0.004);  // platforms raised by 4 mm
  configure_and_activate();
  run_cycles(10);
  for (double v : command_values_) {
    EXPECT_NEAR(v, 0.004, 1e-9);
  }
}

TEST_F(CartesianPoseControllerTest, TracksPoseThroughInverseKinematics)
{
  configure_and_activate();
  // Tilt platform 1 by 0.1 rad about X and rotate platform 2 by 0.2 rad about Y.
  const std::vector<double> pose{0.0, 0.0, 0.14668, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0};
  publish_pose(pose);

  ninedof_controllers::Kinematics kin;
  kin.load(
    ament_index_cpp::get_package_share_directory("ninedof_description") + "/config/geometry.yaml");
  ninedof_controllers::Pose x;
  for (int k = 0; k < 9; ++k) {x[k] = pose[static_cast<size_t>(k)];}
  Eigen::VectorXd q;
  ASSERT_EQ(kin.inverse(x, q), ninedof_controllers::IkStatus::OK);

  // Every intermediate command must be an assembled configuration of the robot.
  ninedof_controllers::Pose x_fk = kin.home();
  Eigen::VectorXd q_cmd(9);
  for (int cycle = 0; cycle < 100; ++cycle) {
    run_cycles(1);
    for (size_t i = 0; i < kLegs; ++i) {q_cmd[static_cast<Eigen::Index>(i)] = command_values_[i];}
    ASSERT_TRUE(kin.forward(q_cmd, x_fk)) << "cycle " << cycle;
  }
  // 0.2 rad at 0.5 rad/s takes 0.4 s: the target is reached after 100 cycles of 10 ms.
  for (size_t i = 0; i < kLegs; ++i) {
    EXPECT_NEAR(command_values_[i], q[static_cast<Eigen::Index>(i)], 1e-9) << joints_[i];
  }
  EXPECT_LT((x_fk - x).cwiseAbs().maxCoeff(), 1e-8);
}

TEST_F(CartesianPoseControllerTest, LimitsAngularVelocity)
{
  configure_and_activate();
  publish_pose({0.0, 0.0, 0.14668, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0});
  ninedof_controllers::Kinematics kin;
  kin.load(
    ament_index_cpp::get_package_share_directory("ninedof_description") + "/config/geometry.yaml");
  // The controller starts from the forward kinematics of q = 0 and, after one
  // cycle, has moved towards the target by 0.5 rad/s * 10 ms = 0.005 rad.
  ninedof_controllers::Pose start = kin.home();
  ASSERT_TRUE(kin.forward(Eigen::VectorXd::Zero(9), start));
  ninedof_controllers::Pose target = kin.home();
  target[7] = 0.2;
  const ninedof_controllers::Pose delta = target - start;
  const ninedof_controllers::Pose x = start + 0.005 / delta.tail<6>().cwiseAbs().maxCoeff() * delta;
  run_cycles(1);
  Eigen::VectorXd q_cmd(9), q_expected;
  for (size_t i = 0; i < kLegs; ++i) {q_cmd[static_cast<Eigen::Index>(i)] = command_values_[i];}
  ASSERT_EQ(kin.inverse(x, q_expected), ninedof_controllers::IkStatus::OK);
  EXPECT_LT((q_cmd - q_expected).cwiseAbs().maxCoeff(), 1e-9);
}

TEST_F(CartesianPoseControllerTest, IgnoresPoseOutsideStroke)
{
  configure_and_activate();
  publish_pose({0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  run_cycles(50);
  for (double v : command_values_) {
    EXPECT_NEAR(v, 0.0, 1e-9);
  }
}
