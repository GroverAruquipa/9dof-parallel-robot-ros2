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

// mujoco_ros2_control plugin for the pick-and-place demo:
//  * /mujoco/qpos         std_msgs/Float64MultiArray  [sim time, qpos...] at ~60 Hz
//  * /mujoco/object_pose  geometry_msgs/PoseStamped   pose of body "block" (world)
//  * /mujoco/grasp        std_msgs/Bool (subscribed)  virtual grasp on/off: while on, a
//    stiff spring-damper (plus gravity compensation) holds the block at the pose
//    relative to platform 1 it had when the grasp was switched on, and the reaction
//    acts on platform 1. mujoco_ros2_control hands plugins a copy of mjData whose
//    external forces (xfrc_applied) reach the physics, so the grasp is a wrench and
//    not a weld constraint. Same law as ninedof_mujoco/sim.py (VirtualGrasp).

#include <atomic>
#include <memory>

#include <mujoco/mujoco.h>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace ninedof_mujoco
{

class SimStatePublisher : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData * /*data*/) override
  {
    node_ = node;
    object_id_ = mj_name2id(model, mjOBJ_BODY, "block");
    holder_id_ = mj_name2id(model, mjOBJ_BODY, "platform_1");

    qpos_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>("/mujoco/qpos", 100);
    pose_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("/mujoco/object_pose", 10);
    grasp_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
      "/mujoco/grasp", 10, [this](const std_msgs::msg::Bool::SharedPtr msg) {
        requested_grasp_.store(msg->data ? 1 : 0);
      });
    RCLCPP_INFO(
      node_->get_logger(), "SimStatePublisher: object id %d, holder id %d", object_id_,
      holder_id_);
    return true;
  }

  void update(const mjModel * model, mjData * data) override
  {
    if (object_id_ >= 0 && holder_id_ >= 0) {
      apply_grasp(model, data);
    }
    // update() runs at the ros2_control rate: publish by simulation time.
    if (data->time - last_publish_ < kPublishPeriod) {
      return;
    }
    last_publish_ = data->time;
    std_msgs::msg::Float64MultiArray qpos;
    qpos.data.reserve(static_cast<size_t>(model->nq) + 1);
    qpos.data.push_back(data->time);
    qpos.data.insert(qpos.data.end(), data->qpos, data->qpos + model->nq);
    qpos_pub_->publish(qpos);

    if (object_id_ >= 0) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = "world";
      pose.header.stamp = rclcpp::Time(static_cast<int64_t>(data->time * 1e9));
      const mjtNum * p = data->xpos + 3 * object_id_;
      const mjtNum * q = data->xquat + 4 * object_id_;
      pose.pose.position.x = p[0];
      pose.pose.position.y = p[1];
      pose.pose.position.z = p[2];
      pose.pose.orientation.w = q[0];
      pose.pose.orientation.x = q[1];
      pose.pose.orientation.y = q[2];
      pose.pose.orientation.z = q[3];
      pose_pub_->publish(pose);
    }
  }

  void cleanup() override
  {
    grasp_sub_.reset();
    qpos_pub_.reset();
    pose_pub_.reset();
  }

private:
  void apply_grasp(const mjModel * model, mjData * data)
  {
    const int request = requested_grasp_.exchange(-1);
    if (request >= 0) {
      grasping_ = request == 1;
      if (grasping_) {
        // Pose of the object in the holder frame at the moment of grasping.
        mjtNum q_inv[4], dp[3];
        mju_negQuat(q_inv, data->xquat + 4 * holder_id_);
        mju_sub3(dp, data->xpos + 3 * object_id_, data->xpos + 3 * holder_id_);
        mju_rotVecQuat(rel_pos_, dp, q_inv);
        mju_mulQuat(rel_quat_, q_inv, data->xquat + 4 * object_id_);
      }
      RCLCPP_INFO(node_->get_logger(), "Grasp %s", grasping_ ? "attached" : "released");
    }
    mjtNum * fo = data->xfrc_applied + 6 * object_id_;
    mjtNum * fh = data->xfrc_applied + 6 * holder_id_;
    mju_zero(fo, 6);
    mju_zero(fh, 6);
    if (!grasping_) {
      return;
    }
    // Target pose of the object, attached to the holder.
    const mjtNum * ph = data->xpos + 3 * holder_id_;
    const mjtNum * po = data->xpos + 3 * object_id_;
    mjtNum p_target[3], q_target[4];
    mju_rotVecQuat(p_target, rel_pos_, data->xquat + 4 * holder_id_);
    mju_addTo3(p_target, ph);
    mju_mulQuat(q_target, data->xquat + 4 * holder_id_, rel_quat_);
    // Body velocities [angular; linear] in world orientation, at the body frames.
    mjtNum vo[6], vh[6];
    mj_objectVelocity(model, data, mjOBJ_BODY, object_id_, vo, 0);
    mj_objectVelocity(model, data, mjOBJ_BODY, holder_id_, vh, 0);
    mjtNum r[3], wxr[3], v_target[3];
    mju_sub3(r, p_target, ph);
    mju_cross(wxr, vh, r);
    mju_add3(v_target, vh + 3, wxr);

    mjtNum force[3], torque[3];
    for (int i = 0; i < 3; ++i) {
      force[i] = kLinStiffness * (p_target[i] - po[i]) + kLinDamping * (v_target[i] - vo[3 + i]);
    }
    force[2] -= model->body_subtreemass[object_id_] * model->opt.gravity[2];
    mjtNum q_obj_inv[4], q_err[4], err[3];
    mju_negQuat(q_obj_inv, data->xquat + 4 * object_id_);
    mju_mulQuat(q_err, q_target, q_obj_inv);
    mju_quat2Vel(err, q_err, 1.0);
    for (int i = 0; i < 3; ++i) {
      torque[i] = kRotStiffness * err[i] + kRotDamping * (vh[i] - vo[i]);
    }
    // xfrc_applied acts at the body centre of mass.
    mjtNum lever_o[3], extra_o[3];
    mju_sub3(lever_o, po, data->xipos + 3 * object_id_);
    mju_cross(extra_o, lever_o, force);
    mju_copy3(fo, force);
    mju_add3(fo + 3, torque, extra_o);
    // Reaction on the holder: -F applied at the object frame origin.
    mjtNum lever_h[3], moment[3];
    mju_sub3(lever_h, po, data->xipos + 3 * holder_id_);
    mju_cross(moment, lever_h, force);
    for (int i = 0; i < 3; ++i) {
      fh[i] = -force[i];
      fh[3 + i] = -(torque[i] + moment[i]);
    }
  }

  // Grasp spring-damper for a block of a few grams. The wrench is updated at
  // the ros2_control rate (100 Hz) and held in between, so the natural
  // frequency is kept at ~30 rad/s with critical damping.
  static constexpr mjtNum kLinStiffness = 4.5;     // N/m
  static constexpr mjtNum kLinDamping = 0.3;       // N s/m
  static constexpr mjtNum kRotStiffness = 2.7e-4;  // N m/rad
  static constexpr mjtNum kRotDamping = 1.8e-5;    // N m s/rad
  static constexpr double kPublishPeriod = 1.0 / 60.0;  // s

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr qpos_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr grasp_sub_;
  std::atomic<int> requested_grasp_{-1};
  int object_id_ = -1;
  int holder_id_ = -1;
  bool grasping_ = false;
  mjtNum rel_pos_[3] = {0, 0, 0};
  mjtNum rel_quat_[4] = {1, 0, 0, 0};
  double last_publish_ = -1.0;
};

}  // namespace ninedof_mujoco

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  ninedof_mujoco::SimStatePublisher, mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
