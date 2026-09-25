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

#include "ninedof_controllers/kinematics.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace ninedof_controllers
{

void Kinematics::load(const std::string & geometry_file)
{
  YAML::Node root = YAML::LoadFile(geometry_file);
  YAML::Node g = root["ninedof"] ? root["ninedof"] : root;
  length_ = g["distal_length"].as<double>();
  stroke_ = g["actuator_stroke"].as<double>();
  home_ = Pose::Zero();
  home_[2] = g["home_z"].as<double>();

  legs_.clear();
  for (const auto & l : g["legs"]) {
    Leg leg;
    leg.name = l["name"].as<std::string>();
    leg.platform = l["platform"].as<int>();
    leg.b0 = {l["base_x"].as<double>(), l["base_y"].as<double>(), l["base_z0"].as<double>()};
    const auto a = l["a"];
    leg.a = {a[0].as<double>(), a[1].as<double>(), a[2].as<double>()};
    if (leg.platform != 1 && leg.platform != 2) {
      throw std::runtime_error("leg " + leg.name + ": platform must be 1 or 2");
    }
    legs_.push_back(leg);
  }
  if (legs_.size() != 9) {
    throw std::runtime_error("expected 9 legs in " + geometry_file);
  }
}

Eigen::Matrix3d Kinematics::rotXYZ(double a1, double a2, double a3)
{
  return (Eigen::AngleAxisd(a1, Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(a2, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(a3, Eigen::Vector3d::UnitZ())).toRotationMatrix();
}

IkStatus Kinematics::inverse(const Pose & x, Eigen::VectorXd & q) const
{
  const Eigen::Vector3d p = x.head<3>();
  const Eigen::Matrix3d Q1 = rotXYZ(x[3], x[4], x[5]);
  const Eigen::Matrix3d Q2 = rotXYZ(x[6], x[7], x[8]);
  q.resize(static_cast<Eigen::Index>(legs_.size()));
  IkStatus status = IkStatus::OK;
  for (size_t i = 0; i < legs_.size(); ++i) {
    const Leg & leg = legs_[i];
    // n_i = p - b_io + Q a_i,  e_i = +Z
    const Eigen::Vector3d n = p - leg.b0 + (leg.platform == 1 ? Q1 : Q2) * leg.a;
    const double ne = n.z();
    const double disc = ne * ne - (n.squaredNorm() - length_ * length_);
    if (disc < 0.0) {
      return IkStatus::NO_REAL_SOLUTION;
    }
    q[static_cast<Eigen::Index>(i)] = ne - std::sqrt(disc);
    if (std::abs(q[static_cast<Eigen::Index>(i)]) > stroke_) {
      status = IkStatus::STROKE_EXCEEDED;
    }
  }
  return status;
}

Eigen::VectorXd Kinematics::constraints(const Pose & x, const Eigen::VectorXd & q) const
{
  const Eigen::Vector3d p = x.head<3>();
  const Eigen::Matrix3d Q1 = rotXYZ(x[3], x[4], x[5]);
  const Eigen::Matrix3d Q2 = rotXYZ(x[6], x[7], x[8]);
  Eigen::VectorXd f(static_cast<Eigen::Index>(legs_.size()));
  for (size_t i = 0; i < legs_.size(); ++i) {
    const Leg & leg = legs_[i];
    const auto k = static_cast<Eigen::Index>(i);
    Eigen::Vector3d b = leg.b0;
    b.z() += q[k];
    const Eigen::Vector3d m = p + (leg.platform == 1 ? Q1 : Q2) * leg.a - b;
    f[k] = m.squaredNorm() - length_ * length_;
  }
  return f;
}

bool Kinematics::forward(const Eigen::VectorXd & q, Pose & x, int max_iter, double tol) const
{
  constexpr double h = 1e-7;
  for (int it = 0; it < max_iter; ++it) {
    const Eigen::VectorXd f = constraints(x, q);
    if (f.cwiseAbs().maxCoeff() < tol) {
      for (int k = 3; k < 9; ++k) {
        x[k] = std::atan2(std::sin(x[k]), std::cos(x[k]));  // angles in (-pi, pi]
      }
      return true;
    }
    Eigen::MatrixXd D(f.size(), 9);
    for (int k = 0; k < 9; ++k) {
      Pose xh = x;
      xh[k] += h;
      D.col(k) = (constraints(xh, q) - f) / h;
    }
    x -= D.completeOrthogonalDecomposition().solve(f);
  }
  return false;
}

}  // namespace ninedof_controllers
