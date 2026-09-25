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

#ifndef NINEDOF_CONTROLLERS__KINEMATICS_HPP_
#define NINEDOF_CONTROLLERS__KINEMATICS_HPP_

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace ninedof_controllers
{

/// Pose x = [px, py, pz, alpha1, alpha2, alpha3, beta1, beta2, beta3] with
/// Q1 = Qx(alpha1) Qy(alpha2) Qz(alpha3) and Q2 = Qx(beta1) Qy(beta2) Qz(beta3).
using Pose = Eigen::Matrix<double, 9, 1>;

struct Leg
{
  std::string name;
  int platform;        // 1 or 2
  Eigen::Vector3d b0;  // position of B_i when q_i = 0
  Eigen::Vector3d a;   // position of A_i in the platform frame
};

enum class IkStatus { OK, NO_REAL_SOLUTION, STROKE_EXCEEDED };

/// Kinematics of the 9-DoF 5PSS-S-4PSS parallel robot (mirror of
/// ninedof_kinematics/kinematics.py). The actuators move along +Z.
class Kinematics
{
public:
  /// Load config/geometry.yaml of ninedof_description. Throws on error.
  void load(const std::string & geometry_file);

  /// Actuator displacements for the pose x (lower branch of Eq. 18).
  IkStatus inverse(const Pose & x, Eigen::VectorXd & q) const;

  /// |A_i - B_i|^2 - l^2 for every leg.
  Eigen::VectorXd constraints(const Pose & x, const Eigen::VectorXd & q) const;

  /// Gauss-Newton forward kinematics starting from x (updated in place).
  bool forward(const Eigen::VectorXd & q, Pose & x, int max_iter = 50, double tol = 1e-12) const;

  static Eigen::Matrix3d rotXYZ(double a1, double a2, double a3);

  const std::vector<Leg> & legs() const {return legs_;}
  double stroke() const {return stroke_;}
  Pose home() const {return home_;}

private:
  std::vector<Leg> legs_;
  double length_ = 0.0;
  double stroke_ = 0.0;
  Pose home_ = Pose::Zero();
};

}  // namespace ninedof_controllers

#endif  // NINEDOF_CONTROLLERS__KINEMATICS_HPP_
