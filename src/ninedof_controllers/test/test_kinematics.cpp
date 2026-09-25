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

#include <string>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "ninedof_controllers/kinematics.hpp"

using ninedof_controllers::IkStatus;
using ninedof_controllers::Kinematics;
using ninedof_controllers::Pose;

namespace
{

// Pose and actuator displacements measured on the exported SolidWorks assembly
// (same data as ninedof_kinematics/test/test_kinematics.py).
Pose measured_pose()
{
  Eigen::Matrix3d Q1, Q2;
  Q1 << 0.981671, -0.132925, -0.136573,
    0.115053, 0.984637, -0.131346,
    0.151935, 0.113226, 0.981884;
  Q2 << 0.955397, -0.243797, 0.166671,
    0.293829, 0.841423, -0.45351,
    -0.029676, 0.482255, 0.875528;
  auto xyz = [](const Eigen::Matrix3d & R) {
      return Eigen::Vector3d(
        std::atan2(-R(1, 2), R(2, 2)), std::asin(R(0, 2)), std::atan2(-R(0, 1), R(0, 0)));
    };
  Pose x;
  x << -0.000184, -0.017543, 0.150456, xyz(Q1), xyz(Q2);
  return x;
}

const double kMeasuredQ[9] = {0.010516, 0.005981, 0.001667, -0.002192, -0.004652,
  0.001669, 0.012577, -0.012929, -0.008713};

class KinematicsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    kin_.load(
      ament_index_cpp::get_package_share_directory("ninedof_description") +
      "/config/geometry.yaml");
  }
  Kinematics kin_;
};

}  // namespace

TEST_F(KinematicsTest, LoadsNineLegs)
{
  ASSERT_EQ(kin_.legs().size(), 9u);
  EXPECT_EQ(kin_.legs()[0].platform, 1);
  EXPECT_EQ(kin_.legs()[8].platform, 2);
}

TEST_F(KinematicsTest, HomeIsZero)
{
  Eigen::VectorXd q;
  ASSERT_EQ(kin_.inverse(kin_.home(), q), IkStatus::OK);
  EXPECT_LT(q.cwiseAbs().maxCoeff(), 1e-3);
}

TEST_F(KinematicsTest, InverseMatchesCadAssembly)
{
  Eigen::VectorXd q;
  ASSERT_EQ(kin_.inverse(measured_pose(), q), IkStatus::OK);
  for (int i = 0; i < 9; ++i) {
    EXPECT_NEAR(q[i], kMeasuredQ[i], 5e-4) << "leg " << i + 1;
  }
}

TEST_F(KinematicsTest, ForwardInvertsInverse)
{
  Pose x = kin_.home();
  x << 0.004, -0.003, 0.15, 0.1, -0.15, 0.05, -0.1, 0.12, 0.2;
  Eigen::VectorXd q;
  ASSERT_EQ(kin_.inverse(x, q), IkStatus::OK);
  Pose x_fk = kin_.home();
  ASSERT_TRUE(kin_.forward(q, x_fk));
  EXPECT_LT((x_fk - x).cwiseAbs().maxCoeff(), 1e-8);
}

TEST_F(KinematicsTest, RejectsUnreachableAndOutOfStroke)
{
  Eigen::VectorXd q;
  Pose far = kin_.home();
  far[0] += 0.2;
  EXPECT_EQ(kin_.inverse(far, q), IkStatus::NO_REAL_SOLUTION);
  Pose high = kin_.home();
  high[2] += 0.03;
  EXPECT_EQ(kin_.inverse(high, q), IkStatus::STROKE_EXCEEDED);
}

TEST(KinematicsLoad, ThrowsOnMissingFile)
{
  Kinematics kin;
  EXPECT_ANY_THROW(kin.load("/does/not/exist.yaml"));
}
