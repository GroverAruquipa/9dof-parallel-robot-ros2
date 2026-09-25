cmake_minimum_required(VERSION 3.16)
project(ninedof_controllers LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  ament_index_cpp
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  yaml_cpp_vendor
)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
foreach(dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${dependency} REQUIRED)
endforeach()
find_package(yaml-cpp REQUIRED)

add_library(ninedof_controllers SHARED
  src/kinematics.cpp
  src/cartesian_pose_controller.cpp
)
target_compile_features(ninedof_controllers PUBLIC cxx_std_17)
target_include_directories(ninedof_controllers PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(ninedof_controllers PUBLIC
  Eigen3::Eigen
  yaml-cpp
  ament_index_cpp::ament_index_cpp
  controller_interface::controller_interface
  hardware_interface::hardware_interface
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  realtime_tools::realtime_tools
  ${std_msgs_TARGETS}
)
pluginlib_export_plugin_description_file(controller_interface ninedof_controllers.xml)

if(BUILD_TESTING)
  find_package(ament_cmake_gmock REQUIRED)
  find_package(controller_manager REQUIRED)
  find_package(ros2_control_test_assets REQUIRED)

  ament_add_gmock(test_kinematics test/test_kinematics.cpp)
  target_link_libraries(test_kinematics ninedof_controllers ament_index_cpp::ament_index_cpp)

  ament_add_gmock(test_cartesian_pose_controller test/test_cartesian_pose_controller.cpp)
  target_link_libraries(test_cartesian_pose_controller
    ninedof_controllers
    ament_index_cpp::ament_index_cpp
    controller_manager::controller_manager
    ros2_control_test_assets::ros2_control_test_assets
  )
endif()

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(TARGETS ninedof_controllers
  EXPORT export_ninedof_controllers
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_ninedof_controllers HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS} eigen3_cmake_module Eigen3)
ament_package()
