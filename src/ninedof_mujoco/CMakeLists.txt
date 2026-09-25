cmake_minimum_required(VERSION 3.16)
project(ninedof_mujoco LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)
find_package(mujoco_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(ninedof_mujoco_plugins SHARED src/sim_state_publisher.cpp)
target_compile_features(ninedof_mujoco_plugins PUBLIC cxx_std_17)
target_link_libraries(ninedof_mujoco_plugins PUBLIC
  mujoco_ros2_control_plugins::mujoco_ros2_control_plugins
  pluginlib::pluginlib
  rclcpp::rclcpp
  ${geometry_msgs_TARGETS}
  ${std_msgs_TARGETS}
)
if(TARGET mujoco::mujoco)
  target_link_libraries(ninedof_mujoco_plugins PUBLIC mujoco::mujoco)
endif()
pluginlib_export_plugin_description_file(mujoco_ros2_control_plugins plugins.xml)

install(TARGETS ninedof_mujoco_plugins
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/pick_place_demo scripts/render_video scripts/showcase_video
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_pick_place test/test_pick_place.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_package()
