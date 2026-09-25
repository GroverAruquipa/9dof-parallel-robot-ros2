cmake_minimum_required(VERSION 3.16)
project(ninedof_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config meshes mujoco urdf DESTINATION share/${PROJECT_NAME})

ament_package()
