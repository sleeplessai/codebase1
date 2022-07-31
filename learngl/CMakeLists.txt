cmake_minimum_required(VERSION 3.12)

project(learnglcn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

set(gl-cxx
    src/window.cpp
    src/triangle.cpp
)

add_library(glad src/glad.c)
add_executable(window src/window.cpp)
add_executable(triangle src/triangle.cpp)

target_link_libraries(window glfw OpenGL glad)
target_link_libraries(triangle glfw OpenGL glad)

