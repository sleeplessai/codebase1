cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(basic_triangle VERSION 0.1)

find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

add_executable(BasicTriangle "BasicTriangle.cpp" "utils.h" "utils.cpp")
add_executable(Rectangle "Rectangle.cpp" "utils.h" "utils.cpp")

target_link_libraries(BasicTriangle PRIVATE GLEW::GLEW)
target_link_libraries(BasicTriangle PRIVATE glfw)

target_link_libraries(Rectangle PRIVATE GLEW::GLEW)
target_link_libraries(Rectangle PRIVATE glfw)
