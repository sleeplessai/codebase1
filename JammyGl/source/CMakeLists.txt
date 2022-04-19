cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(JammyOpenGl VERSION 0.2)

find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
find_package(GLEW REQUIRED)

add_library(jammygl_utils OBJECT "utils.h" "utils.cpp")

add_executable(Triangle "Triangle.cpp")
target_link_libraries(Triangle PRIVATE GLEW::GLEW)
target_link_libraries(Triangle PRIVATE glfw)
target_link_libraries(Triangle PRIVATE jammygl_utils)

add_executable(Rectangle "Rectangle.cpp")
target_link_libraries(Rectangle PRIVATE GLEW::GLEW)
target_link_libraries(Rectangle PRIVATE glfw)
target_link_libraries(Rectangle PRIVATE jammygl_utils)

add_executable(Uniforms "Uniforms.cpp")
target_link_libraries(Uniforms PRIVATE glad::glad)
target_link_libraries(Uniforms PRIVATE glfw)
target_link_libraries(Uniforms PRIVATE jammygl_utils)

add_library(jammygl_renderer OBJECT "renderer.h" "renderer.cpp")
add_library(jammygl_vb OBJECT "vertex_buffer.h" "vertex_buffer.cpp" "vertex_buffer_layout.h")
add_library(jammygl_ib OBJECT "index_buffer.h" "index_buffer.cpp")
add_library(jammygl_va OBJECT "vertex_array.h" "vertex_array.cpp")
target_link_libraries(jammygl_renderer PRIVATE glad::glad)
target_link_libraries(jammygl_vb PRIVATE glad::glad)
target_link_libraries(jammygl_ib PRIVATE glad::glad)
target_link_libraries(jammygl_va PRIVATE glad::glad)

add_executable(Uniforms2 "Uniforms2.cpp")
target_link_libraries(Uniforms2 PRIVATE glad::glad glfw)
target_link_libraries(Uniforms2 PRIVATE jammygl_renderer jammygl_utils jammygl_va jammygl_vb jammygl_ib)
