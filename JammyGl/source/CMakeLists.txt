cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(JammyOpenGl VERSION 0.3.1)

find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

add_executable(Triangle "Triangle.cpp" "deprecated_utils.h" "deprecated_utils.cpp")
target_link_libraries(Triangle PRIVATE GLEW::GLEW)
target_link_libraries(Triangle PRIVATE glfw)

add_executable(Rectangle "Rectangle.cpp" "deprecated_utils.h" "deprecated_utils.cpp")
target_link_libraries(Rectangle PRIVATE GLEW::GLEW)
target_link_libraries(Rectangle PRIVATE glfw)

add_executable(Uniforms "Uniforms.cpp" "deprecated_utils.h" "deprecated_utils.cpp")
target_link_libraries(Uniforms PRIVATE glad::glad)
target_link_libraries(Uniforms PRIVATE glfw)

add_library(jammygl_vb OBJECT "vertex_buffer.h" "vertex_buffer.cpp" "vertex_buffer_layout.h" "gl_helper.h")
target_link_libraries(jammygl_vb PRIVATE glad::glad)

add_library(jammygl_ib OBJECT "index_buffer.h" "index_buffer.cpp" "gl_helper.h")
target_link_libraries(jammygl_ib PRIVATE glad::glad)

add_library(jammygl_va OBJECT "vertex_array.h" "vertex_array.cpp" "gl_helper.h")
target_link_libraries(jammygl_va PRIVATE glad::glad)

add_library(jammygl_shader OBJECT "shader.h" "shader.cpp" "gl_helper.h")
target_link_libraries(jammygl_shader PRIVATE glad::glad)

add_library(jammygl_renderer STATIC "renderer.h" "renderer.cpp" "gl_helper.h")
target_link_libraries(jammygl_renderer PRIVATE glad::glad)
target_link_libraries(jammygl_renderer PRIVATE jammygl_ib jammygl_shader jammygl_va)

add_executable(Uniforms2 "Uniforms2.cpp" "gl_helper.h")
target_link_libraries(Uniforms2 PRIVATE glad::glad glfw)
target_link_libraries(Uniforms2 PRIVATE jammygl_renderer jammygl_vb)

find_path(STB_INCLUDE_DIRS "stb_image.h")
add_library(jammygl_texture OBJECT "texture.h" "texture.cpp" "gl_helper.h")
target_include_directories(jammygl_texture PRIVATE ${STB_INCLUDE_DIRS})
target_link_libraries(jammygl_texture PRIVATE jammygl_renderer)

add_executable(Pentagon "Pentagon.cpp" "gl_helper.h")
target_link_libraries(Pentagon PRIVATE glad::glad glfw)
target_link_libraries(Pentagon PRIVATE jammygl_texture jammygl_renderer jammygl_vb)
