cmake_minimum_required(VERSION 3.12)

project(learnglcn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(OpenGL REQUIRED)
set(link_libs ${link_libs} OpenGL)

find_package(glfw3 REQUIRED)
set(link_libs ${link_libs} glfw)

add_library(Glad src/glad.c)
set(link_libs ${link_libs} Glad)

file(GLOB cpp_source_collection "src/*.cpp")
foreach(cpp_source ${cpp_source_collection})
    set(target ${cpp_source})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" target ${target})
    add_executable(${target} ${cpp_source})
    target_link_libraries(${target} ${link_libs})
endforeach()

file(GLOB shader_collection "glsl/*.fs" "glsl/*.vs" "glsl/*.gs" "glsl/*.cs")
file(COPY ${shader_collection} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/glsl/)
