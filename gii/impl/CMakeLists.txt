file(GLOB_RECURSE main_srcs CONFIGURE_DEPENDS *.cpp *.h)
set(IsolatedExec isolated)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    find_package(OpenGL REQUIRED)
    set(link_libs ${link_libs} OpenGL)
endif()

find_package(glfw3 REQUIRED)
find_package(fmt REQUIRED)

set(link_libs ${link_libs} glad glfw fmt imgui_docking)

add_executable(${IsolatedExec} ${main_srcs})
set_target_properties(${IsolatedExec} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(${IsolatedExec} ${link_libs})
