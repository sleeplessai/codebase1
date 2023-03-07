cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(learnglcn)

include_directories(${CMAKE_SOURCE_DIR}/include)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    find_package(OpenGL REQUIRED)
    set(link_libs ${link_libs} OpenGL)
endif()

find_package(glfw3 REQUIRED)
set(link_libs ${link_libs} glfw)

add_library(Glad lib/glad.c)
set(link_libs ${link_libs} Glad)

add_library(StbImage lib/stb_image.c)
set(link_libs ${link_libs} StbImage)

find_package(fmt REQUIRED)
set(link_libs ${link_libs} fmt)

file(GLOB source_glob "src/*")
set(chapter_collection "")
foreach(item ${source_glob})
    if(IS_DIRECTORY ${item})
        list(APPEND chapter_collection ${item})
        # message(${item})
    endif()
endforeach()

# list(LENGTH chapter_collection n_chapters)
# message(${n_chapters})

foreach(chapter ${chapter_collection})
    file(GLOB cpp_source_collection "${chapter}/*.cpp")
    foreach(cpp_source ${cpp_source_collection})
        # compile binaries
        get_filename_component(target_dir ${chapter} NAME)
        get_filename_component(target ${cpp_source} NAME_WLE)
        add_executable(${target} ${cpp_source})
        target_link_libraries(${target} ${link_libs})
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${target_dir}")

        # copy shaders and assets
        file(GLOB shader_collection "${chapter}/glsl/*.fs" "${chapter}/glsl/*.vs" "${chapter}/glsl/*.gs" "${chapter}/glsl/*.cs")
        file(COPY ${shader_collection} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${target_dir}/glsl/)
        file(GLOB_RECURSE assets_collection "${chapter}/assets/*")
        file(COPY ${assets_collection} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${target_dir}/assets/)
    endforeach()
endforeach()
