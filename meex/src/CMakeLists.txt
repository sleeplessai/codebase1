
# Add source to this project's executable.

set(MeexExecutable meex)

add_executable(${MeexExecutable}
  main.cpp
  vk_engine.cpp
  vk_engine.h
  vk_types.h
  vk_initializers.cpp
  vk_initializers.h
  vk_pipeline.cpp
  vk_pipeline.h
)


set_property(TARGET ${MeexExecutable} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${MeexExecutable}>")

target_include_directories(${MeexExecutable} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${MeexExecutable} vkbootstrap vma glm tinyobjloader imgui stb_image)

target_link_libraries(${MeexExecutable} Vulkan::Vulkan sdl2)

add_dependencies(${MeexExecutable} Shaders)
