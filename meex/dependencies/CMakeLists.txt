## Dependencies
find_package(Vulkan REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/sdl2/cmake)
find_package(SDL2 REQUIRED)

add_library(glm INTERFACE)
target_include_directories(glm INTERFACE glm)

add_library(stb_image STATIC stb_image/stb_image.h stb_image/stb_image.cpp)
target_include_directories(stb_image PUBLIC stb_image)

add_library(tinyobjloader STATIC tinyobjloader/tiny_obj_loader.h tinyobjloader/tiny_obj_loader.cc)
target_include_directories(tinyobjloader PUBLIC tinyobjloader)

add_library(vkbootstrap STATIC vk_bootstrap/VkBootstrap.h vk_bootstrap/VkBootstrap.cpp)
target_link_libraries(vkbootstrap PRIVATE Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)
target_include_directories(vkbootstrap PUBLIC vk_bootstrap)

add_library(vma STATIC vma/vk_mem_alloc.h vma/vk_mem_alloc.cpp)
target_include_directories(vma PUBLIC vma)

file(GLOB imgui_sources CONFIGURE_DEPENDS imgui_docking/*.h imgui_docking/*.cpp)
add_library(imgui STATIC ${imgui_sources})
target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL2::Core)
target_include_directories(imgui PUBLIC imgui_docking)

## Pack
add_library(MeexDependencies INTERFACE)
target_link_libraries(MeexDependencies INTERFACE
  Vulkan::Vulkan SDL2::Main
  glm stb_image tinyobjloader vkbootstrap vma imgui
)

