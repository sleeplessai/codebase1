// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_scene.h>
#include <vk_types.h>

#include <functional>
#include <glm/glm.hpp>
#include <queue>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>
#include <SDL.h>


class Engine {
public:
  Engine() = default;
  Engine(Engine&) = delete;
  Engine& operator=(Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;

  bool is_initialized{false};
  uint32_t frame_count{0};

  VkExtent2D window_extent{1600, 900};
  SDL_Window* window{nullptr};

  //initializes everything in the engine
  void init();
  //shuts down the engine
  void cleanup();
  //draw loop
  void draw();
  //run main loop
  void run();

private:
  // vk core structure
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messenger;
  VkPhysicalDevice _physical_device;
  VkDevice _device;
  VkSurfaceKHR _surface;

  // vk swapchain presentation
  VkSwapchainKHR _swapchain;
  VkFormat _swapchain_image_format;
  std::vector<VkImage> _swapchain_images;
  std::vector<VkImageView> _swapchain_image_views;

  // vk queues and commands
  VkQueue _graphics_queue;
  uint32_t _graphics_queue_family;

  VkCommandPool _command_pool;
  VkCommandBuffer _main_command_buffer;

  VkRenderPass _render_pass;
  std::vector<VkFramebuffer> _framebuffers;

  VkSemaphore _present_semaphore, _render_semaphore;
  VkFence _render_fence;

  // vk pipelines, meshes, depth maps
  VkPipelineLayout _triangle_pipeline_layout;
  VkPipeline _triangle_pipeline;

  VmaAllocator _allocator;
  VkPipelineLayout _mesh_pipeline_layout;
  VkPipeline _mesh_pipeline;

  Scene _scene;

  VkImageView _depth_image_view;
  AllocatedImage _depth_image;
  VkFormat _depth_format;

  struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 render_matrix;
  };

  struct DestroyQueue {
    void push_back(std::function<void()>&& func) {
      __dq.push(func);
    }
    void flush() {
      while (!__dq.empty()) {
        __dq.front()();
        __dq.pop();
      }
    }
  private: std::queue<std::function<void()>> __dq{};
  } _destroy_queue{};

  //void init_imgui();
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_default_renderpass();
  void init_framebuffers();
  void init_sync_structures();
  void init_pipelines();
  void init_scene();
  void draw_objects(VkCommandBuffer cmd, Scene const& scene);

  bool load_shader_module(const char* file_path, VkShaderModule* module);
  void load_meshes();
  void upload_meshes(Mesh& mesh);
};

