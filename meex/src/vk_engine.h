// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <vk_types.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

class VulkanEngine {
public:
  VulkanEngine() = default;
  VulkanEngine(VulkanEngine&) = delete;
  VulkanEngine& operator=(VulkanEngine&) = delete;

  bool _isInitialized{false};
  int _frameNumber{0};

  VkExtent2D _windowExtent{1600, 900};
  struct SDL_Window* _window{nullptr};

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

  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_default_renderpass();
  void init_framebuffers();
  void init_sync_structures();
};

