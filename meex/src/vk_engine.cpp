#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <cmath>
#include <regex>
#include <vk_types.h>
#include <vk_initializers.h>

#include "VkBootstrap.h"
#include <iostream>
#include <vulkan/vulkan_core.h>


#define VK_CHECK(x)                                          \
  do {                                                       \
    VkResult err = x;                                        \
    if (err) {                                               \
      std::cerr << "Detected Vulkan error: " << err << '\n'; \
      std::exit(err);                                        \
    }                                                        \
  } while (0)


void VulkanEngine::init() {
  // We initialize SDL and create a window with it. 
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  _window = SDL_CreateWindow(
    "Vulkan Engine",
    SDL_WINDOWPOS_UNDEFINED,
    SDL_WINDOWPOS_UNDEFINED,
    _windowExtent.width,
    _windowExtent.height,
    window_flags
  );

  // load and init vulkan core structures
  init_vulkan();
  // init swapchainmany images we h
  init_swapchain();
  // query queues and init commands
  init_commands();
  // init renderpass and framebuffer
  init_default_renderpass();
  init_framebuffers();
  // init signals
  init_sync_structures();

  //everything went fine
  _isInitialized = true;
}

void VulkanEngine::cleanup() {
  if (_isInitialized) {
    //vkDestroySemaphore(_device, _present_semaphore, nullptr);
    //vkDestroySemaphore(_device, _render_semaphore, nullptr);
    //vkDestroyFence(_device, _render_fence, nullptr);

    vkDestroyRenderPass(_device, _render_pass, nullptr);
    vkDestroyCommandPool(_device, _command_pool, nullptr);
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    for (size_t i = 0; i < _swapchain_image_views.size(); ++i) {
      vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
      vkDestroyImageView(_device, _swapchain_image_views[i], nullptr);
    }

    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
    vkDestroyInstance(_instance, nullptr);

    SDL_DestroyWindow(_window);
  }
}

void VulkanEngine::draw() {
  //wait until the GPU has finished rendering the last frame. Timeout of 1 second
  VK_CHECK(vkWaitForFences(_device, 1, &_render_fence, true, 1000000000));
  VK_CHECK(vkResetFences(_device, 1, &_render_fence));

  uint32_t swapchain_image_index;
  VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, 1000000000, _present_semaphore, nullptr, &swapchain_image_index));
  VK_CHECK(vkResetCommandBuffer(_main_command_buffer, 0));

  VkCommandBuffer cmd = _main_command_buffer;
  VkCommandBufferBeginInfo cmd_begin_info = {};
  cmd_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cmd_begin_info.pNext = nullptr;
  cmd_begin_info.pInheritanceInfo = nullptr;
  cmd_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VK_CHECK(vkBeginCommandBuffer(cmd, &cmd_begin_info));

  VkClearValue clear_value;
  float flash = std::abs(std::sin(_frameNumber / 120.f));
  clear_value.color = {{ 0.f, 0.f, flash, 1.f }};

  VkRenderPassBeginInfo _rp_begin_info = {};
  _rp_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  _rp_begin_info.pNext = nullptr;
  _rp_begin_info.renderPass = _render_pass;
  _rp_begin_info.renderArea.offset.x = _rp_begin_info.renderArea.offset.y = 0;
  _rp_begin_info.renderArea.extent = _windowExtent;
  _rp_begin_info.framebuffer = _framebuffers[swapchain_image_index];
  _rp_begin_info.clearValueCount = 1;
  _rp_begin_info.pClearValues = &clear_value;

  vkCmdBeginRenderPass(cmd, &_rp_begin_info, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdEndRenderPass(cmd);
  VK_CHECK(vkEndCommandBuffer(cmd));

  // Submit
  VkSubmitInfo submit = {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.pNext = nullptr;

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  submit.pWaitDstStageMask = &wait_stage;
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &_present_semaphore;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &_render_semaphore;

  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;

  VK_CHECK(vkQueueSubmit(_graphics_queue, 1, &submit, _render_fence));


  // Present
  VkPresentInfoKHR present_info = {};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.pNext = nullptr;

  present_info.swapchainCount = 1;
  present_info.pSwapchains = &_swapchain;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &_render_semaphore;
  present_info.pImageIndices = &swapchain_image_index;

  VK_CHECK(vkQueuePresentKHR(_graphics_queue, &present_info));
  _frameNumber++;

}

void VulkanEngine::run() {
  SDL_Event e;
  bool bQuit = false;

  //main loop
  while (!bQuit) {
    //Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      //close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_QUIT) bQuit = true;
    }

    draw();
  }
}

void VulkanEngine::init_vulkan() {
  // Instance
  vkb::InstanceBuilder builder;
  auto inst_ret = builder
    .set_app_name("")
    .request_validation_layers(true)
    .require_api_version(1, 1, 0)
    .use_default_debug_messenger()
    .build();

  vkb::Instance vkb_inst = inst_ret.value();
  _instance = vkb_inst.instance;
  _debug_messenger = vkb_inst.debug_messenger;

  // Surface
  SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

  // Device
  vkb::PhysicalDeviceSelector selector{vkb_inst};
  vkb::PhysicalDevice physical_device = selector
    .set_minimum_version(1, 1)
    .set_surface(_surface)
    .select()
    .value();

  vkb::DeviceBuilder device_builder{physical_device};
  vkb::Device vkb_device = device_builder.build().value();
  _device = vkb_device.device;
  _physical_device = physical_device.physical_device;

  // Queue
  _graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
  _graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();
}

void VulkanEngine::init_swapchain() {
  // Swapchain
  vkb::SwapchainBuilder swapchain_builder{_physical_device, _device, _surface};
  vkb::Swapchain vkb_swapchain = swapchain_builder
    .use_default_format_selection()
    .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
    .set_desired_extent(_windowExtent.width, _windowExtent.height)
    .build()
    .value();

  _swapchain = vkb_swapchain.swapchain;
  _swapchain_images = vkb_swapchain.get_images().value();
  _swapchain_image_views = vkb_swapchain.get_image_views().value();
  _swapchain_image_format = vkb_swapchain.image_format;
}

void VulkanEngine::init_commands() {
  // Commands
  auto command_pool_info = vkinit::command_pool_create_info(_graphics_queue_family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  VK_CHECK(vkCreateCommandPool(_device, &command_pool_info, nullptr, &_command_pool));

  // CommandBuffer
  auto command_alloc_info = vkinit::command_buffer_allocate_info(_command_pool, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  VK_CHECK(vkAllocateCommandBuffers(_device, &command_alloc_info, &_main_command_buffer));

}

void VulkanEngine::init_default_renderpass() {
  VkAttachmentDescription color_attachment = {};

  color_attachment.format = _swapchain_image_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref = {};
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color_attachment_ref.attachment = 0;

  VkSubpassDescription subpass_desc = {};
  subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass_desc.colorAttachmentCount = 1;
  subpass_desc.pColorAttachments = &color_attachment_ref;


  // Render Pass
  VkRenderPassCreateInfo render_pass_info = {};

  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.pNext = nullptr;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass_desc;

  VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_render_pass));

}

void VulkanEngine::init_framebuffers() {
  VkFramebufferCreateInfo framebuffer_info = {};

  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.pNext = nullptr;

  framebuffer_info.renderPass = _render_pass;
  framebuffer_info.attachmentCount = 1;
  framebuffer_info.width = _windowExtent.width;
  framebuffer_info.height = _windowExtent.height;
  framebuffer_info.layers = 1;

  const uint32_t swapchain_image_count = _swapchain_images.size();
  _framebuffers = std::vector<VkFramebuffer>(swapchain_image_count); // TODO: std::array

  for (int i = 0; i < swapchain_image_count; ++i) {
    framebuffer_info.pAttachments = &_swapchain_image_views[i];
    VK_CHECK(vkCreateFramebuffer(_device, &framebuffer_info, nullptr, &_framebuffers[i]));
  }

}

void VulkanEngine::init_sync_structures() {

  VkFenceCreateInfo fence_create_info = {};
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext = nullptr;
  fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VK_CHECK(vkCreateFence(_device, &fence_create_info, nullptr, &_render_fence));

  VkSemaphoreCreateInfo semaphore_create_info = {};
  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext = nullptr;
  semaphore_create_info.flags = 0;

  VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &_render_semaphore));
  VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &_present_semaphore));

}
