#include <vk_engine.h>
#include <vk_scene.h>
#include <vk_pipeline.h>
#include <vk_types.h>
#include <vk_initializers.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <VkBootstrap.h>
#include <SDL_vulkan.h>


#define VK_CHECK(x)                                        \
do {                                                       \
  VkResult err = x;                                        \
  if (err) {                                               \
    std::cerr << "Vulkan error occurred: " << err << '\n'; \
    std::exit(err);                                        \
  }                                                        \
} while (0)


void Engine::init() {
  // We initialize SDL and create a window with it. 
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  window = SDL_CreateWindow(
    "Meex",
    SDL_WINDOWPOS_UNDEFINED,
    SDL_WINDOWPOS_UNDEFINED,
    window_extent.width,
    window_extent.height,
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
  // init pipelines
  init_pipelines();
  // load and upload meshses
  load_meshes();
  // scene
  init_scene();

  //everything went fine
  is_initialized = true;
}

void Engine::cleanup() {
  if (is_initialized) {
    vkWaitForFences(_device, 1, &_render_fence, true, 1000000000);

    for (auto& iv : _swapchain_image_views) {
      vkDestroyImageView(_device, iv, nullptr);
    }

    _destroy_queue.flush();

    vmaDestroyAllocator(_allocator);
    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
    vkDestroyInstance(_instance, nullptr);

    SDL_DestroyWindow(window);
  }
}

void Engine::draw() {
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

  VkClearValue backg_clear;
  float flash = std::abs(std::sin(frame_count / 60.f));
  backg_clear.color = {{ 0.f, flash, 0.f, 1.f }};
  VkClearValue depth_clear;
  depth_clear.depthStencil.depth = 1.0f;

  std::array<VkClearValue, 2> clear_values = {backg_clear, depth_clear};

  VkRenderPassBeginInfo _rp_begin_info = {};
  _rp_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  _rp_begin_info.pNext = nullptr;
  _rp_begin_info.renderPass = _render_pass;
  _rp_begin_info.renderArea.offset.x = _rp_begin_info.renderArea.offset.y = 0;
  _rp_begin_info.renderArea.extent = window_extent;
  _rp_begin_info.framebuffer = _framebuffers[swapchain_image_index];
  _rp_begin_info.clearValueCount = clear_values.size();
  _rp_begin_info.pClearValues = clear_values.data();

  vkCmdBeginRenderPass(cmd, &_rp_begin_info, VK_SUBPASS_CONTENTS_INLINE);

  draw_objects(cmd, _scene);

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
  frame_count++;

}

void Engine::run() {
  SDL_Event e{};
  bool quit{false};

  //main loop
  while (!quit) {
    //Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      if (e.type == SDL_QUIT) {
        quit = true;
      } else if (e.type == SDL_KEYDOWN) {
        if (e.key.keysym.sym == SDLK_ESCAPE) {
          quit = true;
        }
      }
    }

    draw();
  }
}

void Engine::init_vulkan() {
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
  SDL_Vulkan_CreateSurface(window, _instance, &_surface);

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

  // VMAllocator
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.instance = _instance;
  allocator_info.physicalDevice = _physical_device;
  allocator_info.device = _device;

  vmaCreateAllocator(&allocator_info, &_allocator);
}

void Engine::init_swapchain() {
  // Swapchain
  vkb::SwapchainBuilder swapchain_builder{_physical_device, _device, _surface};
  vkb::Swapchain vkb_swapchain = swapchain_builder
    .use_default_format_selection()
    .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
    .set_desired_extent(window_extent.width, window_extent.height)
    .build()
    .value();

  _swapchain = vkb_swapchain.swapchain;
  _swapchain_images = vkb_swapchain.get_images().value();
  _swapchain_image_views = vkb_swapchain.get_image_views().value();
  _swapchain_image_format = vkb_swapchain.image_format;

  // Depthmap
  VkExtent3D depth_extent = {window_extent.width, window_extent.height, 1};
  _depth_format = VK_FORMAT_D32_SFLOAT;
  VkImageCreateInfo depth_info = vkinit::image_create_info(_depth_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depth_extent);
  VmaAllocationCreateInfo alloc_info = {};
  alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  alloc_info.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  vmaCreateImage(_allocator, &depth_info, &alloc_info, &_depth_image.image, &_depth_image.allocation, nullptr);

  VkImageViewCreateInfo depth_view_info = vkinit::image_view_create_info(_depth_format, _depth_image.image, VK_IMAGE_ASPECT_DEPTH_BIT);
  VK_CHECK(vkCreateImageView(_device, &depth_view_info, nullptr, &_depth_image_view));

  _destroy_queue.push_back([this]() {
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    vkDestroyImageView(_device, _depth_image_view, nullptr);
    vmaDestroyImage(_allocator, _depth_image.image, _depth_image.allocation);
  });
}

void Engine::init_commands() {
  // Commands
  auto command_pool_info = vkinit::command_pool_create_info(_graphics_queue_family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  VK_CHECK(vkCreateCommandPool(_device, &command_pool_info, nullptr, &_command_pool));

  // CommandBuffer
  auto command_alloc_info = vkinit::command_buffer_allocate_info(_command_pool, 1, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  VK_CHECK(vkAllocateCommandBuffers(_device, &command_alloc_info, &_main_command_buffer));

  _destroy_queue.push_back([this]() {
    vkDestroyCommandPool(_device, _command_pool, nullptr);
  });
}

void Engine::init_default_renderpass() {
  // Color attachment
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

  // Depth attachment
  VkAttachmentDescription depth_attachment = {};
  depth_attachment.format = _depth_format;
  depth_attachment.flags = 0;
  depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_attachment_ref = {};
  depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depth_attachment_ref.attachment = 1;

  // Subpass and dependencies
  VkSubpassDescription subpass_desc = {};
  subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass_desc.colorAttachmentCount = 1;
  subpass_desc.pColorAttachments = &color_attachment_ref;
  subpass_desc.pDepthStencilAttachment = &depth_attachment_ref;

  VkSubpassDependency color_dep = {};
  color_dep.srcSubpass = VK_SUBPASS_EXTERNAL;
  color_dep.dstSubpass = 0;
  color_dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  color_dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  color_dep.srcAccessMask = 0;
  color_dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkSubpassDependency depth_dep = {};
  depth_dep.srcSubpass = VK_SUBPASS_EXTERNAL;
  depth_dep.dstSubpass = 0;
  depth_dep.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  depth_dep.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  depth_dep.srcAccessMask = 0;
  depth_dep.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  std::array<VkSubpassDependency, 2> dependencies = {color_dep, depth_dep};

  // Render Pass
  std::array<VkAttachmentDescription, 2> attachments = {color_attachment, depth_attachment};
  VkRenderPassCreateInfo render_pass_info = {};

  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.pNext = nullptr;
  render_pass_info.attachmentCount = attachments.size();
  render_pass_info.pAttachments = attachments.data();
  render_pass_info.dependencyCount = dependencies.size();
  render_pass_info.pDependencies = dependencies.data();
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass_desc;

  VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_render_pass));

  _destroy_queue.push_back([this]() {
    vkDestroyRenderPass(_device, _render_pass, nullptr);
  });

}

void Engine::init_framebuffers() {
  VkFramebufferCreateInfo framebuffer_info = {};

  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.pNext = nullptr;

  framebuffer_info.renderPass = _render_pass;
  framebuffer_info.attachmentCount = 1;
  framebuffer_info.width = window_extent.width;
  framebuffer_info.height = window_extent.height;
  framebuffer_info.layers = 1;

  // requesting an image from swapchain will block cpu threads until the image is available
  const uint32_t swapchain_image_count = _swapchain_images.size();
  _framebuffers = std::vector<VkFramebuffer>(swapchain_image_count);

  for (int i = 0; i < swapchain_image_count; ++i) {
    std::array<VkImageView, 2> attachments = {
      _swapchain_image_views[i],
      _depth_image_view
    };
    framebuffer_info.attachmentCount = attachments.size();
    framebuffer_info.pAttachments = attachments.data();

    VK_CHECK(vkCreateFramebuffer(_device, &framebuffer_info, nullptr, &_framebuffers[i]));

    _destroy_queue.push_back([this, i]() {
      vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
    });
  }

}

void Engine::init_sync_structures() {

  VkFenceCreateInfo fence_create_info = vkinit::fence_create_info();

  VK_CHECK(vkCreateFence(_device, &fence_create_info, nullptr, &_render_fence));

  _destroy_queue.push_back([this]() {
    vkDestroyFence(_device, _render_fence, nullptr);
  });

  VkSemaphoreCreateInfo semaphore_create_info = vkinit::semaphore_create_info();

  VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &_render_semaphore));
  VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &_present_semaphore));

  _destroy_queue.push_back([this]() {
    vkDestroySemaphore(_device, _render_semaphore, nullptr);
    vkDestroySemaphore(_device, _present_semaphore, nullptr);
  });

}

bool Engine::load_shader_module(const char* file_path, VkShaderModule* out_module) {
  using dtype = uint32_t;

  std::ifstream file(file_path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  size_t file_size = file.tellg();
  std::vector<dtype> buffer(file_size / sizeof(dtype));
  file.seekg(0);
  file.read((char*)buffer.data(), file_size);
  file.close();

  VkShaderModuleCreateInfo shader_info = {};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.pNext = nullptr;
  shader_info.codeSize = buffer.size() * sizeof(dtype);
  shader_info.pCode = buffer.data();

  if (vkCreateShaderModule(_device, &shader_info, nullptr, out_module) != VK_SUCCESS) {
    return false;
  }

  return true;
}

void Engine::init_pipelines() {
  VkShaderModule triangle_vert_shader;
  if (!load_shader_module("shaders/triangle.vert.spv", &triangle_vert_shader)) {
    std::cerr << "Vertex shader error!\n";
  }
  VkShaderModule triangle_frag_shader;
  if (!load_shader_module("shaders/triangle.frag.spv", &triangle_frag_shader)) {
    std::cerr << "Fragment shader error!\n";
  }

  VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
  VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_triangle_pipeline_layout));

  PipelineBuilder builder;

  // hard-coded triangle pipeline
  builder._shader_stages.push_back(
    vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, triangle_vert_shader));
  builder._shader_stages.push_back(
    vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangle_frag_shader));

  builder._vertex_input_info = vkinit::vertex_input_state_create_info();
  builder._input_assembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

  builder._viewport.x = builder._viewport.y = 0.0f;
  builder._viewport.width = (float)window_extent.width;
  builder._viewport.height = (float)window_extent.height;
  builder._viewport.minDepth = 0.0f;
  builder._viewport.maxDepth = 1.0f;

  builder._scissor.offset = {0, 0};
  builder._scissor.extent = window_extent;

  builder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
  builder._multisampleing = vkinit::multisampling_state_create_info();
  builder._color_blend_attachment = vkinit::color_blend_attachment_state();
  builder._pipeline_layout = _triangle_pipeline_layout;
  builder._depth_stencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

  _triangle_pipeline = builder.build_pipeline(_device, _render_pass);

  // tirangle mesh pipeline
  VkPushConstantRange push_constant;
  push_constant.offset = 0;
  push_constant.size = sizeof(MeshPushConstants);
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkPipelineLayoutCreateInfo mesh_pipeline_layout_info = vkinit::pipeline_layout_create_info();
  mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;
  mesh_pipeline_layout_info.pushConstantRangeCount = 1;

  VK_CHECK(vkCreatePipelineLayout(_device, &mesh_pipeline_layout_info, nullptr, &_mesh_pipeline_layout));

  VertexInputDescription vertex_description = Vertex::get_vertex_input_description();

  builder._vertex_input_info.pVertexBindingDescriptions = vertex_description.bindings.data();
  builder._vertex_input_info.vertexBindingDescriptionCount = vertex_description.bindings.size();
  builder._vertex_input_info.pVertexAttributeDescriptions = vertex_description.attributes.data();
  builder._vertex_input_info.vertexAttributeDescriptionCount = vertex_description.attributes.size();

  builder._shader_stages.clear();

  VkShaderModule trimesh_vert_shader;
  if (!load_shader_module("shaders/tri_mesh.vert.spv", &trimesh_vert_shader)) {
    std::cerr << "Mesh Vertex shader error!\n";
  }
  builder._shader_stages.push_back(
    vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, trimesh_vert_shader));
  builder._shader_stages.push_back(
    vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, triangle_frag_shader));
  builder._pipeline_layout = _mesh_pipeline_layout;

  _mesh_pipeline = builder.build_pipeline(_device, _render_pass);
  _scene.add_material(_mesh_pipeline, _mesh_pipeline_layout, "default_mesh");

  vkDestroyShaderModule(_device, triangle_frag_shader, nullptr);
  vkDestroyShaderModule(_device, triangle_vert_shader, nullptr);
  vkDestroyShaderModule(_device, trimesh_vert_shader, nullptr);

  _destroy_queue.push_back([this]() {
    vkDestroyPipeline(_device, _triangle_pipeline, nullptr);
    vkDestroyPipeline(_device, _mesh_pipeline, nullptr);

    vkDestroyPipelineLayout(_device, _triangle_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(_device, _mesh_pipeline_layout, nullptr);
  });

}

void Engine::load_meshes() {
  Mesh triangle_mesh;
  triangle_mesh.vertices.resize(3);
  triangle_mesh.vertices = {
    {{0.5f, 0.5f, 0.0f},  {}, {0.0f, 0.0f, 0.5f}},
    {{-0.5f, 0.5f, 0.0f}, {}, {0.0f, 0.5f, 0.2f}},
    {{0.0f, -0.5f, 0.0f}, {}, {0.5f, 0.0f, 0.2f}}
  };

  Mesh monkey_mesh;
  monkey_mesh.load_from_obj("assets/monkey_smooth.obj");

  _scene.add_mesh(triangle_mesh, "triangle_mesh");
  _scene.add_mesh(monkey_mesh, "monkey_mesh");

  upload_meshes(_scene.meshes["triangle_mesh"]);
  upload_meshes(_scene.meshes["monkey_mesh"]);
}

void Engine::upload_meshes(Mesh& mesh) {
  VkBufferCreateInfo buffer_info = {};

  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.pNext = nullptr;
  buffer_info.size = sizeof(Vertex) * mesh.vertices.size();
  buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

  VmaAllocationCreateInfo vmalloc_info = {};
  vmalloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

  VK_CHECK(vmaCreateBuffer(
    _allocator,
    &buffer_info,
    &vmalloc_info,
    &mesh.vertex_buffer.buffer,
    &mesh.vertex_buffer.allocation,
    nullptr
  ));

  _destroy_queue.push_back([&mesh, this]() {
    vmaDestroyBuffer(_allocator, mesh.vertex_buffer.buffer, mesh.vertex_buffer.allocation);
  });

  void* data_ptr;
  vmaMapMemory(_allocator, mesh.vertex_buffer.allocation, &data_ptr);
  memcpy(data_ptr, mesh.vertices.data(), buffer_info.size);
  vmaUnmapMemory(_allocator, mesh.vertex_buffer.allocation);

}

void Engine::init_scene() {
  RenderableObject monkey {
    std::make_unique<Mesh>(_scene.meshes["monkey_mesh"]),
    std::make_unique<Material>(_scene.materials["default_mesh"]),
    glm::mat4{1.0f}
  };
  _scene.objects.push_back(std::move(monkey));

  for (int x = -20; x <= 20; ++x) {
    for (int y = -20; y <= 20; ++y) {
      glm::mat4 translate = glm::translate(glm::mat4{1.0f}, glm::vec3{x, 0, y});
      glm::mat4 scale = glm::scale(glm::mat4{1.0f}, glm::vec3{0.2f});
      RenderableObject tri {
        std::make_unique<Mesh>(_scene.meshes["triangle_mesh"]),
        std::make_unique<Material>(_scene.materials["default_mesh"]),
        translate * scale
      };
      _scene.objects.push_back(std::move(tri));
    }
  }
}

void Engine::draw_objects(VkCommandBuffer cmd, Scene const& scene) {
  glm::vec3 cam_pos{0.0f, -5.0f, -10.0f};
  glm::mat4 view = glm::translate(glm::mat4{1.0f}, cam_pos);
  float _aspect = static_cast<float>(window_extent.width) / window_extent.height;
  glm::mat4 projection = glm::perspective(glm::radians(72.0f), _aspect, 0.1f, 200.0f);
  projection[1][1] *= -1;

  Mesh* last_mesh{nullptr};
  Material* last_material{nullptr};

  for (auto& obj : scene.objects) {
    glm::mat4 model = obj.transform_matrix;
    glm::mat4 mvp = projection * view * model;

    MeshPushConstants push_const;
    push_const.render_matrix = mvp;

    if (obj.material.get() != last_material) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, obj.material->pipeline);
      last_material = obj.material.get();
    }
    vkCmdPushConstants(cmd, obj.material->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &push_const);

    VkDeviceSize offset = 0;
    if (obj.mesh.get() != last_mesh) {
      vkCmdBindVertexBuffers(cmd, 0, 1, &obj.mesh->vertex_buffer.buffer, &offset);
      last_mesh = obj.mesh.get();
    }

    vkCmdDraw(cmd, obj.mesh->vertices.size(), 1, 0, 0);
  }

}

