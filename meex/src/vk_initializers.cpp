#include <vk_initializers.h>
#include <vulkan/vulkan_core.h>


namespace vkinit {

VkCommandPoolCreateInfo command_pool_create_info(uint32_t queue_family_index, VkCommandPoolCreateFlags flags) {
  VkCommandPoolCreateInfo info = {};

  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.pNext = nullptr;
  info.queueFamilyIndex = queue_family_index;
  info.flags = flags;

  return info;
}

VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count, VkCommandBufferLevel level) {
  VkCommandBufferAllocateInfo info = {};

  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.pNext = nullptr;
  info.commandPool = pool;
  info.commandBufferCount = 1;
  info.level = level;

  return info;
}


VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule module) {
  VkPipelineShaderStageCreateInfo info = {};

  info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.pNext = nullptr;
  info.stage = stage;
  info.module = module;
  info.pName = "main";   // shader entry point

  return info;
}

VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info() {
  VkPipelineVertexInputStateCreateInfo info = {};

  info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.vertexBindingDescriptionCount = 0;
  info.vertexAttributeDescriptionCount = 0;

  return info;

}

VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info(VkPrimitiveTopology topology) {
  VkPipelineInputAssemblyStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.topology = topology;
  info.primitiveRestartEnable = VK_FALSE;

  // VK_PRIMITIVE_TOPOLOGY_TRIANGLE/POINT/LINE_LIST
  return info;
}

VkPipelineRasterizationStateCreateInfo rasterization_state_create_info(VkPolygonMode polygon_mode) {
  VkPipelineRasterizationStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.depthClampEnable = VK_FALSE;
  //discards all primitives before the rasterization stage if enabled which we don't want
  info.rasterizerDiscardEnable = VK_FALSE;

  // toggle between wireframe and solid drawing.
  info.polygonMode = polygon_mode;
  info.lineWidth = 1.0f;
  //no backface cull
  info.cullMode = VK_CULL_MODE_NONE;
  info.frontFace = VK_FRONT_FACE_CLOCKWISE;
  //no depth bias;
  info.depthBiasEnable = VK_FALSE;
  info.depthBiasConstantFactor = 0.0f;
  info.depthBiasClamp = 0.0f;
  info.depthBiasSlopeFactor = 0.0f;

  return info;
}

VkPipelineMultisampleStateCreateInfo multisampling_state_create_info() {
  VkPipelineMultisampleStateCreateInfo info = {};

  info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.sampleShadingEnable = VK_FALSE;
  info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;  //1spp
  info.minSampleShading = 1.0f;
  info.pSampleMask = nullptr;
  info.alphaToCoverageEnable = VK_FALSE;
  info.alphaToOneEnable = VK_FALSE;

  return info;
}

VkPipelineColorBlendAttachmentState color_blend_attachment_state() {
  VkPipelineColorBlendAttachmentState color_blend_attachment = {};
  color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = VK_FALSE;

  return color_blend_attachment;
}

VkPipelineLayoutCreateInfo pipeline_layout_create_info() {
  VkPipelineLayoutCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.pNext = nullptr;

  info.flags = 0;
  info.setLayoutCount = 0;
  info.pSetLayouts = nullptr;
  info.pushConstantRangeCount = 0;
  info.pPushConstantRanges = nullptr;

  return info;
}

VkFenceCreateInfo fence_create_info() {
  VkFenceCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  return info;
}

VkSemaphoreCreateInfo semaphore_create_info() {
  VkSemaphoreCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0;

  return info;
}

} // vkinit
