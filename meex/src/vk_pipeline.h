#pragma once

#include <vector>
#include <vulkan/vulkan.h>


class PipelineBuilder {
public:
  std::vector<VkPipelineShaderStageCreateInfo> _shader_stages = {};
  VkPipelineVertexInputStateCreateInfo _vertex_input_info = {};
  VkPipelineInputAssemblyStateCreateInfo _input_assembly = {};

  VkViewport _viewport;
  VkRect2D _scissor;

  VkPipelineRasterizationStateCreateInfo _rasterizer = {};
  VkPipelineColorBlendAttachmentState _color_blend_attachment;
  VkPipelineMultisampleStateCreateInfo _multisampleing = {};
  VkPipelineLayout _pipeline_layout;
  VkPipelineDepthStencilStateCreateInfo _depth_stencil = {};

  VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};

