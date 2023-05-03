#pragma once

#include <vk_types.h>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>


struct VertexInputDescription {
  std::vector<VkVertexInputBindingDescription> bindings;
  std::vector<VkVertexInputAttributeDescription> attributes;

  VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec3 color;

  static VertexInputDescription get_vertex_input_description();
};

struct Mesh {
  std::vector<Vertex> vertices;
  AllocatedBuffer vertex_buffer;

  bool load_from_obj(const std::string& filename);
};

