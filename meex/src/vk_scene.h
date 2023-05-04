#pragma once

#include <vk_types.h>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


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

struct Material {
  VkPipeline pipeline;
  VkPipelineLayout layout;
};

struct RenderableObject {
  std::unique_ptr<Mesh> mesh{nullptr};
  std::unique_ptr<Material> material{nullptr};
  glm::mat4 transform_matrix{1.0f};
};

struct Scene {
  std::vector<RenderableObject> objects;
  std::unordered_map<std::string, Mesh> meshes;
  std::unordered_map<std::string, Material> materials;

  Mesh& add_mesh(Mesh& mesh, std::string const& name);
  Material& add_material(VkPipeline pipeline, VkPipelineLayout layout, std::string const& name);
};

