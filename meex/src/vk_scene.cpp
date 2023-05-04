#include <vk_engine.h>
#include <vk_scene.h>
#include <tiny_obj_loader.h>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan_core.h>


VertexInputDescription Vertex::get_vertex_input_description() {
  VertexInputDescription description {};

  VkVertexInputBindingDescription main_binding = {};
  main_binding.binding = 0;
  main_binding.stride = sizeof(Vertex);
  main_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  description.bindings.push_back(main_binding);

  VkVertexInputAttributeDescription position_attrib = {};
  position_attrib.binding = 0;
  position_attrib.location = 0;
  position_attrib.format = VK_FORMAT_R32G32B32_SFLOAT;
  position_attrib.offset = offsetof(Vertex, position);
  description.attributes.push_back(position_attrib);

  VkVertexInputAttributeDescription normal_attrib = {};
  normal_attrib.binding = 0;
  normal_attrib.location = 1;
  normal_attrib.format = VK_FORMAT_R32G32B32_SFLOAT;
  normal_attrib.offset = offsetof(Vertex, normal);
  description.attributes.push_back(normal_attrib);

  VkVertexInputAttributeDescription color_attrib = {};
  color_attrib.binding = 0;
  color_attrib.location = 2;
  color_attrib.format = VK_FORMAT_R32G32B32_SFLOAT;
  color_attrib.offset = offsetof(Vertex, color);
  description.attributes.push_back(color_attrib);

  return description;
}

bool Mesh::load_from_obj(const std::string& filename) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str(), nullptr);
  if (!warn.empty()) {
    std::clog << "tinyobj::warn: " << warn;
  }
  if (!err.empty()) {
    std::cerr << "tinyobj::err: " << err;
    return false;
  }

  for (auto& s : shapes) {
    size_t index_offset = 0;
    for (auto& f : s.mesh.num_face_vertices) {
      constexpr uint32_t fv = 3;
      for (size_t v = 0; v < fv; ++v) {
        tinyobj::index_t idx = s.mesh.indices[index_offset + v];

        // vert position
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
        // vert normal
        tinyobj::real_t nx = attrib.normals[3 * idx.vertex_index + 0];
        tinyobj::real_t ny = attrib.normals[3 * idx.vertex_index + 1];
        tinyobj::real_t nz = attrib.normals[3 * idx.vertex_index + 2];

        Vertex new_vert {{vx, vy, vz}, {nx, ny, nz}, {nx, ny, nz}};
        vertices.push_back(std::move(new_vert));
      }
      index_offset += fv;
    }
  }

  return true;
}

Mesh& Scene::add_mesh(Mesh& mesh, std::string const& name) {
  meshes.emplace(name, mesh);
  return meshes[name];
}

Material& Scene::add_material(VkPipeline pipeline, VkPipelineLayout layout, std::string const& name) {
  materials.emplace(name, Material{pipeline, layout});
  return materials[name];
}


