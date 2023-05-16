#pragma once

#include <glad/glad.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/material.h>
#include <assimp/mesh.h>

#include <fmt/core.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "kit/exception.h"
#include "kit/tex_proc.h"
#include "shader_m.h"


namespace kit {

namespace fs = std::filesystem;

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 tex_coords;
};

struct Texture {
  unsigned int id;
  std::string type;
  std::string path;
};

template <typename C>
concept is_color_t = std::is_same_v<C, aiColor3D> || std::is_same_v<C, aiColor4D> || std::is_same_v<C, glm::vec3> || std::is_same_v<C, glm::vec4>;
template <typename C>
concept has_rgb = requires (C c) { c.r && c.g && c.b; };
template <typename C>
concept has_xyz = requires (C c) { c.x && c.y && c.z; };
template <typename C>
concept Color = is_color_t<C> || has_rgb<C> || has_xyz<C>;

template<Color C>
struct ColorCard {
  C ambient;
  C diffuse;
  C specular;
};

enum struct DrawMode {
  Auto = 0, Color = 1, Texture = 2
};

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<unsigned int> elements;
  std::vector<Texture> textures;
  ColorCard<glm::vec3> colors;

  unsigned int gl_vao{}, gl_vbo{}, gl_ebo{};

  Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& elements, std::vector<Texture>&& textures) {
    this->vertices = vertices;
    this->elements = elements;
    this->textures = textures;
    _upload_mesh();
  }

  Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& elements, ColorCard<glm::vec3>&& color_card) {
    this->vertices = vertices;
    this->elements = elements;
    this->colors = color_card;
    _upload_mesh();
  }

  void draw(const Shader& shader, std::string prefix = {}, DrawMode mode = DrawMode::Auto) {
    if (!prefix.empty() && !prefix.ends_with('.')) {
      prefix.push_back('.');
    }

    switch (mode) {
      case DrawMode::Auto:
        if (!textures.empty()) {
          draw(shader, prefix, DrawMode::Texture);
        } else {
          draw(shader, prefix, DrawMode::Color);
        }
        break;

      case DrawMode::Texture:
        _draw_tex_impl(shader, prefix);
        break;

      case DrawMode::Color:
        _draw_color_impl(shader, prefix);
        break;

      default:
        std::cerr << "Mesh may be not drawn normally.\n";
    };
  }

  void _upload_mesh() {
    glGenVertexArrays(1, &gl_vao);
    glBindVertexArray(gl_vao);

    glGenBuffers(1, &gl_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, gl_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &gl_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size()*sizeof(unsigned int), elements.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tex_coords));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
  }

  void _draw_color_impl(const Shader& shader, const std::string& prefix = {}) {
    shader.setVec3(prefix + "ambient", colors.ambient);
    shader.setVec3(prefix + "diffuse", colors.diffuse);
    shader.setVec3(prefix + "specular", colors.specular);

    shader.use();
    glBindVertexArray(gl_vao);
    glDrawElements(GL_TRIANGLES, elements.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

  void _draw_tex_impl(const Shader& shader, const std::string& prefix = {}) {
    uint32_t ambient_n{1}, diffuse_n{1}, specular_n{1};

    for (uint32_t i = 0; i < textures.size(); ++i) {
      glActiveTexture(GL_TEXTURE0 + i);
      Texture& t = textures[i];
      std::string name{};

      if (t.type == "texture_ambient") {
        fmt::format_to(std::back_inserter(name), "{}{}", t.type, std::to_string(ambient_n++));
      } else if (t.type == "texture_diffuse") {
        fmt::format_to(std::back_inserter(name), "{}{}", t.type, std::to_string(diffuse_n++));
      } else if (t.type == "texture_specular") {
        fmt::format_to(std::back_inserter(name), "{}{}", t.type, std::to_string(specular_n++));
      }
      //fmt::print("sampler2D name: {}\n", prefix + name);

      shader.setInt((prefix + name).c_str(), i);
      glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }
    glActiveTexture(0);

    shader.use();
    glBindVertexArray(gl_vao);
    glDrawElements(GL_TRIANGLES, elements.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

}; // struct Mesh


struct Model {
  std::vector<Mesh> meshes;
  fs::path file_path;
  glm::mat4 model_matrix{1.0f};
  std::unordered_map<std::string, Texture> _texture_cache;

  Model() = default;
  explicit Model(const std::string_view path_sv) {
    load(path_sv);
  }

  void load(const std::string_view path_sv) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path_sv.data(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
      std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << '\n';
      THROW_RT_ERR
    }
    file_path = std::filesystem::path{path_sv};
    _process_node(scene->mRootNode, scene);
  }

  void draw(const Shader& shader, const std::string& prefix = {}, DrawMode mode = DrawMode::Auto) {
    for (auto& mesh : meshes) {
      mesh.draw(shader, prefix, mode);
    }
  }

  void _process_node(aiNode* node, const aiScene* scene) {
    for (int i = 0; i < node->mNumMeshes; ++i) {
      aiMesh* _mesh = scene->mMeshes[node->mMeshes[i]];
      _process_mesh(_mesh, scene);
    }
    for (int i = 0; i < node->mNumChildren; ++i) {
      _process_node(node->mChildren[i], scene);
    }
  }

  void _process_mesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<Vertex> _vertices;
    std::vector<unsigned int> _elements;
    std::vector<Texture> _textures;
    ColorCard<glm::vec3> _color_card;

    for (int i = 0; i < mesh->mNumVertices; ++i) {
      auto& _vert = mesh->mVertices[i];
      auto& _norm = mesh->mNormals[i];

      Vertex v;
      v.position = {_vert.x, _vert.y, _vert.z};
      v.normal = {_norm.x, _norm.y, _norm.z};

      if (mesh->mTextureCoords[0]) {
        auto& _texc = mesh->mTextureCoords[0][i];
        v.tex_coords = {_texc.x, _texc.y};
      } else {
        v.tex_coords = {0.f, 0.f};
      }
      _vertices.push_back(v);
    }

    // Primetives
    for (int i = 0; i < mesh->mNumFaces; ++i) {
      auto& _face = mesh->mFaces[i];
      for (int j = 0; j < _face.mNumIndices; ++j) {
        _elements.push_back(_face.mIndices[j]);
      }
    }

    if (mesh->mMaterialIndex >= 0) {
      aiMaterial* _mtl = scene->mMaterials[mesh->mMaterialIndex];
      _color_card = _load_material_colors(_mtl);

      if (_mtl->GetTextureCount(aiTextureType_AMBIENT) ||
          _mtl->GetTextureCount(aiTextureType_DIFFUSE) ||
          _mtl->GetTextureCount(aiTextureType_SPECULAR)) {

        std::vector<Texture> ambient_maps = _load_material_textures(_mtl, aiTextureType_AMBIENT, "texture_ambient");
        _textures.insert(_textures.end(), ambient_maps.begin(), ambient_maps.end());

        std::vector<Texture> diffuse_maps = _load_material_textures(_mtl, aiTextureType_DIFFUSE, "texture_diffuse");
        _textures.insert(_textures.end(), diffuse_maps.begin(), diffuse_maps.end());

        std::vector<Texture> specular_maps = _load_material_textures(_mtl, aiTextureType_SPECULAR, "texture_specular");

       _textures.insert(_textures.end(), specular_maps.begin(), specular_maps.end());

        // fmt::print("ambient: {}, diffuse: {}, specular: {}\n", ambient_maps.size(), diffuse_maps.size(), specular_maps.size());
      }
    }

    // Construct and store mesh
    if (_textures.empty()) {
      meshes.emplace_back(std::move(_vertices), std::move(_elements), std::move(_color_card));
    } else {
      meshes.emplace_back(std::move(_vertices), std::move(_elements), std::move(_textures));
    }
  }

  ColorCard<glm::vec3> _load_material_colors(aiMaterial* mtl) {
    ColorCard<glm::vec3> card;
    ColorCard<aiColor4D> _card;

    mtl->Get(AI_MATKEY_COLOR_AMBIENT, _card.ambient);
    card.ambient = {_card.ambient.r, _card.ambient.g, _card.ambient.b};

    mtl->Get(AI_MATKEY_COLOR_DIFFUSE, _card.diffuse);
    card.diffuse = {_card.diffuse.r, _card.diffuse.g, _card.diffuse.b};

    mtl->Get(AI_MATKEY_COLOR_SPECULAR, _card.specular);
    card.specular = {_card.specular.r, _card.specular.g, _card.specular.b};

    return card;
  }

  std::vector<Texture> _load_material_textures(aiMaterial* mtl, aiTextureType type, const std::string& type_s) {
    // TODO: need more tests
    std::vector<Texture> _textures;

    for (int i = 0; i < mtl->GetTextureCount(type); ++i) {
      aiString path_as;
      mtl->GetTexture(type, i, &path_as);
      std::string path_s{(file_path.parent_path() / fs::path{path_as.C_Str()}).string()};

      if (_texture_cache.find(path_s) != _texture_cache.end()) {
        _textures.push_back(_texture_cache[path_s]);
      } else {
        _textures.emplace_back(kit::make_texture(path_s), type_s, path_s);
        _texture_cache[path_s] = _textures.back();
      }
    }
    return _textures;
  }

}; // Model

}
