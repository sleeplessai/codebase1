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
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "kit/tex_proc.h"
#include "shader_m.h"


namespace kit {


struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 tex_coords;
};

struct Texture {
  std::string path;
  unsigned int id;
  std::string type;
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

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<unsigned int> elements;
  std::vector<Texture> textures;
  ColorCard<glm::vec3> colors;

  unsigned int vao, vbo, ebo;

  enum struct DrawMode {
    None = 0, Color = 1, Texture = 2
  };

  Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> elements, std::vector<Texture> textures) {
    this->vertices = vertices;
    this->elements = elements;
    this->textures = textures;
    __setup();
  }

  Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> elements, ColorCard<glm::vec3> color_card) {
    this->vertices = vertices;
    this->elements = elements;
    this->colors = color_card;
    __setup();
  }

  void __setup() {
    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &this->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &this->ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size()*sizeof(unsigned int), elements.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tex_coords));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
  }

  void __draw_color_impl(const Shader& shader, const std::string& material_prefix = {}) {
    shader.setVec3(material_prefix + "ambient", colors.ambient);
    shader.setVec3(material_prefix + "diffuse", colors.diffuse);
    shader.setVec3(material_prefix + "specular", colors.specular);

    shader.use();
    glBindVertexArray(this->vao);
    glDrawElements(GL_TRIANGLES, elements.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

  void __draw_tex_impl(const Shader& shader, const std::string& material_prefix = {}) {
    uint32_t diffuse_n{1}, specular_n{1};

    for (uint32_t i = 0; i < textures.size(); ++i) {
      glActiveTexture(GL_TEXTURE0 + i);
      Texture& t = textures[i];
      std::string name{};

      if (t.type == "texture_diffuse") {
        fmt::format_to(std::back_inserter(name), "{}{}", t.type, std::to_string(diffuse_n++));
      } else if (t.type == "texture_specular") {
        fmt::format_to(std::back_inserter(name), "{}{}", t.type, std::to_string(specular_n++));
      }

      shader.setInt((material_prefix + name).c_str(), i);
      glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }
    glActiveTexture(0);

    shader.use();
    glBindVertexArray(this->vao);
    glDrawElements(GL_TRIANGLES, elements.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

  bool draw(const Shader& shader, DrawMode mode = DrawMode::None, std::string material_prefix = {}) {
    if (!material_prefix.empty() && !material_prefix.ends_with('.')) {
      material_prefix.push_back('.');
    }

    if (mode == DrawMode::Color) {
      __draw_color_impl(shader, material_prefix);
    } else if (mode == DrawMode::Texture) {
      __draw_tex_impl(shader, material_prefix);
    } else {
      return false;
    }
    return true;
  }

}; // struct Mesh

struct Model {
  Model() = default;
  Model(const std::string_view path_sv) {
    load_model(path_sv);
  }

  std::vector<Mesh> meshes;
  std::string directory{};

  void load_model(const std::string_view path_sv) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path_sv.data(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
      std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << '\n';
      return;
    }
    process_node(scene->mRootNode, scene);
  }

  void process_node(aiNode* node, const aiScene* scene) {
    for (int i = 0; i < node->mNumMeshes; ++i) {
      aiMesh* _mesh = scene->mMeshes[node->mMeshes[i]];
      process_mesh(_mesh, scene);
    }
    for (int i = 0; i < node->mNumChildren; ++i) {
      process_node(node->mChildren[i], scene);
    }
  }

  void process_mesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<Vertex> _vertices;
    std::vector<unsigned int> _elements;
    std::vector<Texture> _textures;
    ColorCard<glm::vec3> _color_card;

    for (int i = 0; i < mesh->mNumVertices; ++i) {
      auto& _vert = mesh->mVertices[i];
      auto& _norm = mesh->mNormals[i];
      // fmt::print("Vert: {}, {}, {}\n", _vert.x, _vert.y, _vert.z);
      // fmt::print("Norm: {}, {}, {}\n", _norm.x, _norm.y, _norm.z);
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
        // fmt::print("{} ", _face.mIndices[j]);
        _elements.push_back(_face.mIndices[j]);
      }
    }
    putchar('\n');

    if (mesh->mMaterialIndex >= 0) {
      aiMaterial* _mtl = scene->mMaterials[mesh->mMaterialIndex];
      // fmt::print("MtlName: {}\n", _mtl->GetName().data);

      _color_card = load_material_colors(_mtl);
      // fmt::print("Kd: {},{},{}\n", _color_card.diffuse.r, _color_card.diffuse.g, _color_card.diffuse.b);

      if (_mtl->GetTextureCount(aiTextureType_DIFFUSE) && _mtl->GetTextureCount(aiTextureType_SPECULAR)) {
        std::vector<Texture> diffuse_maps = load_material_textures(_mtl, aiTextureType_DIFFUSE, "texture_diffuse");
        _textures.insert(_textures.end(), diffuse_maps.begin(), diffuse_maps.end());
        std::vector<Texture> specular_maps = load_material_textures(_mtl, aiTextureType_SPECULAR, "texture_specular");
        _textures.insert(_textures.end(), specular_maps.begin(), specular_maps.end());
      }
    }
    if (_textures.empty()) {
      this->meshes.emplace_back(_vertices, _elements, _color_card);
      // std::cout << "Colored mesh\n";
    } else {
      this->meshes.emplace_back(_vertices, _elements, _textures);
      // std::cout << "Textured mesh\n";
    }
  }

  ColorCard<glm::vec3> load_material_colors(aiMaterial* mtl) {
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

  std::vector<Texture> load_material_textures(aiMaterial* mtl, aiTextureType type, std::string type_s) {
    // TODO: not tested
    std::vector<Texture> textures;
    for (int i = 0; i < mtl->GetTextureCount(type); ++i) {
      aiString str;
      mtl->GetTexture(type, i, &str);
      Texture texture;
      texture.id = kit::make_texture(std::string(str.data), directory);
      texture.type = type_s;
      texture.path = std::string(str.data);
      textures.push_back(texture);
    }
    return textures;
  }

  void draw(const Shader& shader, const std::string& prefix = {}) {
    for (Mesh& mesh : this->meshes) {
      mesh.draw(shader, Mesh::DrawMode::Color, prefix);
    }
  }
}; // Model

}
