#pragma once

#include "index_buffer.h"
#include "shader.h"
#include "vertex_array.h"

class Renderer {
 public:
  void Clear() const;
  void Draw(const VertexArray& vao, const IndexBuffer& ibo, const Shader& shader) const;
};
