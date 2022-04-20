#pragma once

#include "vertex_buffer.h"

class VertexBufferLayout;

class VertexArray {
 private:
  unsigned int m_RendererId;

 public:
  VertexArray();
  ~VertexArray();

  void AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layer);

  void Bind() const;
  void Unbind() const;
};
