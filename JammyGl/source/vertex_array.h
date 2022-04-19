#pragma once

#include "vertex_buffer.h"
#include "vertex_buffer_layout.h"

//namespace jammygl {
class VertexArray {
 private:
  unsigned int m_RendererID;
 public:
  VertexArray();
  ~VertexArray();

  void AddBuffer(const VertexBuffer& vb, const VertexBufferLayout& layer);

  void Bind() const;
  void Unbind() const;
};
//}  // namespace jammygl