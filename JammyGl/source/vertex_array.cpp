#include "vertex_array.h"
#include "renderer.h"

VertexArray::VertexArray() {
  GlCall(glGenVertexArrays(1, &m_RendererID));
}

VertexArray::~VertexArray() {
  GlCall(glDeleteVertexArrays(1, &m_RendererID));
}

void VertexArray::AddBuffer(const VertexBuffer& vb,
                            const VertexBufferLayout& layout) {
  this->Bind();  
  vb.Bind();
  const auto& elements = layout.GetElements();
  size_t offset = 0;
  for (unsigned int i = 0; i < elements.size(); ++i) {
    const auto& e = elements[i];
    // glVertexAttribPointer links Array_buffer buffer to VAO
    GlCall(glEnableVertexAttribArray(i));
    GlCall(glVertexAttribPointer(i, e.count, e.type, e.normalized,
                                 layout.GetStride(), (const void*)offset));
    offset += e.count * VertexBufferElement::GetSizeofType(e.type);
  }
}

void VertexArray::Bind() const {
  GlCall(glBindVertexArray(m_RendererID));
}

void VertexArray::Unbind() const {
  GlCall(glBindVertexArray(0));
}
