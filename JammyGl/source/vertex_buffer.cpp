#include "vertex_buffer.h"
#include "gl_helper.h"

VertexBuffer::VertexBuffer(const void* data, unsigned int size) {
  GlCall(glGenBuffers(1, &m_Id));
  GlCall(glBindBuffer(GL_ARRAY_BUFFER, m_Id));
  GlCall(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW));
}

VertexBuffer::~VertexBuffer() {
  GlCall(glDeleteBuffers(1, &m_Id));
}

void VertexBuffer::Bind() const {
  GlCall(glBindBuffer(GL_ARRAY_BUFFER, m_Id));
}

void VertexBuffer::Unbind() const {
  GlCall(glBindBuffer(GL_ARRAY_BUFFER, 0));
}
