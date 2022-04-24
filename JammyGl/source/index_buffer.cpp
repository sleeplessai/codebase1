#include "index_buffer.h"
#include "gl_helper.h"

IndexBuffer::IndexBuffer(const unsigned int* data, unsigned int count) : m_Count(count) {
  GlAssert(sizeof(unsigned int) == sizeof(GLuint));

  GlCall(glGenBuffers(1, &m_Id));
  GlCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Id));
  GlCall(glBufferData(GL_ELEMENT_ARRAY_BUFFER, count * sizeof(unsigned int), data, GL_STATIC_DRAW));
}

IndexBuffer::~IndexBuffer() {
  GlCall(glDeleteBuffers(1, &m_Id));
}

void IndexBuffer::Bind() const {
  GlCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Id));
}

void IndexBuffer::Unbind() const {
  GlCall(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}
