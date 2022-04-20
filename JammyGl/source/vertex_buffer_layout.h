#pragma once

#include <glad/glad.h>
#include <vector>
#include "renderer.h"

struct VertexBufferElement {
  unsigned int type;
  unsigned int count;
  unsigned char normalized;

  VertexBufferElement(unsigned int type, unsigned int count, unsigned char normalized)
      : type(type), count(count), normalized(normalized) {}

  static size_t GetSizeofType(unsigned int type) {
    switch (type) {
      case (GL_FLOAT):
        return 4;
      case (GL_UNSIGNED_INT):
        return 4;
      case (GL_UNSIGNED_BYTE):
        return 1;
    }
    MsvcAssert(false);
    return 0;
  }
};

class VertexBufferLayout {
 private:
  std::vector<VertexBufferElement> m_Elements;
  unsigned int m_Stride;

 public:
  VertexBufferLayout() : m_Stride(0) {}

  const auto GetElements() const {
    return m_Elements;
  }
  unsigned int GetStride() const {
    return m_Stride;
  }

  template <typename T>
  void Push(unsigned int count) {
    static_assert(false);
  }

  template <>
  void Push<float>(unsigned int count) {
    m_Elements.emplace_back(GL_FLOAT, count, GL_FALSE);
    m_Stride += sizeof(GLfloat) * count;
  }

  template <>
  void Push<unsigned int>(unsigned int count) {
    m_Elements.emplace_back(GL_UNSIGNED_INT, count, GL_TRUE);
    m_Stride += sizeof(GLuint) * count;
  }

  template <>
  void Push<unsigned char>(unsigned int count) {
    m_Elements.emplace_back(GL_UNSIGNED_BYTE, count, GL_TRUE);
    m_Stride += sizeof(GLubyte) * count;
  }
};
