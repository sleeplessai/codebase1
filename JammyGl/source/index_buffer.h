#pragma once

class IndexBuffer {
 private:
  unsigned int m_RendererId;
  unsigned int m_Count;

 public:
  IndexBuffer(const unsigned int* data, unsigned int count);
  ~IndexBuffer();

  unsigned int GetCount() const {
    return m_Count;
  }

  void Bind() const;
  void Unbind() const;
};
