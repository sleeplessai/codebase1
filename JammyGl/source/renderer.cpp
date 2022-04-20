#include "renderer.h"
#include "vertex_array.h"

#include <iostream>

void ClearGlError() {
  while (glGetError() != GL_NO_ERROR)
    ;
}

bool CheckGlError(const char* function, const char* file, int line) {
  while (GLenum error = glGetError()) {
    std::cerr << "[OpenGL Error] (" << std::hex << error << "): " << function << " " << file << ":"
              << std::dec << line << std::endl;
    // Check hex error code in glew.h header file
    return false;
  }
  return true;
}

void Renderer::Clear() const {
  glClear(GL_COLOR_BUFFER_BIT);
}

void Renderer::Draw(const VertexArray& vao, const IndexBuffer& ibo, const Shader& shader) const {
  vao.Bind();
  ibo.Bind();
  shader.Bind();
  GlCall(glDrawElements(GL_TRIANGLES, ibo.GetCount(), GL_UNSIGNED_INT, nullptr));
}
