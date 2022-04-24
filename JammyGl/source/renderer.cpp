#include "renderer.h"
#include "gl_helper.h"

#include <iostream>

void Renderer::Clear() const {
  glClear(GL_COLOR_BUFFER_BIT);
}

void Renderer::Draw(const VertexArray& vao, const IndexBuffer& ibo, const Shader& shader) const {
  vao.Bind();
  ibo.Bind();
  shader.Bind();
  GlCall(glDrawElements(GL_TRIANGLES, ibo.GetCount(), GL_UNSIGNED_INT, nullptr));
}
