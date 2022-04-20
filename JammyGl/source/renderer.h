#pragma once

#ifndef _glfw3_h_
#include <glad/glad.h>
#endif
#include <GLFW/glfw3.h>

#include "index_buffer.h"
#include "shader.h"
#include "vertex_array.h"

#define MsvcAssert(x) \
  if (!(x))           \
    __debugbreak();

#define GlCall(x) \
  ClearGlError(); \
  x;              \
  MsvcAssert(CheckGlError(#x, __FILE__, __LINE__))

void ClearGlError();
bool CheckGlError(const char* function, const char* file, int line);

class Renderer {
 public:
  void Clear() const;
  void Draw(const VertexArray& vao, const IndexBuffer& ibo, const Shader& shader) const;
};
