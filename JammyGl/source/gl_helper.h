#pragma once

#include <glad/glad.h>
#include <iostream>

#define GlAssert(x) \
  if (!(x))         \
    __debugbreak();

#define GlCall(x) \
  ClearGlError(); \
  x;              \
  GlAssert(CheckGlError(#x, __FILE__, __LINE__))

static void ClearGlError() {
  while (glGetError() != GL_NO_ERROR)
    ;
}

static bool CheckGlError(const char* function, const char* file, int line) {
  while (GLenum error = glGetError()) {
    std::cerr << "[OpenGL Error] (" << std::hex << error << "): " << function << " " << file << ":"
              << std::dec << line << std::endl;
    // Check hex error code in glew.h header file
    return false;
  }
  return true;
}
