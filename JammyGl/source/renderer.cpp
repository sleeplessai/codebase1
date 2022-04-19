#include "renderer.h"
#include <iostream>

void ClearGlError() {
  while (glGetError() != GL_NO_ERROR)
    ;
}

bool CheckGlError(const char* function, const char* file, int line) {
  while (GLenum error = glGetError()) {
    std::cerr << "[OpenGL Error] (" << std::hex << error << "): " << function
              << " " << file << ":" << std::dec << line << std::endl;
    // Check hex error code in glew.h header file
    return false;
  }
  return true;
}
