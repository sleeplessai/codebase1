#pragma once

#ifndef _glfw3_h_
#include <glad/glad.h>
#endif
#include <GLFW/glfw3.h>

#define MsvcAssert(x) \
  if (!(x))           \
    __debugbreak();

#define GlCall(x)                                    \
  {                                                  \
    ClearGlError();                                  \
    x;                                               \
    MsvcAssert(CheckGlError(#x, __FILE__, __LINE__)) \
  }

void ClearGlError();
bool CheckGlError(const char* function, const char* file, int line);
