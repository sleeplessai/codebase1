#ifndef _glfw3_h_
#include <glad/glad.h>
#endif
#include <GLFW/glfw3.h>

//#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <limits>

#include "gl_helper.h"
#include "index_buffer.h"
#include "renderer.h"
#include "shader.h"
#include "texture.h"
#include "vertex_array.h"
#include "vertex_buffer.h"
#include "vertex_buffer_layout.h"

constexpr static float PI = 3.1415927F;

void compute_pentagon_vertex_pos(float* pos, float x0 = 1.0f, float y0 = 0.0f) {
  float R = std::sqrt(x0 * x0 + y0 * y0);
  GlAssert(std::abs(R - 0.f) >= std::numeric_limits<float>::epsilon());

  float theta = std::acos(x0 / R);
  float delta = 0.4f * (float)PI;

  for (int i = 0; i <= 16; i += 4) {
    pos[i] = R * std::cos(theta);
    pos[i + 1] = R * std::sin(theta);
    theta += delta;
    pos[i + 2] = pos[i + 3] = 0.0f;     // texture vertex
  }
  //pos[10] = pos[14] = pos[15] = pos[19] = 1.0f;
}

int main() {
  if (glfwInit() != GLFW_TRUE) {
    std::cerr << "Glfw init error" << std::endl;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(640, 640, "Pentagon", nullptr, nullptr);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  if (!gladLoadGL()) {
    std::cout << "Failed to initialize OpenGL context" << std::endl;
    return -1;
  }
  std::cout << "GL_VERSION: " << glGetString(GL_VERSION) << std::endl;

  {
    float vbo_data[20];
    unsigned int vbo_size = 2 * 5 * sizeof(decltype(vbo_data));
    compute_pentagon_vertex_pos(vbo_data, -0.5f, 0.4f);
    unsigned int ibo_data[9] = {0, 1, 2, 0, 2, 3, 0, 3, 4};

    //GlCall(glEnable(GL_BLEND));
    //GlCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    GlCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE));

    VertexBuffer vbo(vbo_data, vbo_size);
    static VertexArray vao;
    VertexBufferLayout layout;
    layout.Push<float>(2);
    layout.Push<float>(2);
    vao.AddBuffer(vbo, layout);
    IndexBuffer ibo(ibo_data, 9);
    Shader shader{"../../shader/uniforms0.shader"};
    shader.Bind();
    // Texture texture{"../../texture/2018063.jfif"};
    // texture.Bind();
    // shader.SetUniform1i("u_Texture", 0);
    // shader.Unbind();
    Renderer renderer;

    float r_co = 0.0f, delta = 0.01f;

    while (!glfwWindowShouldClose(window)) {
      renderer.Clear();
      renderer.Draw(vao, ibo, shader);
      shader.SetUniform4f("u_Color", r_co, r_co + 0.3f, r_co + 0.3f, 1.0f);
      if (r_co >= 1.0f) {
        delta = -0.01f;
      } else if (r_co <= 0.0f) {
        delta = +0.01f;
      }
      r_co += delta;

      glfwSwapBuffers(window);
      glfwPollEvents();
    }
  }
  glfwTerminate();

  return 0;
}
