#ifndef _glfw3_h_
#include <glad/glad.h>
#endif
#include <GLFW/glfw3.h>

#include <iostream>

#include "index_buffer.h"
#include "renderer.h"
#include "shader.h"
#include "vertex_array.h"
#include "vertex_buffer.h"
#include "vertex_buffer_layout.h"

int main() {
  GLFWwindow* window;

  /* Initialize the library */
  if (!glfwInit())
    return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  /* Create a windowed mode window and its OpenGL context */
  window = glfwCreateWindow(640, 480, "Colorful Rectangle", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  /* Make the window's context current */
  glfwMakeContextCurrent(window);
  // V-sync on
  glfwSwapInterval(1);

  // Gl loading by Glad
  // https://www.khronos.org/opengl/wiki/OpenGL_Loading_Library
  if (!gladLoadGL()) {
    std::cout << "Failed to initialize OpenGL context" << std::endl;
    return -1;
  }
  std::cout << "GL_VERSION: " << glGetString(GL_VERSION) << std::endl;

  // Main scope: vertex buffer, vbo and vbo layout, vao, ibo
  {
    float positions[8] = {-0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f};
    unsigned int indices[6] = {0, 1, 2, 2, 3, 0};

    static VertexArray vao;
    VertexBuffer vbo(positions, 4 * 2 * sizeof(float));
    VertexBufferLayout layout;
    layout.Push<float>(2);
    vao.AddBuffer(vbo, layout);
    IndexBuffer ibo(indices, 6);
    Shader shader(fs::path("../../shader/uniforms0.shader"));
    Renderer renderer;

    float r = 0.0f, delta = 0.01f;

    while (!glfwWindowShouldClose(window)) {
      /* Render here */
      renderer.Clear();
      renderer.Draw(vao, ibo, shader);
      shader.SetUniform4f("u_Color", r, 0.5f, 0.3f, 1.0f);

      if (r >= 1.0f) {
        delta = -0.01f;
      } else if (r <= 0.0f) {
        delta = +0.01f;
      }
      r += delta;

      /* Swap front and back buffers */
      glfwSwapBuffers(window);

      /* Poll for and process events */
      glfwPollEvents();
    }
  }
  glfwTerminate();

  return 0;
}
