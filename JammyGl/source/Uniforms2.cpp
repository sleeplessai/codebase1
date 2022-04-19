#ifndef _glfw3_h_
#include <glad/glad.h>
#endif
#include <GLFW/glfw3.h>

#include <iostream>

#include "index_buffer.h"
#include "renderer.h"
#include "utils.h"
#include "vertex_array.h"
#include "vertex_buffer.h"
#include "vertex_buffer_layout.h"

static uint32_t CompileShader(uint32_t shader_t, const std::string& source) {
  uint32_t id = glCreateShader(shader_t);
  const char* src = source.c_str();
  glShaderSource(id, 1, &src, nullptr);

  glCompileShader(id);
  int result;
  glGetShaderiv(id, GL_COMPILE_STATUS, &result);
  if (result == GL_FALSE) {
    int length;
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
    // alloca() uses to allocate dynamic space on stack heap.
    char* message = (char*)alloca(sizeof(length));
    glGetShaderInfoLog(id, length, &length, message);
    std::cerr << "Failed to compile shader!" << std::endl;
    std::cerr << "Error occurred at "
              << (shader_t == GL_VERTEX_SHADER ? "vertex" : "fragment")
              << std::endl;
    std::cerr << message << std::endl;
    glDeleteShader(id);
    return 0;
  }
  return id;
}

static int CreateShader(const std::string& vertexShader,
                        const std::string& fragmentShader) {
  uint32_t program = glCreateProgram();
  uint32_t vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
  uint32_t fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  glValidateProgram(program);

  glDeleteShader(vs);
  glDeleteShader(fs);

  return program;
}

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

  // Vertex data, Vertex array, Vertex buffer
  {
    float positions[8] = {-0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f};
    uint32_t indices[6] = {0, 1, 2, 2, 3, 0};

    static VertexArray vao;
    VertexBuffer vbo(positions, 4 * 2 * sizeof(float));
    VertexBufferLayout layout;
    layout.Push<float>(2);
    vao.AddBuffer(vbo, layout);
    IndexBuffer ibo(indices, 6);

    fs::path shPath = fs::path("../../shader/uniforms0.shader");
    std::vector basicShader = jammygl::ParseShaderSource(shPath);
    uint32_t shader = CreateShader(basicShader[0], basicShader[1]);
    int32_t location = glGetUniformLocation(shader, "u_Color");
    MsvcAssert(location != -1);

    float r = 0.0f, delta = 0.01f;

    while (!glfwWindowShouldClose(window)) {
      /* Render here */
      glClear(GL_COLOR_BUFFER_BIT);

      GlCall(glUseProgram(shader));
      vao.Bind();
      ibo.Bind();

      GlCall(glUniform4f(location, r, 0.5f, 0.3f, 1.0f));
      if (r >= 1.0f) {
        delta = -0.01f;
      } else if (r <= 0.0f) {
        delta = +0.01f;
      }
      r += delta;

      GlCall(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr));

      /* Swap front and back buffers */
      glfwSwapBuffers(window);

      /* Poll for and process events */
      glfwPollEvents();
    }

    glDeleteProgram(shader);
  }
  glfwTerminate();

  return 0;
}