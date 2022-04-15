#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "utils.h"

static unsigned int CompileShader(unsigned int shader_t,
                                  const std::string& source) {
  unsigned int id = glCreateShader(shader_t);
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
  unsigned int program = glCreateProgram();
  unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
  unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

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

  /* Create a windowed mode window and its OpenGL context */
  window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  /* Make the window's context current */
  glfwMakeContextCurrent(window);

  // Glew init after context
  if (!glewInit() == GLEW_OK) {
    std::cout << "Glew_init error occurred!" << std::endl;
  }
  std::cout << "GL_VERSION: " << glGetString(GL_VERSION) << std::endl;

  // Use buffer-shader to render a triangle

  float positions[6] = {-0.5f, -0.5f, 0.0f, 0.5f, 0.5f, -0.5f};

  unsigned int buffer;
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6, positions, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);

  // shaders definition and compilation

  fs::path vsPath = fs::path("../../shader/vertex0.shader");
  fs::path fsPath = fs::path("../../shader/fragment0.shader");
  std::string vertexShader = jammygl::ReadShaderSource(vsPath);
  std::string fragmentShader = jammygl::ReadShaderSource(fsPath);

  // jammygl::WriteShaderSource(vertexShader);
  // jammygl::WriteShaderSource(fragmentShader);

  unsigned int shader = CreateShader(vertexShader, fragmentShader);
  glUseProgram(shader);

  /* Loop until the user closes the window */
  while (!glfwWindowShouldClose(window)) {
    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);

    // Traditional OpenGL not GLSL
    /*
    glBegin(GL_TRIANGLES);
    glVertex2f(-0.5f, -0.5f);
    glVertex2f( 0.0f, 0.5f);
    glVertex2f(0.5f, -0.5f);
    glEnd();
    */
    glDrawArrays(GL_TRIANGLES, 0, 3);

    /* Swap front and back buffers */
    glfwSwapBuffers(window);

    /* Poll for and process events */
    glfwPollEvents();
  }
  glDeleteProgram(shader);
  glfwTerminate();
  return 0;
}