#include "kit/mesh_proc.h"
#include "kit/cam_inst.h"
#include "shader_m.h"

#include <glm/ext/matrix_transform.hpp>
#include <iostream>
#include <ostream>
#include <vector>
#include <GLFW/glfw3.h>


static struct WindowInfo {
  int width {1600}, height {1000};
  float aspect {1.6f};
  const char* title {"kit::mesh_proc"};
  bool resizable {false};
  bool polygon_mode{false};
  bool depth_test{true};
} wnd_info;

struct DirectionalLight {
  glm::vec3 ambient{1.f}, diffuse{1.f}, specular{1.f}, direction{0.f};
};


GLFWwindow* __Glad_to_be_on_air(int maj_ver=3, int min_ver=3) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, maj_ver);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, min_ver);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(wnd_info.width, wnd_info.height, wnd_info.title, nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSetWindowAttrib(window, GLFW_RESIZABLE, wnd_info.resizable);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD\n";
    return nullptr;
  } // use gl*Api after glad init
  return window;
}

using namespace kit;

void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, true);

  auto& cam = kit::CamInst::get_instance();
  cam.process_keypress(window);
}

int main() {

  GLFWwindow* win = __Glad_to_be_on_air(3,3);
  if (!win) ::exit(-1);

  // __Mesh_func();
  Shader pro {"assets/pro.vert", "assets/pro.frag"};
  Model cornell_box;
  cornell_box.load_model("assets/CornellBox-Original.obj");

  CamInst& cam = CamInst::get_instance();
  cam.position = {0.007245103, 1.3137426, 5.303344};
  cam.target = {0.f, 0.f, 0.f};
  cam.front = {0.f, 0.f, -1.f};
  cam.update();

  if (wnd_info.polygon_mode) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  if (wnd_info.depth_test) glEnable(GL_DEPTH_TEST);

  constexpr float framerate = 1.f / 120;

  while (!glfwWindowShouldClose(win)) {
    cam.tick(framerate);
    process_input(win);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.1f, 0.0f, 0.1f, 1.0f);

    glm::mat4 model{1.0f};
    auto view = glm::lookAt(cam.position, cam.position + cam.front, cam.up);
    auto projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.01f, 120.f);
    auto mvp = projection * view * model;

    pro.setMat4("mvp", mvp);
    pro.setVec3("view_pos", cam.position);

    DirectionalLight lt;
    lt.direction = glm::vec3(0.f, 0.f, -1.f);
    pro.setVec3("lt.ambient", lt.ambient);
    pro.setVec3("lt.diffuse", lt.diffuse);
    pro.setVec3("lt.specular", lt.specular);
    pro.setVec3("lt.direction", lt.direction);

    pro.setBool("_Normal_rgb_check", false);
    cornell_box.draw(pro, "mtl");

    glfwSwapBuffers(win);
    glfwPollEvents();
  }

  return 0;

}

