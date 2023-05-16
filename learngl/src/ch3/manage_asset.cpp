#include "kit/mesh_proc.h"
#include "kit/cam_inst.h"
#include "kit/exception.h"
#include "shader_m.h"

#include <glm/ext/matrix_transform.hpp>
#include <iostream>
#include <ostream>
#include <unordered_map>
#include <vector>
#include <GLFW/glfw3.h>

#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imgui.h>


struct WindowCreateInfo {
  int width{1600}, height{1000};
  float aspect() {
    return static_cast<float>(width)/height;
  };
  std::string title{"Ch3: Asset management"};
  bool resizable{false};
  bool polygon_mode{false};
  bool depth_test{true};
};

struct DirectionalLight {
  glm::vec3 ambient{1.f}, diffuse{1.f}, specular{1.f}, direction{0.f};
};


GLFWwindow* make_window_ptr(WindowCreateInfo* p_create_info=nullptr, int major_ver=3, int minor_ver=3) {
  if (!p_create_info)
    return nullptr;

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major_ver);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor_ver);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(p_create_info->width, p_create_info->height, p_create_info->title.c_str(), nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSetWindowAttrib(window, GLFW_RESIZABLE, p_create_info->resizable);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD\n";
    return nullptr;
  } // use gl*Api after glad init

  // window instance configs
  if (p_create_info->polygon_mode) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  if (p_create_info->depth_test) glEnable(GL_DEPTH_TEST);

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
  WindowCreateInfo window_info = {};
  window_info.polygon_mode = false;
  auto* window_ptr = make_window_ptr(&window_info);
  if (!window_ptr) {
    THROW_RT_ERR
  }

  std::unordered_map<std::string, Shader> programs = {
    //{"color", Shader{"shaders/ch3/pro.vert", "shaders/ch3/pro.frag"}},
    {"tex", Shader{"shaders/ch3/pro.vert", "shaders/ch3/pro1.frag"}},
  };
  Shader& pro = programs["tex"];

  std::unordered_map<std::string, Model> renderables = {
    { "sponza", Model{"assets/models/sponza/sponza.obj"} },
    //{ "monkey_smooth", Model{"assets/models/monkey/monkey_smooth.obj"} },
    //{ "cornell_box", Model{"assets/models/cornell_box/CornellBox-Original.obj"} }
  };

  renderables["sponza"].model_matrix = glm::rotate(
    glm::scale(glm::mat4{1.0f}, glm::vec3{0.1f}),
    glm::radians(0.0f),
    glm::vec3{1.0f, 0.0f, 0.0f}
  );

  DirectionalLight lt;
  lt.direction = glm::vec3(0.f, 0.f, -1.f);
  lt.ambient = glm::vec3{0.5f};
  lt.diffuse = glm::vec3{0.5f};
  lt.specular = glm::vec3{0.0f};

  CamInst& cam = CamInst::get_instance();
  //cam.position = {0.0f, 0.3f, 5.5f};
  cam.position = {-0.98f, 0.676f, 0.0f};
  cam.target = {0.f, 0.f, 0.f};
  //cam.front = {0.0f, 0.f, -1.f};
  cam.front = {1.0f, 0.f, 0.0f};
  cam.update();

  constexpr float framerate = 1.f / 120;

  while (!glfwWindowShouldClose(window_ptr)) {
    cam.tick(framerate);
    process_input(window_ptr);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.2f, 0.0f, 0.2f, 1.0f);

    for (auto&[key, obj] : renderables) {
      float ro_angle = 60.0f * static_cast<float>(glfwGetTime());

      // TODO: use Model self-contained model matrix
      auto model = obj.model_matrix;
      auto view = glm::lookAt(cam.position, cam.position + cam.front, cam.up);
      auto proj = glm::perspective(glm::radians(cam.fovy), window_info.aspect(), 0.01f, 120.f);
      auto mvp = proj * view * model;

      pro.setMat4("mvp", mvp);
      pro.setMat4("model", model);
      pro.setVec3("view_pos", cam.position);

      pro.setVec3("lt.ambient", lt.ambient);
      pro.setVec3("lt.diffuse", lt.diffuse);
      pro.setVec3("lt.specular", lt.specular);
      pro.setVec3("lt.direction", lt.direction);

      pro.setBool("_Normal_as_color", false);
      obj.draw(pro, "mtl", DrawMode::Auto);
    }

    glfwSwapBuffers(window_ptr);
    glfwPollEvents();
  }

  return 0;

}

