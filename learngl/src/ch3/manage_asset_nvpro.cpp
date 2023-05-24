#include "kit/mesh_proc.h"
#include "kit/exception.h"
#include "shader_m.h"

#include <array>
#include <iostream>
#include <ostream>
#include <unordered_map>
#include <vector>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imgui.h>

#include <nvh/cameracontrol.hpp>
#include <nvmath/nvmath.h>
#include <nvmath/nvmath_glsltypes.h>

using namespace kit;
using namespace nvmath;


struct WindowCreateInfo {
  int width{1600}, height{1000};
  float aspect() {
    return static_cast<float>(width)/height;
  };
  std::string title{"Ch3: Asset management (nvpro)"};
  bool resizable{false};
  bool depth_test{true};
  GLenum polygon_mode{GL_FILL};
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
  if (p_create_info->polygon_mode) {
    glPolygonMode(GL_FRONT_AND_BACK, p_create_info->polygon_mode);
  }
  if (p_create_info->depth_test) {
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
  }
  return window;
}

namespace imguih {

inline bool mouse_pos(int x, int y) {
  auto& io = ImGui::GetIO();
  io.AddMousePosEvent((float)x, (float)y);
  return io.WantCaptureMouse;
}

inline bool mouse_button(int button, int action) {
  auto& io = ImGui::GetIO();
  io.AddMouseButtonEvent(button, action == GLFW_PRESS);
  return io.WantCaptureMouse;
}

inline bool mouse_wheel(int wheel) {
  auto& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0, (float)wheel);
  return io.WantCaptureMouse;
}

inline bool key_char(int button) {
  auto& io = ImGui::GetIO();
  io.AddInputCharacter(static_cast<unsigned int>(button));
  return io.WantCaptureKeyboard;
}

} // namespace imguih


int main() {
  WindowCreateInfo window_info = {};
  window_info.polygon_mode = false;
  auto* window_ptr = make_window_ptr(&window_info);
  if (!window_ptr) {
    THROW_RT_ERR
  }
  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  //io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  // Setup Dear ImGui style
  ImGui::StyleColorsLight();
  //Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window_ptr, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  std::unordered_map<std::string, Shader> programs = {
    {"color", Shader{"shaders/ch3/pro.vert", "shaders/ch3/pro.frag"}},
    {"tex", Shader{"shaders/ch3/pro.vert", "shaders/ch3/pro1.frag"}},
  };

  std::unordered_map<std::string, Model> renderables = {
    { "sponza", Model{"assets/models/sponza/sponza.obj"} },
    { "cornell_box", Model{"assets/models/cornell_box/CornellBox-Original.obj"} },
    { "monkey_smooth", Model{"assets/models/monkey/monkey_smooth.obj"} },
  };

  DirectionalLight lt;
  lt.direction = glm::vec3{0.f, 0.f, -1.f};
  lt.ambient = glm::vec3{0.5f};
  lt.diffuse = glm::vec3{0.5f};
  lt.specular = glm::vec3{0.0f};

  constexpr int grid = 64;
  nvh::CameraControl nvcam;
  nvcam.m_useOrbit = true;
  nvcam.m_sceneOrbit = nvmath::vec3(0.0f);
  nvcam.m_sceneDimension = float(grid) * 0.2f;
  nvcam.m_viewMatrix = nvmath::look_at(nvcam.m_sceneOrbit - nvmath::vec3(0, 0, -nvcam.m_sceneDimension), nvcam.m_sceneOrbit, nvmath::vec3(0, 1, 0));

  // UI states
  ImGuiStyle& ui_style = ImGui::GetStyle();
  float spacing = ui_style.ItemInnerSpacing.x;

  vec4 clear_color = {0.12f, 0.0f, 0.16f, 1.0f};
  std::vector<const char*> ui_programs;
  for (const auto& p : programs) {
    ui_programs.push_back(p.first.c_str());
  }
  int ui_curr_program = 0;
  Shader pro = programs[std::string(ui_programs[ui_curr_program])];
  std::vector<const char*> ui_model_objs;
  for (const auto& o : renderables) {
    ui_model_objs.push_back(o.first.c_str());
  }
  int ui_curr_obj = 0;
  Model obj = renderables[std::string(ui_model_objs[ui_curr_obj])];
  float ui_model_scale = 1.0f;
  bool ui_norm_coloring = false;
  static double ui_curr_mouse_x, ui_curr_mouse_y;
  vec2i ui_window_sz{window_info.width, window_info.height};
  static int curr_mouse_button = 0, curr_mouse_wheel = 0;
  std::array<const char*, 3> ui_polygon_modes = {"GL_POINT", "GL_FILL", "GL_LINE"};
  std::array<GLenum, 3> ui_gl_polygon_modes = {GL_POINT, GL_FILL, GL_LINE};
  int ui_curr_ploygon_mode = 1;

  glfwSetMouseButtonCallback(window_ptr, [](GLFWwindow* window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    double mosue_x = 0.0, mouse_y = 0.0;
    glfwGetCursorPos(window, &mosue_x, &mouse_y);

    if (imguih::mouse_pos((int)mosue_x, (int)mouse_y)) {
      return;
    }
    if (action == GLFW_PRESS) {
      if (button == GLFW_MOUSE_BUTTON_LEFT) {
        curr_mouse_button = 1;
      } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        curr_mouse_button = 2;
      }
    }
    if (action == GLFW_RELEASE) {
      curr_mouse_button = 0;
    }
  });
  glfwSetScrollCallback(window_ptr, [](GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    const int mouse_w = (int)yoffset;
    if (imguih::mouse_wheel(mouse_w)) {
      return;
    }
    if (mouse_w > 0) {
      curr_mouse_wheel += 1;
    } else if (mouse_w < 0) {
      curr_mouse_wheel -= 1;
    }
  });
  auto process_input = [](GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, true);

    glfwGetCursorPos(window, &ui_curr_mouse_x, &ui_curr_mouse_y);
  };

  while (!glfwWindowShouldClose(window_ptr)) {
    glfwPollEvents();

    process_input(window_ptr);
    vec2f curr_mouse_pos{ui_curr_mouse_x, ui_curr_mouse_y};
    nvcam.processActions(ui_window_sz, curr_mouse_pos, curr_mouse_button, curr_mouse_wheel);

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
      ImGui::Begin("Basic");

      ImGui::Text("Rendering");
      ImGui::ColorEdit3("clear color", clear_color.vec_array);

      ImGui::Combo("model", &ui_curr_obj, ui_model_objs.data(), ui_model_objs.size());
      obj = renderables[std::string(ui_model_objs[ui_curr_obj])];

      ImGui::SliderFloat("scale factor", &ui_model_scale, 0.2f, 1.2f);

      ImGui::Combo("shader", &ui_curr_program, ui_programs.data(), ui_programs.size());
      pro = programs[std::string(ui_programs[ui_curr_program])];

      ImGui::Checkbox("normal as RGB", &ui_norm_coloring);
      ImGui::Combo("polygon mode", &ui_curr_ploygon_mode, ui_polygon_modes.data(), ui_polygon_modes.size());

      ImGui::Text("Camera");
      ImGui::Checkbox("Orbit", &nvcam.m_useOrbit);
      ImGui::DragFloat3("eye", nvcam.m_sceneOrbit.vec_array);
      ImGui::SliderFloat("dimension", &nvcam.m_sceneDimension, 0.5f, (float)grid / 2);

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
      ImGui::End();
    }
    ImGui::Render();

    glPolygonMode(GL_FRONT_AND_BACK, ui_gl_polygon_modes[ui_curr_ploygon_mode]);
    glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto model = mat4(value_ptr(glm::scale(obj.model_matrix, glm::vec3(ui_model_scale))));
    auto view = nvcam.m_viewMatrix;
    auto proj = perspective(45.0f, window_info.aspect(), 0.1f, 1000.0f);
    auto mvp = proj * view * model;
    auto eye = nvcam.m_sceneOrbit - nvmath::vec3(0, 0, -nvcam.m_sceneDimension);

    pro.setMat4("mvp", glm::make_mat4(mvp.mat_array));
    pro.setMat4("model", glm::make_mat4(model.mat_array));
    pro.setVec3("view_pos", glm::make_vec3(eye.vec_array));

    pro.setVec3("lt.ambient", lt.ambient);
    pro.setVec3("lt.diffuse", lt.diffuse);
    pro.setVec3("lt.specular", lt.specular);
    pro.setVec3("lt.direction", lt.direction);

    pro.setBool("_Normal_as_color", ui_norm_coloring);
    obj.draw(pro, "mtl", DrawMode::Auto);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_ptr);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window_ptr);
  glfwTerminate();

  return 0;
}

