#include "kit/mesh_proc.h"
#include "kit/cam_inst.h"
#include "kit/exception.h"
#include "shader_m.h"

#include <array>
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
    { "monkey_smooth", Model{"assets/models/monkey/monkey_smooth.obj"} },
    { "cornell_box", Model{"assets/models/cornell_box/CornellBox-Original.obj"} }
  };

  renderables["sponza"].model_matrix = glm::rotate(
    glm::scale(glm::mat4{1.0f}, glm::vec3{0.1f}),
    glm::radians(0.0f),
    glm::vec3{1.0f, 0.0f, 0.0f}
  );

  DirectionalLight lt;
  lt.direction = glm::vec3{0.f, 0.f, -1.f};
  lt.ambient = glm::vec3{0.5f};
  lt.diffuse = glm::vec3{0.5f};
  lt.specular = glm::vec3{0.0f};

  CamInst& cam = CamInst::get_instance();
  cam.position = {0.1f, 1.0f, 4.333f};
  //cam.position = {-0.98f, 0.676f, 0.0f};
  cam.target = {0.f, 0.f, 0.f};
  cam.front = {0.0f, 0.f, -1.f};
  //cam.front = {1.0f, 0.f, 0.0f};
  cam.update();

  constexpr float framerate = 1.f / 120;

  // UI states
  ImGuiStyle& ui_style = ImGui::GetStyle();
  float spacing = ui_style.ItemInnerSpacing.x;

  glm::vec4 clear_color = glm::vec4{0.12f, 0.0f, 0.16f, 1.0f};
  float f = 0.0f;
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
  bool ui_ploygon_mode = false;

  while (!glfwWindowShouldClose(window_ptr)) {
    glfwPollEvents();
    cam.tick(framerate);
    process_input(window_ptr);

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui::ShowDemoWindow();
    {
      ImGui::Begin("Basic");                          // Create a window called "Hello, world!" and append into it.

      ImGui::ColorEdit3("Clear color", glm::value_ptr(clear_color));

      ImGui::Combo("Model", &ui_curr_obj, ui_model_objs.data(), ui_model_objs.size());
      obj = renderables[std::string(ui_model_objs[ui_curr_obj])];

      ImGui::SliderFloat("scale", &ui_model_scale, 0.1f, 1.5f);

      ImGui::Combo("Shader", &ui_curr_program, ui_programs.data(), ui_programs.size());
      pro = programs[std::string(ui_programs[ui_curr_program])];

      ImGui::Checkbox("Normal as RGB", &ui_norm_coloring);
      ImGui::Checkbox("Polygon mode", &ui_ploygon_mode);

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
      ImGui::End();
    }
    {
      ImGui::Begin("Camera");

      ImGui::DragFloat3("position", glm::value_ptr(cam.position));
      ImGui::DragFloat3("target", glm::value_ptr(cam.target));
      ImGui::DragFloat3("front", glm::value_ptr(cam.front));

      ImGui::Text("Camera hits");
      ImGui::Text("box/mokey: pos{0.0,0.3,5.5}, front{0,0,-1}");
      ImGui::Text("sponza: pos{-1,0.6,0}, front{1,0,0}");

      ImGui::End();
    }
    ImGui::Render();

    glPolygonMode(GL_FRONT_AND_BACK, ui_ploygon_mode ? GL_LINE : GL_FILL);
    glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //for (auto&[key, obj] : renderables) {
    //float ro_angle = 60.0f * static_cast<float>(glfwGetTime());

    auto model = glm::scale(obj.model_matrix, glm::vec3{ui_model_scale});
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

