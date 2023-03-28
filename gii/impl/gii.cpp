#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <string_view>
#include <fmt/core.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "gii.h"


namespace gii {

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    Gii::get_instance().win_info.set_width(width).set_height(height);
}

void Gii::initilize() {
    /* GLFW Init, GLAD Load */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    this->win_info.set_title("main_ui: window").set_width(1280).set_height(720);
    this->window = glfwCreateWindow(win_info.width, win_info.height, win_info.title.c_str(), nullptr, nullptr);

    glfwMakeContextCurrent(window);
    set_glfw_callbacks();

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    } // use gl*Api after glad init

    const std::string gl_render = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const std::string gl_version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    fmt::print("opengl_device: {}\nopengl_version: {}\n", gl_render, gl_version);

    /* ImGui Init */
    const char* glsl_version = "#version 450";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;     // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;
    io.ConfigViewportsNoTaskBarIcon = true;

    // Setup Dear ImGui style
    if (!gui_conf.dark_mode) {
        ImGui::StyleColorsLight();
    } else {
        ImGui::StyleColorsDark();
    }

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;

    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void Gii::present() {
    while (!glfwWindowShouldClose(this->window)) {
        glfwPollEvents();
        update();
        render();
    }
    terminate();
}

void Gii::update() {

}

void Gii::set_glfw_callbacks() {
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // glfwSetCursorPosCallback(window, mouse_callback);
    // glfwSetScrollCallback(window, scroll_callback);
}

void Gii::render() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    static ImVec4 clear_color = ImVec4(239.0f/255.0f, 248.0f/255.0f, 247.0f/255.0f, 1.0f);
    // fmt::print("{}, {}, {}\n", clear_color.x, clear_color.y, clear_color.z);

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    if (gui_conf.show_demo_window)
        ImGui::ShowDemoWindow(&gui_conf.show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &gui_conf.show_demo_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &gui_conf.show_another_window);
        ImGui::Checkbox("Dark mode", &gui_conf.dark_mode);

        if (gui_conf.dark_mode) {
            ImGui::StyleColorsDark();
        } else {
            ImGui::StyleColorsLight();
        }

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);    // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color);     // Edit 3 floats representing a color
        ImGui::ColorEdit4("color_edit4", reinterpret_cast<float*>(&clear_color));

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }

    // 3. Show another simple window.
    if (gui_conf.show_another_window) {
        ImGui::Begin("Another Window", &gui_conf.show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
        ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
            gui_conf.show_another_window = false;
        ImGui::End();
    }

    // Rendering
    ImGui::Render();
    // int display_w, display_h;
    // glfwGetFramebufferSize(window, &display_w, &display_h);
    // glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
    //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(window);
}

void Gii::terminate() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

Gii::~Gii() {

}

}
