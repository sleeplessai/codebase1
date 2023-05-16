#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string_view>

#include "stb_image.h"
#include "fmt/core.h"

#include "kit/shader_v1.h"
#include "kit/cam_inst.h"


void process_input(GLFWwindow *window);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

static struct WindowInfo {
    int width {800};
    int height {500};
    const char* title {"Ch.2 Phong lighting"};
    float aspect {1.6f};  // 800/500

    void update(GLFWwindow *window) noexcept {
        glfwGetWindowSize(window, &width, &height);
        aspect = static_cast<float>(width) / height;
    }
} wnd_info;

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(wnd_info.width, wnd_info.height, wnd_info.title, nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    } // use gl*Api after glad init
    std::string_view gl_version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    std::string_view gl_render = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    fmt::print("opengl_device: {}\nopengl_version: {}\n", gl_render, gl_version);

    std::vector<float> vertices = {
#include "../assets/ch2/cube_normal.inc"
    };
    glm::vec3 cube_pos {0.0f, 0.0f, 0.0f};
    glm::vec3 light_pos {2.0f, 1.0f, 1.5f};
    glm::vec3 cube_color {1.0f, 0.5f, 0.31f};
    glm::vec3 light_color {1.0f, 1.0f, 1.0f};

    light_pos = {0.0f, 1.1f, -1.5f};

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, reinterpret_cast<void*>(sizeof(float)*3));
    glEnableVertexAttribArray(1);

    unsigned int vao_light;
    glGenVertexArrays(1, &vao_light);
    glBindVertexArray(vao_light);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);

    kit::Shader cube {"shaders/ch2/1-cube.vs", "shaders/ch2/1-cube.fs"};
    kit::Shader phong {"shaders/ch2/2-phong.vs", "shaders/ch2/2-phong.fs"};

    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_DEPTH_TEST);

    // free look camera singleton instance
    kit::CamInst& cam = kit::CamInst::get_instance();
    cam.last_mouse_pos = glm::vec2(wnd_info.width / 2.f, wnd_info.height / 2.f);
    cam.position = {1.1f, 0.36f, 5.2f};
    cam.position = {.25f, 1.51f, 5.12f};
    cam.front = {0.0f, 0.0f, -1.0f};
    cam.target = cube_pos;
    cam.update();

    float frame_time {0.0f}, last_frame {0.0f};

    while (!glfwWindowShouldClose(window)) {
        float curr_frame = static_cast<float>(glfwGetTime());
        cam.tick(curr_frame - last_frame);
        last_frame = curr_frame;
        process_input(window);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 model {1.0f};
        glm::mat4 view {1.0f};
        glm::mat4 projection {1.0f};

        view = glm::lookAt(cam.position, cam.position + cam.front, cam.up);

        // cubric object placement
        phong.use();
        unsigned int uObjectColorLoc = glGetUniformLocation(phong.ID, "uObjectColor");
        unsigned int uLightColorLoc = glGetUniformLocation(phong.ID, "uLightColor");
        unsigned int uLightPosLoc = glGetUniformLocation(phong.ID, "uLightPos");
        unsigned int uViewPosLoc = glGetUniformLocation(phong.ID, "uViewPos");

        glUniform3fv(uObjectColorLoc, 1, glm::value_ptr(cube_color));
        glUniform3fv(uLightColorLoc, 1, glm::value_ptr(light_color));
        glUniform3fv(uLightPosLoc, 1, glm::value_ptr(light_pos));
        glUniform3fv(uViewPosLoc, 1, glm::value_ptr(cam.position));

        glBindVertexArray(vao);
        model = glm::translate(model, cube_pos);
        model = glm::rotate(model, glm::radians(-25.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.0f);

        unsigned int uModelLoc = glGetUniformLocation(phong.ID, "uModel");
        unsigned int uViewLoc = glGetUniformLocation(phong.ID, "uView");
        unsigned int uProjectionLoc = glGetUniformLocation(phong.ID, "uProjection");

        glUniformMatrix4fv(uModelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(uViewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(uProjectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        glDrawArrays(GL_TRIANGLES, 0, 36);

        // light source placement
        cube.use();
        model = glm::mat4(1.0f);
        model = glm::translate(model, light_pos);
        model = glm::scale(model, glm::vec3(0.2f));
        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.0f);

        unsigned int uMvpLoc = glGetUniformLocation(cube.ID, "uMvp");
        uObjectColorLoc = glGetUniformLocation(cube.ID, "uObjectColor");
        uLightColorLoc = glGetUniformLocation(cube.ID, "uLightColor");

        glm::mat4 mvp_mat4 = projection * view * model;
        glUniformMatrix4fv(uMvpLoc, 1, GL_FALSE, glm::value_ptr(mvp_mat4));
        glUniform3fv(uLightColorLoc, 1, glm::value_ptr(light_color));
        glUniform3fv(uObjectColorLoc, 1, glm::value_ptr(glm::vec3(1.0f)));

        glBindVertexArray(vao_light);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;
}

void process_input(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    auto& cam = kit::CamInst::get_instance();
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cam.position += cam.speed * cam.front;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cam.position -= cam.speed * cam.front;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cam.position -= glm::normalize(glm::cross(cam.front, cam.up)) * cam.speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cam.position += glm::normalize(glm::cross(cam.front, cam.up)) * cam.speed;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cam.position -= cam.speed * cam.up;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cam.position += cam.speed * cam.up;

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        std::puts("\nCurrent camera parameters:");
        std::printf("\tpositions: {%.2f, %.2f, %.2f}\n\tfront: {%.2f, %.2f, %.2f}\n\ttarget{%.2f, %.2f, %.2f}\n\n", \
                    cam.position[0], cam.position[1], cam.position[2], \
                    cam.front[0], cam.front[1], cam.front[2], \
                    cam.target[0], cam.target[1], cam.target[2]);

    }
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    wnd_info.update(window);
    fmt::print("Resized to: ({}, {})\n", wnd_info.width, wnd_info.height);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    auto& cam = kit::CamInst::get_instance();

    if (cam.fovy >= 1.0f && cam.fovy <= 100.0f)
        cam.fovy -= static_cast<float>(yoffset * 5.0f);
    if (cam.fovy <= 1.0f)
        cam.fovy = 1.0f;
    if (cam.fovy >= 100.0f)
        cam.fovy = 100.0f;
}

