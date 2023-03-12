#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string_view>

#include "fmt/core.h"

#include "shader_m.h"
#include "kit/cam_inst.h"
#include "kit/tex_proc.h"


void process_input(GLFWwindow *window);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

static struct WindowInfo {
    int width {800};
    int height {500};
    const char* title {"Ch.2 Lighting maps"};
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
    std::string_view gl_version = (const char*)glGetString(GL_VERSION);
    std::string_view gl_render = (const char*)glGetString(GL_RENDERER);
    fmt::print("opengl_device: {}\nopengl_version: {}\n", gl_render, gl_version);

    std::vector<float> vertices = {
#include "assets/cube_norm_tex.inc"
    };
    glm::vec3 object_pos {0.0f, 0.0f, 0.0f};
    glm::vec3 light_pos = {0.0f, 1.1f, -1.5f};
    glm::vec3 object_color {1.0f, 0.5f, 0.31f};
    glm::vec3 light_color {1.0f, 1.0f, 1.0f};

    light_pos = {4.0f, 0.5f, -2.0f};

    struct Material {
        unsigned int diffuse, specular;
        float shininess;
    };
    struct Light {
        glm::vec3 ambient, diffuse, specular, position;
    };
    Material material {
        .diffuse = kit::make_texture("assets/container2.png"),
        .specular = kit::make_texture("assets/container2_specular.png"),
        .shininess = 128.0f
    };
    std::vector<Light> light_bag {
        {
            glm::vec3(0.3f),
            glm::vec3(0.5f),
            glm::vec3(1.0f),
            light_pos
        },
        {
            glm::vec3(1.f),
            glm::vec3(1.f),
            glm::vec3(1.f),
            light_pos
        }
    };
    constexpr std::size_t lidx = 0;

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    unsigned int vao_material;
    glGenVertexArrays(1, &vao_material);
    glBindVertexArray(vao_material);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*8, (void*)(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*8, (void*)(sizeof(float)*3));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float)*8, (void*)(sizeof(float)*6));
    glEnableVertexAttribArray(2);

    unsigned int vao_light;
    glGenVertexArrays(1, &vao_light);
    glBindVertexArray(vao_light);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*8, (void*)(0));
    glEnableVertexAttribArray(0);

    Shader cube_shader {"glsl/1-cube.vs", "glsl/1-cube.fs"};
    Shader material_shader {"glsl/4-tex.vert", "glsl/4-tex.frag"};

    // 纹理与sampler2D变量的关联是通过索引来关联的
    // https://www.jianshu.com/p/484e05a2c816
    material_shader.use();
    material_shader.setInt("material.diffuse", 0);
    material_shader.setInt("material.specular", 1);

    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_DEPTH_TEST);

    // free look camera singleton instance
    kit::CamInst& cam = kit::CamInst::get_instance();
    cam.last_mouse_pos = glm::vec2(wnd_info.width / 2.f, wnd_info.height / 2.f);
    cam.position = {1.1f, 0.36f, 5.2f};
    cam.front = {0.0f, 0.0f, -1.0f};
    cam.target = object_pos;
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

        // activate texture slots
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, material.diffuse);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, material.specular);

        // cubric material object placement
        material_shader.use();
        material_shader.setVec3("view_pos", cam.position);

        material_shader.setFloat("material.shininess", material.shininess);
        material_shader.setVec3("light.ambient", light_bag.at(lidx).ambient);

        material_shader.setVec3("light.diffuse", light_bag.at(lidx).diffuse);
        material_shader.setVec3("light.specular", light_bag.at(lidx).specular);
        material_shader.setVec3("light.position", light_bag.at(lidx).position);

        model = glm::translate(model, object_pos);
        model = glm::rotate(model, glm::radians(-25.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.0f);

        material_shader.setMat4("model", model);
        material_shader.setMat4("view", view);
        material_shader.setMat4("projection", projection);

        glBindVertexArray(vao_material);
        // material_shader drawcall
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // cubric light source placement
        cube_shader.use();
        model = glm::mat4(1.0f);
        model = glm::translate(model, light_pos);
        model = glm::scale(model, glm::vec3(0.2f));
        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.0f);

        glm::mat4 mvp = projection * view * model;
        cube_shader.setMat4("uMvp", mvp);
        cube_shader.setVec3("uObjectColor", light_color);
        cube_shader.setVec3("uLightColor", light_color);

        glBindVertexArray(vao_light);
        // cube_shader drawcall
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
    cam.process(window);
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

