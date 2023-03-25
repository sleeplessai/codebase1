#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <limits>
#include <random>
#include <variant>
#include <vector>
#include <string_view>

#include <fmt/core.h>

#include "shader_m.h"
#include "kit/cam_inst.h"
#include "kit/tex_proc.h"
#include "kit/csv_proc.h"


void process_input(GLFWwindow*);
void framebuffer_size_callback(GLFWwindow*, int, int);
void scroll_callback(GLFWwindow*, double, double);

static struct WindowInfo {
    int width {1600}, height {1000};
    float aspect {1.6f};
    const char* title {"Ch.2 Casters"};

    void update(GLFWwindow *window) noexcept {
        glfwGetWindowSize(window, &width, &height);
        aspect = static_cast<float>(width) / height;
    }
} wnd_info;

auto fetch_point_light_data(int index) {
    kit::CsvDatabase db("assets/point_light_attenuation.csv", "point_light_attenuation");
    db.open();
    db.buffer();
    db.show_meta();
    if (index < 0) index = db.record.size() + index;
    int range = std::get<int>(db.query(index,"Range").value);
    glm::vec3 quad = std::get<glm::vec3>(db.query(index,"ConstantLinearQuadratic").value);
    return std::tuple{range, quad};
}

//#define __Paralleling_light_rendering
#define __Point_light_rendering

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
    std::vector<glm::vec3> cube_pos = {
#include "assets/cube_pos.inc"
    };
    cube_pos.emplace_back(0.0f, 0.0f, 0.0f);
    glm::vec3 object_pos {0.0f};
    glm::vec3 light_pos = {0.0f, 1.1f, -1.5f};
    glm::vec3 object_color {1.0f, 0.5f, 0.31f};
    glm::vec3 light_color {1.0f, 1.0f, 1.0f};

    //light_pos = {4.0f, 0.5f, -2.0f};
    light_pos = {10.f, 0.f, 0.f};

    auto [range, quadratic] = fetch_point_light_data(-7);

    struct Material {
        unsigned int diffuse, specular;
        float shininess;
    };
    struct Light {
        glm::vec3 ambient, diffuse, specular, direction;
    };
    struct PointLight {
        glm::vec3 ambient, diffuse, specular, position, constant_linear_quadratic;
    };
    Material ml {
        .diffuse = kit::make_texture("assets/container2.png"),
        .specular = kit::make_texture("assets/container2_specular.png"),
        .shininess = 128.0f
    };
    std::array<std::variant<Light, PointLight>, 3> lights = {
        Light{
            glm::vec3(1.f),
            glm::vec3(1.f),
            glm::vec3(1.f),
            -light_pos,
            },
        Light{
            glm::vec3(0.3f),
            glm::vec3(0.5f),
            glm::vec3(1.0f),
            glm::vec3(0.f, 0.f, 0.f) - light_pos,
            },
        PointLight{
            glm::vec3(0.5f),
            glm::vec3(0.6f),
            glm::vec3(1.0f),
            glm::vec3(-6.4f,1.0f,0.0f),
            quadratic
            }
    };

#if defined (__Paralleling_light_rendering)
    Light const& lt = std::get<Light>(lights.at(1));
#elif defined (__Point_light_rendering)
    PointLight const& lt = std::get<PointLight>(lights.at(2));
#endif

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

    Shader cube_shader("glsl/1-cube.vs", "glsl/1-cube.fs");

    Shader paralleling_shader("glsl/4-tex.vert", "glsl/5-caster_paralleling.frag");
    paralleling_shader.use();
    paralleling_shader.setInt("material.diffuse", 0);
    paralleling_shader.setInt("material.specular", 1);

    Shader point_shader("glsl/4-tex.vert", "glsl/5-caster_point.frag");
    point_shader.use();
    point_shader.setInt("material.diffuse", 0);
    point_shader.setInt("material.specular", 1);

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

        glm::mat4 view {1.0f};
        glm::mat4 projection {1.0f};

        view = glm::lookAt(cam.position, cam.position + cam.front, cam.up);

        // activate texture slots
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ml.diffuse);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, ml.specular);

#if defined(__Paralleling_light_rendering)
        // cubric material object placement
        paralleling_shader.use();
        paralleling_shader.setVec3("view_pos", cam.position);

        paralleling_shader.setFloat("material.shininess", ml.shininess);
        paralleling_shader.setVec3("light.ambient", lt.ambient);
        paralleling_shader.setVec3("light.diffuse", lt.diffuse);
        paralleling_shader.setVec3("light.specular", lt.specular);
        paralleling_shader.setVec3("light.direction", lt.direction);

        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.f);

        paralleling_shader.setMat4("projection", projection);
        paralleling_shader.setMat4("view", view);

        glm::mat4 model{};
        std::mt19937 g(320);
        constexpr auto y_axis = glm::vec3(0.0f, 1.0f, 0.0f);
        glBindVertexArray(vao_material);

        for (auto& pos3 : cube_pos) {
            model = glm::mat4(1.f);
            model = glm::translate(model, pos3);
            if (pos3 == cube_pos.back()) {
                model = glm::rotate(model, glm::radians(-28.0f), y_axis);
            } else {
                float rand_ro = 90.f * std::generate_canonical<float, std::numeric_limits<float>::digits10>(g);
                // fmt::print("bits:{} real:{}\n", std::numeric_limits<float>::digits10, rand_ro);
                model = glm::rotate(model, glm::radians(rand_ro), y_axis);
            }
            paralleling_shader.setMat4("model", model);
            // paralleling_shader drawcall
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
#elif defined(__Point_light_rendering)
        // point-lighted material placement
        point_shader.use();
        point_shader.setVec3("view_pos", cam.position);

        point_shader.setFloat("material.shininess", ml.shininess);
        point_shader.setVec3("light.ambient", lt.ambient);
        point_shader.setVec3("light.diffuse", lt.diffuse);
        point_shader.setVec3("light.specular", lt.specular);
        point_shader.setVec3("light.position", lt.position);
        point_shader.setVec3("light.constant_linear_quadratic", lt.constant_linear_quadratic);
        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.f);

        point_shader.setMat4("projection", projection);
        point_shader.setMat4("view", view);

        glm::mat4 model{};
        std::mt19937 g(324);
        constexpr auto y_axis = glm::vec3(0.0f, 1.0f, 0.0f);
        glBindVertexArray(vao_material);

        for (auto& pos3 : cube_pos) {
            model = glm::mat4(1.f);
            model = glm::translate(model, pos3);
            if (pos3 == cube_pos.back()) {
                model = glm::rotate(model, glm::radians(-18.0f), y_axis);
            } else {
                float rand_ro = 90.f * std::generate_canonical<float, std::numeric_limits<float>::digits10>(g);
                // fmt::print("bits:{} real:{}\n", std::numeric_limits<float>::digits10, rand_ro);
                model = glm::rotate(model, glm::radians(rand_ro), y_axis);
            }
            point_shader.setMat4("model", model);
            // paralleling_shader drawcall
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

#endif
        // cubric light source placement
        cube_shader.use();
        model = glm::mat4(1.0f);
#if defined (__Paralleling_light_rendering)
        model = glm::translate(model, light_pos);
#elif defined (__Point_light_rendering)
        model = glm::translate(model, lt.position);
#endif
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
    cam.process_keypress(window);
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    wnd_info.update(window);
    fmt::print("Resized to: ({},{})\n", wnd_info.width, wnd_info.height);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    auto& cam = kit::CamInst::get_instance();
    cam.process_scroll(yoffset);
}


