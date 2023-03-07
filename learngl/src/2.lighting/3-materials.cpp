#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string_view>

#include "stb_image.h"
#include "fmt/core.h"

#include "shader_m.h"
#include "kit/free_look_camera.h"


void process_input(GLFWwindow *window);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

static struct WindowInfo {
    int width {800};
    int height {500};
    const char* title {"Ch.2 Materials"};
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
#include "assets/cube_normal.inc"
    };
    glm::vec3 object_pos {0.0f, 0.0f, 0.0f};
    glm::vec3 light_pos = {0.0f, 1.1f, -1.5f};
    glm::vec3 object_color {1.0f, 0.5f, 0.31f};
    glm::vec3 light_color {1.0f, 1.0f, 1.0f};

    light_pos = {4.0f, 0.5f, -2.0f};

    struct Material {
        glm::vec3 ambient, diffuse, specular;
        float shininess;
    };
/*
    ambient材质向量定义了在环境光照下这个表面反射的是什么颜色，通常与表面的颜色相同。
    diffuse材质向量定义了在漫反射光照下表面的颜色。
    漫反射颜色（和环境光照一样）也被设置为我们期望的物体颜色。
    specular材质向量设置的是表面上镜面高光的颜色（或者甚至可能反映一个特定表面的颜色）。
    最后，shininess影响镜面高光的散射/半径。
*/
    struct Light {
        glm::vec3 ambient, diffuse, specular, position;
    };
/*
    一个光源对它的ambient、diffuse和specular光照分量有着不同的强度。
    环境光照通常被设置为一个比较低的强度，因为我们不希望环境光颜色太过主导。
    光源的漫反射分量通常被设置为我们希望光所具有的那个颜色，通常是一个比较明亮的白色。
    镜面光分量通常会保持为vec3(1.0)，以最大强度发光。
*/

    // http://devernay.free.fr/cours/opengl/materials.html
    std::vector<Material> material_vec {
        {
            glm::vec3(1.0f, 0.5f, 0.3f),
            glm::vec3(1.0f, 0.5f, 0.3f),
            glm::vec3(0.5f, 0.5f, 0.5f),
            64.0f
        }, // demo
        {
            glm::vec3(0.5f, 0.f, 0.5f),
            glm::vec3(0.5f, 0.f, 0.5f),
            light_color,
            192.0f
        }, // mine
        {
            glm::vec3(0.0, 0.1, 0.06),
            glm::vec3(0.0, 0.50980392, 0.50980392),
            glm::vec3(0.50196078, 0.50196078, 0.50196078),
            .25f * 128.0f
        }, // cyan plasitc
        {
            glm::vec3(0.0215, 0.1745, 0.0215),
            glm::vec3(0.07568, 0.61424, 0.07568),
            glm::vec3(0.633, 0.727811, 0.633),
            0.6f * 128.0f
        } // emerald
    };
    std::vector<Light> light_vec {
        {
            glm::vec3(0.1f),
            glm::vec3(0.3f),
            glm::vec3(1.f),
            light_pos
        },
        {
            glm::vec3(1.f),
            glm::vec3(1.f),
            glm::vec3(1.f),
            light_pos
        }
    };
    constexpr std::size_t midx = 3, lidx = midx > 2 ? 1 : 0;

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    unsigned int vao_material;
    glGenVertexArrays(1, &vao_material);
    glBindVertexArray(vao_material);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float)*6, (void*)(sizeof(float)*3));
    glEnableVertexAttribArray(1);

    unsigned int vao_light;
    glGenVertexArrays(1, &vao_light);
    glBindVertexArray(vao_light);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)(0));
    glEnableVertexAttribArray(0);

    Shader cube_shader {"glsl/1-cube.vs", "glsl/1-cube.fs"};
    Shader material_shader {"glsl/3-material.vs", "glsl/3-material.fs"};

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

        // cubric material object placement
        material_shader.use();
        material_shader.setVec3("uViewPos", cam.position);

        material_shader.setVec3("material.ambient", material_vec[midx].ambient);
        material_shader.setVec3("material.diffuse", material_vec[midx].diffuse);
        material_shader.setVec3("material.specular", material_vec[midx].specular);
        material_shader.setFloat("material.shininess", material_vec[midx].shininess);

        material_shader.setVec3("light.ambient", light_vec[lidx].ambient);
        material_shader.setVec3("light.diffuse", light_vec[lidx].diffuse);
        material_shader.setVec3("light.specular", light_vec[lidx].specular);
        material_shader.setVec3("light.position", light_vec[lidx].position);

        model = glm::translate(model, object_pos);
        model = glm::rotate(model, glm::radians(-25.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.0f);

        material_shader.setMat4("uModel", model);
        material_shader.setMat4("uView", view);
        material_shader.setMat4("uProjection", projection);

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

