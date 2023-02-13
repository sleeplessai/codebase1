#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

#include "_shader_s.h"
#include "stb_image.h"
#include "fmt/core.h"


void process_input(GLFWwindow *window);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

static struct WindowInfo {
    int width {800};
    int height {500};
    const char* title {"GLCube"};
    float aspect {1.6f};

    void update(GLFWwindow *window) noexcept {
        glfwGetWindowSize(window, &width, &height);
        aspect = static_cast<float>(width) / height;
    }
} wnd_info;

class FreeLookCamera {
public:
    FreeLookCamera(const FreeLookCamera&) = delete;
    FreeLookCamera& operator=(const FreeLookCamera&) = delete;

    static FreeLookCamera& get_instance() {
        static FreeLookCamera cam_inst;
        return cam_inst;
    }

    glm::vec3 position {0.0f};
    glm::vec3 target {0.0f};
    glm::vec3 up {0.0f};
    glm::vec3 front {0.0f};
    glm::vec3 right {0.0f};
    glm::vec3 direction {0.0f};

    float speed {0.0f};
    float pitch {0.0f};
    float yaw {-90.0f};
    float roll {0.0f};
    bool first_mouse_on {true};
    glm::vec2 last_mouse_pos {wnd_info.width / 2.f, wnd_info.height / 2.f};
    float fovy {45.0f};

    static constexpr float sensitivity = 0.2f;

    void update() {
        auto& cam = get_instance();
        cam.direction = glm::normalize(cam.position - cam.target);
        cam.right = glm::normalize(glm::cross({0.0f, 1.0f, 0.0f}, cam.direction));
        cam.up = glm::normalize(glm::cross(cam.direction, cam.right));
    }
    void tick(float frame_time) {
        auto& cam = get_instance();
        cam.speed = 10.0f * frame_time;
    }

    ~FreeLookCamera() {
        std::clog << "Camera instance destructed.\n";
    }

private:
    FreeLookCamera() {
    }
};

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(wnd_info.width, wnd_info.height, wnd_info.title, nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    } // use gl*Api after glad init
    std::string gl_version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    std::string gl_render = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    fmt::print("opengl_device: {}\nopengl_version: {}\n", gl_render, gl_version);

    std::vector<float> vertices = {
#include "assets/cube.inc"
    };
    glm::vec3 cube_pos {0.0f, 0.0f, 0.0f};
    glm::vec3 light_pos {2.0f, 0.0f, 0.0f};
    glm::vec4 cube_color {1.0f, 0.5f, 0.31f, 1.0f};
    glm::vec4 light_color {1.0f, 1.0f, 1.0f, 1.0f};

    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);

    unsigned int vao_light;
    glGenVertexArrays(1, &vao_light);
    glBindVertexArray(vao_light);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);

    Shader shader {"glsl/cube.vs", "glsl/cube.fs"};

    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_DEPTH_TEST);

    // free look camera singleton instance
    FreeLookCamera& cam = FreeLookCamera::get_instance();
    cam.position = {1.1f, 0.36f, 5.2f};
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

        shader.use();
        glm::mat4 model {1.0f};
        glm::mat4 view {1.0f};
        glm::mat4 projection {1.0f};

        view = glm::lookAt(cam.position, cam.position + cam.front, cam.up);

        unsigned int uObjectColor = glGetUniformLocation(shader.ID, "uObjectColor");
        unsigned int uLightColor = glGetUniformLocation(shader.ID, "uLightColor");

        glUniform4fv(uObjectColor, 1, glm::value_ptr(cube_color));
        glUniform4fv(uLightColor, 1, glm::value_ptr(light_color));

        // object placement
        glBindVertexArray(vao);
        model = glm::translate(model, cube_pos);
        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.0f);

        unsigned int uMvp = glGetUniformLocation(shader.ID, "uMvp");
        glm::mat4 mvp_mat4 = projection * view * model;
        glUniformMatrix4fv(uMvp, 1, GL_FALSE, glm::value_ptr(mvp_mat4));

        glDrawArrays(GL_TRIANGLES, 0, 36);

        // light source
        model = glm::mat4(1.0f);
        model = glm::translate(model, light_pos);
        model = glm::scale(model, glm::vec3(0.2f));
        projection = glm::perspective(glm::radians(cam.fovy), wnd_info.aspect, 0.1f, 100.0f);

        mvp_mat4 = projection * view * model;
        glUniformMatrix4fv(uMvp, 1, GL_FALSE, glm::value_ptr(mvp_mat4));
        glUniform4fv(uObjectColor, 1, glm::value_ptr(glm::vec4(1.0f)));

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

    auto& cam = FreeLookCamera::get_instance();
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

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    return;
    auto& cam = FreeLookCamera::get_instance();

    float xposf = static_cast<float>(xpos);
    float yposf = static_cast<float>(ypos);

    if (cam.first_mouse_on) {
        cam.last_mouse_pos.x = xposf;
        cam.last_mouse_pos.y = yposf;
        cam.first_mouse_on = false;
    }

    float xoffset = xposf - cam.last_mouse_pos.x;
    float yoffset = cam.last_mouse_pos.y - yposf;
    cam.last_mouse_pos.x = xposf;
    cam.last_mouse_pos.y = yposf;

    xoffset *= cam.sensitivity;
    yoffset *= cam.sensitivity;

    cam.yaw   += xoffset;
    cam.pitch += yoffset;

    if (cam.pitch > 89.0f)
        cam.pitch = 89.0f;
    if (cam.pitch < -89.0f)
        cam.pitch = -89.0f;

    glm::vec3 front {};
    front.x = cos(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch));
    front.y = sin(glm::radians(cam.pitch));
    front.z = sin(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch));
    cam.front = glm::normalize(front);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    auto& cam = FreeLookCamera::get_instance();

    if (cam.fovy >= 1.0f && cam.fovy <= 100.0f)
        cam.fovy -= static_cast<float>(yoffset * 5.0f);
    if (cam.fovy <= 1.0f)
        cam.fovy = 1.0f;
    if (cam.fovy >= 100.0f)
        cam.fovy = 100.0f;
}

