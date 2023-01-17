#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "_shader_s.h"
#include "stb_image.h"

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void process_input(GLFWwindow *window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

constexpr int window_width = 800;
constexpr int window_height = 500;

class FreeLookCamera {
public:
    FreeLookCamera(const FreeLookCamera&) = delete;
    FreeLookCamera& operator=(const FreeLookCamera&) = delete;

    static FreeLookCamera& get_instance() {
        static FreeLookCamera cam_inst;
        return cam_inst;
    }

    glm::vec3 position{0.0f};
    glm::vec3 target{0.0f};
    glm::vec3 up{0.0f};
    glm::vec3 front{0.0f};
    glm::vec3 right{0.0f};
    glm::vec3 direction{0.0f};

    float speed{0.0f};
    float pitch{0.0f}, yaw{-90.0f}, roll{0.0f};
    bool first_mouse_on{true};
    glm::vec2 last_mouse_pos{320.0f, 320.0f};
    float fovy{45.0f};

    static constexpr float sensitivity = 0.2f;

    void update() {
        auto& cam = get_instance();
        cam.direction = glm::normalize(cam.position - cam.target);
        cam.right = glm::normalize(glm::cross({0.0f, 1.0f, 0.0f}, cam.direction));
        cam.up = glm::normalize(glm::cross(cam.direction, cam.right));
    }
    void tick(float frame_time) {
        auto& cam = get_instance();
        cam.speed = 6.0f * frame_time;
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
    glfwInitHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwInitHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwInitHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "GLCube", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    } // use gl*Api after glad init
    const unsigned char* gl_version = glGetString(GL_VERSION);
    const unsigned char* gl_render = glGetString(GL_RENDERER);
    std::printf("opengl_device: %\nopengl_version: %s\n", gl_render, gl_version);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, channels;
    unsigned char* tex_data = stbi_load("assets/container.jpg", &width, &height, &channels, 0);
    std::cout << "tex_size: w,h,c = " << width << ',' << height << ',' << channels << std::endl;

    if (tex_data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cout << "Failed to load texture\n";
    }
    stbi_image_free(tex_data);

    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
        0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };
    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);

    Shader sp("glsl/glcube.vs", "glsl/glcube.fs");

    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    constexpr glm::vec3 cubePositions[] = {
        glm::vec3( 0.0f,  0.0f,  1.2f),
        glm::vec3( 2.0f,  1.3f, -1.8f),
        glm::vec3(-1.5f, -2.2f, -1.4f),
        glm::vec3(-1.3f, -2.0f, -1.3f),
        glm::vec3( 1.2f, -0.4f, -1.2f),
        glm::vec3(-1.7f,  1.0f, -1.5f),
        glm::vec3( 0.6f, -2.0f, -1.3f),
        glm::vec3( 1.5f,  2.0f, -1.5f),
        glm::vec3( 0.75f,  0.2f, -1.5f),
        glm::vec3(-1.3f,  1.0f, -1.5f)
    };

    // Free look camera singleton instance
    FreeLookCamera& camera = FreeLookCamera::get_instance();
    camera.position = {0.0f, 0.0f, 3.0f};
    camera.front = {0.0f, 0.0f, -1.0f};
    camera.target = {0.0f, 0.0f, 0.0f};
    camera.update();

    float frame_time{0.0f}, last_frame{0.0f};

    while (!glfwWindowShouldClose(window)) {
        float curr_frame = static_cast<float>(glfwGetTime());
        camera.tick(curr_frame - last_frame);
        last_frame = curr_frame;
        process_input(window);

        glEnable(GL_DEPTH_TEST);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        sp.use();
        glm::mat4 model{1.0f};
        glm::mat4 view{1.0f};
        glm::mat4 projection{1.0f};

        view = glm::lookAt(camera.position, camera.position + camera.front, camera.up);

        glBindVertexArray(vao);
        for (int i = 0; i < 10; ++i) {
            model = glm::translate(model, cubePositions[i]);
            model = glm::rotate(model, glm::radians(-50.0f), {1.0f, 1.0f, 0.0f});
            model = glm::rotate(model, glm::radians(30.0f), {0.0f, 1.0f, 0.0f});
            model = glm::rotate(model, static_cast<float>(glfwGetTime())/3.0f, glm::vec3(0.0f, 0.0f, 1.0f));
            float angle = 20.0f * i;
            model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
            projection = glm::perspective(glm::radians(camera.fovy), static_cast<float>(window_width)/window_height, 0.1f, 100.0f);

            unsigned int uModel = glGetUniformLocation(sp.ID, "uModel");
            unsigned int uView = glGetUniformLocation(sp.ID, "uView");
            unsigned int uProjection = glGetUniformLocation(sp.ID, "uProjection");

            glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(uView, 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(uProjection, 1, GL_FALSE, glm::value_ptr(projection));

            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
}

void process_input(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    auto& camera = FreeLookCamera::get_instance();
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.position += camera.speed * camera.front;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.position -= camera.speed * camera.front;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.position -= glm::normalize(glm::cross(camera.front, camera.up)) * camera.speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.position += glm::normalize(glm::cross(camera.front, camera.up)) * camera.speed;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
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

    if(cam.pitch > 89.0f)
        cam.pitch = 89.0f;
    if(cam.pitch < -89.0f)
        cam.pitch = -89.0f;

    glm::vec3 front{};
    front.x = cos(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch));
    front.y = sin(glm::radians(cam.pitch));
    front.z = sin(glm::radians(cam.yaw)) * cos(glm::radians(cam.pitch));
    cam.front = glm::normalize(front);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    auto& cam = FreeLookCamera::get_instance();

    if (cam.fovy >= 1.0f && cam.fovy <= 45.0f)
        cam.fovy -= static_cast<float>(yoffset);
    if (cam.fovy <= 1.0f)
        cam.fovy = 1.0f;
    if (cam.fovy >= 45.0f)
        cam.fovy = 45.0f;
}
