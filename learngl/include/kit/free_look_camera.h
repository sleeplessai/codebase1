#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fmt/core.h>

namespace kit {

class CamInst {
public:
    CamInst(const CamInst&) = delete;
    CamInst& operator=(const CamInst&) = delete;

    static CamInst& get_instance() {
        static CamInst cam_inst;
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
    glm::vec2 last_mouse_pos {};
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
    void process_keypress(GLFWwindow* window) {
        auto& cam = get_instance();

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
            fmt::print("\nCurrent camera parameters:");
            fmt::print("\tpositions: {},{},{}\n", cam.position[0], cam.position[1], cam.position[2]);
            fmt::print("\tfront: {},{},{}\n", cam.front[0], cam.front[1], cam.front[2]);
            fmt::print("\ttarget: {},{},{}\n", cam.target[0], cam.target[1], cam.target[2]);
        }
    }
    void process_scroll(float const yoffset) {
        auto& cam = get_instance();

        if (cam.fovy >= 1.0f && cam.fovy <= 100.0f)
            cam.fovy -= yoffset * 5.0f;
        if (cam.fovy <= 1.0f)
            cam.fovy = 1.0f;
        if (cam.fovy >= 100.0f)
            cam.fovy = 100.0f;
    }

    ~CamInst() {
        //std::clog << "Camera instance destructed.\n";
    }

private:
    CamInst() = default;
};

using FreeLookCamera = CamInst; // alias

}
