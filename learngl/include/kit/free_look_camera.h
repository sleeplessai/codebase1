#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


namespace kit {

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

    ~FreeLookCamera() {
        //std::clog << "Camera instance destructed.\n";
    }

private:
    FreeLookCamera() {
    }
};

using CamInst = FreeLookCamera;

}
