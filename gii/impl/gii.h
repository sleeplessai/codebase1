#pragma once

#include <GLFW/glfw3.h>
#include <string>
#include <string_view>


namespace gii {

struct GiiWindowInfo {
    int width {800};
    int height {600};
    std::string title = "window";

    GiiWindowInfo& set_width(int width) {
        this->width = width;
        return *this;
    }
    GiiWindowInfo& set_height(int height) {
        this->height = height;
        return *this;
    }
    GiiWindowInfo& set_title(std::string_view title) {
        this->title = title;
        return *this;
    }
};

struct GuiConfig {
    bool show_demo_window {false};
    bool show_another_window {false};
    bool dark_mode {false};
};

class Gii {
public:
    Gii(const Gii&) = delete;
    Gii& operator=(const Gii&) = delete;

    static Gii& get_instance() {
        static Gii main_ui;
        return main_ui;
    }

    void initilize();
    void present();
    ~Gii();

    // members
    GLFWwindow* window {nullptr};
    GiiWindowInfo win_info {};
    GuiConfig gui_conf {};

private:
    Gii() = default;
    void update();
    void render();
    void terminate();
    void set_glfw_callbacks();
};

}
