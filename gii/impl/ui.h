#pragma once

#include <GLFW/glfw3.h>
#include <string>
#include <string_view>


namespace ui {

struct MainUIWindowInfo {
    int width {800};
    int height {600};
    std::string title = "window";

    MainUIWindowInfo& set_width(int width) {
        this->width = width;
        return *this;
    }
    MainUIWindowInfo& set_height(int height) {
        this->height = height;
        return *this;
    }
    MainUIWindowInfo& set_title(std::string_view title) {
        this->title = title;
        return *this;
    }
};

struct GuiConfig {
    bool show_demo_window {false};
    bool show_another_window {false};
    bool dark_mode {false};
};

class MainUI {
public:
    MainUI(const MainUI&) = delete;
    MainUI& operator=(const MainUI&) = delete;

    static MainUI& get_instance() {
        static MainUI main_ui;
        return main_ui;
    }

    void initilize();
    void present();
    ~MainUI();

    // members
    GLFWwindow* window {nullptr};
    MainUIWindowInfo win_info {};
    GuiConfig gui_conf {};

private:
    MainUI() = default;
    void update();
    void render();
    void terminate();
    void set_glfw_callbacks();
};

}
