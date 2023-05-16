#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <array>
#include <iostream>
#include <vector>
#include <cmath>

#include <_shader_s.h>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

int main(int argc, char* argv[]) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(600, 600, "Polygon", nullptr, nullptr);
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD\n";
        return -1;
    }
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    auto compute_polygon_vertices = [] (float r, const int n) {
        std::vector<float> vv;
        const float _pi = std::acos(-1), _2pi = _pi * 2.0f;
        float x = r, y = 0.0f, tt = 0.0, dt = _2pi / n;
        for (int i = 0; i < n; ++i) {
            //std::cout << tt << std::endl;
            x = r * std::cos(tt);
            y = r * std::sin(tt);
            tt = std::fmod(tt + dt, _2pi);
            vv.push_back(x);
            vv.push_back(y);
            vv.push_back(0.0f);
        }
        return vv;
    };

    auto compute_polygon_indices = [] (int n, int o = 0) {
        if (o < 0) o = 0;
        if (o >= n) o = n - 1;
        //std::printf("o = %d\n", o);
        std::vector<unsigned int> iv;
        for (int i = 0; i < n - 2; ++i) {
            iv.push_back(o - i);
            iv.push_back((o + 1) % n);
            iv.push_back((o + 2) % n);
            //std::printf("%d %d %d\n", o - i, (o + 1) % n, (o + 2) % n);
            o++;
        }
        return iv;
    };

    int polygon_n = 17;
    auto polygon_v = compute_polygon_vertices(0.618, polygon_n);
    auto polygon_i = compute_polygon_indices(polygon_n, polygon_n / 1.1);

    std::array<unsigned int, 1> vbo;
    glGenBuffers(vbo.size(), vbo.data());
    glBindBuffer(GL_ARRAY_BUFFER, vbo.at(0));
    glBufferData(GL_ARRAY_BUFFER, polygon_v.size() * sizeof(float), polygon_v.data(), GL_STATIC_DRAW);

    std::array<unsigned int, 1> ebo;
    glGenBuffers(ebo.size(), ebo.data());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.at(0));
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, polygon_v.size() * sizeof(unsigned int), polygon_i.data(), GL_STATIC_DRAW);

    std::array<unsigned int, 1> vao;
    glGenVertexArrays(vao.size(), vao.data());
    glBindVertexArray(vao.at(0));
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);
    glEnableVertexAttribArray(0);

    Shader program("shaders/ch1/polygon.vs", "shaders/ch1/polygon.fs");
    program.use().setBool("uWhite", true);
    glUniform4f(glGetUniformLocation(program.ID, "uColor"), 0.0f, 0.6f, 0.3f, 1.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        program.use();
        glBindVertexArray(vao.at(0));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.at(0));
        glDrawElements(GL_TRIANGLES, polygon_i.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

