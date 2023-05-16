#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <array>
#include <iostream>
#include <vector>
#include <cmath>

#include <_shader_s.h>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int _window_size = 600;
    GLFWwindow *window = glfwCreateWindow(_window_size, _window_size, "Triangle2", nullptr, nullptr);

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

    const float _Pi = std::acos(-1.0f), _2Pi = 2.0f * _Pi;
    std::vector<float> vv;
    std::vector<unsigned int> iv = {0, 1, 2};
    auto compute_equilateral_triangle_buffers = [&vv, &iv, &_2Pi] (float sl, float ro) {
        float r = sl / std::sqrt(3.0f) * 2.0f;
        ro = std::fmod(ro, _2Pi);
        for (int i = 0; i < 3; i++) {
            float x = r * std::cos(ro), y = r * std::sin(ro);
            vv.insert(vv.end(), {x, y, 0.0f, 0.0f, 0.0f, 0.0f});
            ro = std::fmod(ro + _2Pi / 3.0f, _2Pi);
            vv[7 * i + 3] = 1.0f;
        }
        //for (auto ix = vv.begin(); ix != vv.end(); ++ix) std::cout << *ix << ' ';
        //std::putchar('\n');
    };

    compute_equilateral_triangle_buffers(_Pi / 5, _Pi / 9);
    //std::cout << vv.size() << ' ' << iv.size() << std::endl;

    std::array<GLuint, 1> vbo;
    glGenBuffers(vbo.size(), vbo.data());
    glBindBuffer(GL_ARRAY_BUFFER, vbo.at(0));
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vv.size(), vv.data(), GL_STATIC_DRAW);

    std::array<GLuint, 1> ebo;
    glGenBuffers(ebo.size(), ebo.data());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.at(0));
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * iv.size(), iv.data(), GL_STATIC_DRAW);

    std::array<GLuint, 1> vao;
    glGenVertexArrays(vao.size(), vao.data());
    glBindVertexArray(vao.at(0));
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, reinterpret_cast<void*>(sizeof(GLfloat) * 3));
    glEnableVertexAttribArray(1);

    Shader program("shaders/ch1/triangle1.vs", "shaders/ch1/triangle1.fs");

    // colorful contour: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        program.use();
        //glBindVertexArray(vao.at(0));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.at(0));
        glDrawElements(GL_TRIANGLES, iv.size(), GL_UNSIGNED_INT, 0);

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
