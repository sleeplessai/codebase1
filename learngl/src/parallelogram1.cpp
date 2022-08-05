#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include <shader_s.h>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

int main() {
    /* Init */
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "Parallelogram1", nullptr, nullptr);

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

    /* Render */
    // data on cpu
    float vertices[] = {
        -0.2f,  0.5f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
         0.2f, -0.5f, 0.0f,
    };
    unsigned int indices[] = {0, 1, 2, 1, 2, 3};

    // vbo, ebo(ibo)
    unsigned int vbo, ebo;
    glGenBuffers(1, &vbo); //  &vbo can be array to gen multiple buffers
    glBindBuffer(GL_ARRAY_BUFFER, vbo);             // vbo buffering
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glGenBuffers(1, &ebo); //  ebo for vertex data indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);     // ebo buffering
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // vao
    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao); // bind once
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), static_cast<void*>(0));
    // location, vertex_abbr_value_num(vec3), dtype, norm_to_0-1, offset
    glEnableVertexAttribArray(1);

    // shader
    Shader shader("build/glsl/triangle.vs", "build/glsl/triangle.fs");

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();
        glBindVertexArray(vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

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
