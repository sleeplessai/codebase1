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

int main() {
    glfwInit();
    glfwInitHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwInitHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwInitHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(640, 640, "Transforms", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    } // use gl*Api after glad init
    const unsigned char* gl_version = glGetString(GL_VERSION);
    const unsigned char* gl_render = glGetString(GL_RENDERER);
    std::printf("opengl_device: %s\nopengl_version: %s\n", gl_render, gl_version);

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
        -0.5f,  0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, // LT
         0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // RT
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // LB
         0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f  // RB
    };
    unsigned int indices[] = {0, 1, 2, 1, 2, 3};
    unsigned int vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    unsigned int ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 3));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 8, (void*)(sizeof(float) * 6));
    glEnableVertexAttribArray(2);

    Shader sp("glsl/texture1.vs", "glsl/texture1.fs");

    constexpr bool EnableMvpTransform = true;
    if (EnableMvpTransform) std::clog << std::boolalpha << EnableMvpTransform << '\n';

    while (!glfwWindowShouldClose(window)) {
        process_input(window);

        glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        sp.use();
        sp.setBool("uUseMvp", EnableMvpTransform);

        if (EnableMvpTransform) {
            glm::mat4 model{1.0f};
            glm::mat4 view{1.0f};
            glm::mat4 projection{1.0f};

            model = glm::rotate(model, glm::radians(-50.0f), {1.0f, 0.0f, 0.0f});
            view = glm::translate(view, {0.0f, 0.0f, -3.2f});
            projection = glm::perspective(glm::radians(45.0f), 1.0f, 1.0f, 100.0f);

            unsigned int uModel = glGetUniformLocation(sp.ID, "uModel");
            unsigned int uView = glGetUniformLocation(sp.ID, "uView");
            unsigned int uProjection = glGetUniformLocation(sp.ID, "uProjection");

            glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix4fv(uView, 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(uProjection, 1, GL_FALSE, glm::value_ptr(projection));
        } else {
            glm::mat4 transform = glm::mat4(1.0f);
            // transform = glm::rotate(transform, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            transform = glm::rotate(transform, static_cast<float>(glfwGetTime()), glm::vec3(0.0f, 0.0f, 1.0f));
            transform = glm::scale(transform, glm::vec3(0.75f, 0.5f, 0.0f));

            unsigned int uTransformLoc = glGetUniformLocation(sp.ID, "uTransform");
            // std::clog << "uTrans: " << uTransformLoc << '\n';
            glUniformMatrix4fv(uTransformLoc, 1, GL_FALSE, glm::value_ptr(transform));
        }

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

void process_input(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

