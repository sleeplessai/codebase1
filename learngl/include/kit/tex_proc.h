#pragma once

#include <cstdlib>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdexcept>

#include <fmt/core.h>
#include "stb_image.h"


namespace kit {

unsigned int make_texture(const std::string& tex_image, bool error_exit = false) {

    int width{}, height{}, channels{};
    auto* tex_data = stbi_load(tex_image.c_str(), &width, &height, &channels, 0);
    fmt::print("tex_size: w,h,c = {},{},{}\n", width, height, channels);

    if (!tex_data) {
        fmt::print("Failed to load texture, texture image data is null.");
        if (error_exit) ::exit(-1);
    }

    GLint tex_fmt{};
    if (channels == 4) {
        tex_fmt = GL_RGBA;
    } else if (channels == 3) {
        tex_fmt = GL_RGB;
    } else {
        fmt::print("Unsupported texture image channel {}.", channels);
        if (error_exit) ::exit(-1);
    }

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, tex_fmt, width, height, 0, tex_fmt, GL_UNSIGNED_BYTE, tex_data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(tex_data);

    return texture;
}

}
