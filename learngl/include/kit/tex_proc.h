#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <fmt/core.h>
#include "kit/exception.h"
#include "stb_image.h"
#include <filesystem>
#include <iostream>


namespace kit {

inline unsigned int make_texture(const std::string& tex_image, bool restrict = false) {

    int width{}, height{}, channels{};
    auto* tex_data = stbi_load(tex_image.c_str(), &width, &height, &channels, 0);
    // fmt::print("tex_size: w,h,c = {},{},{}\n", width, height, channels);

    if (!tex_data) {
        fmt::print("Failed to load texture {}, texture image data is null.\n", tex_image);
        if (restrict) {
          THROW_RT_ERR
        }
    }

    GLint tex_fmt{};
    if (channels == 4) {
        tex_fmt = GL_RGBA;
    } else if (channels == 3) {
        tex_fmt = GL_RGB;
    } else if (channels == 1) {
        tex_fmt = GL_RED;
    } else {
        fmt::print("Unsupported texture image channel {}.\n", channels);
        if (restrict) {
          THROW_RT_ERR
        }
    }

    unsigned int tex_id;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, tex_fmt, width, height, 0, tex_fmt, GL_UNSIGNED_BYTE, tex_data);
    // glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(tex_data);

    return tex_id;
}

}
