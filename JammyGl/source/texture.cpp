#include "texture.h"
#include "gl_helper.h"
#include "renderer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Texture::Texture(const fs::path& filepath)
    : m_Id(0), m_FilePath(filepath), m_LocalBuffer(nullptr), m_Width(0), m_Height(0), m_Bpp(0) {
  stbi_set_flip_vertically_on_load(1);
  m_LocalBuffer = stbi_load(filepath.string().c_str(), &m_Width, &m_Height, &m_Bpp, 4);

  GlCall(glGenTextures(1, &m_Id));
  GlCall(glBindTexture(GL_TEXTURE_2D, m_Id));

  GlCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  GlCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  GlCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  GlCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

  GlCall(glTexImage2D(
      GL_TEXTURE_2D, 0, GL_RGBA8, m_Width, m_Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_LocalBuffer));
  GlCall(glBindTexture(GL_TEXTURE_2D, 0));

  if (m_LocalBuffer) {
    stbi_image_free(m_LocalBuffer);
  }
}

Texture::~Texture() {
  GlCall(glDeleteTextures(1, &m_Id));
}

void Texture::Bind(unsigned int slot) const {
  GlCall(glActiveTexture(GL_TEXTURE0 + slot));
  GlCall(glBindTexture(GL_TEXTURE_2D, m_Id));
}

void Texture::Unbind() const {
  GlCall(glBindTexture(GL_TEXTURE_2D, 0));
}
