#pragma once

#include <filesystem>

namespace fs = std::filesystem;

class Texture {
 private:
  unsigned int m_Id;
  fs::path m_FilePath;
  unsigned char* m_LocalBuffer;
  int m_Width, m_Height, m_Bpp;

 public:
  explicit Texture(const fs::path& filepath);
  ~Texture();

  void Bind(unsigned int slot = 0) const;
  void Unbind() const;

  inline int GetWidth() const {
    return m_Width;
  }

  inline int GetHeight() const {
    return m_Height;
  }
};
