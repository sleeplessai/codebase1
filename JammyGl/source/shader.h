#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

class Shader {
 private:
  unsigned int m_Id;
  fs::path m_FilePath;
  std::vector<std::string> m_ShaderSource;
  std::unordered_map<std::string, int> m_UniformLocaltionCache;

  void ParseShaderSource();
  void CreateShader();
  unsigned int CompileShader(unsigned int shader_t, const std::string& source);
  int GetUniformLocation(const std::string& u_Var);

 public:
  explicit Shader(const fs::path& filepath);
  ~Shader();

  void Bind() const;
  void Unbind() const;

  void SetUniform4f(const std::string& u_Var, float v0, float v1, float v2, float v3);
  void SetUniform1i(const std::string& u_Var, int v0);
};
