#include "shader.h"
#include "gl_helper.h"
#include "shader.h"

#include <fstream>
#include <iostream>

void _WriteShaderSource(const std::vector<std::string>& src_v) {
  std::printf("## Vertex shader Begin ##\n%s## Vertex shader End ##\n\n", src_v.at(0).c_str());
  std::printf("## Fragment shader Begin ##\n%s## Fragment shader End ##\n", src_v.at(1).c_str());
}

enum class Shader_t { Null = -1, Vertex = 0, Fragment = 1 };

Shader::Shader(const fs::path& filepath)
    : m_FilePath(filepath), m_Id(0), m_ShaderSource(std::vector<std::string>(2)) {
  ParseShaderSource();
  CreateShader();
}

Shader::~Shader() {
  GlCall(glDeleteProgram(this->m_Id));
}

void Shader::ParseShaderSource() {
  std::vector<std::string>& src_v = m_ShaderSource;
  std::ifstream fin(m_FilePath, std::ios::in);

  Shader_t curr_type = Shader_t::Null;
  std::string t, s;

  while (!fin.eof()) {
    std::getline(fin, t);

    if (t.find("#shader") != std::string::npos) {
      if (!s.empty()) {
        src_v[(int)curr_type] = s;
        s.clear();
        curr_type = Shader_t::Null;
      }
      if (t.find("vertex") != std::string::npos) {
        curr_type = Shader_t::Vertex;
      } else if (t.find("fragment") != std::string::npos) {
        curr_type = Shader_t::Fragment;
      }
    } else {
      s += t + '\n';
    }
  }
  if (!s.empty()) {
    src_v[(int)curr_type] = s;
  }

  fin.close();
}

void Shader::CreateShader() {
  unsigned int& program = this->m_Id;
  program = glCreateProgram();
  unsigned int vs = CompileShader(GL_VERTEX_SHADER, m_ShaderSource[(int)Shader_t::Vertex]);
  unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, m_ShaderSource[(int)Shader_t::Fragment]);

  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  glValidateProgram(program);

  glDeleteShader(vs);
  glDeleteShader(fs);
}

unsigned int Shader::CompileShader(unsigned int shader_t, const std::string& source) {
  unsigned int id = glCreateShader(shader_t);
  const char* src = source.c_str();

  glShaderSource(id, 1, &src, nullptr);
  glCompileShader(id);

  int result;
  glGetShaderiv(id, GL_COMPILE_STATUS, &result);
  if (result == GL_FALSE) {
    int length;
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
    // alloca() uses to allocate dynamic space on stack heap.
    char* message = (char*)alloca(sizeof(length));
    glGetShaderInfoLog(id, length, &length, message);
    std::cerr << "[GLSL Error] Failed to compile shader! Error occurred at "
              << (shader_t == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader" << std::endl;
    std::cerr << message << std::endl;
    glDeleteShader(id);
    return 0;
  }
  return id;
}

int Shader::GetUniformLocation(const std::string& u_Var) {
  if (m_UniformLocaltionCache.find(u_Var) != m_UniformLocaltionCache.end()) {
    return m_UniformLocaltionCache[u_Var];
  }
  GlCall(GLint location = glGetUniformLocation(m_Id, u_Var.c_str()));
  if (location == -1) {
    std::cerr << "[GLSL Warning] Uniform " << u_Var << " doesn\'t exist! " << __FILE__ << ':'
              << __LINE__ << std::endl;
  }
  m_UniformLocaltionCache[u_Var] = location;
  return location;
}

void Shader::Bind() const {
  GlCall(glUseProgram(this->m_Id));
}

void Shader::Unbind() const {
  GlCall(glUseProgram(0));
}

void Shader::SetUniform4f(const std::string& u_Var, float v0, float v1, float v2, float v3) {
  GlCall(glUniform4f(GetUniformLocation(u_Var.c_str()), v0, v1, v2, v3));
}

void Shader::SetUniform1i(const std::string& u_Var, int v0) {
  GlCall(glUniform1i(GetUniformLocation(u_Var.c_str()), v0));
}

void Shader::SetUniformMat4f(const std::string& u_Var, const glm::mat4& matrix) {
  GlCall(glUniformMatrix4fv(GetUniformLocation(u_Var), 1, GL_FALSE, &matrix[0][0]));
}
