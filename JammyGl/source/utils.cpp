#include "utils.h"

std::string jammygl::ReadShaderSource(const fs::path& path) {
  std::string src{};
  std::ifstream fin(path, std::ios::in);
  while (!fin.eof()) {
    std::string t;
    std::getline(fin, t);
    src += t + '\n';
  }

  fin.close();
  return src;
}

std::vector<std::string> jammygl::ParseShaderSource(const fs::path& path) {
  std::vector<std::string> src_v(2);
  std::ifstream fin(path, std::ios::in);

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
  return src_v;
}

void jammygl::WriteShaderSource(const std::string& source) {
  std::cout << source;
}

void jammygl::WriteShaderSource(const std::vector<std::string>& source_v) {
  std::cout << "## Vertex shader Begin ##" << std::endl;
  jammygl::WriteShaderSource(source_v.at(0));
  std::cout << "## Vertex shader End ##" << std::endl << std::endl;

  std::cout << "## Fragment shader Begin ##" << std::endl;
  jammygl::WriteShaderSource(source_v.at(1));
  std::cout << "## Fragment shader End ##" << std::endl;
}

