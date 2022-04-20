#include "deprecated_utils.h"

#include <fstream>
#include <iostream>

std::string __read_single_shader(const fs::path& fpath) {
  std::string src;
  std::ifstream fin(fpath, std::ios::in);

  while (!fin.eof()) {
    std::string t;
    std::getline(fin, t);
    src += t + '\n';
  }

  fin.close();
  return src;
}

std::vector<std::string> __parse_vertex_and_fragment_shaders(const fs::path& fpath) {
  enum class Shader_t { Null = -1, Vertex = 0, Fragment = 1 };

  std::vector<std::string> src_v(2);
  std::ifstream fin(fpath, std::ios::in);

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

void __print_vertex_and_fragment_shaders(const std::vector<std::string>& src_v) {
  std::printf("## Vertex shader Begin ##\n%s## Vertex shader End ##\n\n", src_v.at(0).c_str());
  std::printf("## Fragment shader Begin ##\n%s## Fragment shader End ##\n", src_v.at(1).c_str());
}
