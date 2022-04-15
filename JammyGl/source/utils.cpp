#include "utils.h"

std::string jammygl::ReadShaderSource(fs::path& path) {
  std::string src{};
  auto fin = std::ifstream(path, std::ios::in);
  while (!fin.eof()) {
    std::string t;
    std::getline(fin, t);
    src += t + '\n';
  }
  fin.close();
  return src;
}

void jammygl::WriteShaderSource(const std::string& source) {
  std::cout << source;
}
