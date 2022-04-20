#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::string __read_single_shader(const fs::path& fpath);
std::vector<std::string> __parse_vertex_and_fragment_shaders(const fs::path& fpath);
void __print_vertex_and_fragment_shaders(const std::vector<std::string>& src_v);
