#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace jammygl {

enum class Shader_t { Null = -1, Vertex = 0, Fragment = 1 };

std::string ReadShaderSource(const fs::path& path);
std::vector<std::string> ParseShaderSource(const fs::path& path);

void WriteShaderSource(const std::string& source);
void WriteShaderSource(const std::vector<std::string>& source_v);

}  // namespace jammygl
