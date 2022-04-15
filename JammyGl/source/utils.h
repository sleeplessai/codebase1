#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace jammygl {
std::string ReadShaderSource(fs::path& path);
void WriteShaderSource(const std::string& source);

}  // namespace jammygl
