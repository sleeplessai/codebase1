#include <chrono>
#include <cmath>
#include <concepts>
#include <fstream>
#include <iterator>
#include <numbers>
#include <random>
#include <type_traits>

#include <fmt/core.h>
#include <glm/glm.hpp>

namespace glm {
template <typename VecT>
concept vec3_or_vec2 = std::is_same_v<glm::vec3, VecT> || std::is_same_v<glm::vec2, VecT>;
};

template <glm::vec3_or_vec2 VecT, std::floating_point DataT>
VecT&& rand_spherial_coords(const VecT center, const DataT radius) {
    constexpr DataT Pi = std::numbers::pi_v<DataT>;

    DataT seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::mt19937 g(seed);
    std::uniform_real_distribution<DataT> urd(0.f, 2.f * Pi);

    DataT p = urd(g), q = urd(g);
    DataT sin_p = std::sin(p), sin_q = std::sin(q), cos_p = std::cos(p), cos_q = std::cos(q);
    VecT s{};

    if constexpr (std::is_same_v<VecT, glm::vec3>) {
        s.x = sin_p * cos_q;
        s.y = sin_p * sin_q;
        s.z = cos_p;
    } else if constexpr (std::is_same_v<VecT, glm::vec2>) {
        s.x = sin_p;
        s.y = cos_p;
    }
    return std::move(center + radius * s);
}

int main(const int argc, const char** argv) {
    std::string out_path{"assets/cube_pos.inc"};
    std::fstream out_stream(out_path, std::ios::out);

    out_stream << "// generate via `gencubes`\n";
    for (int i = 0; i < std::atoi(argv[1]); ++i) {
        auto o = rand_spherial_coords(glm::vec3{0.f}, 1.f);
        std::string rt_fmt_str = fmt::vformat("glm::vec3({}f,{}f,{}f),\n", fmt::make_format_args(o.x, o.y, o.z));
        out_stream << rt_fmt_str;
    }
    return 0;
}
