#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <string>
#include <type_traits>
#include <variant>

#include <fmt/core.h>
#include "kit/csv_proc.h"


namespace kit {

std::vector<std::string> __split(const std::string& s, char delim) {
    std::vector<std::string> res;
    for (auto token : std::ranges::split_view{s, delim}) {
        res.emplace_back(token.begin(), token.end());
    }
    return res;
}

std::string __remove_space(std::string& s) {
    std::string ns{};
    for (auto c : s) {
        if (c == ' ' || c == '\t') continue;
        ns.push_back(c);
    }
    s = std::string{ns};
    return ns;
}

constexpr static auto __Meta_proc_func = [](CsvDatabase::Var& var_lr, auto stox_func, auto iter, std::string const& id, int size) {
    auto value = stox_func(iter);
    var_lr.value = value;
    var_lr.id = id;
    var_lr.size = size;
};
constexpr static auto __str_ctor = [](auto _s) {
    return std::string(*_s);
};
constexpr static auto __int_ctor = [](auto _s)  {
    return std::stoi(*_s);
};
constexpr static auto __float_ctor = [](auto _s) {
    return std::stof(*_s);
};
constexpr static auto __glm_vec4_ctor = [](auto _s) {
    return glm::vec4(std::stof(*_s), std::stof(*(_s+1)), std::stof(*(_s+2)), std::stof(*(_s+3)));
};
constexpr static auto __glm_vec3_ctor = [](auto _s) {
    return glm::vec3(std::stof(*_s), std::stof(*(_s+1)), std::stof(*(_s+2)));
};
constexpr static auto __glm_vec2_ctor = [](auto _s) {
    return glm::vec2(std::stof(*_s), std::stof(*(_s+1)));
};

CsvDatabase::CsvDatabase(const std::string& path, const std::string& name) : path(path), name(name) {
};

bool CsvDatabase::open() {
    std::ifstream ifs(path.data());
    if (!ifs.is_open()) {
        fmt::print("Failed to open file: {}.\n", path);
        return false;
    }
    for (int line = 0; line < 2; ++line) {
        std::string meta_s{};
        std::getline(ifs, meta_s);
        auto tokens = __split(__remove_space(meta_s), ',');
        for (auto ti = tokens.begin(); ti != tokens.end() && !(*ti).empty(); ++ti) {
            if (line == 0) {
                meta.emplace_back(Meta_t{});
                meta.at(ti - tokens.begin()).id = *ti;
                lut[*ti] = ti - tokens.begin();
            } else {
                meta.at(ti - tokens.begin()).type = *ti;
            }
        }
    }
    is_opened = true;
    return true;
}

void CsvDatabase::buffer() {
    std::ifstream ifs(path.data());
    std::string fbuf;
    for (int _ : {0,1}) std::getline(ifs, fbuf);
    fbuf.clear();

    for (int record_idx = 0; std::getline(ifs, fbuf); ++record_idx) {
        auto _rec = __split(__remove_space(fbuf), ',');
        auto iter = _rec.begin();
        record.emplace_back(Record_t{});

        Var _var{};
        bool is_parsed = true;
        for (Meta_t const& m : meta) {
            // fmt::print("**parsing: {}; type_is: {}; value_is: {}\n", m.id, m.type, *iter);
            if (m.id == "string" || m.type == "str") {
                __Meta_proc_func(_var, __str_ctor, iter, m.id, 1);
            } else if (m.type == "int") {
                __Meta_proc_func(_var, __int_ctor, iter, m.id, 1);
            } else if (m.type == "float") {
                __Meta_proc_func(_var, __float_ctor, iter, m.id, 1);
            } else if (m.type == "vec4") {
                __Meta_proc_func(_var, __glm_vec4_ctor, iter, m.id, 4);
            } else if (m.type == "vec3") {
                __Meta_proc_func(_var, __glm_vec3_ctor, iter, m.id, 3);
            } else if (m.type == "vec2") {
                __Meta_proc_func(_var, __glm_vec2_ctor, iter, m.id, 2);
            } else {
                fmt::print("Unregistered type {}\n", m.type);
                is_parsed = false;
            }
            if (is_parsed) {
                record.at(record_idx).push_back(_var);
                iter += _var.size;
            }
        }
    }
}


CsvDatabase::Record_t& CsvDatabase::pick(size_t index) {
    return record.at(index);
}

CsvDatabase::Var& CsvDatabase::query(size_t index, const std::string& key) {
    CsvDatabase::Record_t& rec = pick(index);
    size_t var_index = lut[key];
    return rec.at(var_index);
}

std::vector<std::string> CsvDatabase::keys() const noexcept {
    std::vector<std::string> keys;
    for (Meta_t const& m : meta) keys.push_back(m.id);
    return keys;
}

void CsvDatabase::show_meta() const noexcept {
    if (meta.empty()) {
        std::cout << "\tMeta data is empty, call open() to read.\n";
        return;
    }
    std::cout << "Database " << std::quoted(name) << " meta:\n\t";
    for (const auto& m : meta) {
        std::cout << std::quoted(m.id) << ' ';
    }
    std::putchar('\n');
    std::cout << '\t' << record.size() << " record(s) in total.\n";
    std::cout << "\tGet value by calling `std::get<Type>(Db.query(Index, Key).value)`.\n";
}

}

int main() {
    std::array DB = {
        kit::CsvDatabase{"../2.lighting/assets/phong_material_samples.csv", "phong_material_samples"},
        kit::CsvDatabase{"../2.lighting/assets/gl_spotlight_smoothness.csv", "gl_spotlight_smoothness"},
        kit::CsvDatabase{"../2.lighting/assets/point_light_attenuation.csv", "point_light_attenuation"},
    };

    for (auto& db : DB) {
        if (db.open()) {
            db.buffer();
            db.show_meta();
        }
    }
/*
Database "phong_material_samples" meta:
"Name" "Ambient" "Diffuse" "Specular" "Shininess" 
Database "gl_spotlight_smoothness" meta:
"Theta" "ThetaInDegrees" "InnerCutoff" "InnerCutoffInDegrees" "OuterCutoff" "OuterCutoffInDegrees" "Epsilon" "Intensity" 
Database "point_light_attenuation" meta:
"Range" "ConstantLinearQuadratic" 
*/
    std::cout << std::get<float>(DB[1].query(4, "InnerCutoff").value) << std::endl;
    std::cout << std::get<float>(DB[1].query(1, "Epsilon").value) << std::endl;

    std::cout << std::get<glm::vec3>(DB[2].query(3, "ConstantLinearQuadratic").value).z << std::endl;
    std::cout << std::get<glm::vec3>(DB[2].query(5, "ConstantLinearQuadratic").value).x << std::endl;


    return 0;
}
