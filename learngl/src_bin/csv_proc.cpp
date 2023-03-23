#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

#include <fmt/core.h>
#include "kit/csv_proc.h"

namespace kit {

std::vector<std::string> __split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream token_stream(s);
    for (std::string token; std::getline(token_stream, token, delimiter); ) {
        tokens.push_back(token);
    }
    return tokens;
}
void __remove_space(std::string& s) {
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
}

CsvDatabase::CsvDatabase(std::string_view path, std::string_view name) : path(path), name(name) {
};

bool CsvDatabase::open() {
    std::ifstream ifs;
    ifs.open(path.data());
    if (!ifs.is_open()) {
        fmt::print("Failed to open file {}.\n", path);
        return false;
    }
    fmt::print("CsvDatabase {} opened.\n", name);
    std::string head;
    std::getline(ifs, head);
    __remove_space(head);
    fmt::print("{}\n", head);

    auto vv = __split(head, ',');
    for (auto& v : vv) {
        auto var = __split(v, ':');
        header.emplace_back(var.at(0), var.at(1));
        // fmt::print("{} => {}\n", std::get<0>(header.back()), std::get<1>(header.back()));
    }
    return true;
}

void CsvDatabase::buffer() {
    std::ifstream ifs;
    ifs.open(path.data());
    std::string data;
    std::getline(ifs, data);

#define __Meta_proc_block \
    record.at(record_idx).push_back(v); \
    id.push_back(x.id); \
    type.push_back(meta_type); \
    offset.push_back(x.offset); \
    iter += x.offset;

#define __Glm_vec2_initializer {std::stof(*iter), std::stof(*(iter+1))}
#define __Glm_vec3_initializer {std::stof(*iter), std::stof(*(iter+1)), std::stof(*(iter+2))}
#define __Glm_vec4_initializer {std::stof(*iter), std::stof(*(iter+1)), std::stof(*(iter+2)), std::stof(*(iter+3))}
#define __Glm_vec_initializer(__dim) __Glm_vec##__dim##_initializer

#define __Glm_vec_proc_if_block(__type, __dim) if (meta_type == #__type#__dim) { \
    Var<glm::__type##__dim> x{meta_id, __dim}; \
    glm::__type##__dim v __Glm_vec_initializer(__dim); \
    __Meta_proc_block \
}

#define __Literal_proc_if_block(__type, __std_stox, __offset) if (meta_type == #__type) { \
    Var<__type> x{meta_id, __offset}; \
    __type v = __std_stox(*iter); \
    __Meta_proc_block \
}

#define __String_proc_block if (meta_type == "string" || meta_type == "str") { \
    Var<std::string> x{meta_id, 1}; \
    std::string v = *iter; \
    __Meta_proc_block \
}

    for (int record_idx = 0; std::getline(ifs, data); ++record_idx) {
        __remove_space(data);
        auto _data = __split(data, ',');
        auto iter = _data.begin();
        record.emplace_back(Record{});

        for (const auto& [meta_id, meta_type] : header) {
            // fmt::print("***parsing: {}\n***type_is: {}\n", meta_id, meta_type);
            __String_proc_block else
            __Literal_proc_if_block(int, std::stoi, 1) else
            __Literal_proc_if_block(float, std::stof, 1) else
            __Glm_vec_proc_if_block(vec, 2) else
            __Glm_vec_proc_if_block(vec, 3) else
            __Glm_vec_proc_if_block(vec, 4) else
            // if (meta_type == "add_new_type_here") {} else
            {
                fmt::print("Unregistered type {}.\n", meta_type);
            }

            if (iter == _data.end()) break;
        }
    }
}

}

int main() {
    //kit::CsvDatabase db1{"../2.lighting/assets/point_light_attenuation.csv", "point_light"};
    //if (db1.open()) db1.buffer();

    kit::CsvDatabase db2{"../2.lighting/assets/phong_materil_samples.csv", "phong_material"};
    if (db2.open()) db2.buffer();

    //kit::CsvDatabase db3{"../2.lighting/assets/gl_spotlight_smoothness.csv", "spotlight_smoothness"};
    //if (db3.open()) db3.buffer();

    //kit::CsvDatabase db0{"../2.lighting/assets/test.csv", "Test"};
    //if (db0.open()) db0.buffer();

    return 0;
}
