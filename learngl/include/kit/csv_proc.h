#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <glm/glm.hpp>
#include <fmt/core.h>


namespace kit {

#define __Supported_types \
    std::string, int, int64_t, float, double, \
    glm::vec4, glm::vec3, glm::vec2

template<typename T, typename... Types>
concept __type_in_types = (std::is_same_v<T, Types> || ...);

template<typename T>
concept __var_data_type = __type_in_types<T, __Supported_types>;

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

struct CsvDatabase {
    CsvDatabase(const CsvDatabase&) = delete;
    CsvDatabase& operator=(const CsvDatabase&) = delete;

    CsvDatabase(const std::string& path, const std::string& name) : path(path), name(name) {
    }
    struct Var {
        std::string id{};
        int size{0};
        std::variant<__Supported_types> value;
    };
    using Record_t = std::vector<Var>;
    using Meta_t = struct { std::string id, type; };
    using LookupTable_t = std::unordered_map<std::string, size_t>;
    using __Var_Variant_t = std::variant<__Supported_types>;

    std::string name{}, path{};
    bool is_opened{false};
    std::vector<Meta_t> meta;
    std::vector<Record_t> record;
    LookupTable_t lut;

    constexpr static auto __Meta_proc_func = [](Var& var_lr, auto stox_func, auto iter, std::string const& id, int size) {
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

    bool open() {
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

    void buffer() {
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

    void show_meta() const noexcept {
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

    std::vector<std::string> keys() const noexcept {
        std::vector<std::string> keys;
        for (Meta_t const& m : meta) keys.push_back(m.id);
        return keys;
    }

    Record_t& pick(size_t index) {
        return record.at(index);
    }

    Var& query(size_t index, const std::string& key = {}) {
        CsvDatabase::Record_t& rec = pick(index);
        size_t var_index = lut[key];
        return rec.at(var_index);
    }
}; // struct CsvDatabase

} // namespace kit

