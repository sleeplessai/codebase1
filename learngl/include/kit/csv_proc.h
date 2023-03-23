#include <glm/glm.hpp>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>


namespace kit {

#define __Supported_types \
    std::string, int, int64_t, float, double, \
    glm::vec4, glm::vec3, glm::vec2

template<typename T, typename... Types>
concept __type_in_types = (std::is_same_v<T, Types> || ...);

template<typename T>
concept __var_data_type = __type_in_types<T, __Supported_types>;

struct CsvDatabase {
    CsvDatabase(const CsvDatabase&) = delete;
    CsvDatabase& operator=(const CsvDatabase&) = delete;

    CsvDatabase(const std::string& path, const std::string& name);

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

    bool open();
    void buffer();
    void show_meta() const noexcept;
    std::vector<std::string> keys() const noexcept;

    Record_t& pick(size_t);
    Var& query(size_t index, const std::string& key = {});

};


}
