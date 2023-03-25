#define __Deprecated_codes_

#include <glm/glm.hpp>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>


namespace kit {

#define __Supported_types int,float,std::string,glm::vec4,glm::vec3,glm::vec2

template<typename T, typename... Types>
concept __type_in_types = (std::is_same_v<T, Types> || ...);

template<typename T>
concept __var_data_type = __type_in_types<T, __Supported_types>;

struct CsvDatabase {
    CsvDatabase() = default;
    explicit CsvDatabase(std::string_view path, std::string_view name);

    CsvDatabase(CsvDatabase&&) = default;
    CsvDatabase& operator=(CsvDatabase&&) = default;
    CsvDatabase(const CsvDatabase&) = delete;
    CsvDatabase& operator=(const CsvDatabase&) = delete;

    template<__var_data_type T>
    struct Var {
        std::string id{};
        int offset{0};
        explicit Var(std::string_view id, int offset) : id(id), offset(offset) {}
    };
    using Record = std::vector<std::variant<__Supported_types>>;
    using Tss_t = std::tuple<std::string, std::string>;

    std::vector<std::string> id;
    std::vector<int> offset;
    std::vector<std::string> type;
    std::vector<Record> record;

    std::string_view name{}, path{};
    std::vector<Tss_t> header{};

    bool open();
    void buffer();

    template<typename QueryT>
    int query(QueryT key);
    int random_pick();
};


}
