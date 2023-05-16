#include <iostream>
#include "kit/csv_proc.h"


int main() {
    std::clog << "***Test Only, Please Ingnore~\n";
    std::array DB = {
        kit::CsvDatabase{"assets/common/phong_material_samples.csv", "phong_material_samples"},
        kit::CsvDatabase{"assets/common/gl_spotlight_smoothness.csv", "gl_spotlight_smoothness"},
        kit::CsvDatabase{"assets/common/point_light_attenuation.csv", "point_light_attenuation"},
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
    std::clog << std::get<float>(DB[1].query(4, "InnerCutoff").value) << std::endl;
    std::clog << std::get<float>(DB[1].query(1, "Epsilon").value) << std::endl;

    std::clog << std::get<glm::vec3>(DB[2].query(3, "ConstantLinearQuadratic").value).z << std::endl;
    std::clog << std::get<glm::vec3>(DB[2].query(5, "ConstantLinearQuadratic").value).x << std::endl;

    return 0;
}
