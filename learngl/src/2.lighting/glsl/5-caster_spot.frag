#version 330 core

uniform vec3 view_pos;

in vec3 FragPos;
in vec3 Norm;
in vec2 TexCoords;

out vec4 FragColor;

struct Material {
    sampler2D diffuse, specular;
    float shininess;
};
struct SpotLight {
    vec3 ambient, diffuse, specular, position, direction;
    float cutoff;
};

uniform Material material;
uniform SpotLight light;


void main() {

    vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));

    vec3 norm_dir = normalize(Norm);
    vec3 light_dir = normalize(light.position - FragPos);
    float diff = max(dot(light_dir, norm_dir), 0.f);
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));

    vec3 reflect_dir = normalize(reflect(-light_dir, norm_dir));
    vec3 view_dir = normalize(view_pos - FragPos);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.f), material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));

    float theta = dot(light_dir, normalize(-light.direction));

    if (theta > light.cutoff) {
        FragColor = vec4(ambient + diffuse + specular, 1.f);
    } else {
        FragColor = vec4(ambient, 1.f);
    }
}

