#version 330 core

uniform vec3 uViewPos;

in vec3 FragPos;
in vec3 Norm;

out vec4 FragColor;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

struct Light {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 position;
};

uniform Material material;
uniform Light light;


void main() {
    vec3 ambient = material.ambient * light.ambient;

    vec3 norm = normalize(Norm);
    vec3 light_dir = normalize(light.position - FragPos);
    float diff = max(dot(light_dir, norm), 0.f);
    vec3 diffuse = material.diffuse * diff * light.diffuse;

    vec3 reflect_dir = normalize(reflect(-light_dir, norm));
    vec3 view_dir = normalize(uViewPos - FragPos);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.f), material.shininess);
    vec3 specular = material.specular * spec * light.specular;

    FragColor = vec4(vec3(ambient + diffuse + specular), 1.f);
}
