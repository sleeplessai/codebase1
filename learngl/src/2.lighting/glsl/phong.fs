#version 330 core

out vec4 outColor;
in vec3 Normal;
in vec3 FragPos;

uniform vec3 uLightColor;
uniform vec3 uObjectColor;
uniform vec3 uLightPos;
uniform vec3 uViewPos;

void main() {
    //outColor = uLightColor * uObjectColor;
    // ambient
    float ambientStrength = 0.1f;
    vec3 ambient = ambientStrength * uLightColor;

    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(uLightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0f);
    vec3 diffuse = diff * uLightColor;

    // specular
    float specularStrength = 0.5f;
    vec3 viewDir = normalize(uViewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0f), 256);
    vec3 specular = specularStrength * spec * uLightColor;

    // combine
    vec3 result = (ambient + diffuse + specular) * uObjectColor;
    outColor = vec4(result, 1.0f);
}
