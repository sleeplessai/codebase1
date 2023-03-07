#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel, uView, uProjection;
out vec3 Normal;
out vec3 FragPos;

void main() {
    FragPos = vec3(uModel * vec4(aPos, 1.0f));
    //Normal = aNormal;
    Normal = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0f);
}
