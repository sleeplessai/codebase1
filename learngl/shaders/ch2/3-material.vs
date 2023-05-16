#version 330 core

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNorm;

out vec3 Norm;
out vec3 FragPos;

void main() {
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.f);
    Norm = mat3(transpose(inverse(uModel))) * aNorm;
    FragPos = vec3(uModel * vec4(aPos, 1.f));
}
