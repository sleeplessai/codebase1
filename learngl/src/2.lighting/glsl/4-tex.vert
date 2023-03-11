#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNorm;
layout (location = 2) in vec2 aTex;

out vec3 Norm;
out vec3 FragPos;
out vec2 TexCoords;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.f);
    Norm = mat3(transpose(inverse(model))) * aNorm;
    FragPos = vec3(model * vec4(aPos, 1.f));
    TexCoords = aTex;
}
