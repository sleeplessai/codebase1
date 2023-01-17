#version 330 core
layout (location = 0) in vec3 vPos;
out vec4 vColor; // pass to fragment shader

uniform vec4 uColor;

void main() {
    gl_Position = vec4(vPos, 1.0f);
    vColor = uColor;
}
