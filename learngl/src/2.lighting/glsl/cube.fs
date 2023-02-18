#version 330 core

out vec4 outColor;

uniform vec3 uLightColor;
uniform vec3 uObjectColor;

void main() {
    outColor = vec4(uLightColor * uObjectColor, 1.0f);
}
