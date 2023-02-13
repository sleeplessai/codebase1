#version 330 core

out vec4 outColor;

uniform vec4 uLightColor;
uniform vec4 uObjectColor;

void main() {
    outColor = uLightColor * uObjectColor;
}
