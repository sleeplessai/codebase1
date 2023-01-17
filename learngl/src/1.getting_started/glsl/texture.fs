#version 330 core

in vec3 vColor;
in vec2 texCoord;
out vec4 outColor;

uniform sampler2D outTexture;

void main() {
    //outColor = vec4(0.5f, 0.5f, 0.5f, 1.0f);
    outColor = texture(outTexture, texCoord) * vec4(vColor, 1.0f);
}
