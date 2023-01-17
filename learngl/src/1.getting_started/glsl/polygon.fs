#version 330 core
out vec4 outColor;
in vec4 vColor; // obtain from vertex shader

uniform bool uWhite;

void main() {
    vec4 whiteOffset = vec4(vec3(1.0, 1.0, 1.0) - vColor.xyz, 0.0);
    if (uWhite) {
        outColor = vColor + whiteOffset;
    } else {
        outColor = vColor;
    }
}
