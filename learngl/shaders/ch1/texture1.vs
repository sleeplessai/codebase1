#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;

out vec3 vColor;
out vec2 texCoord;

uniform bool uUseMvp;
uniform mat4 uTransform;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

void main() {
    if (uUseMvp) {
        gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0f);
    } else {
        gl_Position = uTransform * vec4(aPos, 1.0f);
    }
    vColor = aColor;
    texCoord = aTexCoord;
    // texCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
